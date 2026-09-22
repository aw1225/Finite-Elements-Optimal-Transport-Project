#!/usr/bin/env python3
"""Collect stopping-endpoint and matched-update m-sensitivity comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPERIMENT = "run_affine_mcheck_B460_v1"
EXPECTED_M = {8, 16, 32}
REQUIRED_SUFFIXES = [
    "_hierarchy.csv", "_fine_to_tol_trace.csv", "_heat_counts.csv",
    "_cost_profile.csv", "_fine_to_tol_science_audit.csv",
    "_fine_to_tol_audit_heat_counts.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def one_row(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def discover(results_root: Path) -> list[Path]:
    return sorted(
        p for p in results_root.glob("affine_mcheck_m*_B460_*")
        if p.is_dir()
    )


def load_run(run_dir: Path) -> dict:
    required = [run_dir / "source_used.py", run_dir / "source_sha256.txt",
                run_dir / "environment_manifest.txt", run_dir / "pbs_environment.txt",
                run_dir / "petsc_log_view.txt", run_dir / "stdout_forced.txt",
                run_dir / "stderr_forced.txt"]
    required += [run_dir / f"{EXPERIMENT}{suffix}" for suffix in REQUIRED_SUFFIXES]
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir}: missing {', '.join(missing)}")

    hierarchy = one_row(run_dir / f"{EXPERIMENT}_hierarchy.csv")
    trace = read_rows(run_dir / f"{EXPERIMENT}_fine_to_tol_trace.csv")
    audit = one_row(run_dir / f"{EXPERIMENT}_fine_to_tol_science_audit.csv")
    if not trace:
        raise ValueError(f"{run_dir}: empty trace")
    m = int(hierarchy["m"])
    if m not in EXPECTED_M:
        raise ValueError(f"{run_dir}: unexpected m={m}")
    if any(int(row["m"]) != m for row in trace) or int(audit["m"]) != m:
        raise ValueError(f"{run_dir}: inconsistent m reporting")
    source_hash_recorded = (run_dir / "source_sha256.txt").read_text().split()[0]
    source_hash_actual = sha256(run_dir / "source_used.py")
    if source_hash_recorded != source_hash_actual:
        raise ValueError(f"{run_dir}: source hash mismatch")
    final = trace[-1]
    return {"dir": run_dir, "m": m, "hierarchy": hierarchy, "trace": trace,
            "audit": audit, "final": final, "source_sha256": source_hash_actual}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true",
                        help="permit collection of an m=16 smoke run before all m values exist")
    args = parser.parse_args()
    runs = [load_run(path) for path in discover(args.results_root)]
    if not runs:
        raise SystemExit("no affine_mcheck result directories found")
    by_m = {}
    for run in runs:
        if run["m"] in by_m:
            raise SystemExit(f"multiple result directories for m={run['m']}; select a clean results root")
        by_m[run["m"]] = run
    present = set(by_m)
    if not args.allow_partial and present != EXPECTED_M:
        raise SystemExit(f"expected m={sorted(EXPECTED_M)}, found {sorted(present)}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    endpoints = []
    for m in sorted(by_m):
        run = by_m[m]
        final, audit, h = run["final"], run["audit"], run["hierarchy"]
        endpoints.append({
            "m": m, "tau_physical": h["tau_physical"], "dt_physical": h["dt_physical"],
            "updates_at_stop": final["updates"], "residual_p_at_stop": final["residual_p"],
            "target_residual": final["target_residual"], "reached_target": final["reached_target"],
            "iteration_seconds": final["iteration_seconds"], "total_seconds": final["total_seconds"],
            "map_l2_physical": audit.get("map_l2_physical", ""),
            "monge_defect_physical": audit.get("monge_defect_physical", ""),
            "det_min": audit.get("det_min", ""), "tangled_area": audit.get("tangled_area", ""),
            "source_sha256": run["source_sha256"], "result_directory": run["dir"].name,
        })
    write_rows(args.output_dir / "affine_mcheck_v1_stopping_endpoints.csv", endpoints)

    update_sets = [{int(row["updates"]) for row in run["trace"]} for run in by_m.values()]
    common_updates = sorted(set.intersection(*update_sets))
    matched = []
    for update in common_updates:
        for m in sorted(by_m):
            row = next(row for row in by_m[m]["trace"] if int(row["updates"]) == update)
            matched.append({
                "updates": update, "m": m, "tau_physical": row["tau_physical"],
                "dt_physical": row["dt_physical"], "residual_p": row["residual_p"],
                "reached_target": row["reached_target"],
                "iteration_seconds": row["iteration_seconds"], "total_seconds": row["total_seconds"],
                "result_directory": by_m[m]["dir"].name,
            })
    write_rows(args.output_dir / "affine_mcheck_v1_matched_updates.csv", matched)
    report = {
        "experiment": EXPERIMENT, "m_present": sorted(present),
        "complete_design": present == EXPECTED_M,
        "common_updates": common_updates,
        "comparison_outputs": ["affine_mcheck_v1_stopping_endpoints.csv", "affine_mcheck_v1_matched_updates.csv"],
        "runs": [{"m": r["m"], "result_directory": r["dir"].name,
                  "source_sha256": r["source_sha256"]} for r in runs],
    }
    (args.output_dir / "affine_mcheck_v1_collection_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

