#!/usr/bin/env python3
"""Mine existing T1--T5 CSV outputs for safeguard/plateau diagnostics.
No Firedrake import and no reruns are required.
"""
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get(
    "T4_RUN_ROOT",
    "/rds/general/user/aw1225/ephemeral/ot_project_profile",
))
RESULTS = ROOT / "results"
OUTDIR = Path(os.environ.get(
    "T4_SAFETY_OUTDIR",
    str(Path.home() / "ot_project" / "notes_FAS" / "postreview_safety_mining"),
))

KEY_RE = re.compile(
    r"(safe_log|floor|clamp|nonpositive|non_positive|nonfinite|denominator|variance|residual_[pq]|reference_(change|certified|residual)|stable_checks|gauge_shift)",
    re.I,
)


def classify(path: str):
    s = path.lower()
    rules = [
        ("T1", ["t1_", "nonlinear_crescent"]),
        ("T2", ["t2_", "split_target", "split_map"]),
        ("T3", ["t3_", "disk_to_annulus"]),
        ("T4", ["t4_", "high_frequency_square"]),
        ("T5", ["t5_", "thin_neck"]),
    ]
    for label, pats in rules:
        if any(p in s for p in pats):
            return label
    return None


def as_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    if not RESULTS.exists():
        raise SystemExit(f"Results root not found: {RESULTS}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in RESULTS.rglob("*.csv"):
        test = classify(str(path))
        if test is None:
            continue
        n = path.name.lower()
        if not any(tag in n for tag in ["science_audit", "reference_trace", "summary", "audit", "conditional"]):
            continue
        try:
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    selected = {k: v for k, v in row.items() if k and KEY_RE.search(k) and v not in (None, "")}
                    if not selected:
                        continue
                    base = {
                        "test": test,
                        "file": str(path),
                        "row_index": i,
                    }
                    base.update(selected)
                    rows.append(base)
        except Exception as e:
            print(f"WARN could not parse {path}: {e}", file=sys.stderr)

    # Union schema.
    fields = ["test", "file", "row_index"]
    seen = set(fields)
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    inv = OUTDIR / "safety_audit_inventory.csv"
    with inv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    # Aggregate whether safeguard-like quantities are ever non-zero by test/key.
    agg = {}
    for r in rows:
        test = r["test"]
        for k, v in r.items():
            if k in {"test", "file", "row_index"}:
                continue
            x = as_float(v)
            if x is None or not math.isfinite(x):
                continue
            a = agg.setdefault((test, k), {"count": 0, "nonzero": 0, "min": x, "max": x, "max_abs": abs(x)})
            a["count"] += 1
            a["nonzero"] += int(x != 0.0)
            a["min"] = min(a["min"], x)
            a["max"] = max(a["max"], x)
            a["max_abs"] = max(a["max_abs"], abs(x))
    summary = []
    for (test, key), a in sorted(agg.items()):
        summary.append({"test": test, "field": key, **a})
    sout = OUTDIR / "safety_audit_summary.csv"
    with sout.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test", "field", "count", "nonzero", "min", "max", "max_abs"])
        w.writeheader(); w.writerows(summary)

    print("PASTE_SAFETY_BEGIN")
    print(f"inventory_rows={len(rows)}")
    print(f"inventory_csv={inv}")
    print(f"summary_csv={sout}")
    for test in ["T1", "T2", "T3", "T4", "T5"]:
        interesting = [x for x in summary if x["test"] == test and re.search(r"safe_log|floor|clamp|nonpositive|nonfinite|denominator", x["field"], re.I)]
        if not interesting:
            print(f"{test}: no_matching_safety_fields_found")
            continue
        nz = [(x["field"], x["nonzero"], x["max_abs"]) for x in interesting if x["nonzero"]]
        print(f"{test}: safety_fields={len(interesting)} nonzero_fields={len(nz)}")
        for item in nz[:20]:
            print(f"  NONZERO {item[0]} count={item[1]} max_abs={item[2]}")
    print("PASTE_SAFETY_END")


if __name__ == "__main__":
    main()
