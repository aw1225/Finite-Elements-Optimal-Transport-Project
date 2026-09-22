#!/usr/bin/env python
from __future__ import print_function
import argparse
import csv
import glob
import math
import os
import sys


def read_rows(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def one(pattern):
    hits = glob.glob(pattern)
    if len(hits) != 1:
        raise RuntimeError("Expected exactly one match for %r, got %r" % (pattern, hits))
    return hits[0]


def ffloat(x):
    return float(x)


def pair_name(coarse, fine):
    return "B%d_B%d" % (int(coarse), int(fine))


def load_run(result_dir):
    trace_path = one(os.path.join(result_dir, "*_fas_fixed_cycles_trace.csv"))
    summary_path = one(os.path.join(result_dir, "*_fas_fixed_cycles_summary.csv"))
    hierarchy_path = one(os.path.join(result_dir, "*_hierarchy.csv"))
    audit_path = one(os.path.join(result_dir, "*_fas_fixed_cycles_science_audit.csv"))

    trace = read_rows(trace_path)
    summary = read_rows(summary_path)[0]
    hierarchy = read_rows(hierarchy_path)
    audit = read_rows(audit_path)[0]

    bridges = sorted(int(float(r["bridge_n"])) for r in hierarchy)
    if len(bridges) != 2 or bridges[1] != 2 * bridges[0]:
        raise RuntimeError("Not a two-level factor-two hierarchy: %r" % (bridges,))
    if any(int(float(r.get("heat_substeps", "32"))) != 32 for r in hierarchy):
        raise RuntimeError("Hierarchy is not m=32")
    if len(trace) != 6 or [int(r["cycle"]) for r in trace] != list(range(6)):
        raise RuntimeError("Expected cycle rows 0..5 in %s" % trace_path)

    residuals = [float(r["residual_p"]) for r in trace]
    rho = (residuals[-1] / residuals[0]) ** (1.0 / 5.0)
    return {
        "result_dir": result_dir,
        "pair": pair_name(bridges[0], bridges[1]),
        "coarse_n": bridges[0],
        "fine_n": bridges[1],
        "residuals": residuals,
        "rho": rho,
        "reached_target": int(float(trace[-1]["reached_target"])),
        "iteration_seconds": float(trace[-1]["iteration_seconds"]),
        "map_l2_physical": float(audit.get("map_l2_physical", "nan")),
        "monge_defect_physical": float(audit.get("monge_defect_physical", "nan")),
        "det_min": float(audit.get("det_min", "nan")),
        "tangled_area": float(audit.get("tangled_area", "nan")),
        "trace_path": trace_path,
        "summary_path": summary_path,
        "audit_path": audit_path,
    }


def load_baseline(path):
    rows = read_rows(path)
    by = {}
    for r in rows:
        by.setdefault(r["pair"], {})[int(r["cycle"])] = float(r["residual_p"])
    out = {}
    for pair, d in by.items():
        if sorted(d) != list(range(6)):
            raise RuntimeError("Baseline %s does not have cycles 0..5" % pair)
        vals = [d[k] for k in range(6)]
        out[pair] = {
            "residuals": vals,
            "rho": (vals[-1] / vals[0]) ** (1.0 / 5.0),
        }
    return out


def rel_diff(a, b):
    den = 0.5 * (abs(a) + abs(b))
    return abs(a - b) / den if den > 0.0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dirs", nargs="+", help="Two m=32 result directories")
    ap.add_argument("--baseline", default="canonical_m16_rq1_baseline.csv")
    ap.add_argument("--out", default="affine_fas_m32_refinement_control_summary.csv")
    args = ap.parse_args()

    runs = [load_run(d) for d in args.result_dirs]
    runs.sort(key=lambda r: r["coarse_n"])
    primary = [r for r in runs if r["pair"] in ("B230_B460", "B460_B920")]
    if len(primary) != 2:
        raise RuntimeError("Primary analysis needs B230_B460 and B460_B920")

    baseline = load_baseline(args.baseline) if os.path.exists(args.baseline) else {}

    with open(args.out, "w") as f:
        fields = [
            "pair", "m", "r0", "r1", "r2", "r3", "r4", "r5",
            "rho_0_to_5", "m16_rho_0_to_5", "rho_change_vs_m16_rel",
            "reached_target_after_5", "iteration_seconds",
            "map_l2_physical", "monge_defect_physical", "det_min", "tangled_area",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in runs:
            b = baseline.get(r["pair"])
            w.writerow({
                "pair": r["pair"],
                "m": 32,
                "r0": r["residuals"][0], "r1": r["residuals"][1],
                "r2": r["residuals"][2], "r3": r["residuals"][3],
                "r4": r["residuals"][4], "r5": r["residuals"][5],
                "rho_0_to_5": r["rho"],
                "m16_rho_0_to_5": "" if b is None else b["rho"],
                "rho_change_vs_m16_rel": "" if b is None else rel_diff(r["rho"], b["rho"]),
                "reached_target_after_5": r["reached_target"],
                "iteration_seconds": r["iteration_seconds"],
                "map_l2_physical": r["map_l2_physical"],
                "monge_defect_physical": r["monge_defect_physical"],
                "det_min": r["det_min"],
                "tangled_area": r["tangled_area"],
            })

    a, b = primary
    rho_spread = rel_diff(a["rho"], b["rho"])
    norm_a = [x / a["residuals"][0] for x in a["residuals"]]
    norm_b = [x / b["residuals"][0] for x in b["residuals"]]
    curve_spread = max(rel_diff(x, y) for x, y in zip(norm_a[1:], norm_b[1:]))

    print("=== m=32 affine FAS refinement control ===")
    for r in primary:
        print("%s rho_0_to_5=%.9g r5=%.9g reached_target=%d" % (
            r["pair"], r["rho"], r["residuals"][-1], r["reached_target"]))
        if r["pair"] in baseline:
            brho = baseline[r["pair"]]["rho"]
            print("  m16 rho=%.9g  relative change=%.3f%%" % (
                brho, 100.0 * rel_diff(r["rho"], brho)))

    print("m32 cross-refinement rho relative spread = %.3f%%" % (100.0 * rho_spread))
    print("m32 max normalized-curve relative spread (cycles 1..5) = %.3f%%" % (100.0 * curve_spread))
    print("Preregistered descriptive thresholds: rho spread <=5%%; normalized-curve spread <=10%%.")
    if rho_spread <= 0.05 and curve_spread <= 0.10:
        print("INTERPRETATION: supports approximately refinement-insensitive FAS contraction at m=32 over B230->B460 and B460->B920.")
    else:
        print("INTERPRETATION: temporal refinement exposes a material bridge-refinement dependence; do not claim refinement-insensitivity at m=32.")
    print("Wrote %s" % args.out)


if __name__ == "__main__":
    main()
