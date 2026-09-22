#!/usr/bin/env python3
"""Small human-readable collector for the frozen T4 v50 result directory."""
import csv
import glob
import json
import os
import sys


def one(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def read_one(path):
    if not path:
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    summary = read_one(one(os.path.join(root, "*reference_fas_summary_with_map.csv")))
    if summary is None:
        summary = read_one(one(os.path.join(root, "*reference_fas_summary.csv")))
    audit = read_one(one(os.path.join(root, "*science_audit.csv")))
    cert = read_one(one(os.path.join(root, "*benchmark_certificate.csv")))
    trajectory_path = one(os.path.join(root, "*fas_discrete_error_trajectory.csv"))
    trajectory = []
    if trajectory_path:
        with open(trajectory_path, newline="") as f:
            trajectory = list(csv.DictReader(f))

    payload = {
        "result_dir": os.path.abspath(root),
        "certificate": cert,
        "summary": summary,
        "science_audit": audit,
        "trajectory": trajectory,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if summary:
        print("\nT4 headline")
        keys = [
            "cycles", "reached_target", "initial_residual_p", "final_residual_p",
            "geomean_residual_contraction", "initial_discrete_reference_error",
            "final_discrete_reference_error", "geomean_discrete_error_contraction",
            "reference_certified", "reference_updates", "reference_residual_p",
            "observed_heat_actions", "expected_heat_actions", "heat_count_match",
            "map_l2_physical", "oscillatory_displacement_gain",
            "oscillatory_displacement_cosine_alignment", "det_min", "tangled_area",
        ]
        for k in keys:
            if k in summary:
                print(f"{k}={summary[k]}")
    if audit:
        print("\nT4 map/spectral audit")
        keys = [
            "map_l2_physical", "map_linf_physical", "jacobian_frobenius_l2",
            "det_l2_error", "det_min", "det_max", "tangled_area",
            "monge_defect_physical", "rms_conditional_spread_physical",
            "exact_oscillatory_displacement_rms_physical",
            "numerical_displacement_rms_physical", "oscillatory_displacement_gain",
            "oscillatory_displacement_cosine_alignment", "barycentric_support_escape_mass",
            "map_denominator_nonfinite_count",
        ]
        for k in keys:
            if k in audit:
                print(f"{k}={audit[k]}")


if __name__ == "__main__":
    main()
