#!/usr/bin/env python3
import csv
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get(
    "T4_RUN_ROOT",
    "/rds/general/user/aw1225/ephemeral/ot_project_profile",
))


def latest_result():
    items = list((ROOT / "results").glob("t4_supplemental_diagnostics_v1_*"))
    if not items:
        raise SystemExit("No t4_supplemental_diagnostics_v1_* result directory found")
    return max(items, key=lambda p: p.stat().st_mtime)


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_result()
    p = d / "supplemental_summary.json"
    if not p.exists():
        raise SystemExit(f"Missing {p}")
    s = json.loads(p.read_text())
    print("PASTE_BACK_BEGIN")
    for k in [
        "source_sha256", "reference_certified", "reference_updates",
        "reference_residual_p", "reference_change_rel",
        "E_map_reference_exact", "E_map_fas1_reference", "E_map_fas1_exact",
        "E_map_fas1_reference_over_reference_exact",
        "smooth3_heat_count_vector", "smooth3_residual_p", "smooth3_state_error",
        "smooth3_mode6_probe_rss", "smooth3_mode3_probe_rss",
        "fas1_heat_count_vector", "fas1_residual_p", "fas1_state_error",
        "fas1_mode6_probe_rss", "fas1_mode3_probe_rss",
        "fas_over_smooth3_residual_ratio", "fas_over_smooth3_state_error_ratio",
        "fas_over_smooth3_mode6_ratio", "fas_over_smooth3_mode3_ratio",
    ]:
        print(f"{k}={s.get(k)}")
    for a in s.get("adjoint", []):
        print(
            f"adjoint_B{a.get('bridge_n')}_mass_global={a.get('mass_global_probe_relative_defect')} "
            f"median={a.get('mass_active_pair_median_relative_defect')} "
            f"p95={a.get('mass_active_pair_p95_relative_defect')} "
            f"max={a.get('mass_active_pair_max_relative_defect')}"
        )
    print("baseline_reproduction_checks=" + json.dumps(s.get("baseline_reproduction_checks", {}), sort_keys=True))
    print("PASTE_BACK_END")
    print("RESULT_DIR=" + str(d))


if __name__ == "__main__":
    main()
