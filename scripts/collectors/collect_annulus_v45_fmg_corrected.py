#!/usr/bin/env python3
import csv
from pathlib import Path

RESULT_ROOT = Path("/rds/general/user/aw1225/ephemeral/ot_project_profile/results")
JOBIDS = Path("notes_FAS/annulus_v45_fmg_corrected_N0_N3_jobids_2026-09-08.csv")
PREFIX = "run_annulus_fmg_corrected_v45"
EXPECTED_SHA = "784682d55abc22f361f8b849ced6a3f6b005e42fb8086cf4a50483e691746901"

FIELDS = [
    "N", "jobid", "source_match", "B920_entry_residual", "top_cycles",
    "reached_target", "residual_p", "iteration_seconds", "total_seconds",
    "bottom_prep_seconds", "middle_prep_seconds", "entry_check_seconds",
    "prep_seconds", "top_solve_seconds", "bottom_heat_actions",
    "middle_heat_actions", "top_heat_actions", "expected_bottom_heat_actions",
    "expected_middle_heat_actions", "expected_top_heat_actions", "heat_count_match",
    "full_residual_p", "full_residual_q", "source_marginal_l2_physical",
    "target_marginal_l2_physical", "map_l2_physical", "map_linf_physical",
    "radial_l2_physical", "tangential_l2_physical", "polar_jacobian_l2",
    "det_l2", "det_min", "det_max", "tangled_area", "raw_hole_escape_mass",
    "buffered_hole_escape_mass", "raw_outer_escape_mass",
    "buffered_outer_escape_mass", "inner_map_l2_physical",
    "middle_map_l2_physical", "outer_map_l2_physical", "monge_defect_physical",
    "rms_conditional_spread_physical", "map_denominator_min",
    "map_denominator_n_le_0", "map_denominator_n_le_tiny", "trace_cycles",
    "trace_residuals", "stderr_lines",
]

def read_last(path):
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open()))
    return rows[-1] if rows else {}

def source_hash(path):
    if not path.exists():
        return ""
    parts = path.read_text().split()
    return parts[0] if parts else ""

print(",".join(FIELDS))
for manifest in csv.DictReader(JOBIDS.open()):
    run_dir = RESULT_ROOT / manifest["run_id"]
    summary = read_last(run_dir / f"{PREFIX}_fmg_summary.csv")
    audit = read_last(run_dir / f"{PREFIX}_fmg_to_tol_science_audit.csv")
    trace_path = run_dir / f"{PREFIX}_fmg_to_tol_trace.csv"
    trace = list(csv.DictReader(trace_path.open())) if trace_path.exists() else []
    stderr = run_dir / "stderr_forced.txt"
    row = {key: "" for key in FIELDS}
    row.update({"N": manifest["N"], "jobid": manifest["jobid"]})
    row["source_match"] = int(source_hash(run_dir / "source_sha256.txt") == EXPECTED_SHA)
    row["stderr_lines"] = len(stderr.read_text().splitlines()) if stderr.exists() else "NO_STDERR"
    if not summary:
        row["B920_entry_residual"] = "NO_SUMMARY"
    else:
        for key in FIELDS:
            if key in summary:
                row[key] = summary[key]
            if key in audit:
                row[key] = audit[key]
        row["trace_cycles"] = "|".join(item.get("cycle", "") for item in trace)
        row["trace_residuals"] = "|".join(item.get("residual_p", "") for item in trace)
    print(",".join(str(row[key]) for key in FIELDS))