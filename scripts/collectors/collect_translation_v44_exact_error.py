#!/usr/bin/env python3
import csv
from pathlib import Path

RESULT_ROOT = Path("/rds/general/user/aw1225/ephemeral/ot_project_profile/results")
JOBIDS = Path("notes_FAS/translation_v44_exact_error_S0_S13_jobids_2026-09-08.csv")
PREFIX = "run_translation_exact_error_trajectory_v44"
EXPECTED_SHA = "ddb492662317c98e2add53983524e89e6cada1974a569b3e421ba1cb489cd5cc"

FIELDS = [
    "S", "M", "jobid", "source_match", "checkpoint", "cycle",
    "production_residual_p", "residual_ratio_to_entry", "map_l2_physical",
    "map_l2_ratio_to_entry", "map_linf_physical", "map_linf_ratio_to_entry",
    "det_min", "det_max", "tangled_area", "frozen_max", "diagnostic_seconds",
    "top_cycles", "reached_target", "iteration_seconds", "total_seconds",
    "heat_count_match", "stderr_lines",
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
for manifest_row in csv.DictReader(JOBIDS.open()):
    run_dir = RESULT_ROOT / manifest_row["run_id"]
    summary = read_last(run_dir / f"{PREFIX}_fmg_summary.csv")
    trajectory_path = run_dir / f"{PREFIX}_exact_error_trajectory.csv"
    trajectory = list(csv.DictReader(trajectory_path.open())) if trajectory_path.exists() else []
    stderr = run_dir / "stderr_forced.txt"
    common = {
        "S": manifest_row["S"], "M": manifest_row["M"], "jobid": manifest_row["jobid"],
        "source_match": int(source_hash(run_dir / "source_sha256.txt") == EXPECTED_SHA),
        "stderr_lines": len(stderr.read_text().splitlines()) if stderr.exists() else "NO_STDERR",
    }
    if not trajectory:
        row = {key: "" for key in FIELDS}
        row.update(common)
        row["checkpoint"] = "NO_TRAJECTORY"
        print(",".join(str(row[key]) for key in FIELDS))
        continue
    for checkpoint in trajectory:
        row = {key: "" for key in FIELDS}
        row.update(common)
        for key in FIELDS:
            if key in summary:
                row[key] = summary[key]
            if key in checkpoint:
                row[key] = checkpoint[key]
        print(",".join(str(row[key]) for key in FIELDS))