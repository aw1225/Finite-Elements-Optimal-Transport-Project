#!/usr/bin/env python3
import csv, glob, json, os, sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit("usage: collect_t3_v49.py RESULT_DIR")
root = Path(sys.argv[1])
if not root.is_dir():
    raise SystemExit(f"not a directory: {root}")

patterns = [
    "*benchmark_certificate.csv",
    "*mesh_metadata.csv",
    "*pushforward_moments.csv",
    "*reference_fas_summary_with_map.csv",
    "*reference_fas_summary.csv",
    "*fas_discrete_error_trajectory.csv",
    "*centre_bins.csv",
    "*science_audit.csv",
    "*production_heat_counts.csv",
    "*reference_trace.csv",
]
report = {"result_dir": str(root), "files": {}}
for pat in patterns:
    for fn in sorted(root.glob(pat)):
        with fn.open(newline="") as f:
            rows = list(csv.DictReader(f))
        report["files"][fn.name] = rows

sha = root / "source_sha256.txt"
if sha.exists():
    report["source_sha256"] = sha.read_text().strip()

print(json.dumps(report, indent=2))
