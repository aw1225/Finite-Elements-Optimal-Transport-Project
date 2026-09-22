from __future__ import annotations
import csv, math, os
from pathlib import Path

ROOT = Path("/rds/general/user/aw1225/ephemeral/ot_project_profile/results")
JOBS = Path("notes_FAS/translation_v46_discrete_reference_jobids_2026-09-08.csv")
PREFIX = "run_translation_discrete_reference_validation_v46"
OUT = Path("notes_FAS/translation_v46_discrete_reference_results_2026-09-08.csv")

def one(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))[-1]

rows=[]
for job in csv.DictReader(JOBS.open()):
    d=ROOT/job["run_id"]
    summary=d/f"{PREFIX}_summary.csv"
    stderr=d/"stderr_forced.txt"
    source=d/"source_sha256.txt"
    row={"coarse_B":job["coarse_B"],"fine_B":job["fine_B"],"jobid":job["jobid"],"run_id":job["run_id"]}
    row["source_match"]=int(source.exists() and source.read_text().split()[0]==job["source_sha"])
    row["stderr_lines"]=len(stderr.read_text().splitlines()) if stderr.exists() else "NO_STDERR"
    if summary.exists():
        row.update(one(summary))
        certified = str(row.get("reference_certified", "0")) == "1"
        row["status"] = "CERTIFIED" if certified else "PROVISIONAL_REFERENCE_NOT_CERTIFIED"
        row["certified_geomean_error_contraction"] = row.get("geomean_error_contraction", "") if certified else ""
    else:
        row["status"]="NO_SUMMARY"
    rows.append(row)
fields=[]
for row in rows:
    for key in row:
        if key not in fields: fields.append(key)
with OUT.open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
print(",".join(fields))
for row in rows: print(",".join(str(row.get(k,"")) for k in fields))