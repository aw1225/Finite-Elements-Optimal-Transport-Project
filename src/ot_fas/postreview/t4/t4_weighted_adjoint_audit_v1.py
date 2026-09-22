#!/usr/bin/env python3
import os
import time
from pathlib import Path

from t4_diag_common import (
    adjoint_audit,
    build_single_level,
    load_frozen_t4,
    write_json,
)


def main():
    t4 = load_frozen_t4()
    comm = t4.COMM
    B = int(os.environ.get("T4_ADJOINT_B", "230"))
    if B not in (230, 460, 920):
        raise ValueError("T4_ADJOINT_B must be one of 230, 460, 920")
    root = os.environ.get(
        "T4_RUN_ROOT",
        "/rds/general/user/aw1225/ephemeral/ot_project_profile",
    )
    jobid = os.environ.get("PBS_JOBID", f"manual_{int(time.time())}")
    result_dir = os.environ.get(
        "T4_DIAG_RESULT_DIR",
        os.path.join(root, "results", f"t4_weighted_adjoint_B{B}_v1_{jobid}"),
    )
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(result_dir)

    if comm.rank == 0:
        print("T4_ADJOINT_RESULT_DIR", result_dir)
        print("T4_FROZEN_SOURCE", t4._supplemental_source_path)
        print("T4_FROZEN_SHA256", t4._supplemental_source_sha256)
        print("T4_ADJOINT_BRIDGE", B)

    level, V0, V1, mu0, mu1 = build_single_level(t4, B, result_dir)
    summary = adjoint_audit(level, result_dir, label=f"B{B}")
    summary.update({
        "source_sha256": t4._supplemental_source_sha256,
        "source_path": t4._supplemental_source_path,
        "result_dir": result_dir,
    })
    write_json(os.path.join(result_dir, "adjoint_summary.json"), summary, comm)

    if comm.rank == 0:
        print("PASTE_ADJOINT_BEGIN")
        for k in [
            "bridge_n",
            "mass_global_probe_relative_defect",
            "mass_active_pair_median_relative_defect",
            "mass_active_pair_p95_relative_defect",
            "mass_active_pair_max_relative_defect",
            "mu_weighted_global_probe_relative_defect",
        ]:
            print(f"{k}={summary[k]}")
        print("PASTE_ADJOINT_END")
        print("RESULT_DIR=" + result_dir)


if __name__ == "__main__":
    main()
