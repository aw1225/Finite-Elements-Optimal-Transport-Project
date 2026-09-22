#!/usr/bin/env python3
"""Post-review T4 diagnostic extension.

Scientific status: supplemental/post-review diagnostic.  The frozen v50 solver
source is imported byte-for-byte and its SHA is verified.  No solver parameter,
regularisation parameter, hierarchy, or stopping threshold is tuned here.

Outputs:
  * fine-only B920 reference trace and reconstructed reference map;
  * direct FAS-to-reference map error;
  * reference-to-exact and FAS-to-exact map errors;
  * cold vs three-fine-sweep vs one-full-FAS-cycle state/spectral comparison;
  * matrix-free weighted-adjoint probe audit at B230/B460/B920;
  * compact summary designed to paste back into ChatGPT.
"""

import json
import math
import os
import time
from pathlib import Path

from firedrake import Function
from mpi4py import MPI

from t4_diag_common import (
    adjoint_audit,
    barycentric_map_physical,
    build_full_problem,
    count_vector,
    gauge_aligned_error,
    hierarchy_levels,
    load_frozen_t4,
    make_zero_state,
    map_error_exact,
    map_error_pair,
    reference_norm,
    snapshot_heat_counts,
    state_diagnostics,
    write_json,
    write_rows,
)

REF_CHECK_BLOCK = 6
REF_MIN_FINE_UPDATES = 60
REF_MAX_FINE_UPDATES = 240
REF_RESIDUAL_TARGET = 1.0e-6
REF_REL_CHANGE_TARGET = 1.0e-8
REF_STABLE_CHECKS_REQUIRED = 2

ORIGINAL_BASELINE = {
    "reference_updates": 96,
    "reference_residual_p": 6.841164570371364e-11,
    "fas1_residual_p": 2.182027712827593e-4,
    "fas1_state_error": 0.006872241679525263,
    "fas1_mode6_probe_rss": 0.0006706581606850891,
    "fas1_mode3_probe_rss": 3.337469823006887e-6,
    "fas1_map_exact_l2": 0.017221020138544388,
}


def rel_diff(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1.0e-300)


def run_reference(fine, p, q, bp, bq, result_dir, comm):
    rows = []
    p_prev = Function(p.function_space(), name="ref_p_prev").assign(p)
    q_prev = Function(q.function_space(), name="ref_q_prev").assign(q)
    stable_checks = 0
    previous_change = None
    certified = False

    fine.reset_counters()
    for updates in range(REF_CHECK_BLOCK, REF_MAX_FINE_UPDATES + 1, REF_CHECK_BLOCK):
        fine.smoother(p, q, bp, bq, REF_CHECK_BLOCK, f"supp_ref_to_{updates}")
        fine.gauge_fix(p, q)
        rp, rq = fine.residual_norms(p, q, bp, bq)

        ch = gauge_aligned_error(fine, p, q, p_prev, q_prev, prefix=f"ref_change_{updates}")
        change_abs = ch["combined_error"]
        state_norm = reference_norm(fine, p, q)
        change_rel = change_abs / max(state_norm, 1.0e-300)
        contraction = (
            change_abs / previous_change
            if previous_change is not None and previous_change > 0.0
            else math.nan
        )
        tail = (
            change_abs * contraction / max(1.0 - contraction, 1.0e-300)
            if math.isfinite(contraction) and 0.0 <= contraction < 1.0
            else math.nan
        )
        qualifies = (
            updates >= REF_MIN_FINE_UPDATES
            and rp <= REF_RESIDUAL_TARGET
            and change_rel <= REF_REL_CHANGE_TARGET
        )
        stable_checks = stable_checks + 1 if qualifies else 0
        certified = stable_checks >= REF_STABLE_CHECKS_REQUIRED
        row = {
            "fine_bridge_n": int(fine.bridge_n),
            "updates": updates,
            "residual_p": rp,
            "residual_q": rq,
            "change_p": ch["p_error"],
            "change_q": ch["q_error"],
            "change_abs": change_abs,
            "change_rel": change_rel,
            "change_contraction": contraction,
            "estimated_remaining_tail": tail,
            "change_gauge_shift": ch["gauge_shift"],
            "stable_checks": stable_checks,
            "reference_certified": int(certified),
        }
        rows.append(row)
        if comm.rank == 0:
            print("T4_SUPP_REFERENCE_ROW", row)

        p_prev.assign(p)
        q_prev.assign(q)
        previous_change = change_abs
        if certified:
            break

    write_rows(Path(result_dir) / "supplemental_reference_trace.csv", rows, comm)
    write_rows(Path(result_dir) / "supplemental_reference_heat_counts.csv", snapshot_heat_counts(fine), comm)
    if not certified:
        raise RuntimeError(
            "Supplemental B920 reference did not satisfy the frozen numerical qualification rule "
            f"within {REF_MAX_FINE_UPDATES} updates. Last row={rows[-1]}"
        )
    return rows[-1], rows


def reset_state(p, q, bp, bq):
    p.assign(0.0)
    q.assign(0.0)
    bp.assign(0.0)
    bq.assign(0.0)


def main():
    t4 = load_frozen_t4()
    comm = t4.COMM
    root = os.environ.get(
        "T4_RUN_ROOT",
        "/rds/general/user/aw1225/ephemeral/ot_project_profile",
    )
    jobid = os.environ.get("PBS_JOBID", f"manual_{int(time.time())}")
    result_dir = os.environ.get(
        "T4_DIAG_RESULT_DIR",
        os.path.join(root, "results", f"t4_supplemental_diagnostics_v1_{jobid}"),
    )
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(result_dir)

    if comm.rank == 0:
        print("T4_SUPPLEMENTAL_RESULT_DIR", result_dir)
        print("T4_FROZEN_SOURCE", t4._supplemental_source_path)
        print("T4_FROZEN_SHA256", t4._supplemental_source_sha256)
        print("T4_SUPPLEMENTAL_STATUS=post_review_diagnostic_no_retuning")

    total_t0 = time.perf_counter()
    fine, immediate_coarse, V0, V1 = build_full_problem(t4, result_dir)
    levels = hierarchy_levels(fine)
    observed_hierarchy = [int(x.bridge_n) for x in levels]
    if observed_hierarchy != [230, 460, 920]:
        raise RuntimeError(f"Expected frozen T4 hierarchy [230,460,920], got {observed_hierarchy}")

    p, q, bp, bq = make_zero_state(V0, V1)

    # ------------------------------------------------------------------
    # 1. Fine-only reference and direct reference map
    # ------------------------------------------------------------------
    ref_t0 = time.perf_counter()
    ref_last, ref_rows = run_reference(fine, p, q, bp, bq, result_dir, comm)
    reference_seconds = comm.allreduce(time.perf_counter() - ref_t0, op=MPI.MAX)
    p_ref = Function(V0, name="supp_p_ref").assign(p)
    q_ref = Function(V1, name="supp_q_ref").assign(q)

    T_ref0, T_ref1, ref_map_health = barycentric_map_physical(fine, q_ref, "reference")
    T_ref = (T_ref0, T_ref1)
    E_ref_exact = map_error_exact(t4, fine, T_ref)

    # Cold metrics to reproduce original T4 cycle-zero diagnostic.
    reset_state(p, q, bp, bq)
    cold = state_diagnostics(fine, p, q, bp, bq, p_ref, q_ref, "cold")

    # ------------------------------------------------------------------
    # 2. Three finest-level Sinkhorn sweeps only
    # ------------------------------------------------------------------
    reset_state(p, q, bp, bq)
    fine.reset_counters()
    sm_t0 = time.perf_counter()
    fine.smoother(p, q, bp, bq, 3, "supplemental_three_fine_sweeps")
    fine.gauge_fix(p, q)
    smoothing_seconds = comm.allreduce(time.perf_counter() - sm_t0, op=MPI.MAX)
    smoothing_cycle_counts = snapshot_heat_counts(fine)  # snapshot BEFORE diagnostic heat calls
    smoothing_count_vector = count_vector(smoothing_cycle_counts)
    smooth3 = state_diagnostics(fine, p, q, bp, bq, p_ref, q_ref, "smooth3")
    T_s0, T_s1, smooth_map_health = barycentric_map_physical(fine, q, "smooth3")
    T_smooth = (T_s0, T_s1)
    E_smooth_ref = map_error_pair(fine, T_smooth, T_ref)
    E_smooth_exact = map_error_exact(t4, fine, T_smooth)

    # ------------------------------------------------------------------
    # 3. One complete frozen three-level FAS cycle from the same cold state
    # ------------------------------------------------------------------
    reset_state(p, q, bp, bq)
    fine.reset_counters()
    fas_t0 = time.perf_counter()
    fine.fas_cycle_perf(p, q, bp, bq)
    fine.gauge_fix(p, q)
    fas_seconds = comm.allreduce(time.perf_counter() - fas_t0, op=MPI.MAX)
    fas_cycle_counts = snapshot_heat_counts(fine)  # snapshot BEFORE diagnostic heat calls
    fas_count_vector = count_vector(fas_cycle_counts)
    expected_fas_counts = [16, 10, 8]
    fas_count_match = int(fas_count_vector == expected_fas_counts)
    if not fas_count_match:
        raise RuntimeError(
            f"Frozen one-cycle T4 heat-count invariant failed: observed={fas_count_vector}, "
            f"expected={expected_fas_counts}"
        )
    fas1 = state_diagnostics(fine, p, q, bp, bq, p_ref, q_ref, "fas1")
    T_f0, T_f1, fas_map_health = barycentric_map_physical(fine, q, "fas1")
    T_fas = (T_f0, T_f1)
    E_fas_ref = map_error_pair(fine, T_fas, T_ref)
    E_fas_exact = map_error_exact(t4, fine, T_fas)

    mechanism_rows = [cold, smooth3, fas1]
    write_rows(Path(result_dir) / "mechanism_comparison.csv", mechanism_rows, comm)
    write_rows(Path(result_dir) / "smooth3_heat_counts.csv", smoothing_cycle_counts, comm)
    write_rows(Path(result_dir) / "fas1_cycle_heat_counts.csv", fas_cycle_counts, comm)

    # ------------------------------------------------------------------
    # 4. Weighted-adjoint probe audit on all already-factorised levels
    # ------------------------------------------------------------------
    adjoint_summaries = []
    for lev in levels:
        if comm.rank == 0:
            print(f"T4_SUPP_ADJOINT_BEGIN B{lev.bridge_n}")
        adj = adjoint_audit(lev, result_dir, label=f"B{lev.bridge_n}")
        adjoint_summaries.append(adj)
        if comm.rank == 0:
            print("T4_SUPP_ADJOINT_SUMMARY", adj)
    write_rows(Path(result_dir) / "adjoint_all_levels_summary.csv", adjoint_summaries, comm)

    # ------------------------------------------------------------------
    # 5. Compact result and reproducibility checks
    # ------------------------------------------------------------------
    baseline_checks = {
        "reference_updates_abs_difference": abs(int(ref_last["updates"]) - ORIGINAL_BASELINE["reference_updates"]),
        "reference_residual_p_relative_difference": rel_diff(ref_last["residual_p"], ORIGINAL_BASELINE["reference_residual_p"]),
        "fas1_residual_p_relative_difference": rel_diff(fas1["residual_p"], ORIGINAL_BASELINE["fas1_residual_p"]),
        "fas1_state_error_relative_difference": rel_diff(fas1["state_error_combined"], ORIGINAL_BASELINE["fas1_state_error"]),
        "fas1_mode6_relative_difference": rel_diff(fas1["mode6_probe_rss"], ORIGINAL_BASELINE["fas1_mode6_probe_rss"]),
        "fas1_mode3_relative_difference": rel_diff(fas1["mode3_probe_rss"], ORIGINAL_BASELINE["fas1_mode3_probe_rss"]),
        "fas1_map_exact_relative_difference": rel_diff(E_fas_exact, ORIGINAL_BASELINE["fas1_map_exact_l2"]),
    }

    summary = {
        "scientific_status": "post_review_supplemental_diagnostic_no_retuning",
        "source_path": t4._supplemental_source_path,
        "source_sha256": t4._supplemental_source_sha256,
        "result_dir": result_dir,
        "hierarchy": observed_hierarchy,
        "tau_physical": float(t4.TAU_FINE_PHYSICAL),
        "pre_smooth": int(t4.PRE_SMOOTH),
        "coarse_smooth": int(t4.COARSE_SMOOTH),
        "post_smooth": int(t4.POST_SMOOTH),
        "reference_certified": int(ref_last["reference_certified"]),
        "reference_updates": int(ref_last["updates"]),
        "reference_residual_p": float(ref_last["residual_p"]),
        "reference_residual_q": float(ref_last["residual_q"]),
        "reference_change_rel": float(ref_last["change_rel"]),
        "reference_estimated_remaining_tail": float(ref_last["estimated_remaining_tail"]),
        "reference_seconds": float(reference_seconds),
        "E_map_reference_exact": E_ref_exact,
        "E_map_smooth3_reference": E_smooth_ref,
        "E_map_smooth3_exact": E_smooth_exact,
        "E_map_fas1_reference": E_fas_ref,
        "E_map_fas1_exact": E_fas_exact,
        "E_map_fas1_reference_over_reference_exact": E_fas_ref / max(E_ref_exact, 1.0e-300),
        "cold_residual_p": cold["residual_p"],
        "cold_state_error": cold["state_error_combined"],
        "cold_mode6_probe_rss": cold["mode6_probe_rss"],
        "cold_mode3_probe_rss": cold["mode3_probe_rss"],
        "smooth3_seconds": float(smoothing_seconds),
        "smooth3_heat_count_vector": smoothing_count_vector,
        "smooth3_residual_p": smooth3["residual_p"],
        "smooth3_state_error": smooth3["state_error_combined"],
        "smooth3_mode6_probe_rss": smooth3["mode6_probe_rss"],
        "smooth3_mode3_probe_rss": smooth3["mode3_probe_rss"],
        "fas1_seconds": float(fas_seconds),
        "fas1_heat_count_vector": fas_count_vector,
        "fas1_heat_count_match": fas_count_match,
        "fas1_residual_p": fas1["residual_p"],
        "fas1_state_error": fas1["state_error_combined"],
        "fas1_mode6_probe_rss": fas1["mode6_probe_rss"],
        "fas1_mode3_probe_rss": fas1["mode3_probe_rss"],
        "fas_over_smooth3_residual_ratio": fas1["residual_p"] / max(smooth3["residual_p"], 1.0e-300),
        "fas_over_smooth3_state_error_ratio": fas1["state_error_combined"] / max(smooth3["state_error_combined"], 1.0e-300),
        "fas_over_smooth3_mode6_ratio": fas1["mode6_probe_rss"] / max(smooth3["mode6_probe_rss"], 1.0e-300),
        "fas_over_smooth3_mode3_ratio": fas1["mode3_probe_rss"] / max(smooth3["mode3_probe_rss"], 1.0e-300),
        "reference_map_health": ref_map_health,
        "smooth3_map_health": smooth_map_health,
        "fas1_map_health": fas_map_health,
        "adjoint": adjoint_summaries,
        "baseline_reproduction_checks": baseline_checks,
        "total_seconds": float(comm.allreduce(time.perf_counter() - total_t0, op=MPI.MAX)),
    }
    write_json(Path(result_dir) / "supplemental_summary.json", summary, comm)
    write_rows(Path(result_dir) / "supplemental_summary.csv", [{
        k: (json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v)
        for k, v in summary.items()
    }], comm)

    if comm.rank == 0:
        print("PASTE_BACK_BEGIN")
        keys = [
            "source_sha256",
            "reference_certified",
            "reference_updates",
            "reference_residual_p",
            "reference_change_rel",
            "E_map_reference_exact",
            "E_map_fas1_reference",
            "E_map_fas1_exact",
            "E_map_fas1_reference_over_reference_exact",
            "smooth3_heat_count_vector",
            "smooth3_residual_p",
            "smooth3_state_error",
            "smooth3_mode6_probe_rss",
            "smooth3_mode3_probe_rss",
            "fas1_heat_count_vector",
            "fas1_residual_p",
            "fas1_state_error",
            "fas1_mode6_probe_rss",
            "fas1_mode3_probe_rss",
            "fas_over_smooth3_residual_ratio",
            "fas_over_smooth3_state_error_ratio",
            "fas_over_smooth3_mode6_ratio",
            "fas_over_smooth3_mode3_ratio",
        ]
        for k in keys:
            print(f"{k}={summary[k]}")
        for a in adjoint_summaries:
            print(
                f"adjoint_B{a['bridge_n']}_mass_global={a['mass_global_probe_relative_defect']} "
                f"median={a['mass_active_pair_median_relative_defect']} "
                f"p95={a['mass_active_pair_p95_relative_defect']} "
                f"max={a['mass_active_pair_max_relative_defect']}"
            )
        print("baseline_reproduction_checks=" + json.dumps(baseline_checks, sort_keys=True))
        print("PASTE_BACK_END")
        print("RESULT_DIR=" + result_dir)


if __name__ == "__main__":
    main()
