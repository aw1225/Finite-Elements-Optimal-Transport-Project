#!/usr/bin/env python3
"""Shared helpers for post-review T4 supplemental diagnostics.

This file DOES NOT modify the frozen T4 source.  It imports the authoritative
v50 source, verifies its SHA256, and adds diagnostics around the existing
Firedrake operators.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from mpi4py import MPI
from firedrake import (
    Constant,
    Function,
    PeriodicUnitSquareMesh,
    SpatialCoordinate,
    as_vector,
    assemble,
    cos,
    dx,
    exp,
    max_value,
    sin,
    sqrt,
)

EXPECTED_T4_SHA256 = "f59b123df8a4902a04390c5b267b28a82007ba3963b9d0015fc80b5ad86ca269"
DEFAULT_T4_SOURCE = os.path.expanduser(
    "~/ot_project/code_FAS/cx3_t4_high_frequency_square_stress_v50.py"
)


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_frozen_t4():
    path = os.path.abspath(os.path.expanduser(os.environ.get("T4_FROZEN_SOURCE", DEFAULT_T4_SOURCE)))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Frozen T4 source not found: {path}\n"
            "Set T4_FROZEN_SOURCE to the authoritative v50 source path."
        )
    observed = sha256_file(path)
    allow = os.environ.get("T4_ALLOW_SOURCE_SHA_MISMATCH", "0") == "1"
    if observed != EXPECTED_T4_SHA256 and not allow:
        raise RuntimeError(
            "Frozen T4 source SHA mismatch. Refusing to run a scientific diagnostic.\n"
            f" expected={EXPECTED_T4_SHA256}\n observed={observed}\n path={path}\n"
            "If this is an intentionally archived but byte-different copy, verify it manually and set "
            "T4_ALLOW_SOURCE_SHA_MISMATCH=1."
        )
    spec = importlib.util.spec_from_file_location("frozen_t4_v50", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import frozen T4 source from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._supplemental_source_path = path
    mod._supplemental_source_sha256 = observed
    return mod


def rank0(comm) -> bool:
    return comm.rank == 0


def write_rows(path: str | os.PathLike[str], rows: Sequence[dict], comm) -> None:
    if not rows or not rank0(comm):
        return
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_json(path: str | os.PathLike[str], obj, comm) -> None:
    if not rank0(comm):
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return str(x)


def global_dof_max(fn, comm) -> float:
    a = np.asarray(fn.dat.data_ro)
    local = float(a.max()) if a.size else -math.inf
    return float(comm.allreduce(local, op=MPI.MAX))


def global_dof_min(fn, comm) -> float:
    a = np.asarray(fn.dat.data_ro)
    local = float(a.min()) if a.size else math.inf
    return float(comm.allreduce(local, op=MPI.MIN))


def global_nonfinite_count(fn, comm) -> int:
    a = np.asarray(fn.dat.data_ro)
    return int(comm.allreduce(int((~np.isfinite(a)).sum()), op=MPI.SUM))


def hierarchy_levels(fine) -> list:
    out = []
    if getattr(fine, "coarse", None) is not None:
        out.extend(hierarchy_levels(fine.coarse))
    out.append(fine)
    return out


def _call_build_support(t4, result_dir: str):
    fn = t4.build_support_spaces_and_measures
    sig = inspect.signature(fn)
    kwargs = {}
    for name in sig.parameters:
        if name in {"result_dir", "output_dir", "run_dir"}:
            kwargs[name] = result_dir
    ans = fn(**kwargs)
    if not isinstance(ans, tuple) or len(ans) < 4:
        raise RuntimeError("build_support_spaces_and_measures did not return at least (V0,V1,mu0,mu1)")
    return ans[0], ans[1], ans[2], ans[3]


def build_full_problem(t4, result_dir: str):
    """Build the frozen hierarchy, passing a result directory if the source accepts one."""
    fn = t4.build_problem
    sig = inspect.signature(fn)
    kwargs = {}
    if "coarse_tau_ratio" in sig.parameters:
        kwargs["coarse_tau_ratio"] = 1.0
    for name in sig.parameters:
        if name in {"result_dir", "output_dir", "run_dir"}:
            kwargs[name] = result_dir
    ans = fn(**kwargs)
    if not isinstance(ans, tuple) or len(ans) < 4:
        raise RuntimeError("build_problem did not return at least (fine,coarse,V0,V1)")
    return ans[0], ans[1], ans[2], ans[3]


def build_single_level(t4, bridge_n: int, result_dir: str):
    """Build one heat level without constructing the B920 hierarchy."""
    V0, V1, mu0, mu1 = _call_build_support(t4, result_dir)
    original_levels = list(t4.BRIDGE_LEVELS)
    t4.BRIDGE_LEVELS = [int(bridge_n)]
    try:
        mesh = PeriodicUnitSquareMesh(int(bridge_n), int(bridge_n))
        level = t4.FASLevel(
            f"supplemental_adjoint_B{bridge_n}",
            0,
            mesh,
            float(t4.TAU_FINE_PHYSICAL),
            V0,
            V1,
            mu0,
            mu1,
            coarse=None,
        )
    finally:
        t4.BRIDGE_LEVELS = original_levels
    # FASLevel stores bridge_n at construction, so restoring the module list is safe.
    return level, V0, V1, mu0, mu1


def make_zero_state(V0, V1):
    p = Function(V0, name="supp_p").assign(0.0)
    q = Function(V1, name="supp_q").assign(0.0)
    bp = Function(V0, name="supp_bp").assign(0.0)
    bq = Function(V1, name="supp_bq").assign(0.0)
    return p, q, bp, bq


def reference_norm(level, p, q) -> float:
    val = float(assemble(p * p * level.mu0 * dx) + assemble(q * q * level.mu1 * dx))
    return math.sqrt(max(val, 0.0))


def gauge_aligned_error(level, p, q, p_ref, q_ref, prefix="ga") -> dict:
    dp = Function(p.function_space(), name=f"{prefix}_dp")
    dq = Function(q.function_space(), name=f"{prefix}_dq")
    dp.interpolate(p - p_ref)
    dq.interpolate(q - q_ref)
    a = float(assemble(dp * level.mu0 * dx))
    b = float(assemble(dq * level.mu1 * dx))
    shift = 0.5 * (a - b)
    dp.interpolate(p - p_ref - Constant(shift))
    dq.interpolate(q - q_ref + Constant(shift))
    ep = math.sqrt(max(float(assemble(dp * dp * level.mu0 * dx)), 0.0))
    eq = math.sqrt(max(float(assemble(dq * dq * level.mu1 * dx)), 0.0))
    return {
        "p_error": ep,
        "q_error": eq,
        "combined_error": math.sqrt(ep * ep + eq * eq),
        "gauge_shift": shift,
        "dp": dp,
        "dq": dq,
    }


def physical_coordinate(mesh, width: float):
    x = SpatialCoordinate(mesh)
    return as_vector((
        Constant(width) * (x[0] - Constant(0.5)),
        Constant(width) * (x[1] - Constant(0.5)),
    ))


def _probe_exprs(mesh, width: float):
    xp = physical_coordinate(mesh, width)
    x, y = xp[0], xp[1]
    def cc(m):
        w = Constant(2.0 * math.pi * m)
        return cos(w * x) * cos(w * y)
    def ss(m):
        w = Constant(2.0 * math.pi * m)
        return sin(w * x) * sin(w * y)
    mix_a = (
        Constant(0.61) * cos(Constant(2.0 * math.pi) * x)
        + Constant(0.37) * sin(Constant(4.0 * math.pi) * y)
        + Constant(0.23) * cc(3)
    )
    mix_b = (
        Constant(0.53) * sin(Constant(2.0 * math.pi) * (x + y))
        - Constant(0.31) * cos(Constant(4.0 * math.pi) * (x - y))
        + Constant(0.19) * ss(6)
    )
    return [
        ("one", Constant(1.0)),
        ("x", x),
        ("y", y),
        ("xy", x * y),
        ("m3_cc", cc(3)),
        ("m3_ss", ss(3)),
        ("m6_cc", cc(6)),
        ("m6_ss", ss(6)),
        ("mix_A", mix_a),
        ("mix_B", mix_b),
    ]


def _normalised_probe_functions(V, width: float, prefix: str):
    rows = []
    for name, expr in _probe_exprs(V.mesh(), width):
        f = Function(V, name=f"{prefix}_{name}").interpolate(expr)
        norm = math.sqrt(max(float(assemble(f * f * dx)), 0.0))
        if norm <= 1.0e-14:
            raise RuntimeError(f"Probe {name} has near-zero L2 norm")
        f.interpolate(f / Constant(norm))
        rows.append((name, f))
    return rows


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    return float(np.percentile(np.asarray(values, dtype=float), q))


def adjoint_audit(level, result_dir: str, label: str | None = None) -> dict:
    """Matrix-free support-mass adjoint audit for one directional heat pair.

    The headline pairing is the ordinary FE L2 pairing, i.e. the mass-matrix
    weighted algebraic adjoint test.  A mu-weighted sensitivity pairing is also
    reported, but is not the defining discrete-adjoint criterion.
    """
    comm = getattr(level, "V0").mesh().comm
    width = float(getattr(level, "tau_physical", 0.0) * 0.0 + 4.0)
    # Prefer source-module constant if present through level geometry convention.
    # All T4 v50 runs use W=4; caller can override by attaching bridge_width_physical.
    width = float(getattr(level, "bridge_width_physical", width))

    src = _normalised_probe_functions(level.V0, width, "adj_src")
    tgt = _normalised_probe_functions(level.V1, width, "adj_tgt")

    H01 = {}
    H10 = {}
    for name, u in src:
        out = Function(level.V1, name=f"H01_{name}")
        level.heat_scaling_to_target(u, out)
        H01[name] = out
    for name, v in tgt:
        out = Function(level.V0, name=f"H10_{name}")
        level.heat_scaling_to_source(v, out)
        H10[name] = out

    rows = []
    mass_diffs, mass_scales = [], []
    mu_diffs, mu_scales = [], []
    active_etas, active_mu_etas = [], []
    for sname, u in src:
        for tname, v in tgt:
            a = float(assemble(H01[sname] * v * dx))
            b = float(assemble(u * H10[tname] * dx))
            scale = abs(a) + abs(b)
            diff = abs(a - b)
            eta = diff / max(scale, 1.0e-300)

            am = float(assemble(H01[sname] * v * level.mu1 * dx))
            bm = float(assemble(u * H10[tname] * level.mu0 * dx))
            muscale = abs(am) + abs(bm)
            mudiff = abs(am - bm)
            mueta = mudiff / max(muscale, 1.0e-300)

            mass_diffs.append(diff)
            mass_scales.append(a * a + b * b)
            mu_diffs.append(mudiff)
            mu_scales.append(am * am + bm * bm)
            if scale > 1.0e-10:
                active_etas.append(eta)
            if muscale > 1.0e-10:
                active_mu_etas.append(mueta)

            rows.append({
                "bridge_n": int(level.bridge_n),
                "source_probe": sname,
                "target_probe": tname,
                "mass_pair_H01u_v": a,
                "mass_pair_u_H10v": b,
                "mass_abs_defect": diff,
                "mass_symmetric_relative_defect": eta,
                "mass_pair_active_gt_1e-10": int(scale > 1.0e-10),
                "mu_pair_H01u_v": am,
                "mu_pair_u_H10v": bm,
                "mu_abs_defect": mudiff,
                "mu_symmetric_relative_defect": mueta,
                "mu_pair_active_gt_1e-10": int(muscale > 1.0e-10),
            })

    global_mass = math.sqrt(sum(d * d for d in mass_diffs)) / max(
        math.sqrt(sum(mass_scales)), 1.0e-300
    )
    global_mu = math.sqrt(sum(d * d for d in mu_diffs)) / max(
        math.sqrt(sum(mu_scales)), 1.0e-300
    )
    summary = {
        "bridge_n": int(level.bridge_n),
        "probe_count_each_side": len(src),
        "pair_count": len(rows),
        "headline_pairing": "ordinary_FE_L2_mass_pairing",
        "mass_global_probe_relative_defect": global_mass,
        "mass_active_pair_median_relative_defect": float(np.median(active_etas)) if active_etas else math.nan,
        "mass_active_pair_p95_relative_defect": percentile(active_etas, 95),
        "mass_active_pair_max_relative_defect": max(active_etas) if active_etas else math.nan,
        "mass_active_pair_count": len(active_etas),
        "mu_weighted_global_probe_relative_defect": global_mu,
        "mu_active_pair_median_relative_defect": float(np.median(active_mu_etas)) if active_mu_etas else math.nan,
        "mu_active_pair_p95_relative_defect": percentile(active_mu_etas, 95),
        "mu_active_pair_max_relative_defect": max(active_mu_etas) if active_mu_etas else math.nan,
        "mu_active_pair_count": len(active_mu_etas),
        "interpretation_note": (
            "Finite probe audit only: small defect supports approximate adjointness on these probes; "
            "it does not prove an exact common discrete variational structure."
        ),
    }
    stem = label or f"B{int(level.bridge_n)}"
    write_rows(Path(result_dir) / f"adjoint_{stem}_pairs.csv", rows, comm)
    write_rows(Path(result_dir) / f"adjoint_{stem}_summary.csv", [summary], comm)
    return summary


def gauge_aligned_fields(level, p, q, p_ref, q_ref, prefix: str):
    g = gauge_aligned_error(level, p, q, p_ref, q_ref, prefix=prefix)
    return g["dp"], g["dq"], g


def spectral_probe_metrics(level, p, q, p_ref, q_ref, mode: int, prefix: str) -> dict:
    dp, dq, ga = gauge_aligned_fields(level, p, q, p_ref, q_ref, prefix)
    width = 4.0
    xs = physical_coordinate(level.V0.mesh(), width)
    xt = physical_coordinate(level.V1.mesh(), width)
    omega = Constant(2.0 * math.pi * int(mode))

    def probes(x):
        return [
            cos(omega * x[0]) * cos(omega * x[1]),
            sin(omega * x[0]) * sin(omega * x[1]),
            cos(omega * x[0]) * sin(omega * x[1]),
            sin(omega * x[0]) * cos(omega * x[1]),
        ]

    pcoef, qcoef = [], []
    for i, psi in enumerate(probes(xs)):
        den = math.sqrt(max(float(assemble(psi * psi * level.mu0 * dx)), 1.0e-300))
        pcoef.append(float(assemble(dp * psi * level.mu0 * dx)) / den)
    for i, psi in enumerate(probes(xt)):
        den = math.sqrt(max(float(assemble(psi * psi * level.mu1 * dx)), 1.0e-300))
        qcoef.append(float(assemble(dq * psi * level.mu1 * dx)) / den)
    rss = math.sqrt(sum(c * c for c in pcoef) + sum(c * c for c in qcoef))
    out = {
        "mode": int(mode),
        "combined_probe_rss": rss,
        "p_probe_rss": math.sqrt(sum(c * c for c in pcoef)),
        "q_probe_rss": math.sqrt(sum(c * c for c in qcoef)),
        "combined_state_error": ga["combined_error"],
        "gauge_shift": ga["gauge_shift"],
    }
    for i, c in enumerate(pcoef):
        out[f"p_c{i}"] = c
    for i, c in enumerate(qcoef):
        out[f"q_c{i}"] = c
    return out


def state_diagnostics(level, p, q, bp, bq, p_ref, q_ref, label: str) -> dict:
    rp, rq = level.residual_norms(p, q, bp, bq)
    ga = gauge_aligned_error(level, p, q, p_ref, q_ref, prefix=f"{label}_state")
    high = spectral_probe_metrics(level, p, q, p_ref, q_ref, 6, prefix=f"{label}_m6")
    low = spectral_probe_metrics(level, p, q, p_ref, q_ref, 3, prefix=f"{label}_m3")
    return {
        "label": label,
        "residual_p": rp,
        "residual_q": rq,
        "state_error_p": ga["p_error"],
        "state_error_q": ga["q_error"],
        "state_error_combined": ga["combined_error"],
        "state_gauge_shift": ga["gauge_shift"],
        "mode6_probe_rss": high["combined_probe_rss"],
        "mode3_probe_rss": low["combined_probe_rss"],
    }


def exact_map_physical_expr(t4, V0):
    width = float(getattr(t4, "BRIDGE_WIDTH_PHYSICAL", 4.0))
    xu = SpatialCoordinate(V0.mesh())
    xp = as_vector((
        Constant(width) * (xu[0] - Constant(0.5)),
        Constant(width) * (xu[1] - Constant(0.5)),
    ))
    if hasattr(t4, "exact_map_physical"):
        try:
            return t4.exact_map_physical(xp)
        except TypeError:
            return t4.exact_map_physical(xp[0], xp[1])

    mode = int(getattr(t4, "OSC_MODE", getattr(t4, "T4_MODE", 6)))
    beta = float(getattr(t4, "OSC_BETA", getattr(t4, "T4_BETA", 8.0)))
    omega = 2.0 * math.pi * mode
    om = Constant(omega)

    def qfun(z):
        return (
            (-z * z / om + Constant(2.0 / omega**3 + 1.0 / (4.0 * omega))) * cos(om * z)
            + Constant(2.0 / omega**2) * z * sin(om * z)
        )

    def qprime(z):
        return (z * z - Constant(0.25)) * sin(om * z)

    return as_vector((
        xp[0] + Constant(beta) * qprime(xp[0]) * qfun(xp[1]),
        xp[1] + Constant(beta) * qfun(xp[0]) * qprime(xp[1]),
    ))


def barycentric_map_physical(level, q, label: str):
    """Reconstruct E[Y|X] from the same directional heat action used by T4."""
    comm = level.V0.mesh().comm
    width = 4.0
    q_shift = global_dof_max(q, comm)
    weight = Function(level.V1, name=f"{label}_weight").interpolate(exp(q - Constant(q_shift)))
    y = SpatialCoordinate(level.V1.mesh())
    den = Function(level.V0, name=f"{label}_den")
    n0 = Function(level.V0, name=f"{label}_num0")
    n1 = Function(level.V0, name=f"{label}_num1")
    w0 = Function(level.V1, name=f"{label}_weighted_y0").interpolate(y[0] * weight)
    w1 = Function(level.V1, name=f"{label}_weighted_y1").interpolate(y[1] * weight)
    level.heat_scaling_to_source(weight, den)
    level.heat_scaling_to_source(w0, n0)
    level.heat_scaling_to_source(w1, n1)

    dmin = global_dof_min(den, comm)
    dmax = global_dof_max(den, comm)
    dnonfinite = global_nonfinite_count(den, comm)
    if dnonfinite or dmin <= 0.0:
        raise RuntimeError(
            f"Barycentric denominator unhealthy for {label}: min={dmin}, max={dmax}, nonfinite={dnonfinite}"
        )
    t0u = Function(level.V0, name=f"{label}_T0_unit").interpolate(n0 / max_value(den, Constant(1.0e-300)))
    t1u = Function(level.V0, name=f"{label}_T1_unit").interpolate(n1 / max_value(den, Constant(1.0e-300)))
    t0p = Function(level.V0, name=f"{label}_T0_phys").interpolate(
        Constant(width) * (t0u - Constant(0.5))
    )
    t1p = Function(level.V0, name=f"{label}_T1_phys").interpolate(
        Constant(width) * (t1u - Constant(0.5))
    )
    return t0p, t1p, {
        "map_q_shift": q_shift,
        "map_denominator_min": dmin,
        "map_denominator_max": dmax,
        "map_denominator_nonfinite_count": dnonfinite,
    }


def map_error_pair(level, Ta: Tuple[Function, Function], Tb: Tuple[Function, Function]) -> float:
    e2 = float(assemble(((Ta[0] - Tb[0]) ** 2 + (Ta[1] - Tb[1]) ** 2) * level.mu0 * dx))
    return math.sqrt(max(e2, 0.0))


def map_error_exact(t4, level, T: Tuple[Function, Function]) -> float:
    Tex = exact_map_physical_expr(t4, level.V0)
    e2 = float(assemble(((T[0] - Tex[0]) ** 2 + (T[1] - Tex[1]) ** 2) * level.mu0 * dx))
    return math.sqrt(max(e2, 0.0))


def snapshot_heat_counts(fine) -> list:
    if hasattr(fine, "counter_rows"):
        rows = fine.counter_rows()
        out = []
        for r in rows:
            out.append({k: _json_default(v) for k, v in r.items()})
        return out
    out = []
    for lev in hierarchy_levels(fine):
        out.append({
            "bridge_n": int(lev.bridge_n),
            "heat_source_calls": int(lev.heat_source_calls),
            "heat_target_calls": int(lev.heat_target_calls),
            "heat_total_calls": int(lev.heat_source_calls + lev.heat_target_calls),
            "backward_euler_solves": int(lev.backward_euler_solves),
        })
    return out


def count_vector(rows: list) -> list:
    ans = []
    for row in rows:
        if "heat_total_calls" in row:
            ans.append(int(row["heat_total_calls"]))
        else:
            ans.append(int(row.get("heat_source_calls", 0)) + int(row.get("heat_target_calls", 0)))
    return ans
