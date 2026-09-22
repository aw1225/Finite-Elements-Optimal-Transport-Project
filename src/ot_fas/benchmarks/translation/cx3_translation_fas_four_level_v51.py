from firedrake import *
from mpi4py import MPI
import csv
import math
import os
import time

EXPERIMENT_NAME = "run_translation_exact_error_trajectory_v44"

DISK_REFINEMENT = 7
DISK_CG_DEGREE = 2
BRIDGE_CG_DEGREE = 2
N_SUB = 16

BRIDGE_COARSE_N = int(os.environ.get("FAS_BRIDGE_COARSE_N", "230"))
N_BRIDGE_REFINEMENTS = int(os.environ.get("FAS_N_BRIDGE_REFINEMENTS", "2"))
BRIDGE_LEVELS = [
    BRIDGE_COARSE_N * 2**level
    for level in range(N_BRIDGE_REFINEMENTS + 1)
]
BRIDGE_WIDTH_PHYSICAL = 4.0
TAU_FINE_PHYSICAL = 2.373046875e-3

SOURCE_CENTER_PHYSICAL = (-0.15, -0.125)
TARGET_CENTER_PHYSICAL = (+0.15, +0.125)
DISK_RADIUS_PHYSICAL = 0.6

LOG_CLAMP = Constant(120.0)
TINY = Constant(1e-300)

FAS_CYCLES = int(os.environ.get("FAS_CYCLES", "1"))
FMG_N = int(os.environ.get("FAS_FMG_N", "0"))  # legacy equal-N compatibility
FMG_BOTTOM_UPDATES = int(os.environ.get("FAS_FMG_BOTTOM_UPDATES", str(7 * FMG_N)))
FMG_MIDDLE_CYCLES = int(os.environ.get("FAS_FMG_MIDDLE_CYCLES", str(FMG_N)))
FMG_TOP_MAX_CYCLES = int(os.environ.get("FAS_FMG_TOP_MAX_CYCLES", str(FAS_CYCLES)))
PRE_SMOOTH = int(os.environ.get("FAS_PRE_SMOOTH", "2"))
COARSE_SMOOTH = int(os.environ.get("FAS_COARSE_SMOOTH", "7"))
POST_SMOOTH = int(os.environ.get("FAS_POST_SMOOTH", "1"))
RUN_MODE = os.environ.get("FAS_RUN_MODE", "sign_test")
SMOOTHER_TEST_UPDATES = int(os.environ.get("FAS_SMOOTHER_TEST_UPDATES", "1"))
FINE_ONLY_UPDATES = int(os.environ.get("FAS_FINE_ONLY_UPDATES", "10"))
CORRECTION_DAMPING = float(os.environ.get("FAS_CORRECTION_DAMPING", "1.0"))
COARSE_SWEEP_VALUES = [int(x) for x in os.environ.get("FAS_COARSE_SWEEP_VALUES", "0,2,5,10,20,30").split(",") if x.strip()]
COARSE_TAU_RATIO_VALUES = [float(x) for x in os.environ.get("FAS_COARSE_TAU_RATIO_VALUES", "1,2,3,4").split(",") if x.strip()]
COARSE_TAU_RATIO = float(os.environ.get("FAS_COARSE_TAU_RATIO", "1.0"))
TARGET_RESIDUAL = float(os.environ.get("FAS_TARGET_RESIDUAL", "4.31e-3"))
AUDIT_WRITE_VTK = int(os.environ.get("FAS_WRITE_VTK", "0"))
MAX_FINE_ONLY_UPDATES = int(os.environ.get("FAS_MAX_FINE_ONLY_UPDATES", "60"))
FINE_CHECK_BLOCK = int(os.environ.get("FAS_FINE_CHECK_BLOCK", "3"))

COMM = COMM_WORLD

solver_params = {
    "ksp_type": "preonly",
    "pc_type": "cholesky",
    "pc_factor_mat_solver_type": "mumps",
}


def unit_center(center_physical):
    return (
        0.5 + center_physical[0] / BRIDGE_WIDTH_PHYSICAL,
        0.5 + center_physical[1] / BRIDGE_WIDTH_PHYSICAL,
    )


def unit_radius(radius_physical):
    return radius_physical / BRIDGE_WIDTH_PHYSICAL


def disk_mesh_unit(center_physical, radius_physical, refinement):
    cx, cy = unit_center(center_physical)
    radius = unit_radius(radius_physical)
    mesh = UnitDiskMesh(refinement)
    mesh.coordinates.interpolate(
        radius * mesh.coordinates + as_vector([Constant(cx), Constant(cy)])
    )
    return mesh


def build_support_spaces_and_measures():
    mesh0 = disk_mesh_unit(
        SOURCE_CENTER_PHYSICAL,
        DISK_RADIUS_PHYSICAL,
        DISK_REFINEMENT,
    )
    mesh1 = disk_mesh_unit(
        TARGET_CENTER_PHYSICAL,
        DISK_RADIUS_PHYSICAL,
        DISK_REFINEMENT,
    )

    V0 = FunctionSpace(mesh0, "CG", DISK_CG_DEGREE)
    V1 = FunctionSpace(mesh1, "CG", DISK_CG_DEGREE)

    mu0 = Function(V0, name="mu0").interpolate(1.0)
    mu1 = Function(V1, name="mu1").interpolate(1.0)

    mu0.assign(mu0 / assemble(mu0 * dx))
    mu1.assign(mu1 / assemble(mu1 * dx))

    return V0, V1, mu0, mu1


def build_bridge_hierarchy():
    base = PeriodicUnitSquareMesh(BRIDGE_COARSE_N, BRIDGE_COARSE_N)
    return MeshHierarchy(base, N_BRIDGE_REFINEMENTS)


def safe_log(expr):
    return ln(max_value(expr, TINY))


def clamp_expr(expr):
    return min_value(max_value(expr, -LOG_CLAMP), LOG_CLAMP)


def write_rows(path, rows):
    if COMM.rank != 0 or not rows:
        return
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def l2_norm(function):
    return float(math.sqrt(assemble(function * function * dx)))


def linf_norm(function):
    data = function.dat.data_ro
    local = float(abs(data).max()) if data.size else 0.0
    return float(COMM.allreduce(local, op=MPI.MAX))


def global_dof_max(function):
    data = function.dat.data_ro
    local = float(data.max()) if data.size else -math.inf
    return float(COMM.allreduce(local, op=MPI.MAX))


def global_dof_min(function):
    data = function.dat.data_ro
    local = float(data.min()) if data.size else math.inf
    return float(COMM.allreduce(local, op=MPI.MIN))


class FASLevel:
    def __init__(self, name, level_index, mesh, tau_physical, V0, V1, mu0, mu1, coarse=None):
        self.name = name
        self.level_index = level_index
        self.mesh = mesh
        self.bridge_n = BRIDGE_LEVELS[level_index]
        self.h_physical = BRIDGE_WIDTH_PHYSICAL / self.bridge_n
        self.tau_physical = float(tau_physical)
        self.tau_unit = self.tau_physical / (BRIDGE_WIDTH_PHYSICAL ** 2)
        self.tau = Constant(self.tau_unit)
        self.sub_tau = Constant(self.tau_unit / N_SUB)
        self.V0 = V0
        self.V1 = V1
        self.mu0 = mu0
        self.mu1 = mu1
        self.coarse = coarse

        self.Vb = FunctionSpace(mesh, "CG", BRIDGE_CG_DEGREE)
        self.rhs_source = Function(self.Vb, name=f"{name}_rhs_source")
        self.heat_source_bridge = Function(self.Vb, name=f"{name}_heat_source_bridge")
        self.rhs_target = Function(self.Vb, name=f"{name}_rhs_target")
        self.heat_target_bridge = Function(self.Vb, name=f"{name}_heat_target_bridge")

        trial = TrialFunction(self.Vb)
        test = TestFunction(self.Vb)
        self.sub_tau = Constant(self.tau_unit / N_SUB)
        a = (trial * test + self.sub_tau * dot(grad(trial), grad(test))) * dx

        L_source = self.rhs_source * test * dx
        problem_source = LinearVariationalProblem(
            a, L_source, self.heat_source_bridge, constant_jacobian=True
        )
        self.solver_source = LinearVariationalSolver(
            problem_source, solver_parameters=solver_params
        )

        L_target = self.rhs_target * test * dx
        problem_target = LinearVariationalProblem(
            a, L_target, self.heat_target_bridge, constant_jacobian=True
        )
        self.solver_target = LinearVariationalSolver(
            problem_target, solver_parameters=solver_params
        )

        self.exp_p = Function(V0, name=f"{name}_exp_p")
        self.exp_q = Function(V1, name=f"{name}_exp_q")
        self.Hq_on_0 = Function(V0, name=f"{name}_Hq_on_0")
        self.Hp_on_1 = Function(V1, name=f"{name}_Hp_on_1")

        self.Ap = Function(V0, name=f"{name}_A_p")
        self.Aq = Function(V1, name=f"{name}_A_q")
        self.rp = Function(V0, name=f"{name}_r_p")
        self.rq = Function(V1, name=f"{name}_r_q")

        self.rows = []
        self.heat_source_calls = 0
        self.heat_target_calls = 0
        self.backward_euler_solves = 0

        self.source_input_interp_seconds = 0.0
        self.source_solver_seconds = 0.0
        self.source_rhs_assign_seconds = 0.0
        self.source_output_interp_seconds = 0.0
        self.source_heat_total_seconds = 0.0
        self.source_logheat_total_seconds = 0.0
        self.source_first_heat_seconds = 0.0
        self.source_first_logheat_seconds = 0.0

        self.target_input_interp_seconds = 0.0
        self.target_solver_seconds = 0.0
        self.target_rhs_assign_seconds = 0.0
        self.target_output_interp_seconds = 0.0
        self.target_heat_total_seconds = 0.0
        self.target_logheat_total_seconds = 0.0
        self.target_first_heat_seconds = 0.0
        self.target_first_logheat_seconds = 0.0

        self.entry_rp = Function(self.V0, name=f"entry_rp_L{self.level_index}")
        self.entry_rq = Function(self.V1, name=f"entry_rq_L{self.level_index}")
        self.delta_p = Function(self.V0, name=f"delta_p_L{self.level_index}")
        self.delta_q = Function(self.V1, name=f"delta_q_L{self.level_index}")
        self.p_corrected = Function(self.V0, name=f"p_corrected_L{self.level_index}")
        self.q_corrected = Function(self.V1, name=f"q_corrected_L{self.level_index}")

        self.identity_p = Function(self.V0, name=f"identity_p_L{self.level_index}")
        self.identity_q = Function(self.V1, name=f"identity_q_L{self.level_index}")
        self.marginal_p = Function(self.V0, name=f"marginal_p_L{self.level_index}")
        self.marginal_q = Function(self.V1, name=f"marginal_q_L{self.level_index}")
        self.repeat_source_1 = Function(self.V0, name=f"repeat_source_1_L{self.level_index}")
        self.repeat_source_2 = Function(self.V0, name=f"repeat_source_2_L{self.level_index}")
        self.repeat_source_delta = Function(self.V0, name=f"repeat_source_delta_L{self.level_index}")
        self.repeat_target_1 = Function(self.V1, name=f"repeat_target_1_L{self.level_index}")
        self.repeat_target_2 = Function(self.V1, name=f"repeat_target_2_L{self.level_index}")
        self.repeat_target_delta = Function(self.V1, name=f"repeat_target_delta_L{self.level_index}")
        self.log_repeat_source_delta = Function(self.V0, name=f"log_repeat_source_delta_L{self.level_index}")
        self.log_repeat_target_delta = Function(self.V1, name=f"log_repeat_target_delta_L{self.level_index}")

        self.p_shifted = Function(self.V0, name=f"p_shifted_L{self.level_index}")
        self.q_shifted = Function(self.V1, name=f"q_shifted_L{self.level_index}")
        self.exp_p_shifted = Function(self.V0, name=f"exp_p_shifted_L{self.level_index}")
        self.exp_q_shifted = Function(self.V1, name=f"exp_q_shifted_L{self.level_index}")
        self.Hp_shifted_on_1 = Function(self.V1, name=f"Hp_shifted_on_1_L{self.level_index}")
        self.Hq_shifted_on_0 = Function(self.V0, name=f"Hq_shifted_on_0_L{self.level_index}")
        self.logHp_on_1 = Function(self.V1, name=f"logHp_on_1_L{self.level_index}")
        self.logHq_on_0 = Function(self.V0, name=f"logHq_on_0_L{self.level_index}")
        self.log_marginal_p = Function(self.V0, name=f"log_marginal_p_L{self.level_index}")
        self.log_marginal_q = Function(self.V1, name=f"log_marginal_q_L{self.level_index}")
        self.logHq_repeat_1 = Function(self.V0, name=f"logHq_repeat_1_L{self.level_index}")
        self.logHq_repeat_2 = Function(self.V0, name=f"logHq_repeat_2_L{self.level_index}")
        self.logHp_repeat_1 = Function(self.V1, name=f"logHp_repeat_1_L{self.level_index}")
        self.logHp_repeat_2 = Function(self.V1, name=f"logHp_repeat_2_L{self.level_index}")
        self.logheat_repeat_source_delta = Function(self.V0, name=f"logheat_repeat_source_delta_L{self.level_index}")
        self.logheat_repeat_target_delta = Function(self.V1, name=f"logheat_repeat_target_delta_L{self.level_index}")
        self.log_mu0 = Function(self.V0, name=f"log_mu0_L{self.level_index}").interpolate(safe_log(self.mu0))
        self.log_mu1 = Function(self.V1, name=f"log_mu1_L{self.level_index}").interpolate(safe_log(self.mu1))

        if coarse is not None:
            self.restricted_p = Function(self.V0, name=f"restricted_p_L{self.level_index}")
            self.restricted_q = Function(self.V1, name=f"restricted_q_L{self.level_index}")
            self.restricted_rp = Function(self.V0, name=f"restricted_rp_L{self.level_index}")
            self.restricted_rq = Function(self.V1, name=f"restricted_rq_L{self.level_index}")
            self.coarse_Ap = Function(coarse.V0, name=f"coarse_Ap_from_L{self.level_index}")
            self.coarse_Aq = Function(coarse.V1, name=f"coarse_Aq_from_L{self.level_index}")
            self.coarse_bp = Function(coarse.V0, name=f"coarse_bp_from_L{self.level_index}")
            self.coarse_bq = Function(coarse.V1, name=f"coarse_bq_from_L{self.level_index}")
            self.coarse_p = Function(coarse.V0, name=f"coarse_p_from_L{self.level_index}")
            self.coarse_q = Function(coarse.V1, name=f"coarse_q_from_L{self.level_index}")

    def hierarchy_row(self):
        local_cells = self.mesh.cell_set.size
        global_cells = COMM.allreduce(local_cells, op=MPI.SUM)
        return {
            "level": self.level_index,
            "name": self.name,
            "bridge_n": self.bridge_n,
            "global_cells": global_cells,
            "h_physical": self.h_physical,
            "tau_physical": self.tau_physical,
            "tau_unit": self.tau_unit,
            "tau_over_h2": self.tau_physical / (self.h_physical ** 2),
            "sqrt_2tau_over_h": math.sqrt(2.0 * self.tau_physical) / self.h_physical,
        }

    def heat_scaling_to_source(self, scaling, output):
        first_call = self.heat_source_calls == 0
        heat_t0 = time.perf_counter()

        # Start each complete heat action from a defined solution state.
        self.heat_source_bridge.assign(0.0)

        self.heat_source_calls += 1
        self.backward_euler_solves += N_SUB

        t0 = time.perf_counter()
        self.rhs_source.interpolate(
            scaling,
            allow_missing_dofs=True,
            default_missing_val=0.0,
        )
        self.source_input_interp_seconds += time.perf_counter() - t0

        for _ in range(N_SUB):
            t0 = time.perf_counter()
            self.solver_source.solve()
            self.source_solver_seconds += time.perf_counter() - t0

            t0 = time.perf_counter()
            self.rhs_source.assign(self.heat_source_bridge)
            self.source_rhs_assign_seconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        output.interpolate(self.heat_source_bridge, allow_missing_dofs=False)
        self.source_output_interp_seconds += time.perf_counter() - t0

        heat_elapsed = time.perf_counter() - heat_t0
        self.source_heat_total_seconds += heat_elapsed
        if first_call:
            self.source_first_heat_seconds = heat_elapsed

        return output

    def heat_scaling_to_target(self, scaling, output):
        first_call = self.heat_target_calls == 0
        heat_t0 = time.perf_counter()

        # Symmetric with the source action; do not reset inside the BE loop.
        self.heat_target_bridge.assign(0.0)

        self.heat_target_calls += 1
        self.backward_euler_solves += N_SUB

        t0 = time.perf_counter()
        self.rhs_target.interpolate(
            scaling,
            allow_missing_dofs=True,
            default_missing_val=0.0,
        )
        self.target_input_interp_seconds += time.perf_counter() - t0

        for _ in range(N_SUB):
            t0 = time.perf_counter()
            self.solver_target.solve()
            self.target_solver_seconds += time.perf_counter() - t0

            t0 = time.perf_counter()
            self.rhs_target.assign(self.heat_target_bridge)
            self.target_rhs_assign_seconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        output.interpolate(self.heat_target_bridge, allow_missing_dofs=False)
        self.target_output_interp_seconds += time.perf_counter() - t0

        heat_elapsed = time.perf_counter() - heat_t0
        self.target_heat_total_seconds += heat_elapsed
        if first_call:
            self.target_first_heat_seconds = heat_elapsed

        return output

    def stable_log_heat_to_source(self, q, out_logH):
        first_call = self.heat_source_calls == 0
        logheat_t0 = time.perf_counter()

        q_shift = global_dof_max(q)
        q_shift_c = Constant(q_shift)
        self.q_shifted.interpolate(q - q_shift_c)
        self.exp_q_shifted.interpolate(exp(self.q_shifted))
        self.heat_scaling_to_source(self.exp_q_shifted, self.Hq_shifted_on_0)
        out_logH.interpolate(q_shift_c + safe_log(self.Hq_shifted_on_0))

        logheat_elapsed = time.perf_counter() - logheat_t0
        self.source_logheat_total_seconds += logheat_elapsed
        if first_call:
            self.source_first_logheat_seconds = logheat_elapsed

        return q_shift

    def stable_log_heat_to_target(self, p, out_logH):
        first_call = self.heat_target_calls == 0
        logheat_t0 = time.perf_counter()

        p_shift = global_dof_max(p)
        p_shift_c = Constant(p_shift)
        self.p_shifted.interpolate(p - p_shift_c)
        self.exp_p_shifted.interpolate(exp(self.p_shifted))
        self.heat_scaling_to_target(self.exp_p_shifted, self.Hp_shifted_on_1)
        out_logH.interpolate(p_shift_c + safe_log(self.Hp_shifted_on_1))

        logheat_elapsed = time.perf_counter() - logheat_t0
        self.target_logheat_total_seconds += logheat_elapsed
        if first_call:
            self.target_first_logheat_seconds = logheat_elapsed

        return p_shift

    def apply_operator(self, p, q, out_p, out_q):
        self.stable_log_heat_to_source(q, self.logHq_on_0)
        out_p.interpolate(p - self.log_mu0 + self.logHq_on_0)
        self.stable_log_heat_to_target(p, self.logHp_on_1)
        out_q.interpolate(q - self.log_mu1 + self.logHp_on_1)
        return out_p, out_q

    def residual(self, p, q, b_p, b_q, out_p, out_q):
        self.apply_operator(p, q, self.Ap, self.Aq)
        out_p.interpolate(b_p - self.Ap)
        out_q.interpolate(b_q - self.Aq)
        return out_p, out_q

    def gauge_fix(self, p, q):
        p_mean = float(assemble(p * self.mu0 * dx))
        q_mean = float(assemble(q * self.mu1 * dx))
        shift = 0.5 * (p_mean - q_mean)

        self.p_corrected.interpolate(p - Constant(shift))
        self.q_corrected.interpolate(q + Constant(shift))
        p.assign(self.p_corrected)
        q.assign(self.q_corrected)
        return shift

    def residual_norms(self, p, q, b_p, b_q):
        self.residual(p, q, b_p, b_q, self.rp, self.rq)
        return l2_norm(self.rp), l2_norm(self.rq)

    def p_residual_norm(self, p, q, b_p):
        self.stable_log_heat_to_source(q, self.logHq_on_0)
        self.rp.interpolate(
            b_p
            - (
                p
                - self.log_mu0
                + self.logHq_on_0
            )
        )
        return l2_norm(self.rp)

    def smoother(self, p, q, b_p, b_q, updates, label):
        COMM.Barrier()
        t0 = time.perf_counter()
        last_q_shift = math.nan
        last_p_shift = math.nan
        for _ in range(int(updates)):
            last_q_shift = self.stable_log_heat_to_source(q, self.logHq_on_0)
            p.interpolate(b_p + self.log_mu0 - self.logHq_on_0)
            self.gauge_fix(p, q)

            last_p_shift = self.stable_log_heat_to_target(p, self.logHp_on_1)
            q.interpolate(b_q + self.log_mu1 - self.logHp_on_1)
            self.gauge_fix(p, q)
        COMM.Barrier()
        elapsed = COMM.allreduce(time.perf_counter() - t0, op=MPI.MAX)
        self.rows.append({
            "level": self.name,
            "event": f"smoother_{label}",
            "updates": int(updates),
            "seconds": elapsed,
            "last_p_shift": last_p_shift,
            "last_q_shift": last_q_shift,
        })

    def restrict_state(self, p, q):
        self.restricted_p.assign(p)
        self.restricted_q.assign(q)
        return self.restricted_p, self.restricted_q

    def restrict_residual(self):
        self.restricted_rp.assign(self.rp)
        self.restricted_rq.assign(self.rq)
        return self.restricted_rp, self.restricted_rq

    def build_coarse_rhs(self, restricted_p, restricted_q, restricted_rp, restricted_rq):
        self.coarse.apply_operator(restricted_p, restricted_q, self.coarse_Ap, self.coarse_Aq)
        self.coarse_bp.interpolate(self.coarse_Ap + restricted_rp)
        self.coarse_bq.interpolate(self.coarse_Aq + restricted_rq)
        return self.coarse_bp, self.coarse_bq

    def prolong_correction(self, p, q, coarse_p, coarse_q, restricted_p, restricted_q):
        self.delta_p.interpolate(coarse_p - restricted_p)
        self.delta_q.interpolate(coarse_q - restricted_q)
        self.p_corrected.interpolate(p + Constant(CORRECTION_DAMPING) * self.delta_p)
        self.q_corrected.interpolate(q + Constant(CORRECTION_DAMPING) * self.delta_q)
        p.assign(self.p_corrected)
        q.assign(self.q_corrected)
        return l2_norm(self.delta_p), l2_norm(self.delta_q)

    def field_stats(self, field, prefix):
        data = field.dat.data_ro
        local_min = float(data.min()) if data.size else math.inf
        local_max = float(data.max()) if data.size else -math.inf

        global_min = COMM.allreduce(local_min, op=MPI.MIN)
        global_max = COMM.allreduce(local_max, op=MPI.MAX)

        return {
            f"{prefix}_min": global_min,
            f"{prefix}_max": global_max,
            f"{prefix}_below_neg_clamp_area": float(
                assemble(conditional(field < -LOG_CLAMP, 1.0, 0.0) * dx)
            ),
            f"{prefix}_above_pos_clamp_area": float(
                assemble(conditional(field > LOG_CLAMP, 1.0, 0.0) * dx)
            ),
        }

    def heat_stats(self, field, prefix):
        data = field.dat.data_ro
        local_min = float(data.min()) if data.size else math.inf
        local_max = float(data.max()) if data.size else -math.inf

        global_min = COMM.allreduce(local_min, op=MPI.MIN)
        global_max = COMM.allreduce(local_max, op=MPI.MAX)

        return {
            f"{prefix}_min": global_min,
            f"{prefix}_max": global_max,
            f"{prefix}_nonpositive_area": float(
                assemble(conditional(field <= 0.0, 1.0, 0.0) * dx)
            ),
            f"{prefix}_safe_log_floor_area": float(
                assemble(conditional(field <= TINY, 1.0, 0.0) * dx)
            ),
        }

    def marginal_residual_norms(self, p, q):
        self.stable_log_heat_to_source(q, self.logHq_on_0)
        self.log_marginal_p.interpolate(p + self.logHq_on_0)
        self.marginal_p.interpolate(exp(clamp_expr(self.log_marginal_p)) - self.mu0)

        self.stable_log_heat_to_target(p, self.logHp_on_1)
        self.log_marginal_q.interpolate(q + self.logHp_on_1)
        self.marginal_q.interpolate(exp(clamp_expr(self.log_marginal_q)) - self.mu1)

        return l2_norm(self.marginal_p), l2_norm(self.marginal_q)

    def heat_repeatability(self, p, q):
        q_shift_1 = self.stable_log_heat_to_source(q, self.logHq_repeat_1)
        q_shift_2 = self.stable_log_heat_to_source(q, self.logHq_repeat_2)
        self.logheat_repeat_source_delta.interpolate(
            self.logHq_repeat_2 - self.logHq_repeat_1
        )

        p_shift_1 = self.stable_log_heat_to_target(p, self.logHp_repeat_1)
        p_shift_2 = self.stable_log_heat_to_target(p, self.logHp_repeat_2)
        self.logheat_repeat_target_delta.interpolate(
            self.logHp_repeat_2 - self.logHp_repeat_1
        )

        return {
            "heat_repeat_source_log_l2": l2_norm(self.logheat_repeat_source_delta),
            "heat_repeat_target_log_l2": l2_norm(self.logheat_repeat_target_delta),
            "logheat_repeat_source_l2": l2_norm(self.logheat_repeat_source_delta),
            "logheat_repeat_source_linf": linf_norm(self.logheat_repeat_source_delta),
            "logheat_repeat_target_l2": l2_norm(self.logheat_repeat_target_delta),
            "logheat_repeat_target_linf": linf_norm(self.logheat_repeat_target_delta),
            "q_shift_1": q_shift_1,
            "q_shift_2": q_shift_2,
            "p_shift_1": p_shift_1,
            "p_shift_2": p_shift_2,
            "q_shifted_min": global_dof_min(self.q_shifted),
            "q_shifted_max": global_dof_max(self.q_shifted),
            "p_shifted_min": global_dof_min(self.p_shifted),
            "p_shifted_max": global_dof_max(self.p_shifted),
            "exp_q_shifted_min": global_dof_min(self.exp_q_shifted),
            "exp_q_shifted_max": global_dof_max(self.exp_q_shifted),
            "exp_p_shifted_min": global_dof_min(self.exp_p_shifted),
            "exp_p_shifted_max": global_dof_max(self.exp_p_shifted),
            "Hq_shifted_min": global_dof_min(self.Hq_shifted_on_0),
            "Hp_shifted_min": global_dof_min(self.Hp_shifted_on_1),
        }

    def smoother_test(self, p, q, b_p, b_q, updates):
        rows = []

        def add_row(label, sweep, gauge_shift="", cached_p="", cached_q="", repeat=None, extra=None):
            fas_p, fas_q = self.residual_norms(p, q, b_p, b_q)
            marginal_p, marginal_q = self.marginal_residual_norms(p, q)

            row = {
                "mode": "smoother_test",
                "label": label,
                "sweep": sweep,
                "gauge_shift": gauge_shift,
                "cached_identity_p": cached_p,
                "cached_identity_q": cached_q,
                "fas_residual_p": fas_p,
                "fas_residual_q": fas_q,
                "marginal_residual_p": marginal_p,
                "marginal_residual_q": marginal_q,
            }
            row.update(self.field_stats(p, "p"))
            row.update(self.field_stats(q, "q"))
            if repeat:
                row.update(repeat)
            if extra:
                row.update(extra)

            rows.append(row)
            if COMM.rank == 0:
                print("FAS_SMOOTHER_TEST", row)

        add_row("initial", 0)

        for sweep in range(1, int(updates) + 1):
            q_shift = self.stable_log_heat_to_source(q, self.logHq_on_0)
            Hq_stats = self.heat_stats(self.Hq_shifted_on_0, "Hq_shifted"); Hq_stats["q_shift"] = q_shift
            p.interpolate(b_p + self.log_mu0 - self.logHq_on_0)
            self.identity_p.interpolate(p - b_p - self.log_mu0 + self.logHq_on_0)
            cached_p = l2_norm(self.identity_p)

            add_row("after_p_update_before_gauge", sweep, cached_p=cached_p, extra=Hq_stats)

            shift_p = self.gauge_fix(p, q)
            add_row("after_p_gauge", sweep, gauge_shift=shift_p, cached_p=cached_p)

            p_shift = self.stable_log_heat_to_target(p, self.logHp_on_1)
            Hp_stats = self.heat_stats(self.Hp_shifted_on_1, "Hp_shifted"); Hp_stats["p_shift"] = p_shift
            q.interpolate(b_q + self.log_mu1 - self.logHp_on_1)
            self.identity_q.interpolate(q - b_q - self.log_mu1 + self.logHp_on_1)
            cached_q = l2_norm(self.identity_q)

            add_row("after_q_update_before_gauge", sweep, cached_q=cached_q, extra=Hp_stats)

            shift_q = self.gauge_fix(p, q)
            repeat = self.heat_repeatability(p, q)
            add_row("after_q_gauge", sweep, gauge_shift=shift_q, cached_q=cached_q, repeat=repeat)

        return rows


    def sign_test(self, p, q, b_p, b_q, pre_smooth):
        before_p, before_q = self.residual_norms(p, q, b_p, b_q)

        if pre_smooth:
            self.smoother(p, q, b_p, b_q, pre_smooth, "sign_test_pre_smooth")

        self.gauge_fix(p, q)
        after_pre_p, after_pre_q = self.residual_norms(p, q, b_p, b_q)

        restricted_p, restricted_q = self.restrict_state(p, q)
        restricted_rp, restricted_rq = self.restrict_residual()
        coarse_bp, coarse_bq = self.build_coarse_rhs(
            restricted_p, restricted_q, restricted_rp, restricted_rq
        )

        self.coarse.residual(
            restricted_p, restricted_q,
            coarse_bp, coarse_bq,
            self.entry_rp, self.entry_rq,
        )

        self.delta_p.interpolate(self.entry_rp - restricted_rp)
        self.delta_q.interpolate(self.entry_rq - restricted_rq)

        rp_norm = l2_norm(restricted_rp)
        rq_norm = l2_norm(restricted_rq)
        sign_abs_p = l2_norm(self.delta_p)
        sign_abs_q = l2_norm(self.delta_q)
        sign_rel_p = sign_abs_p / max(rp_norm, 1.0e-30)
        sign_rel_q = sign_abs_q / max(rq_norm, 1.0e-30)

        row = {
            "mode": "sign_test",
            "before_residual_p": before_p,
            "before_residual_q": before_q,
            "after_pre_residual_p": after_pre_p,
            "after_pre_residual_q": after_pre_q,
            "coarse_entry_residual_p": l2_norm(self.entry_rp),
            "coarse_entry_residual_q": l2_norm(self.entry_rq),
            "restricted_fine_residual_p": rp_norm,
            "restricted_fine_residual_q": rq_norm,
            "fas_rhs_sign_error_p": sign_abs_p,
            "fas_rhs_sign_error_q": sign_abs_q,
            "fas_rhs_sign_relative_error_p": sign_rel_p,
            "fas_rhs_sign_relative_error_q": sign_rel_q,
        }

        if COMM.rank == 0:
            print("FAS_SIGN_TEST", row)

        return row


    def reset_counters(self):
        self.heat_source_calls = 0
        self.heat_target_calls = 0
        self.backward_euler_solves = 0

        self.source_input_interp_seconds = 0.0
        self.source_solver_seconds = 0.0
        self.source_rhs_assign_seconds = 0.0
        self.source_output_interp_seconds = 0.0
        self.source_heat_total_seconds = 0.0
        self.source_logheat_total_seconds = 0.0
        self.source_first_heat_seconds = 0.0
        self.source_first_logheat_seconds = 0.0

        self.target_input_interp_seconds = 0.0
        self.target_solver_seconds = 0.0
        self.target_rhs_assign_seconds = 0.0
        self.target_output_interp_seconds = 0.0
        self.target_heat_total_seconds = 0.0
        self.target_logheat_total_seconds = 0.0
        self.target_first_heat_seconds = 0.0
        self.target_first_logheat_seconds = 0.0

        if self.coarse is not None:
            self.coarse.reset_counters()

    def counter_row(self):
        def mpi_max(value):
            return COMM.allreduce(value, op=MPI.MAX)

        source_calls = mpi_max(self.heat_source_calls)
        target_calls = mpi_max(self.heat_target_calls)
        total_calls = mpi_max(self.heat_source_calls + self.heat_target_calls)
        be_solves = mpi_max(self.backward_euler_solves)

        source_input = mpi_max(self.source_input_interp_seconds)
        source_solver = mpi_max(self.source_solver_seconds)
        source_assign = mpi_max(self.source_rhs_assign_seconds)
        source_output = mpi_max(self.source_output_interp_seconds)
        source_heat = mpi_max(self.source_heat_total_seconds)
        source_logheat = mpi_max(self.source_logheat_total_seconds)
        source_first_heat = mpi_max(self.source_first_heat_seconds)
        source_first_logheat = mpi_max(self.source_first_logheat_seconds)

        target_input = mpi_max(self.target_input_interp_seconds)
        target_solver = mpi_max(self.target_solver_seconds)
        target_assign = mpi_max(self.target_rhs_assign_seconds)
        target_output = mpi_max(self.target_output_interp_seconds)
        target_heat = mpi_max(self.target_heat_total_seconds)
        target_logheat = mpi_max(self.target_logheat_total_seconds)
        target_first_heat = mpi_max(self.target_first_heat_seconds)
        target_first_logheat = mpi_max(self.target_first_logheat_seconds)

        heat_total_seconds = mpi_max(
            self.source_heat_total_seconds + self.target_heat_total_seconds
        )
        logheat_total_seconds = mpi_max(
            self.source_logheat_total_seconds + self.target_logheat_total_seconds
        )
        solver_total_seconds = mpi_max(
            self.source_solver_seconds + self.target_solver_seconds
        )
        assign_total_seconds = mpi_max(
            self.source_rhs_assign_seconds + self.target_rhs_assign_seconds
        )
        input_interp_total_seconds = mpi_max(
            self.source_input_interp_seconds + self.target_input_interp_seconds
        )
        output_interp_total_seconds = mpi_max(
            self.source_output_interp_seconds + self.target_output_interp_seconds
        )

        source_be_solves = source_calls * N_SUB
        target_be_solves = target_calls * N_SUB

        return {
            "level": self.name,
            "bridge_n": self.bridge_n,
            "heat_source_calls": source_calls,
            "heat_target_calls": target_calls,
            "heat_total_calls": total_calls,
            "backward_euler_solves": be_solves,
            "source_be_solves": source_be_solves,
            "target_be_solves": target_be_solves,

            "source_input_interp_seconds": source_input,
            "source_solver_seconds": source_solver,
            "source_rhs_assign_seconds": source_assign,
            "source_output_interp_seconds": source_output,
            "source_heat_total_seconds": source_heat,
            "source_logheat_total_seconds": source_logheat,
            "source_first_heat_seconds": source_first_heat,
            "source_first_logheat_seconds": source_first_logheat,

            "target_input_interp_seconds": target_input,
            "target_solver_seconds": target_solver,
            "target_rhs_assign_seconds": target_assign,
            "target_output_interp_seconds": target_output,
            "target_heat_total_seconds": target_heat,
            "target_logheat_total_seconds": target_logheat,
            "target_first_heat_seconds": target_first_heat,
            "target_first_logheat_seconds": target_first_logheat,

            "input_interp_total_seconds": input_interp_total_seconds,
            "solver_total_seconds": solver_total_seconds,
            "rhs_assign_total_seconds": assign_total_seconds,
            "output_interp_total_seconds": output_interp_total_seconds,
            "heat_total_seconds": heat_total_seconds,
            "logheat_total_seconds": logheat_total_seconds,

            "mean_source_heat_seconds": source_heat / max(source_calls, 1),
            "mean_target_heat_seconds": target_heat / max(target_calls, 1),
            "mean_heat_seconds": heat_total_seconds / max(total_calls, 1),

            "mean_source_logheat_seconds": source_logheat / max(source_calls, 1),
            "mean_target_logheat_seconds": target_logheat / max(target_calls, 1),
            "mean_logheat_seconds": logheat_total_seconds / max(total_calls, 1),

            "mean_source_solver_seconds_per_be": source_solver / max(source_be_solves, 1),
            "mean_target_solver_seconds_per_be": target_solver / max(target_be_solves, 1),
            "mean_solver_seconds_per_be": solver_total_seconds / max(be_solves, 1),

            "mean_rhs_assign_seconds_per_be": assign_total_seconds / max(be_solves, 1),
            "mean_input_interp_seconds_per_heat": input_interp_total_seconds / max(total_calls, 1),
            "mean_output_interp_seconds_per_heat": output_interp_total_seconds / max(total_calls, 1),
            "mean_logheat_wrapper_overhead_seconds": max(
                logheat_total_seconds - heat_total_seconds,
                0.0,
            ) / max(total_calls, 1),

            "solver_fraction_of_heat": solver_total_seconds / max(heat_total_seconds, 1.0e-300),
            "rhs_assign_fraction_of_heat": assign_total_seconds / max(heat_total_seconds, 1.0e-300),
            "transfer_fraction_of_heat": (
                input_interp_total_seconds + output_interp_total_seconds
            ) / max(heat_total_seconds, 1.0e-300),
        }

    def counter_rows(self):
        rows = []
        if self.coarse is not None:
            rows.extend(self.coarse.counter_rows())
        rows.append(self.counter_row())
        return rows

    def fas_cycle_perf(self, p, q, b_p, b_q):
        if COMM.rank == 0:
            print(
                "FAS_ENTER_PERF",
                f"level={self.level_index}",
                f"B={self.bridge_n}",
                f"has_coarse={int(self.coarse is not None)}",
            )

        if self.coarse is None:
            if COMM.rank == 0:
                print(
                    "FAS_BOTTOM_PERF_ENTER",
                    f"level={self.level_index}",
                    f"B={self.bridge_n}",
                    f"bottom_sweeps={COARSE_SMOOTH}",
                )
            self.smoother(
                p, q,
                b_p, b_q,
                COARSE_SMOOTH,
                f"level{self.level_index}_bottom_perf",
            )
            self.gauge_fix(p, q)
            return

        self.smoother(
            p, q,
            b_p, b_q,
            PRE_SMOOTH,
            f"level{self.level_index}_pre_perf",
        )
        self.gauge_fix(p, q)

        self.residual(p, q, b_p, b_q, self.rp, self.rq)

        restricted_p, restricted_q = self.restrict_state(p, q)
        restricted_rp, restricted_rq = self.restrict_residual()
        coarse_bp, coarse_bq = self.build_coarse_rhs(
            restricted_p, restricted_q, restricted_rp, restricted_rq
        )

        self.coarse_p.assign(restricted_p)
        self.coarse_q.assign(restricted_q)

        if COMM.rank == 0:
            print(
                "FAS_RECURSE_PERF_CALL",
                f"from_level={self.level_index}",
                f"from_B={self.bridge_n}",
                f"to_level={self.coarse.level_index}",
                f"to_B={self.coarse.bridge_n}",
            )

        before_source = self.coarse.heat_source_calls
        before_target = self.coarse.heat_target_calls

        self.coarse.fas_cycle_perf(
            self.coarse_p,
            self.coarse_q,
            coarse_bp,
            coarse_bq,
        )

        after_source = self.coarse.heat_source_calls
        after_target = self.coarse.heat_target_calls

        if COMM.rank == 0:
            print(
                "FAS_RECURSE_PERF_RETURN",
                f"from_B={self.bridge_n}",
                f"to_B={self.coarse.bridge_n}",
                f"source_delta={after_source - before_source}",
                f"target_delta={after_target - before_target}",
                f"total_delta={(after_source + after_target) - (before_source + before_target)}",
            )

        self.prolong_correction(
            p, q,
            self.coarse_p, self.coarse_q,
            restricted_p, restricted_q,
        )

        self.gauge_fix(p, q)

        self.smoother(
            p, q,
            b_p, b_q,
            POST_SMOOTH,
            f"level{self.level_index}_post_perf",
        )
        self.gauge_fix(p, q)

    def fas_cycle(self, p, q, b_p, b_q, depth=0):
        before_p, before_q = self.residual_norms(p, q, b_p, b_q)

        if self.coarse is None:
            self.smoother(
                p, q,
                b_p, b_q,
                COARSE_SMOOTH,
                f"level{self.level_index}_bottom",
            )
            self.gauge_fix(p, q)
            after_p, after_q = self.residual_norms(p, q, b_p, b_q)

            return {
                "level": self.level_index,
                "depth": depth,
                "kind": "bottom",
                "before_p": before_p,
                "before_q": before_q,
                "after_pre_p": before_p,
                "after_pre_q": before_q,
                "after_correction_p": after_p,
                "after_correction_q": after_q,
                "correction_p": 0.0,
                "correction_q": 0.0,
                "after_cycle_p": after_p,
                "after_cycle_q": after_q,
            }

        self.smoother(
            p, q,
            b_p, b_q,
            PRE_SMOOTH,
            f"level{self.level_index}_pre",
        )
        self.gauge_fix(p, q)

        after_pre_p, after_pre_q = self.residual_norms(p, q, b_p, b_q)

        restricted_p, restricted_q = self.restrict_state(p, q)
        restricted_rp, restricted_rq = self.restrict_residual()
        coarse_bp, coarse_bq = self.build_coarse_rhs(
            restricted_p, restricted_q, restricted_rp, restricted_rq
        )

        self.coarse_p.assign(restricted_p)
        self.coarse_q.assign(restricted_q)

        coarse_row = self.coarse.fas_cycle(
            self.coarse_p, self.coarse_q,
            coarse_bp, coarse_bq,
            depth + 1,
        )

        correction_p, correction_q = self.prolong_correction(
            p, q,
            self.coarse_p, self.coarse_q,
            restricted_p, restricted_q,
        )

        self.gauge_fix(p, q)
        after_correction_p, after_correction_q = self.residual_norms(p, q, b_p, b_q)

        self.smoother(
            p, q,
            b_p, b_q,
            POST_SMOOTH,
            f"level{self.level_index}_post",
        )
        self.gauge_fix(p, q)

        after_cycle_p, after_cycle_q = self.residual_norms(p, q, b_p, b_q)

        return {
            "level": self.level_index,
            "depth": depth,
            "kind": "fas",
            "before_p": before_p,
            "before_q": before_q,
            "after_pre_p": after_pre_p,
            "after_pre_q": after_pre_q,
            "after_correction_p": after_correction_p,
            "after_correction_q": after_correction_q,
            "coarse_kind": coarse_row["kind"],
            "coarse_after_p": coarse_row["after_cycle_p"],
            "coarse_after_q": coarse_row["after_cycle_q"],
            "correction_p": correction_p,
            "correction_q": correction_q,
            "correction_damping": CORRECTION_DAMPING,
            "after_cycle_p": after_cycle_p,
            "after_cycle_q": after_cycle_q,
        }




COUNTER_FIELDS = (
    "heat_source_calls", "heat_target_calls", "backward_euler_solves",
    "source_input_interp_seconds", "source_solver_seconds",
    "source_rhs_assign_seconds", "source_output_interp_seconds",
    "source_heat_total_seconds", "source_logheat_total_seconds",
    "source_first_heat_seconds", "source_first_logheat_seconds",
    "target_input_interp_seconds", "target_solver_seconds",
    "target_rhs_assign_seconds", "target_output_interp_seconds",
    "target_heat_total_seconds", "target_logheat_total_seconds",
    "target_first_heat_seconds", "target_first_logheat_seconds",
)


def snapshot_counters(fine):
    snapshots = []
    level = fine
    while level is not None:
        snapshots.append((level, {name: getattr(level, name) for name in COUNTER_FIELDS}))
        level = level.coarse
    return snapshots


def restore_counters(snapshots):
    for level, values in snapshots:
        for name, value in values.items():
            setattr(level, name, value)


def function_difference_stats(current, reference, name):
    difference = Function(current.function_space(), name=f"v44_{name}_difference")
    difference.interpolate(current - reference)
    return {
        f"{name}_frozen_l2": l2_norm(difference),
        f"{name}_frozen_linf": linf_norm(difference),
    }


def exact_error_checkpoint(fine, p, q, b_p, b_q, checkpoint, cycle, residual_p):
    audit_t0 = time.perf_counter()
    counter_state = snapshot_counters(fine)
    p_reference = Function(p.function_space(), name=f"v44_p_reference_{checkpoint}").assign(p)
    q_reference = Function(q.function_space(), name=f"v44_q_reference_{checkpoint}").assign(q)
    bp_reference = Function(b_p.function_space(), name=f"v44_bp_reference_{checkpoint}").assign(b_p)
    bq_reference = Function(b_q.function_space(), name=f"v44_bq_reference_{checkpoint}").assign(b_q)

    V0 = fine.V0
    V1 = fine.V1
    x = SpatialCoordinate(V0.mesh())
    y = SpatialCoordinate(V1.mesh())
    width = BRIDGE_WIDTH_PHYSICAL
    shift_x_unit = (TARGET_CENTER_PHYSICAL[0] - SOURCE_CENTER_PHYSICAL[0]) / width
    shift_y_unit = (TARGET_CENTER_PHYSICAL[1] - SOURCE_CENTER_PHYSICAL[1]) / width

    q_shift = global_dof_max(q)
    q_shifted = Function(V1, name=f"v44_{checkpoint}_q_shifted")
    exp_q_shifted = Function(V1, name=f"v44_{checkpoint}_exp_q_shifted")
    q_shifted.interpolate(q - Constant(q_shift))
    exp_q_shifted.interpolate(exp(q_shifted))

    denominator = Function(V0, name=f"v44_{checkpoint}_denominator")
    y0_weight = Function(V1, name=f"v44_{checkpoint}_y0_weight")
    y1_weight = Function(V1, name=f"v44_{checkpoint}_y1_weight")
    numerator0 = Function(V0, name=f"v44_{checkpoint}_numerator0")
    numerator1 = Function(V0, name=f"v44_{checkpoint}_numerator1")
    fine.heat_scaling_to_source(exp_q_shifted, denominator)
    y0_weight.interpolate(y[0] * exp_q_shifted)
    y1_weight.interpolate(y[1] * exp_q_shifted)
    fine.heat_scaling_to_source(y0_weight, numerator0)
    fine.heat_scaling_to_source(y1_weight, numerator1)

    T0 = Function(V0, name=f"v44_{checkpoint}_T0")
    T1 = Function(V0, name=f"v44_{checkpoint}_T1")
    T0.interpolate(numerator0 / max_value(denominator, TINY))
    T1.interpolate(numerator1 / max_value(denominator, TINY))
    T0_exact = x[0] + Constant(shift_x_unit)
    T1_exact = x[1] + Constant(shift_y_unit)
    map_error = Function(V0, name=f"v44_{checkpoint}_map_error")
    map_error.interpolate(sqrt((T0 - T0_exact) ** 2 + (T1 - T1_exact) ** 2))

    T_field = as_vector([T0, T1])
    det_space = FunctionSpace(V0.mesh(), "DG", 0)
    det_T = Function(det_space, name=f"v44_{checkpoint}_det_DT")
    det_T.interpolate(det(grad(T_field)))

    row = {
        "checkpoint": checkpoint,
        "cycle": cycle,
        "production_residual_p": residual_p,
        "map_l2_physical": width * math.sqrt(float(assemble(map_error ** 2 * fine.mu0 * dx))),
        "map_linf_physical": width * global_dof_max(map_error),
        "det_min": global_dof_min(det_T),
        "det_max": global_dof_max(det_T),
        "tangled_area": float(
            assemble(conditional(lt(det(grad(T_field)), 0.0), 1.0, 0.0) * dx(domain=V0.mesh()))
        ),
    }
    row.update(function_difference_stats(p, p_reference, "p"))
    row.update(function_difference_stats(q, q_reference, "q"))
    row.update(function_difference_stats(b_p, bp_reference, "b_p"))
    row.update(function_difference_stats(b_q, bq_reference, "b_q"))
    row["frozen_max"] = max(
        row["p_frozen_linf"], row["q_frozen_linf"],
        row["b_p_frozen_linf"], row["b_q_frozen_linf"],
    )
    restore_counters(counter_state)
    row["diagnostic_seconds"] = COMM.allreduce(time.perf_counter() - audit_t0, op=MPI.MAX)
    return row


def add_checkpoint_ratios(rows):
    residual0 = rows[0]["production_residual_p"]
    l2_error0 = rows[0]["map_l2_physical"]
    linf_error0 = rows[0]["map_linf_physical"]
    for row in rows:
        row["residual_ratio_to_entry"] = row["production_residual_p"] / max(residual0, 1.0e-300)
        row["map_l2_ratio_to_entry"] = row["map_l2_physical"] / max(l2_error0, 1.0e-300)
        row["map_linf_ratio_to_entry"] = row["map_linf_physical"] / max(linf_error0, 1.0e-300)

def translation_map_audit(fine, q, result_dir, method):
    V0 = fine.V0
    V1 = fine.V1
    x = SpatialCoordinate(V0.mesh())
    y = SpatialCoordinate(V1.mesh())

    width = BRIDGE_WIDTH_PHYSICAL
    shift_x_unit = (TARGET_CENTER_PHYSICAL[0] - SOURCE_CENTER_PHYSICAL[0]) / width
    shift_y_unit = (TARGET_CENTER_PHYSICAL[1] - SOURCE_CENTER_PHYSICAL[1]) / width

    q_shift = global_dof_max(q)
    q_shifted = Function(V1, name=f"{method}_q_shifted_for_map")
    exp_q_shifted = Function(V1, name=f"{method}_exp_q_shifted_for_map")
    q_shifted.interpolate(q - Constant(q_shift))
    exp_q_shifted.interpolate(exp(q_shifted))

    Hden = Function(V0, name=f"{method}_map_denominator")
    fine.heat_scaling_to_source(exp_q_shifted, Hden)

    y0_weight = Function(V1, name=f"{method}_y0_weight")
    y1_weight = Function(V1, name=f"{method}_y1_weight")
    ysq_weight = Function(V1, name=f"{method}_ysq_weight")

    y0_weight.interpolate(y[0] * exp_q_shifted)
    y1_weight.interpolate(y[1] * exp_q_shifted)
    ysq_weight.interpolate((y[0] ** 2 + y[1] ** 2) * exp_q_shifted)

    Hnum0 = Function(V0, name=f"{method}_map_num0")
    Hnum1 = Function(V0, name=f"{method}_map_num1")
    Hnumsq = Function(V0, name=f"{method}_map_numsq")

    fine.heat_scaling_to_source(y0_weight, Hnum0)
    fine.heat_scaling_to_source(y1_weight, Hnum1)
    fine.heat_scaling_to_source(ysq_weight, Hnumsq)

    T0 = Function(V0, name=f"{method}_T0_unit")
    T1 = Function(V0, name=f"{method}_T1_unit")
    T0.interpolate(Hnum0 / max_value(Hden, TINY))
    T1.interpolate(Hnum1 / max_value(Hden, TINY))

    T0_exact = x[0] + Constant(shift_x_unit)
    T1_exact = x[1] + Constant(shift_y_unit)

    map_error_unit = Function(V0, name=f"{method}_map_error_unit")
    map_error_unit.interpolate(
        sqrt((T0 - T0_exact) ** 2 + (T1 - T1_exact) ** 2)
    )

    map_l2_unit = math.sqrt(float(assemble(map_error_unit ** 2 * fine.mu0 * dx)))
    map_l2_physical = width * map_l2_unit
    map_linf_unit = global_dof_max(map_error_unit)
    map_linf_physical = width * map_linf_unit

    disp_x_unit = float(assemble((T0 - x[0]) * fine.mu0 * dx))
    disp_y_unit = float(assemble((T1 - x[1]) * fine.mu0 * dx))

    T_field = as_vector([T0, T1])
    det_space = FunctionSpace(V0.mesh(), "DG", 0)
    det_T = Function(det_space, name=f"{method}_det_DT")
    det_T.interpolate(det(grad(T_field)))

    det_min = global_dof_min(det_T)
    det_max = global_dof_max(det_T)
    tangled_area = float(
        assemble(conditional(lt(det(grad(T_field)), 0.0), 1.0, 0.0) * dx(domain=V0.mesh()))
    )

    E_y_sq = Function(V0, name=f"{method}_E_y_sq_unit")
    variance_raw = Function(V0, name=f"{method}_variance_raw_unit")
    variance = Function(V0, name=f"{method}_variance_unit")
    E_y_sq.interpolate(Hnumsq / max_value(Hden, TINY))
    variance_raw.interpolate(E_y_sq - (T0 ** 2 + T1 ** 2))
    variance.interpolate(max_value(variance_raw, 0.0))

    monge_defect_unit = float(assemble(variance * fine.mu0 * dx))
    monge_defect_physical = width * width * monge_defect_unit
    rms_conditional_spread_physical = width * math.sqrt(max(monge_defect_unit, 0.0))

    row = {
        "map_q_shift": q_shift,
        "map_denominator_min": global_dof_min(Hden),
        "map_denominator_max": global_dof_max(Hden),
        "expected_shift_x_unit": shift_x_unit,
        "expected_shift_y_unit": shift_y_unit,
        "mean_displacement_x_unit": disp_x_unit,
        "mean_displacement_y_unit": disp_y_unit,
        "mean_displacement_x_physical": width * disp_x_unit,
        "mean_displacement_y_physical": width * disp_y_unit,
        "map_l2_unit": map_l2_unit,
        "map_l2_physical": map_l2_physical,
        "map_linf_unit": map_linf_unit,
        "map_linf_physical": map_linf_physical,
        "det_min": det_min,
        "det_max": det_max,
        "tangled_area": tangled_area,
        "monge_defect_unit": monge_defect_unit,
        "monge_defect_physical": monge_defect_physical,
        "rms_conditional_spread_physical": rms_conditional_spread_physical,
    }

    if AUDIT_WRITE_VTK:
        Vvec = VectorFunctionSpace(V0.mesh(), "CG", DISK_CG_DEGREE)
        T_vec = Function(Vvec, name=f"{method}_T_unit").interpolate(as_vector([T0, T1]))
        T_exact_vec = Function(Vvec, name=f"{method}_T_exact_unit").interpolate(
            as_vector([T0_exact, T1_exact])
        )
        VTKFile(os.path.join(result_dir, f"{EXPERIMENT_NAME}_{method}_source_map.pvd")).write(
            T_vec, T_exact_vec, map_error_unit, det_T, variance
        )

    if COMM.rank == 0:
        print("FAS_TRANSLATION_MAP_AUDIT", row)

    return row

def final_science_audit(method, fine, p, q, b_p, b_q, performance_row, result_dir):
    audit_t0 = time.perf_counter()
    fine.reset_counters()

    full_rp, full_rq = fine.residual_norms(p, q, b_p, b_q)
    marginal_p_unit, marginal_q_unit = fine.marginal_residual_norms(p, q)

    width = BRIDGE_WIDTH_PHYSICAL
    row = {
        "method": method,
        "target_residual": TARGET_RESIDUAL,
        "performance_total_seconds": performance_row.get("total_seconds", ""),
        "performance_iteration_seconds": performance_row.get("iteration_seconds", ""),
        "performance_reached_target": performance_row.get("reached_target", ""),
        "performance_cycles": performance_row.get("cycle", ""),
        "performance_updates": performance_row.get("updates", ""),
        "performance_residual_p": performance_row.get("residual_p", ""),
        "full_residual_p": full_rp,
        "full_residual_q": full_rq,
        "source_marginal_l2_unit": marginal_p_unit,
        "target_marginal_l2_unit": marginal_q_unit,
        "source_marginal_l2_physical": marginal_p_unit / width,
        "target_marginal_l2_physical": marginal_q_unit / width,
        "audit_write_vtk": AUDIT_WRITE_VTK,
    }

    row.update(fine.field_stats(p, "p"))
    row.update(fine.field_stats(q, "q"))
    row.update(translation_map_audit(fine, q, result_dir, method))

    for name in [
        "log_marginal_p",
        "log_marginal_q",
        "marginal_p",
        "marginal_q",
        "logHq_on_0",
        "logHp_on_1",
        "Hq_shifted_on_0",
        "Hp_shifted_on_1",
    ]:
        field = getattr(fine, name, None)
        if field is not None:
            row.update(fine.field_stats(field, name))

    row.update(
        fine.heat_stats(
            fine.Hq_shifted_on_0,
            "Hq_shifted_heat",
        )
    )
    row.update(
        fine.heat_stats(
            fine.Hp_shifted_on_1,
            "Hp_shifted_heat",
        )
    )

    row["audit_seconds"] = COMM.allreduce(time.perf_counter() - audit_t0, op=MPI.MAX)

    write_rows(
        os.path.join(result_dir, f"{EXPERIMENT_NAME}_{method}_science_audit.csv"),
        [row],
    )
    
    write_rows(
        os.path.join(result_dir, f"{EXPERIMENT_NAME}_{method}_audit_heat_counts.csv"),
        fine.counter_rows(),
    )

    if COMM.rank == 0:
        print("FAS_SCIENCE_AUDIT", row)

    return row


def hierarchy_level_objects(level):
    levels = []
    if level.coarse is not None:
        levels.extend(hierarchy_level_objects(level.coarse))
    levels.append(level)
    return levels


def hierarchy_rows(level):
    return [item.hierarchy_row() for item in hierarchy_level_objects(level)]


def assert_requested_hierarchy_is_active(fine):
    levels = hierarchy_level_objects(fine)
    observed = [level.bridge_n for level in levels]
    expected = list(BRIDGE_LEVELS)

    if observed != expected:
        raise RuntimeError(f"Expected hierarchy {expected}, got {observed}")

    if len(expected) >= 3:
        bottom = levels[0]
        bottom_calls = bottom.heat_source_calls + bottom.heat_target_calls
        if bottom_calls <= 0:
            raise RuntimeError(
                f"Three-level FAS requested but B{bottom.bridge_n} performed zero heat actions."
            )


def build_problem(coarse_tau_ratio=1.0):
    V0, V1, mu0, mu1 = build_support_spaces_and_measures()

    bridge_hierarchy = build_bridge_hierarchy()

    levels = []
    coarse = None

    for level, mesh in enumerate(bridge_hierarchy):
        tau_physical = (
            TAU_FINE_PHYSICAL * coarse_tau_ratio
            if level == 0
            else TAU_FINE_PHYSICAL
        )

        current = FASLevel(
            f"level{level}_B{BRIDGE_LEVELS[level]}",
            level,
            mesh,
            tau_physical,
            V0,
            V1,
            mu0,
            mu1,
            coarse=coarse,
        )

        levels.append(current)
        coarse = current

    fine = levels[-1]
    immediate_coarse = levels[-2] if len(levels) >= 2 else None

    return fine, immediate_coarse, V0, V1


def build_fine_only_problem():
    V0, V1, mu0, mu1 = build_support_spaces_and_measures()

    bridge_meshes = list(build_bridge_hierarchy())
    fine_index = len(BRIDGE_LEVELS) - 1

    fine = FASLevel(
        f"level{fine_index}_B{BRIDGE_LEVELS[fine_index]}",
        fine_index,
        bridge_meshes[fine_index],
        TAU_FINE_PHYSICAL,
        V0,
        V1,
        mu0,
        mu1,
        coarse=None,
    )

    return fine, None, V0, V1


def fmg_level_triplet(fine):
    levels = hierarchy_level_objects(fine)
    if len(levels) != 3:
        raise RuntimeError(f"FMG requires exactly 3 levels, got {len(levels)}")
    if levels[-1] is not fine:
        raise RuntimeError("FMG hierarchy ordering error: finest level is not last")
    return levels[0], levels[1], levels[2]


def fmg_heat_action_count_row(fine, bottom_updates, middle_cycles, top_cycles):
    bottom, middle, top = fmg_level_triplet(fine)
    levels = [bottom, middle, top]
    actual = [level.heat_source_calls + level.heat_target_calls for level in levels]

    prep_active = bottom_updates > 0 or middle_cycles > 0

    if not prep_active:
        expected = [16 * top_cycles, 10 * top_cycles, 9 * top_cycles]
    else:
        expected = [
            2 * bottom_updates + 16 * middle_cycles + 16 * top_cycles,
            8 * middle_cycles + 10 * top_cycles,
            1 + 9 * top_cycles,
        ]

    row = {
        "fmg_N": FMG_N,
        "fmg_bottom_updates": bottom_updates,
        "fmg_middle_cycles": middle_cycles,
        "top_cycles": top_cycles,
        "bottom_bridge_n": bottom.bridge_n,
        "middle_bridge_n": middle.bridge_n,
        "top_bridge_n": top.bridge_n,
        "bottom_heat_actions": actual[0],
        "middle_heat_actions": actual[1],
        "top_heat_actions": actual[2],
        "expected_bottom_heat_actions": expected[0],
        "expected_middle_heat_actions": expected[1],
        "expected_top_heat_actions": expected[2],
        "heat_count_match": int(actual == expected),
    }

    if actual != expected:
        raise RuntimeError(
            "FMG heat-count assertion failed: "
            f"observed={actual}, expected={expected}, "
            f"bottom_updates={bottom_updates}, "
            f"middle_cycles={middle_cycles}, "
            f"top_cycles={top_cycles}"
        )

    return row

def main():
    result_dir = os.environ.get("RESULT_DIR", os.getcwd())
    os.makedirs(result_dir, exist_ok=True)

    COMM.Barrier()
    total_t0 = time.perf_counter()

    if RUN_MODE == "fine_to_tol":
        fine, coarse, V0, V1 = build_fine_only_problem()
        hierarchy = [fine.hierarchy_row()]
    else:
        fine, coarse, V0, V1 = build_problem(COARSE_TAU_RATIO)
        hierarchy = hierarchy_rows(fine)
    write_rows(
        os.path.join(result_dir, f"{EXPERIMENT_NAME}_hierarchy.csv"),
        hierarchy,
    )

    p = Function(V0, name="p").assign(0.0)
    q = Function(V1, name="q").assign(0.0)
    b_p = Function(V0, name="b_p_finest").assign(0.0)
    b_q = Function(V1, name="b_q_finest").assign(0.0)

    if COMM.rank == 0:
        print(f"EXPERIMENT_NAME={EXPERIMENT_NAME}")
        print(f"FAS_CYCLES={FAS_CYCLES} PRE={PRE_SMOOTH} COARSE={COARSE_SMOOTH} POST={POST_SMOOTH} COARSE_TAU_RATIO={COARSE_TAU_RATIO}")
        print("===== FAS hierarchy =====")
        for row in hierarchy:
            print(
                "FAS_HIERARCHY "
                f"level={row['level']} "
                f"B={row['bridge_n']} "
                f"global_cells={row['global_cells']} "
                f"h_physical={row['h_physical']:.8e} "
                f"tau_physical={row['tau_physical']:.8e} "
                f"tau_unit={row['tau_unit']:.8e} "
                f"tau_over_h2={row['tau_over_h2']:.8e} "
                f"sqrt_2tau_over_h={row['sqrt_2tau_over_h']:.6f}"
            )

    initial_p, initial_q = fine.residual_norms(p, q, b_p, b_q)
    if COMM.rank == 0:
        print("FAS_INITIAL_RESIDUAL", f"p={initial_p}", f"q={initial_q}")

    if COMM.rank == 0:
        print(
            f"FAS_RUN_MODE={RUN_MODE} FAS_CYCLES={FAS_CYCLES} "
            f"PRE={PRE_SMOOTH} COARSE={COARSE_SMOOTH} POST={POST_SMOOTH}"
        )


    if RUN_MODE == "fmg_to_tol":
        fine.reset_counters()
        bottom, middle, top = fmg_level_triplet(fine)
        rows = []
        exact_error_rows = []
        iteration_t0 = time.perf_counter()

        bottom_prep_seconds = 0.0
        middle_prep_seconds = 0.0
        entry_check_seconds = 0.0
        # For the cold S=0 control, the corrected initial p-residual is the
        # B920 entry value. Prepared runs overwrite this after prolongation.
        entry_residual_p = initial_p

        prep_active = FMG_BOTTOM_UPDATES > 0 or FMG_MIDDLE_CYCLES > 0

        if COMM.rank == 0:
            print(
                "FMG_CONFIGURATION",
                f"legacy_N={FMG_N}",
                f"bottom_updates={FMG_BOTTOM_UPDATES}",
                f"middle_cycles={FMG_MIDDLE_CYCLES}",
                f"top_max={FMG_TOP_MAX_CYCLES}",
                f"target={TARGET_RESIDUAL}",
                f"hierarchy={BRIDGE_LEVELS}",
            )

        if FMG_BOTTOM_UPDATES > 0:
            stage_t0 = time.perf_counter()
            bottom.smoother(
                p, q,
                b_p, b_q,
                FMG_BOTTOM_UPDATES,
                f"fmg_bottom_prep_S{FMG_BOTTOM_UPDATES}_B{bottom.bridge_n}",
            )
            bottom.gauge_fix(p, q)
            bottom_prep_seconds = COMM.allreduce(time.perf_counter() - stage_t0, op=MPI.MAX)

        if FMG_MIDDLE_CYCLES > 0:
            stage_t0 = time.perf_counter()
            for prep in range(1, FMG_MIDDLE_CYCLES + 1):
                middle.fas_cycle_perf(p, q, b_p, b_q)
                middle.gauge_fix(p, q)
            middle_prep_seconds = COMM.allreduce(time.perf_counter() - stage_t0, op=MPI.MAX)

        if prep_active:
            stage_t0 = time.perf_counter()
            entry_residual_p = fine.p_residual_norm(p, q, b_p)
            entry_check_seconds = COMM.allreduce(time.perf_counter() - stage_t0, op=MPI.MAX)

        prep_seconds = bottom_prep_seconds + middle_prep_seconds + entry_check_seconds
        checkpoint_row = exact_error_checkpoint(
            fine, p, q, b_p, b_q, "entry", 0, entry_residual_p
        )
        exact_error_rows.append(checkpoint_row)
        diagnostic_seconds = checkpoint_row["diagnostic_seconds"]
        iteration_t0 += diagnostic_seconds
        total_t0 += diagnostic_seconds
        if COMM.rank == 0:
            print("EXACT_ERROR_CHECKPOINT", checkpoint_row)

        top_t0 = time.perf_counter()
        top_cycles = 0

        if prep_active and entry_residual_p <= TARGET_RESIDUAL:
            now = time.perf_counter()
            row = {
                "mode": "fmg_to_tol",
                "fmg_N": FMG_N,
                "fmg_bottom_updates": FMG_BOTTOM_UPDATES,
                "fmg_middle_cycles": FMG_MIDDLE_CYCLES,
                "cycle": 0,
                "top_cycles": 0,
                "B920_entry_residual": entry_residual_p,
                "residual_p": entry_residual_p,
                "residual_q": "",
                "target_residual": TARGET_RESIDUAL,
                "reached_target": 1,
                "bottom_prep_seconds": bottom_prep_seconds,
                "middle_prep_seconds": middle_prep_seconds,
                "entry_check_seconds": entry_check_seconds,
                "prep_seconds": prep_seconds,
                "top_solve_seconds": 0.0,
                "iteration_seconds": COMM.allreduce(now - iteration_t0, op=MPI.MAX),
                "total_seconds": COMM.allreduce(now - total_t0, op=MPI.MAX),
                "coarse_tau_ratio": COARSE_TAU_RATIO,
                "correction_damping": CORRECTION_DAMPING,
            }
            rows.append(row)
        else:
            for cycle in range(1, FMG_TOP_MAX_CYCLES + 1):
                fine.fas_cycle_perf(p, q, b_p, b_q)
                top_cycles = cycle
                residual_p = fine.p_residual_norm(p, q, b_p)
                now = time.perf_counter()
                row = {
                    "mode": "fmg_to_tol",
                    "fmg_N": FMG_N,
                    "fmg_bottom_updates": FMG_BOTTOM_UPDATES,
                    "fmg_middle_cycles": FMG_MIDDLE_CYCLES,
                    "cycle": cycle,
                    "top_cycles": cycle,
                    "B920_entry_residual": entry_residual_p,
                    "residual_p": residual_p,
                    "residual_q": "",
                    "target_residual": TARGET_RESIDUAL,
                    "reached_target": int(residual_p <= TARGET_RESIDUAL),
                    "bottom_prep_seconds": bottom_prep_seconds,
                    "middle_prep_seconds": middle_prep_seconds,
                    "entry_check_seconds": entry_check_seconds,
                    "prep_seconds": prep_seconds,
                    "top_solve_seconds": COMM.allreduce(now - top_t0, op=MPI.MAX),
                    "iteration_seconds": COMM.allreduce(now - iteration_t0, op=MPI.MAX),
                    "total_seconds": COMM.allreduce(now - total_t0, op=MPI.MAX),
                    "coarse_tau_ratio": COARSE_TAU_RATIO,
                    "correction_damping": CORRECTION_DAMPING,
                }
                rows.append(row)
                if COMM.rank == 0:
                    print("FMG_TO_TOL_ROW", row)

                checkpoint_row = exact_error_checkpoint(
                    fine, p, q, b_p, b_q, f"cycle_{cycle}", cycle, residual_p
                )
                exact_error_rows.append(checkpoint_row)
                diagnostic_seconds = checkpoint_row["diagnostic_seconds"]
                top_t0 += diagnostic_seconds
                iteration_t0 += diagnostic_seconds
                total_t0 += diagnostic_seconds
                if COMM.rank == 0:
                    print("EXACT_ERROR_CHECKPOINT", checkpoint_row)

                if residual_p <= TARGET_RESIDUAL:
                    break

        if not rows:
            raise RuntimeError("FMG produced no trace rows")

        assert_requested_hierarchy_is_active(fine)
        count_row = fmg_heat_action_count_row(
            fine,
            FMG_BOTTOM_UPDATES,
            FMG_MIDDLE_CYCLES,
            top_cycles,
        )
        rows[-1].update(count_row)
        add_checkpoint_ratios(exact_error_rows)

        write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_fmg_to_tol_trace.csv"), rows)
        write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_fmg_summary.csv"), [rows[-1]])
        write_rows(
            os.path.join(result_dir, f"{EXPERIMENT_NAME}_exact_error_trajectory.csv"),
            exact_error_rows,
        )

        profile_rows = fine.counter_rows()
        write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_heat_counts.csv"), profile_rows)
        write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_cost_profile.csv"), profile_rows)
        final_science_audit("fmg_to_tol", fine, p, q, b_p, b_q, rows[-1], result_dir)

        if COMM.rank == 0:
            print("FMG_TO_TOL_FINAL", rows[-1])
            print(f"RESULT_DIR={result_dir}")
        return

    if RUN_MODE == "fas_to_tol":
        fine.reset_counters()
        rows = []
        iteration_t0 = time.perf_counter()

        for cycle in range(1, FAS_CYCLES + 1):
            fine.fas_cycle_perf(p, q, b_p, b_q)
            residual_p = fine.p_residual_norm(p, q, b_p)
            residual_q = ""
            now = time.perf_counter()
            row = {
                "mode": "fas_to_tol",
                "cycle": cycle,
                "fine_equivalent_sweeps": cycle * (PRE_SMOOTH + POST_SMOOTH),
                "coarse_sweeps": cycle * COARSE_SMOOTH,
                "residual_p": residual_p,
                "residual_q": residual_q,
                "target_residual": TARGET_RESIDUAL,
                "reached_target": int(residual_p <= TARGET_RESIDUAL),
                "iteration_seconds": now - iteration_t0,
                "total_seconds": now - total_t0,
                "coarse_tau_ratio": COARSE_TAU_RATIO,
                "correction_damping": CORRECTION_DAMPING,
            }
            rows.append(row)
            if COMM.rank == 0:
                print("FAS_TO_TOL_ROW", row)
            if residual_p <= TARGET_RESIDUAL:
                break

        write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_fas_to_tol_trace.csv"), rows)

        assert_requested_hierarchy_is_active(fine)

        profile_rows = fine.counter_rows()
        write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_heat_counts.csv"), profile_rows)
        write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_cost_profile.csv"), profile_rows)
        final_science_audit("fas_to_tol", fine, p, q, b_p, b_q, rows[-1], result_dir)
        if COMM.rank == 0:
            print("FAS_TO_TOL_FINAL", rows[-1])
            print(f"RESULT_DIR={result_dir}")
        return

    if RUN_MODE == "fine_to_tol":
        fine.reset_counters()
        rows = []
        updates = 0
        iteration_t0 = time.perf_counter()

        while updates < MAX_FINE_ONLY_UPDATES:
            block = min(FINE_CHECK_BLOCK, MAX_FINE_ONLY_UPDATES - updates)
            fine.smoother(
                p, q,
                b_p, b_q,
                block,
                f"fine_to_tol_block_{updates + block}",
            )
            fine.gauge_fix(p, q)
            updates += block

            residual_p = fine.p_residual_norm(p, q, b_p)
            residual_q = ""
            now = time.perf_counter()
            row = {
                "mode": "fine_to_tol",
                "updates": updates,
                "residual_p": residual_p,
                "residual_q": residual_q,
                "target_residual": TARGET_RESIDUAL,
                "reached_target": int(residual_p <= TARGET_RESIDUAL),
                "iteration_seconds": now - iteration_t0,
                "total_seconds": now - total_t0,
                "coarse_tau_ratio": COARSE_TAU_RATIO,
            }
            rows.append(row)
            if COMM.rank == 0:
                print("FINE_TO_TOL_ROW", row)
            if residual_p <= TARGET_RESIDUAL:
                break

        write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_fine_to_tol_trace.csv"), rows)
        profile_rows = fine.counter_rows()
        write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_heat_counts.csv"), profile_rows)
        write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_cost_profile.csv"), profile_rows)
        final_science_audit("fine_to_tol", fine, p, q, b_p, b_q, rows[-1], result_dir)
        if COMM.rank == 0:
            print("FINE_TO_TOL_FINAL", rows[-1])
            print(f"RESULT_DIR={result_dir}")
        return

    if RUN_MODE == "smoother_test":
        rows = fine.smoother_test(p, q, b_p, b_q, SMOOTHER_TEST_UPDATES)
        write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_smoother_test.csv"), rows)
        if COMM.rank == 0:
            print("Smoother-test complete: no FAS correction cycle was applied.")
            print(f"RESULT_DIR={result_dir}")
        return

    if RUN_MODE == "sign_test":
        row = fine.sign_test(p, q, b_p, b_q, PRE_SMOOTH)
        write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_sign_test.csv"), [row])
        if COMM.rank == 0:
            print("Smoke/sign-test complete: no FAS correction cycle was applied.")
            print(f"RESULT_DIR={result_dir}")
        return

    if RUN_MODE == "fine_only":
        fine.smoother(p, q, b_p, b_q, FINE_ONLY_UPDATES, "fine_only")
        fine.gauge_fix(p, q)
        final_p, final_q = fine.residual_norms(p, q, b_p, b_q)
        row = {
            "mode": "fine_only",
            "updates": FINE_ONLY_UPDATES,
            "final_residual_p": final_p,
            "final_residual_q": final_q,
        }
        write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_fine_only.csv"), [row])
        if COMM.rank == 0:
            print("FAS_FINE_ONLY", row)
            print(f"RESULT_DIR={result_dir}")
        return

    if RUN_MODE == "coarse_tau_sweep":
        rows = []

        for ratio in COARSE_TAU_RATIO_VALUES:
            fine, coarse, V0, V1 = build_problem()

            coarse.tau_physical = TAU_FINE_PHYSICAL * ratio
            coarse.tau_unit = coarse.tau_physical / (BRIDGE_WIDTH_PHYSICAL ** 2)
            coarse.tau.assign(coarse.tau_unit)
            coarse.sub_tau.assign(coarse.tau_unit / N_SUB)

            p = Function(V0, name=f"p_ratio_{ratio:g}").assign(0.0)
            q = Function(V1, name=f"q_ratio_{ratio:g}").assign(0.0)
            b_p = Function(V0, name=f"b_p_ratio_{ratio:g}").assign(0.0)
            b_q = Function(V1, name=f"b_q_ratio_{ratio:g}").assign(0.0)

            before_p, before_q = fine.residual_norms(p, q, b_p, b_q)

            cycle_detail = fine.fas_cycle(
                p,
                q,
                b_p,
                b_q,
            )

            row = {
                "mode": "coarse_tau_sweep",
                "coarse_tau_ratio": ratio,
                "coarse_tau_physical": coarse.tau_physical,
                "coarse_tau_unit": coarse.tau_unit,
                "coarse_smooth": COARSE_SMOOTH,
                "pre_smooth": PRE_SMOOTH,
                "post_smooth": POST_SMOOTH,
                "correction_damping": CORRECTION_DAMPING,
                "initial_p": before_p,
                "initial_q": before_q,
            }
            row.update(cycle_detail)
            rows.append(row)

            if COMM.rank == 0:
                print("FAS_COARSE_TAU_ROW", row)

        write_rows(
            os.path.join(result_dir, f"{EXPERIMENT_NAME}_coarse_tau_sweep.csv"),
            rows,
        )

        COMM.Barrier()
        total_runtime = COMM.allreduce(time.perf_counter() - total_t0, op=MPI.MAX)

        all_timer_rows = []
        for row_level in [fine, coarse]:
            all_timer_rows.extend(row_level.rows)
        all_timer_rows.append({
            "level": "global",
            "event": "total",
            "seconds": total_runtime,
        })

        write_rows(
            os.path.join(result_dir, f"{EXPERIMENT_NAME}_timers.csv"),
            all_timer_rows,
        )

        if COMM.rank == 0:
            print("FAS_TOTAL_RUNTIME", total_runtime)
            print("RESULT_DIR=" + result_dir)
        return

    if RUN_MODE == "coarse_sweep":
        p_pre = Function(V0, name="p_pre").assign(0.0)
        q_pre = Function(V1, name="q_pre").assign(0.0)
        p_trial = Function(V0, name="p_trial").assign(0.0)
        q_trial = Function(V1, name="q_trial").assign(0.0)

        fine.smoother(p, q, b_p, b_q, PRE_SMOOTH, "coarse_sweep_pre")
        fine.gauge_fix(p, q)

        p_pre.assign(p)
        q_pre.assign(q)

        after_pre_p, after_pre_q = fine.residual_norms(p_pre, q_pre, b_p, b_q)

        restricted_p, restricted_q = fine.restrict_state(p_pre, q_pre)
        restricted_rp, restricted_rq = fine.restrict_residual()
        coarse_bp, coarse_bq = fine.build_coarse_rhs(
            restricted_p, restricted_q, restricted_rp, restricted_rq
        )

        fine.coarse.residual(
            restricted_p, restricted_q,
            coarse_bp, coarse_bq,
            fine.entry_rp, fine.entry_rq,
        )
        coarse_entry_p = l2_norm(fine.entry_rp)
        coarse_entry_q = l2_norm(fine.entry_rq)
        coarse_entry_combined = math.sqrt(coarse_entry_p ** 2 + coarse_entry_q ** 2)

        rows = []
        for ncoarse in COARSE_SWEEP_VALUES:
            fine.coarse_p.assign(restricted_p)
            fine.coarse_q.assign(restricted_q)

            if ncoarse:
                fine.coarse.smoother(
                    fine.coarse_p,
                    fine.coarse_q,
                    coarse_bp,
                    coarse_bq,
                    ncoarse,
                    f"coarse_sweep_{ncoarse}",
                )
                fine.coarse.gauge_fix(fine.coarse_p, fine.coarse_q)

            coarse_exit_p, coarse_exit_q = fine.coarse.residual_norms(
                fine.coarse_p,
                fine.coarse_q,
                coarse_bp,
                coarse_bq,
            )
            coarse_exit_combined = math.sqrt(coarse_exit_p ** 2 + coarse_exit_q ** 2)

            p_trial.assign(p_pre)
            q_trial.assign(q_pre)

            correction_p, correction_q = fine.prolong_correction(
                p_trial,
                q_trial,
                fine.coarse_p,
                fine.coarse_q,
                restricted_p,
                restricted_q,
            )
            fine.gauge_fix(p_trial, q_trial)

            after_correction_p, after_correction_q = fine.residual_norms(
                p_trial,
                q_trial,
                b_p,
                b_q,
            )

            fine.smoother(
                p_trial,
                q_trial,
                b_p,
                b_q,
                POST_SMOOTH,
                f"coarse_sweep_post_{ncoarse}",
            )
            fine.gauge_fix(p_trial, q_trial)

            after_post_p, after_post_q = fine.residual_norms(
                p_trial,
                q_trial,
                b_p,
                b_q,
            )

            row = {
                "mode": "coarse_sweep",
                "coarse_sweeps": ncoarse,
                "correction_damping": CORRECTION_DAMPING,
                "after_pre_p": after_pre_p,
                "after_pre_q": after_pre_q,
                "coarse_entry_p": coarse_entry_p,
                "coarse_entry_q": coarse_entry_q,
                "coarse_exit_p": coarse_exit_p,
                "coarse_exit_q": coarse_exit_q,
                "coarse_residual_ratio": coarse_exit_combined / max(coarse_entry_combined, 1.0e-300),
                "correction_p": correction_p,
                "correction_q": correction_q,
                "after_correction_p": after_correction_p,
                "after_correction_q": after_correction_q,
                "after_post_p": after_post_p,
                "after_post_q": after_post_q,
            }
            rows.append(row)

            if COMM.rank == 0:
                print("FAS_COARSE_SWEEP_ROW", row)

        write_rows(
            os.path.join(result_dir, f"{EXPERIMENT_NAME}_coarse_sweep.csv"),
            rows,
        )

        COMM.Barrier()
        total_runtime = COMM.allreduce(time.perf_counter() - total_t0, op=MPI.MAX)
        all_rows = fine.rows + coarse.rows + [{
            "level": "global",
            "event": "total",
            "seconds": total_runtime,
        }]
        write_rows(
            os.path.join(result_dir, f"{EXPERIMENT_NAME}_timers.csv"),
            all_rows,
        )

        if COMM.rank == 0:
            print("FAS_TOTAL_RUNTIME", total_runtime)
            print("RESULT_DIR=" + result_dir)
        return

    if RUN_MODE != "fas_cold":
        raise ValueError(f"Unknown FAS_RUN_MODE={RUN_MODE!r}")

    cycle_rows = []
    for cycle in range(FAS_CYCLES):
        if COMM.rank == 0:
            print(f"===== FAS cycle {cycle + 1} =====")

        cycle_detail = fine.fas_cycle(
            p,
            q,
            b_p,
            b_q,
        )

        row = {"cycle": cycle + 1}
        row.update(cycle_detail)
        cycle_rows.append(row)

        if COMM.rank == 0:
            print(
                "FAS_CYCLE_SUMMARY",
                f"cycle={cycle + 1}",
                f"before_p={cycle_detail['before_p']}",
                f"after_pre_p={cycle_detail['after_pre_p']}",
                f"after_correction_p={cycle_detail['after_correction_p']}",
                f"after_cycle_p={cycle_detail['after_cycle_p']}",
                f"after_cycle_q={cycle_detail['after_cycle_q']}",
            )

    COMM.Barrier()
    total_runtime = COMM.allreduce(time.perf_counter() - total_t0, op=MPI.MAX)

    all_rows = fine.rows + coarse.rows + [{
        "level": "global",
        "event": "total",
        "seconds": total_runtime,
    }]

    write_rows(
        os.path.join(result_dir, f"{EXPERIMENT_NAME}_timers.csv"),
        all_rows,
    )
    write_rows(
        os.path.join(result_dir, f"{EXPERIMENT_NAME}_cycle_residuals.csv"),
        cycle_rows,
    )

    if COMM.rank == 0:
        print("FAS_TOTAL_RUNTIME", total_runtime)
        print("RESULT_DIR=" + result_dir)


if __name__ == "__main__":
    main()
