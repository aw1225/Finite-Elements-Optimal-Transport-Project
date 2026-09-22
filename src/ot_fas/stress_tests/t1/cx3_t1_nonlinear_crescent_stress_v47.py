from firedrake import *
from mpi4py import MPI
import csv
import math
import os
import time
import numpy as np

EXPERIMENT_NAME = "run_t1_nonlinear_crescent_stress_v47"

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

# T1 manufactured smooth non-convex crescent benchmark.
# Physical coordinates live in [-BRIDGE_WIDTH_PHYSICAL/2, BRIDGE_WIDTH_PHYSICAL/2]^2
# and are mapped to the unit periodic bridge by x_unit = 0.5 + x_phys / width.
DISK_RADIUS_PHYSICAL = 0.6
CRESCENT_KAPPA = 1.9  # 1/length; 2*kappa*R = 2.28 > 1 certifies curvature sign change.

# Strongly-convex manufactured Brenier map
#   phi(x) = 1/2 x^T A x + d.x
#          + sum_j s_j L^2/alpha_j^2 log cosh(alpha_j a_j.x/L)
# so T_exact = grad(phi), and DT_exact = Hess(phi) = A + PSD rank-one terms.
MAP_BASE_EIG_MIN = 0.62
MAP_BASE_EIG_MAX = 1.18
MAP_BASE_ANGLE_DEG = 28.0
MAP_DIRECTION1_DEG = 10.0
MAP_DIRECTION2_DEG = 118.0
MAP_S1 = 0.62
MAP_S2 = 0.46
MAP_ALPHA1 = 2.7
MAP_ALPHA2 = 3.4
MAP_LENGTH_SCALE = DISK_RADIUS_PHYSICAL
MAP_TRANSLATION_PHYSICAL = (0.25, -0.08)

CERT_MAX_AREA_REL_MISMATCH = float(os.environ.get("T1_CERT_MAX_AREA_REL_MISMATCH", "5e-3"))
CERT_MAX_PUSHFORWARD_MOMENT_ERROR = float(os.environ.get("T1_CERT_MAX_PUSHFORWARD_MOMENT_ERROR", "5e-3"))
CERT_MIN_BRIDGE_MARGIN_UNIT = float(os.environ.get("T1_CERT_MIN_BRIDGE_MARGIN_UNIT", "0.05"))
CERT_MIN_EXACT_JACOBIAN_EIG = float(os.environ.get("T1_CERT_MIN_EXACT_JACOBIAN_EIG", "0.50"))

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

REF_CHECK_BLOCK = int(os.environ.get("FAS_REF_CHECK_BLOCK", "6"))
REF_MIN_FINE_UPDATES = int(os.environ.get("FAS_REF_MIN_FINE_UPDATES", "60"))
REF_MAX_FINE_UPDATES = int(os.environ.get("FAS_REF_MAX_FINE_UPDATES", "120"))
REF_RESIDUAL_TARGET = float(os.environ.get("FAS_REF_RESIDUAL_TARGET", "1e-6"))
REF_CHANGE_REL_TOL = float(os.environ.get("FAS_REF_CHANGE_REL_TOL", "1e-8"))
REF_STABLE_CHECKS = int(os.environ.get("FAS_REF_STABLE_CHECKS", "2"))

COMM = COMM_WORLD

solver_params = {
    "ksp_type": "preonly",
    "pc_type": "cholesky",
    "pc_factor_mat_solver_type": "mumps",
}


def physical_to_unit_vector(v):
    return as_vector([
        Constant(0.5) + v[0] / Constant(BRIDGE_WIDTH_PHYSICAL),
        Constant(0.5) + v[1] / Constant(BRIDGE_WIDTH_PHYSICAL),
    ])


def unit_to_physical_vector(v):
    return as_vector([
        Constant(BRIDGE_WIDTH_PHYSICAL) * (v[0] - Constant(0.5)),
        Constant(BRIDGE_WIDTH_PHYSICAL) * (v[1] - Constant(0.5)),
    ])


def _unit_direction(angle_deg):
    angle = math.radians(angle_deg)
    return (math.cos(angle), math.sin(angle))


_A_THETA = math.radians(MAP_BASE_ANGLE_DEG)
_A_C = math.cos(_A_THETA)
_A_S = math.sin(_A_THETA)
_A00 = MAP_BASE_EIG_MIN * _A_C ** 2 + MAP_BASE_EIG_MAX * _A_S ** 2
_A11 = MAP_BASE_EIG_MIN * _A_S ** 2 + MAP_BASE_EIG_MAX * _A_C ** 2
_A01 = (MAP_BASE_EIG_MIN - MAP_BASE_EIG_MAX) * _A_C * _A_S
_A10 = _A01
_A1 = _unit_direction(MAP_DIRECTION1_DEG)
_A2 = _unit_direction(MAP_DIRECTION2_DEG)


def source_crescent_physical_from_reference(z):
    """Map reference UnitDisk coordinates z to the physical smooth crescent."""
    xb = Constant(DISK_RADIUS_PHYSICAL) * z[0]
    yb = Constant(DISK_RADIUS_PHYSICAL) * z[1]
    return as_vector([
        xb + Constant(CRESCENT_KAPPA) * yb * yb,
        yb,
    ])


def exact_map_physical(x):
    """Exact manufactured quadratic-cost Brenier map T=grad(phi), physical coordinates."""
    a1 = as_vector([Constant(_A1[0]), Constant(_A1[1])])
    a2 = as_vector([Constant(_A2[0]), Constant(_A2[1])])
    A = as_matrix([
        [Constant(_A00), Constant(_A01)],
        [Constant(_A10), Constant(_A11)],
    ])
    L = Constant(MAP_LENGTH_SCALE)
    z1 = Constant(MAP_ALPHA1) * dot(a1, x) / L
    z2 = Constant(MAP_ALPHA2) * dot(a2, x) / L
    nl1 = Constant(MAP_S1 * MAP_LENGTH_SCALE / MAP_ALPHA1) * tanh(z1) * a1
    nl2 = Constant(MAP_S2 * MAP_LENGTH_SCALE / MAP_ALPHA2) * tanh(z2) * a2
    d = as_vector([
        Constant(MAP_TRANSLATION_PHYSICAL[0]),
        Constant(MAP_TRANSLATION_PHYSICAL[1]),
    ])
    return dot(A, x) + nl1 + nl2 + d


def exact_DT_physical(x):
    """Exact Hessian of the strongly convex potential; also the Jacobian DT."""
    a1 = as_vector([Constant(_A1[0]), Constant(_A1[1])])
    a2 = as_vector([Constant(_A2[0]), Constant(_A2[1])])
    A = as_matrix([
        [Constant(_A00), Constant(_A01)],
        [Constant(_A10), Constant(_A11)],
    ])
    L = Constant(MAP_LENGTH_SCALE)
    z1 = Constant(MAP_ALPHA1) * dot(a1, x) / L
    z2 = Constant(MAP_ALPHA2) * dot(a2, x) / L
    t1 = tanh(z1)
    t2 = tanh(z2)
    w1 = Constant(MAP_S1) * (Constant(1.0) - t1 * t1)
    w2 = Constant(MAP_S2) * (Constant(1.0) - t2 * t2)
    return A + w1 * outer(a1, a1) + w2 * outer(a2, a2)


def exact_map_unit_from_source_unit(x_unit):
    return physical_to_unit_vector(exact_map_physical(unit_to_physical_vector(x_unit)))


def _global_mesh_bbox(mesh):
    data = mesh.coordinates.dat.data_ro
    local_min_x = float(data[:, 0].min()) if data.size else math.inf
    local_max_x = float(data[:, 0].max()) if data.size else -math.inf
    local_min_y = float(data[:, 1].min()) if data.size else math.inf
    local_max_y = float(data[:, 1].max()) if data.size else -math.inf
    return (
        COMM.allreduce(local_min_x, op=MPI.MIN),
        COMM.allreduce(local_max_x, op=MPI.MAX),
        COMM.allreduce(local_min_y, op=MPI.MIN),
        COMM.allreduce(local_max_y, op=MPI.MAX),
    )


def _build_transformed_disk_mesh(kind, refinement):
    """Build source or exact target mesh and certify discrete coordinate-map orientation."""
    if kind not in {"source", "target"}:
        raise ValueError(kind)
    mesh = UnitDiskMesh(refinement)
    coord_space = mesh.coordinates.function_space()
    z = Function(coord_space, name=f"{kind}_reference_coordinates").assign(mesh.coordinates)
    source_phys = source_crescent_physical_from_reference(z)
    physical = source_phys if kind == "source" else exact_map_physical(source_phys)
    mapped = Function(coord_space, name=f"{kind}_mapped_coordinates").interpolate(
        physical_to_unit_vector(physical)
    )

    # Evaluate the Jacobian of the *discrete P1 coordinate map* before replacing coordinates.
    Vdet = FunctionSpace(mesh, "DG", 0)
    coord_det = Function(Vdet, name=f"{kind}_coordinate_map_det").interpolate(det(grad(mapped)))
    coord_det_min = global_dof_min(coord_det)
    coord_det_max = global_dof_max(coord_det)
    if coord_det_min <= 0.0:
        raise RuntimeError(
            f"T1 {kind} coordinate transform inverted a cell: det_min={coord_det_min}"
        )
    mesh.coordinates.assign(mapped)
    return mesh, {
        f"{kind}_coordinate_map_det_min": coord_det_min,
        f"{kind}_coordinate_map_det_max": coord_det_max,
    }


def global_nonfinite_count(function):
    data = function.dat.data_ro
    local = int(np.count_nonzero(~np.isfinite(data)))
    return int(COMM.allreduce(local, op=MPI.SUM))


def build_support_spaces_and_measures(return_metadata=False):
    mesh0, source_meta = _build_transformed_disk_mesh("source", DISK_REFINEMENT)
    mesh1, target_meta = _build_transformed_disk_mesh("target", DISK_REFINEMENT)

    V0 = FunctionSpace(mesh0, "CG", DISK_CG_DEGREE)
    V1 = FunctionSpace(mesh1, "CG", DISK_CG_DEGREE)

    x = SpatialCoordinate(mesh0)
    x_phys = unit_to_physical_vector(x)
    det_exact_expr = det(exact_DT_physical(x_phys))

    mu0 = Function(V0, name="mu0").interpolate(det_exact_expr)
    mu1 = Function(V1, name="mu1").interpolate(1.0)

    raw_source_mass = float(assemble(mu0 * dx(domain=mesh0)))
    raw_target_area = float(assemble(mu1 * dx(domain=mesh1)))
    if raw_source_mass <= 0.0 or raw_target_area <= 0.0:
        raise RuntimeError(
            f"T1 invalid support mass/area: source_raw={raw_source_mass}, target_area={raw_target_area}"
        )

    mu0.assign(mu0 / raw_source_mass)
    mu1.assign(mu1 / raw_target_area)

    metadata = {}
    metadata.update(source_meta)
    metadata.update(target_meta)
    metadata.update({
        "source_det_integral_unit": raw_source_mass,
        "target_area_unit": raw_target_area,
        "area_change_relative_mismatch": abs(raw_source_mass - raw_target_area) / raw_target_area,
    })

    if return_metadata:
        return V0, V1, mu0, mu1, metadata
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




def manufactured_map_audit(fine, q, result_dir, method):
    V0 = fine.V0
    V1 = fine.V1
    x = SpatialCoordinate(V0.mesh())
    y = SpatialCoordinate(V1.mesh())
    width = BRIDGE_WIDTH_PHYSICAL

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
    T_field = as_vector([T0, T1])

    T_exact_unit = exact_map_unit_from_source_unit(x)
    T0_exact = T_exact_unit[0]
    T1_exact = T_exact_unit[1]

    map_error_unit = Function(V0, name=f"{method}_map_error_unit")
    map_error_unit.interpolate(sqrt((T0 - T0_exact) ** 2 + (T1 - T1_exact) ** 2))
    map_l2_unit = math.sqrt(max(float(assemble(map_error_unit ** 2 * fine.mu0 * dx)), 0.0))
    map_linf_unit = global_dof_max(map_error_unit)

    source_phys = unit_to_physical_vector(x)
    DT_exact = exact_DT_physical(source_phys)
    DT_num = grad(T_field)
    det_space = FunctionSpace(V0.mesh(), "DG", 0)

    det_T = Function(det_space, name=f"{method}_det_DT")
    det_T.interpolate(det(DT_num))
    det_exact = Function(det_space, name=f"{method}_det_DT_exact")
    det_exact.interpolate(det(DT_exact))

    jac_frob_sq = sum(
        (DT_num[i, j] - DT_exact[i, j]) ** 2
        for i in range(2) for j in range(2)
    )
    jacobian_frobenius_l2 = math.sqrt(max(float(assemble(jac_frob_sq * fine.mu0 * dx)), 0.0))
    det_error_l2 = math.sqrt(max(float(assemble((det(DT_num) - det(DT_exact)) ** 2 * fine.mu0 * dx)), 0.0))

    det_min = global_dof_min(det_T)
    det_max = global_dof_max(det_T)
    exact_det_min = global_dof_min(det_exact)
    exact_det_max = global_dof_max(det_exact)
    tangled_area = float(
        assemble(conditional(lt(det(DT_num), 0.0), 1.0, 0.0) * dx(domain=V0.mesh()))
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

    mean_disp_x = float(assemble((T0 - x[0]) * fine.mu0 * dx))
    mean_disp_y = float(assemble((T1 - x[1]) * fine.mu0 * dx))

    row = {
        "map_q_shift": q_shift,
        "map_denominator_min": global_dof_min(Hden),
        "map_denominator_max": global_dof_max(Hden),
        "map_denominator_nonfinite_count": global_nonfinite_count(Hden),
        "mean_displacement_x_physical": width * mean_disp_x,
        "mean_displacement_y_physical": width * mean_disp_y,
        "map_l2_unit": map_l2_unit,
        "map_l2_physical": width * map_l2_unit,
        "map_linf_unit": map_linf_unit,
        "map_linf_physical": width * map_linf_unit,
        "jacobian_frobenius_l2": jacobian_frobenius_l2,
        "det_error_l2": det_error_l2,
        "det_min": det_min,
        "det_max": det_max,
        "exact_det_min": exact_det_min,
        "exact_det_max": exact_det_max,
        "tangled_area": tangled_area,
        "monge_defect_unit": monge_defect_unit,
        "monge_defect_physical": monge_defect_physical,
        "rms_conditional_spread_physical": rms_conditional_spread_physical,
    }

    if AUDIT_WRITE_VTK:
        Vvec = VectorFunctionSpace(V0.mesh(), "CG", DISK_CG_DEGREE)
        T_vec = Function(Vvec, name=f"{method}_T_unit").interpolate(T_field)
        T_exact_vec = Function(Vvec, name=f"{method}_T_exact_unit").interpolate(T_exact_unit)
        VTKFile(os.path.join(result_dir, f"{EXPERIMENT_NAME}_{method}_source_map.pvd")).write(
            T_vec, T_exact_vec, map_error_unit, det_T, det_exact, variance
        )

    if COMM.rank == 0:
        print("T1_MANUFACTURED_MAP_AUDIT", row)
    return row


def _pushforward_moment_rows(V0, V1, mu0, mu1):
    x = SpatialCoordinate(V0.mesh())
    y = SpatialCoordinate(V1.mesh())
    Tx = exact_map_unit_from_source_unit(x)
    source_exprs = [
        ("one", Constant(1.0)),
        ("x", Tx[0]),
        ("y", Tx[1]),
        ("x2", Tx[0] ** 2),
        ("xy", Tx[0] * Tx[1]),
        ("y2", Tx[1] ** 2),
        ("sin2pix", sin(Constant(2.0 * math.pi) * Tx[0])),
        ("cos2piy", cos(Constant(2.0 * math.pi) * Tx[1])),
    ]
    target_exprs = [
        Constant(1.0), y[0], y[1], y[0] ** 2, y[0] * y[1], y[1] ** 2,
        sin(Constant(2.0 * math.pi) * y[0]), cos(Constant(2.0 * math.pi) * y[1]),
    ]
    rows = []
    for (name, sx), ty in zip(source_exprs, target_exprs):
        source_value = float(assemble(sx * mu0 * dx))
        target_value = float(assemble(ty * mu1 * dx))
        abs_diff = abs(source_value - target_value)
        rows.append({
            "moment": name,
            "source_pushforward_value": source_value,
            "target_value": target_value,
            "abs_difference": abs_diff,
        })
    return rows


def benchmark_certificate(V0, V1, mu0, mu1, metadata, result_dir):
    mesh0 = V0.mesh()
    mesh1 = V1.mesh()
    x = SpatialCoordinate(mesh0)
    x_phys = unit_to_physical_vector(x)
    H = exact_DT_physical(x_phys)
    trH = H[0, 0] + H[1, 1]
    disc = sqrt((H[0, 0] - H[1, 1]) ** 2 + Constant(4.0) * H[0, 1] ** 2)
    lambda_min_expr = Constant(0.5) * (trH - disc)
    lambda_max_expr = Constant(0.5) * (trH + disc)

    Vdg = FunctionSpace(mesh0, "DG", 0)
    lam_min = Function(Vdg, name="T1_exact_lambda_min").interpolate(lambda_min_expr)
    lam_max = Function(Vdg, name="T1_exact_lambda_max").interpolate(lambda_max_expr)
    det_exact = Function(Vdg, name="T1_exact_det").interpolate(det(H))

    source_bbox = _global_mesh_bbox(mesh0)
    target_bbox = _global_mesh_bbox(mesh1)
    all_bbox = [source_bbox[0], source_bbox[1], source_bbox[2], source_bbox[3],
                target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3]]
    bridge_margin = min(
        source_bbox[0], 1.0 - source_bbox[1], source_bbox[2], 1.0 - source_bbox[3],
        target_bbox[0], 1.0 - target_bbox[1], target_bbox[2], 1.0 - target_bbox[3],
    )

    moment_rows = _pushforward_moment_rows(V0, V1, mu0, mu1)
    max_moment_error = max(row["abs_difference"] for row in moment_rows)

    # A second direct pushforward consistency check for the density formula.
    target_area = metadata["target_area_unit"]
    ideal_mu0 = det(H) / Constant(target_area)
    rho_diff = Function(V0, name="T1_pushforward_density_identity_error").interpolate(mu0 - ideal_mu0)
    pushforward_density_rel_l2 = l2_norm(rho_diff) / max(l2_norm(mu0), 1.0e-300)

    source_mass = float(assemble(mu0 * dx))
    target_mass = float(assemble(mu1 * dx))
    curvature_indicator = 1.0 - 2.0 * CRESCENT_KAPPA * DISK_RADIUS_PHYSICAL
    nonconvexity_ratio = 2.0 * CRESCENT_KAPPA * DISK_RADIUS_PHYSICAL

    row = {
        "benchmark": "T1_nonlinear_crescent",
        "disk_radius_physical": DISK_RADIUS_PHYSICAL,
        "crescent_kappa": CRESCENT_KAPPA,
        "nonconvexity_ratio_2kR": nonconvexity_ratio,
        "curvature_sign_change_indicator_1_minus_2kR": curvature_indicator,
        "analytic_exact_map_lambda_min_lower_bound": MAP_BASE_EIG_MIN,
        "numerical_exact_map_lambda_min": global_dof_min(lam_min),
        "numerical_exact_map_lambda_max": global_dof_max(lam_max),
        "exact_det_min": global_dof_min(det_exact),
        "exact_det_max": global_dof_max(det_exact),
        "source_density_min": global_dof_min(mu0),
        "source_density_max": global_dof_max(mu0),
        "target_density_min": global_dof_min(mu1),
        "target_density_max": global_dof_max(mu1),
        "source_density_nonfinite_count": global_nonfinite_count(mu0),
        "target_density_nonfinite_count": global_nonfinite_count(mu1),
        "source_mass": source_mass,
        "target_mass": target_mass,
        "mass_absolute_mismatch": abs(source_mass - target_mass),
        "source_bbox_xmin_unit": source_bbox[0],
        "source_bbox_xmax_unit": source_bbox[1],
        "source_bbox_ymin_unit": source_bbox[2],
        "source_bbox_ymax_unit": source_bbox[3],
        "target_bbox_xmin_unit": target_bbox[0],
        "target_bbox_xmax_unit": target_bbox[1],
        "target_bbox_ymin_unit": target_bbox[2],
        "target_bbox_ymax_unit": target_bbox[3],
        "bridge_margin_unit": bridge_margin,
        "max_pushforward_moment_abs_error": max_moment_error,
        "pushforward_density_identity_relative_l2": pushforward_density_rel_l2,
    }
    row.update(metadata)

    checks = {
        "nonconvex_source_certified": curvature_indicator < 0.0,
        "source_coordinate_map_positive": metadata["source_coordinate_map_det_min"] > 0.0,
        "target_coordinate_map_positive": metadata["target_coordinate_map_det_min"] > 0.0,
        "exact_map_strongly_convex": global_dof_min(lam_min) >= CERT_MIN_EXACT_JACOBIAN_EIG,
        "positive_source_density": global_dof_min(mu0) > 0.0,
        "positive_target_density": global_dof_min(mu1) > 0.0,
        "finite_densities": global_nonfinite_count(mu0) == 0 and global_nonfinite_count(mu1) == 0,
        "mass_match": abs(source_mass - target_mass) <= 1.0e-10,
        "area_change_match": metadata["area_change_relative_mismatch"] <= CERT_MAX_AREA_REL_MISMATCH,
        "pushforward_moments_match": max_moment_error <= CERT_MAX_PUSHFORWARD_MOMENT_ERROR,
        "bridge_margin_safe": bridge_margin >= CERT_MIN_BRIDGE_MARGIN_UNIT,
    }
    for key, value in checks.items():
        row[key] = int(value)
    row["certificate_pass"] = int(all(checks.values()))

    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_benchmark_certificate.csv"), [row])
    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_pushforward_moments.csv"), moment_rows)

    if AUDIT_WRITE_VTK:
        VTKFile(os.path.join(result_dir, f"{EXPERIMENT_NAME}_source_geometry.pvd")).write(mu0)
        VTKFile(os.path.join(result_dir, f"{EXPERIMENT_NAME}_target_geometry.pvd")).write(mu1)

    if COMM.rank == 0:
        print("T1_BENCHMARK_CERTIFICATE", row)
        for mrow in moment_rows:
            print("T1_PUSHFORWARD_MOMENT", mrow)

    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"T1 benchmark certificate failed checks: {failed}; row={row}")
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
    row.update(manufactured_map_audit(fine, q, result_dir, method))

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

def gauge_aligned_distance(level, p, q, p_ref, q_ref):
    dp = Function(p.function_space(), name="v46_dp")
    dq = Function(q.function_space(), name="v46_dq")
    dp.interpolate(p - p_ref)
    dq.interpolate(q - q_ref)
    a = float(assemble(dp * level.mu0 * dx))
    b = float(assemble(dq * level.mu1 * dx))
    shift = 0.5 * (a - b)
    dp.interpolate(p - p_ref - Constant(shift))
    dq.interpolate(q - q_ref + Constant(shift))
    ep = math.sqrt(max(float(assemble(dp * dp * level.mu0 * dx)), 0.0))
    eq = math.sqrt(max(float(assemble(dq * dq * level.mu1 * dx)), 0.0))
    return ep, eq, math.sqrt(ep * ep + eq * eq), shift


def reference_norm(level, p, q):
    value = float(assemble(p * p * level.mu0 * dx) + assemble(q * q * level.mu1 * dx))
    return math.sqrt(max(value, 0.0))


def run_discrete_reference_validation(result_dir, fine, coarse, V0, V1, p, q, b_p, b_q):
    if coarse is None or coarse.coarse is not None:
        raise RuntimeError("v46 requires exactly one coarse and one fine bridge level")
    if REF_CHECK_BLOCK <= 0 or REF_MAX_FINE_UPDATES <= 0:
        raise ValueError("reference block and maximum updates must be positive")

    fine.reset_counters()
    ref_rows = []
    previous_p = Function(V0, name="v46_previous_p").assign(p)
    previous_q = Function(V1, name="v46_previous_q").assign(q)
    stable_checks = 0
    updates = 0
    reference_certified = False

    while updates < REF_MAX_FINE_UPDATES:
        block = min(REF_CHECK_BLOCK, REF_MAX_FINE_UPDATES - updates)
        fine.smoother(p, q, b_p, b_q, block, "v46_reference")
        fine.gauge_fix(p, q)
        updates += block
        residual_p, residual_q = fine.residual_norms(p, q, b_p, b_q)
        change_p, change_q, change_abs, change_shift = gauge_aligned_distance(
            fine, p, q, previous_p, previous_q
        )
        norm = reference_norm(fine, p, q)
        change_rel = change_abs / max(norm, 1.0e-30)
        criteria = (
            updates >= REF_MIN_FINE_UPDATES
            and residual_p <= REF_RESIDUAL_TARGET
            and change_rel <= REF_CHANGE_REL_TOL
        )
        stable_checks = stable_checks + 1 if criteria else 0
        reference_certified = stable_checks >= REF_STABLE_CHECKS
        ref_rows.append({
            "coarse_bridge_n": coarse.bridge_n,
            "fine_bridge_n": fine.bridge_n,
            "updates": updates,
            "residual_p": residual_p,
            "residual_q": residual_q,
            "change_p": change_p,
            "change_q": change_q,
            "change_abs": change_abs,
            "change_rel": change_rel,
            "change_gauge_shift": change_shift,
            "stable_checks": stable_checks,
            "reference_certified": int(reference_certified),
        })
        previous_p.assign(p)
        previous_q.assign(q)
        if reference_certified:
            break

    p_ref = Function(V0, name="v46_p_ref").assign(p)
    q_ref = Function(V1, name="v46_q_ref").assign(q)
    reference_heat_rows = fine.counter_rows()
    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_reference_trace.csv"), ref_rows)
    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_reference_heat_counts.csv"), reference_heat_rows)

    p.assign(0.0)
    q.assign(0.0)
    b_p.assign(0.0)
    b_q.assign(0.0)
    fine.reset_counters()

    initial_residual = fine.p_residual_norm(p, q, b_p)
    ep0, eq0, error0, shift0 = gauge_aligned_distance(fine, p, q, p_ref, q_ref)
    trajectory = [{
        "coarse_bridge_n": coarse.bridge_n,
        "fine_bridge_n": fine.bridge_n,
        "cycle": 0,
        "residual_p": initial_residual,
        "residual_ratio_to_initial": 1.0,
        "residual_contraction_from_previous": "",
        "p_error_to_reference": ep0,
        "q_error_to_reference": eq0,
        "combined_error_to_reference": error0,
        "error_ratio_to_initial": 1.0,
        "error_contraction_from_previous": "",
        "gauge_alignment_shift": shift0,
        "target_residual": TARGET_RESIDUAL,
        "reached_target": int(initial_residual <= TARGET_RESIDUAL),
    }]
    # Cycle-zero diagnostics are not part of the production action count.
    fine.reset_counters()
    previous_residual = initial_residual
    previous_error = error0

    for cycle in range(1, FAS_CYCLES + 1):
        fine.fas_cycle_perf(p, q, b_p, b_q)
        residual_p = fine.p_residual_norm(p, q, b_p)
        ep, eq, error, shift = gauge_aligned_distance(fine, p, q, p_ref, q_ref)
        trajectory.append({
            "coarse_bridge_n": coarse.bridge_n,
            "fine_bridge_n": fine.bridge_n,
            "cycle": cycle,
            "residual_p": residual_p,
            "residual_ratio_to_initial": residual_p / initial_residual,
            "residual_contraction_from_previous": residual_p / previous_residual,
            "p_error_to_reference": ep,
            "q_error_to_reference": eq,
            "combined_error_to_reference": error,
            "error_ratio_to_initial": error / error0,
            "error_contraction_from_previous": error / previous_error,
            "gauge_alignment_shift": shift,
            "target_residual": TARGET_RESIDUAL,
            "reached_target": int(residual_p <= TARGET_RESIDUAL),
        })
        previous_residual = residual_p
        previous_error = error
        if residual_p <= TARGET_RESIDUAL:
            break

    cycles = len(trajectory) - 1
    counter_rows = fine.counter_rows()
    observed_coarse = coarse.heat_source_calls + coarse.heat_target_calls
    observed_fine = fine.heat_source_calls + fine.heat_target_calls
    expected_coarse = 16 * cycles
    expected_fine = 9 * cycles
    heat_count_match = int(observed_coarse == expected_coarse and observed_fine == expected_fine)
    if not heat_count_match:
        raise RuntimeError(
            f"v46 heat-count failure: observed={(observed_coarse, observed_fine)} "
            f"expected={(expected_coarse, expected_fine)}"
        )

    final_error = trajectory[-1]["combined_error_to_reference"]
    summary = {
        "coarse_bridge_n": coarse.bridge_n,
        "fine_bridge_n": fine.bridge_n,
        "cycles": cycles,
        "reached_target": trajectory[-1]["reached_target"],
        "initial_residual_p": initial_residual,
        "final_residual_p": trajectory[-1]["residual_p"],
        "initial_error": error0,
        "final_error": final_error,
        "geomean_residual_contraction": (trajectory[-1]["residual_p"] / initial_residual) ** (1.0 / max(cycles, 1)),
        "geomean_error_contraction": (final_error / error0) ** (1.0 / max(cycles, 1)),
        "reference_updates": updates,
        "reference_residual_p": ref_rows[-1]["residual_p"],
        "reference_residual_q": ref_rows[-1]["residual_q"],
        "reference_change_abs": ref_rows[-1]["change_abs"],
        "reference_change_rel": ref_rows[-1]["change_rel"],
        "reference_certified": int(reference_certified),
        "reference_uncertainty_over_final_error": ref_rows[-1]["change_abs"] / max(final_error, 1.0e-300),
        "observed_coarse_heat_actions": observed_coarse,
        "observed_fine_heat_actions": observed_fine,
        "expected_coarse_heat_actions": expected_coarse,
        "expected_fine_heat_actions": expected_fine,
        "heat_count_match": heat_count_match,
    }
    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_fas_error_trajectory.csv"), trajectory)
    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_fas_heat_counts.csv"), counter_rows)
    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_summary.csv"), [summary])
    if COMM.rank == 0:
        print("V46_FINAL", summary)
        print(f"RESULT_DIR={result_dir}")

def main():
    result_dir = os.environ.get("RESULT_DIR", os.getcwd())
    os.makedirs(result_dir, exist_ok=True)

    COMM.Barrier()
    total_t0 = time.perf_counter()

    if RUN_MODE == "benchmark_certificate":
        V0, V1, mu0, mu1, metadata = build_support_spaces_and_measures(return_metadata=True)
        benchmark_certificate(V0, V1, mu0, mu1, metadata, result_dir)
        if COMM.rank == 0:
            print(f"RESULT_DIR={result_dir}")
        return

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
        print(
            "T1_CONFIGURATION",
            f"R={DISK_RADIUS_PHYSICAL}", f"kappa={CRESCENT_KAPPA}",
            f"A_eigs=({MAP_BASE_EIG_MIN},{MAP_BASE_EIG_MAX})",
            f"A_angle={MAP_BASE_ANGLE_DEG}",
            f"s=({MAP_S1},{MAP_S2})",
            f"alpha=({MAP_ALPHA1},{MAP_ALPHA2})",
            f"translation={MAP_TRANSLATION_PHYSICAL}",
        )
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

    if RUN_MODE == "discrete_reference_validation":
        run_discrete_reference_validation(
            result_dir, fine, coarse, V0, V1, p, q, b_p, b_q
        )
        return

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

        write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_fmg_to_tol_trace.csv"), rows)
        write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_fmg_summary.csv"), [rows[-1]])

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

        assert_requested_hierarchy_is_active(fine)

        if len(hierarchy_level_objects(fine)) == 3:
            count_row = fmg_heat_action_count_row(fine, 0, 0, len(rows))
            rows[-1].update(count_row)

        write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_fas_to_tol_trace.csv"), rows)
        write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_fas_summary.csv"), [rows[-1]])

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
