from firedrake import *
from mpi4py import MPI
import csv
import math
import os
import time

EXPERIMENT_NAME = "run_annulus_fmg_corrected_v45"

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
TAU_FINE_PHYSICAL = float(os.environ.get("FAS_TAU_FINE_PHYSICAL", "5.0e-3"))

SOURCE_CENTER_PHYSICAL = (-0.10, 0.0)
TARGET_CENTER_PHYSICAL = (+0.15, 0.0)
SOURCE_INNER_RADIUS_PHYSICAL = 0.25
SOURCE_OUTER_RADIUS_PHYSICAL = 0.65
TARGET_INNER_RADIUS_PHYSICAL = 0.40
TARGET_OUTER_RADIUS_PHYSICAL = math.sqrt(
    TARGET_INNER_RADIUS_PHYSICAL ** 2
    + SOURCE_OUTER_RADIUS_PHYSICAL ** 2
    - SOURCE_INNER_RADIUS_PHYSICAL ** 2
)
ANNULUS_MESH_SIZE_PHYSICAL = float(os.environ.get("FAS_SUPPORT_MESH_SIZE_PHYSICAL", "0.035"))
ANNULUS_ESCAPE_BUFFER_PHYSICAL = float(
    os.environ.get("FAS_ANNULUS_ESCAPE_BUFFER_PHYSICAL", str(2.0 * math.sqrt(TAU_FINE_PHYSICAL)))
)

LOG_CLAMP = Constant(120.0)
TINY = Constant(1e-300)

FAS_CYCLES = int(os.environ.get("FAS_CYCLES", "10"))
PRE_SMOOTH = int(os.environ.get("FAS_PRE_SMOOTH", "2"))
COARSE_SMOOTH = int(os.environ.get("FAS_COARSE_SMOOTH", "7"))
POST_SMOOTH = int(os.environ.get("FAS_POST_SMOOTH", "1"))
FMG_N = int(os.environ.get("FAS_FMG_N", "0"))
FMG_TOP_MAX_CYCLES = int(os.environ.get("FAS_FMG_TOP_MAX_CYCLES", str(FAS_CYCLES)))

RUN_MODE = os.environ.get("FAS_RUN_MODE", "sign_test")
SMOOTHER_TEST_UPDATES = int(os.environ.get("FAS_SMOOTHER_TEST_UPDATES", "1"))
FINE_ONLY_UPDATES = int(os.environ.get("FAS_FINE_ONLY_UPDATES", "10"))
CORRECTION_DAMPING = float(os.environ.get("FAS_CORRECTION_DAMPING", "1.0"))
COARSE_SWEEP_VALUES = [int(x) for x in os.environ.get("FAS_COARSE_SWEEP_VALUES", "0,2,5,10,20,30").split(",") if x.strip()]
COARSE_TAU_RATIO_VALUES = [float(x) for x in os.environ.get("FAS_COARSE_TAU_RATIO_VALUES", "1,2,3,4").split(",") if x.strip()]
COARSE_TAU_RATIO = float(os.environ.get("FAS_COARSE_TAU_RATIO", "1.0"))
TARGET_RESIDUAL = float(os.environ.get("FAS_TARGET_RESIDUAL", "4.31e-3"))
AUDIT_WRITE_VTK = int(os.environ.get("FAS_WRITE_VTK", "0"))
MAX_FINE_ONLY_UPDATES = int(os.environ.get("FAS_MAX_FINE_ONLY_UPDATES", "90"))
FINE_CHECK_BLOCK = int(os.environ.get("FAS_FINE_CHECK_BLOCK", "3"))
CONTINUATION_PREFINAL_BRIDGE_N = int(os.environ.get("FAS_CONTINUATION_PREFINAL_BRIDGE_N", "230"))
CONTINUATION_PREFINAL_TAU_PHYSICAL = float(
    os.environ.get("FAS_CONTINUATION_PREFINAL_TAU_PHYSICAL", str(TAU_FINE_PHYSICAL))
)
CONTINUATION_PREFINAL_UPDATES = int(os.environ.get("FAS_CONTINUATION_PREFINAL_UPDATES", "5"))
CONTINUATION_TRANSFER_MODE = os.environ.get("FAS_CONTINUATION_TRANSFER_MODE", "log_scaling")

COMM = COMM_WORLD

solver_params = {
    "ksp_type": "preonly",
    "pc_type": "cholesky",
    "pc_factor_mat_solver_type": "mumps",
}


def physical_to_unit_expr(x_phys, y_phys):
    return as_vector([
        Constant(0.5) + x_phys / Constant(BRIDGE_WIDTH_PHYSICAL),
        Constant(0.5) + y_phys / Constant(BRIDGE_WIDTH_PHYSICAL),
    ])


def physical_to_unit_scalar(value):
    return 0.5 + value / BRIDGE_WIDTH_PHYSICAL


def annulus_target_radius_physical_expr(r):
    return sqrt(
        Constant(TARGET_INNER_RADIUS_PHYSICAL ** 2 - SOURCE_INNER_RADIUS_PHYSICAL ** 2)
        + r * r
    )


def annulus_exact_map_unit_expr(x):
    width = BRIDGE_WIDTH_PHYSICAL
    x0_phys = width * (x[0] - Constant(0.5))
    x1_phys = width * (x[1] - Constant(0.5))
    dx0 = x0_phys - Constant(SOURCE_CENTER_PHYSICAL[0])
    dx1 = x1_phys - Constant(SOURCE_CENTER_PHYSICAL[1])
    r = sqrt(dx0 * dx0 + dx1 * dx1)
    srad = annulus_target_radius_physical_expr(r)
    scale = srad / max_value(r, Constant(1.0e-14))
    y0_phys = Constant(TARGET_CENTER_PHYSICAL[0]) + scale * dx0
    y1_phys = Constant(TARGET_CENTER_PHYSICAL[1]) + scale * dx1
    return physical_to_unit_expr(y0_phys, y1_phys)


def annulus_polar_frame_physical_expr(x):
    width = BRIDGE_WIDTH_PHYSICAL
    x0_phys = width * (x[0] - Constant(0.5))
    x1_phys = width * (x[1] - Constant(0.5))
    dx0 = x0_phys - Constant(SOURCE_CENTER_PHYSICAL[0])
    dx1 = x1_phys - Constant(SOURCE_CENTER_PHYSICAL[1])
    r = sqrt(dx0 * dx0 + dx1 * dx1)
    r_safe = max_value(r, Constant(1.0e-14))
    srad = annulus_target_radius_physical_expr(r_safe)
    er0 = dx0 / r_safe
    er1 = dx1 / r_safe
    et0 = -er1
    et1 = er0
    lambda_r = r_safe / srad
    lambda_t = srad / r_safe
    return r_safe, srad, er0, er1, et0, et1, lambda_r, lambda_t


def write_structured_annulus_msh(path, mesh_size_physical):
    radial_width = SOURCE_OUTER_RADIUS_PHYSICAL - SOURCE_INNER_RADIUS_PHYSICAL
    n_radial = max(5, int(math.ceil(radial_width / mesh_size_physical)))
    n_theta = max(32, int(math.ceil(2.0 * math.pi * SOURCE_OUTER_RADIUS_PHYSICAL / mesh_size_physical)))
    if n_theta % 4:
        n_theta += 4 - (n_theta % 4)

    nodes = []
    for i in range(n_radial + 1):
        radius = SOURCE_INNER_RADIUS_PHYSICAL + radial_width * i / n_radial
        for j in range(n_theta):
            theta = 2.0 * math.pi * j / n_theta
            x_phys = SOURCE_CENTER_PHYSICAL[0] + radius * math.cos(theta)
            y_phys = SOURCE_CENTER_PHYSICAL[1] + radius * math.sin(theta)
            nodes.append((
                physical_to_unit_scalar(x_phys),
                physical_to_unit_scalar(y_phys),
                0.0,
            ))

    def node_id(i, j):
        return i * n_theta + (j % n_theta) + 1

    elements = []
    eid = 1
    inner_tag = 11
    outer_tag = 12
    surface_tag = 1

    for j in range(n_theta):
        elements.append((eid, 1, inner_tag, inner_tag, (node_id(0, j + 1), node_id(0, j))))
        eid += 1

    for j in range(n_theta):
        elements.append((eid, 1, outer_tag, outer_tag, (node_id(n_radial, j), node_id(n_radial, j + 1))))
        eid += 1

    for i in range(n_radial):
        for j in range(n_theta):
            n00 = node_id(i, j)
            n01 = node_id(i, j + 1)
            n10 = node_id(i + 1, j)
            n11 = node_id(i + 1, j + 1)
            elements.append((eid, 2, surface_tag, surface_tag, (n00, n10, n11)))
            eid += 1
            elements.append((eid, 2, surface_tag, surface_tag, (n00, n11, n01)))
            eid += 1

    with open(path, "w") as handle:
        handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        handle.write("$PhysicalNames\n3\n")
        handle.write(f'1 {inner_tag} "inner_boundary"\n')
        handle.write(f'1 {outer_tag} "outer_boundary"\n')
        handle.write(f'2 {surface_tag} "source_annulus"\n')
        handle.write("$EndPhysicalNames\n")
        handle.write(f"$Nodes\n{len(nodes)}\n")
        for idx, (x, y, z) in enumerate(nodes, start=1):
            handle.write(f"{idx} {x:.16e} {y:.16e} {z:.16e}\n")
        handle.write("$EndNodes\n")
        handle.write(f"$Elements\n{len(elements)}\n")
        for elem_id, elem_type, physical, geometric, elem_nodes in elements:
            node_text = " ".join(str(n) for n in elem_nodes)
            handle.write(f"{elem_id} {elem_type} 2 {physical} {geometric} {node_text}\n")
        handle.write("$EndElements\n")


def annulus_mesh_pair_unit(mesh_size_physical, level_name):
    work_dir = os.environ.get("RESULT_DIR", os.getcwd())
    os.makedirs(work_dir, exist_ok=True)
    msh_path = os.path.join(work_dir, f"source_annulus_{level_name}.msh")

    if COMM.rank == 0:
        write_structured_annulus_msh(msh_path, mesh_size_physical)
    COMM.Barrier()

    source_mesh = Mesh(msh_path)
    x = SpatialCoordinate(source_mesh)
    Vc = source_mesh.coordinates.function_space()
    target_coords = Function(Vc, name="target_annulus_coordinates")
    target_coords.interpolate(annulus_exact_map_unit_expr(x))
    target_mesh = Mesh(Function(target_coords))
    return source_mesh, target_mesh


def build_support_spaces_and_measures():
    mesh0, mesh1 = annulus_mesh_pair_unit(ANNULUS_MESH_SIZE_PHYSICAL, "fine")
    V0 = FunctionSpace(mesh0, "CG", DISK_CG_DEGREE)
    V1 = FunctionSpace(mesh1, "CG", DISK_CG_DEGREE)

    rho0 = Function(V0, name="rho0_raw").interpolate(1.0)
    rho1 = Function(V1, name="rho1_raw").interpolate(1.0)
    mass0 = assemble(rho0 * dx)
    mass1 = assemble(rho1 * dx)

    mu0 = Function(V0, name="mu0").interpolate(rho0 / mass0)
    mu1 = Function(V1, name="mu1").interpolate(rho1 / mass1)

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


def global_dof_count_le(function, threshold):
    data = function.dat.data_ro
    local = int((data <= threshold).sum()) if data.size else 0
    return int(COMM.allreduce(local, op=MPI.SUM))


def physical_radius_function(V, center_physical, name):
    x = SpatialCoordinate(V.mesh())
    width = BRIDGE_WIDTH_PHYSICAL
    x0_phys = width * (x[0] - Constant(0.5))
    x1_phys = width * (x[1] - Constant(0.5))
    radius = Function(V, name=name)
    radius.interpolate(
        sqrt(
            (x0_phys - Constant(center_physical[0])) ** 2
            + (x1_phys - Constant(center_physical[1])) ** 2
        )
    )
    return radius


def support_geometry_rows(V0, V1, mu0, mu1):
    source_r = physical_radius_function(V0, SOURCE_CENTER_PHYSICAL, "source_radius_physical")
    target_r = physical_radius_function(V1, TARGET_CENTER_PHYSICAL, "target_radius_physical")
    return [
        {
            "support": "source",
            "cells": COMM.allreduce(int(V0.mesh().num_cells()), op=MPI.SUM),
            "cg_dofs": V0.dim(),
            "mass": float(assemble(mu0 * dx(domain=V0.mesh()))),
            "radius_min": global_dof_min(source_r),
            "radius_max": global_dof_max(source_r),
            "expected_inner_radius": SOURCE_INNER_RADIUS_PHYSICAL,
            "expected_outer_radius": SOURCE_OUTER_RADIUS_PHYSICAL,
        },
        {
            "support": "target",
            "cells": COMM.allreduce(int(V1.mesh().num_cells()), op=MPI.SUM),
            "cg_dofs": V1.dim(),
            "mass": float(assemble(mu1 * dx(domain=V1.mesh()))),
            "radius_min": global_dof_min(target_r),
            "radius_max": global_dof_max(target_r),
            "expected_inner_radius": TARGET_INNER_RADIUS_PHYSICAL,
            "expected_outer_radius": TARGET_OUTER_RADIUS_PHYSICAL,
        },
    ]


class FASLevel:
    def __init__(self, name, level_index, mesh, tau_physical, V0, V1, mu0, mu1, coarse=None, bridge_n=None):
        self.name = name
        self.level_index = level_index
        self.mesh = mesh
        self.bridge_n = BRIDGE_LEVELS[level_index] if bridge_n is None else int(bridge_n)
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

        self.heat_source_calls += 1
        self.backward_euler_solves += N_SUB

        t0 = time.perf_counter()
        self.rhs_source.interpolate(
            scaling,
            allow_missing_dofs=True,
            default_missing_val=0.0,
        )
        self.source_input_interp_seconds += time.perf_counter() - t0

        # Each complete heat action starts from the same zero initial state.
        self.heat_source_bridge.assign(0.0)
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

        self.heat_target_calls += 1
        self.backward_euler_solves += N_SUB

        t0 = time.perf_counter()
        self.rhs_target.interpolate(
            scaling,
            allow_missing_dofs=True,
            default_missing_val=0.0,
        )
        self.target_input_interp_seconds += time.perf_counter() - t0

        # Each complete heat action starts from the same zero initial state.
        self.heat_target_bridge.assign(0.0)
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




def annulus_band_errors(T0_phys, T1_phys, T0_exact_phys, T1_exact_phys, x, fine):
    width = BRIDGE_WIDTH_PHYSICAL
    x0_phys = width * (x[0] - Constant(0.5))
    x1_phys = width * (x[1] - Constant(0.5))
    rx = x0_phys - Constant(SOURCE_CENTER_PHYSICAL[0])
    ry = x1_phys - Constant(SOURCE_CENTER_PHYSICAL[1])
    r = sqrt(rx * rx + ry * ry)
    band_width = Constant((SOURCE_OUTER_RADIUS_PHYSICAL - SOURCE_INNER_RADIUS_PHYSICAL) / 3.0)

    inner = conditional(le(r, Constant(SOURCE_INNER_RADIUS_PHYSICAL) + band_width), 1.0, 0.0)
    middle = conditional(
        le(r, Constant(SOURCE_INNER_RADIUS_PHYSICAL) + 2.0 * band_width),
        conditional(gt(r, Constant(SOURCE_INNER_RADIUS_PHYSICAL) + band_width), 1.0, 0.0),
        0.0,
    )
    outer = conditional(gt(r, Constant(SOURCE_INNER_RADIUS_PHYSICAL) + 2.0 * band_width), 1.0, 0.0)

    err2 = (T0_phys - T0_exact_phys) ** 2 + (T1_phys - T1_exact_phys) ** 2
    out = {}
    for label, mask in [("inner", inner), ("middle", middle), ("outer", outer)]:
        mass = float(assemble(mask * fine.mu0 * dx))
        val = float(assemble(err2 * mask * fine.mu0 * dx))
        out[f"{label}_mass"] = mass
        out[f"{label}_map_l2_physical"] = math.sqrt(max(val, 0.0) / max(mass, 1.0e-300))
    return out


def annulus_map_audit(fine, q, result_dir, method):
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

    T_exact = annulus_exact_map_unit_expr(x)
    T0_exact = T_exact[0]
    T1_exact = T_exact[1]

    T0_phys = width * (T0 - Constant(0.5))
    T1_phys = width * (T1 - Constant(0.5))
    T0_exact_phys = width * (T0_exact - Constant(0.5))
    T1_exact_phys = width * (T1_exact - Constant(0.5))

    map_error_unit = Function(V0, name=f"{method}_map_error_unit")
    map_error_unit.interpolate(sqrt((T0 - T0_exact) ** 2 + (T1 - T1_exact) ** 2))
    map_l2_unit = math.sqrt(max(float(assemble(map_error_unit ** 2 * fine.mu0 * dx)), 0.0))
    map_l2_physical = width * map_l2_unit
    map_linf_unit = global_dof_max(map_error_unit)
    map_linf_physical = width * map_linf_unit

    _, _, er0, er1, et0, et1, lambda_r, lambda_t = annulus_polar_frame_physical_expr(x)
    ex_phys = T0_phys - T0_exact_phys
    ey_phys = T1_phys - T1_exact_phys
    radial_error = ex_phys * er0 + ey_phys * er1
    tangential_error = ex_phys * et0 + ey_phys * et1
    radial_l2_physical = math.sqrt(max(float(assemble(radial_error ** 2 * fine.mu0 * dx)), 0.0))
    tangential_l2_physical = math.sqrt(max(float(assemble(tangential_error ** 2 * fine.mu0 * dx)), 0.0))

    T_field = as_vector([T0, T1])
    J = grad(T_field)
    det_space = FunctionSpace(V0.mesh(), "DG", 0)
    det_T = Function(det_space, name=f"{method}_det_DT")
    det_T.interpolate(det(J))

    det_min = global_dof_min(det_T)
    det_max = global_dof_max(det_T)
    det_l2 = math.sqrt(max(float(assemble((det(J) - Constant(1.0)) ** 2 * fine.mu0 * dx)), 0.0))
    tangled_area = float(assemble(conditional(lt(det(J), 0.0), 1.0, 0.0) * dx(domain=V0.mesh())))

    Jrr = er0 * (J[0, 0] * er0 + J[0, 1] * er1) + er1 * (J[1, 0] * er0 + J[1, 1] * er1)
    Jrt = er0 * (-J[0, 0] * er1 + J[0, 1] * er0) + er1 * (-J[1, 0] * er1 + J[1, 1] * er0)
    Jtr = -er1 * (J[0, 0] * er0 + J[0, 1] * er1) + er0 * (J[1, 0] * er0 + J[1, 1] * er1)
    Jtt = -er1 * (-J[0, 0] * er1 + J[0, 1] * er0) + er0 * (-J[1, 0] * er1 + J[1, 1] * er0)

    polar_jacobian_l2 = math.sqrt(max(float(assemble((
        (Jrr - lambda_r) ** 2
        + Jrt ** 2
        + Jtr ** 2
        + (Jtt - lambda_t) ** 2
    ) * fine.mu0 * dx)), 0.0))

    target_rx = T0_phys - Constant(TARGET_CENTER_PHYSICAL[0])
    target_ry = T1_phys - Constant(TARGET_CENTER_PHYSICAL[1])
    target_r = sqrt(target_rx * target_rx + target_ry * target_ry)
    raw_hole_escape_mass = float(assemble(
        conditional(lt(target_r, Constant(TARGET_INNER_RADIUS_PHYSICAL)), 1.0, 0.0) * fine.mu0 * dx
    ))
    raw_outer_escape_mass = float(assemble(
        conditional(gt(target_r, Constant(TARGET_OUTER_RADIUS_PHYSICAL)), 1.0, 0.0) * fine.mu0 * dx
    ))
    buffered_hole_escape_mass = float(assemble(
        conditional(lt(target_r, Constant(TARGET_INNER_RADIUS_PHYSICAL - ANNULUS_ESCAPE_BUFFER_PHYSICAL)), 1.0, 0.0) * fine.mu0 * dx
    ))
    buffered_outer_escape_mass = float(assemble(
        conditional(gt(target_r, Constant(TARGET_OUTER_RADIUS_PHYSICAL + ANNULUS_ESCAPE_BUFFER_PHYSICAL)), 1.0, 0.0) * fine.mu0 * dx
    ))

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
        "source_center_x_physical": SOURCE_CENTER_PHYSICAL[0],
        "source_center_y_physical": SOURCE_CENTER_PHYSICAL[1],
        "target_center_x_physical": TARGET_CENTER_PHYSICAL[0],
        "target_center_y_physical": TARGET_CENTER_PHYSICAL[1],
        "source_inner_radius_physical": SOURCE_INNER_RADIUS_PHYSICAL,
        "source_outer_radius_physical": SOURCE_OUTER_RADIUS_PHYSICAL,
        "target_inner_radius_physical": TARGET_INNER_RADIUS_PHYSICAL,
        "target_outer_radius_physical": TARGET_OUTER_RADIUS_PHYSICAL,
        "annulus_mesh_size_physical": ANNULUS_MESH_SIZE_PHYSICAL,
        "annulus_escape_buffer_physical": ANNULUS_ESCAPE_BUFFER_PHYSICAL,
        "map_q_shift": q_shift,
        "map_denominator_min": global_dof_min(Hden),
        "map_denominator_max": global_dof_max(Hden),
        "map_denominator_n_le_0": global_dof_count_le(Hden, 0.0),
        "map_denominator_n_le_tiny": global_dof_count_le(Hden, 1.0e-300),
        "map_l2_unit": map_l2_unit,
        "map_l2_physical": map_l2_physical,
        "map_linf_unit": map_linf_unit,
        "map_linf_physical": map_linf_physical,
        "radial_l2_physical": radial_l2_physical,
        "tangential_l2_physical": tangential_l2_physical,
        "polar_jacobian_l2": polar_jacobian_l2,
        "det_l2": det_l2,
        "det_min": det_min,
        "det_max": det_max,
        "tangled_area": tangled_area,
        "raw_hole_escape_mass": raw_hole_escape_mass,
        "buffered_hole_escape_mass": buffered_hole_escape_mass,
        "raw_outer_escape_mass": raw_outer_escape_mass,
        "buffered_outer_escape_mass": buffered_outer_escape_mass,
        "monge_defect_unit": monge_defect_unit,
        "monge_defect_physical": monge_defect_physical,
        "rms_conditional_spread_physical": rms_conditional_spread_physical,
    }
    row.update(annulus_band_errors(T0_phys, T1_phys, T0_exact_phys, T1_exact_phys, x, fine))

    if AUDIT_WRITE_VTK:
        Vvec = VectorFunctionSpace(V0.mesh(), "CG", DISK_CG_DEGREE)
        T_vec = Function(Vvec, name=f"{method}_T_unit").interpolate(as_vector([T0, T1]))
        T_exact_vec = Function(Vvec, name=f"{method}_T_exact_unit").interpolate(T_exact)
        VTKFile(os.path.join(result_dir, f"{EXPERIMENT_NAME}_{method}_source_map.pvd")).write(
            T_vec, T_exact_vec, map_error_unit, det_T, variance
        )

    if COMM.rank == 0:
        print("FAS_ANNULUS_MAP_AUDIT", row)

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
    row.update(annulus_map_audit(fine, q, result_dir, method))

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

def build_bridge_continuation_problem():
    V0, V1, mu0, mu1 = build_support_spaces_and_measures()

    if CONTINUATION_PREFINAL_BRIDGE_N == BRIDGE_COARSE_N:
        bridge_hierarchy = build_bridge_hierarchy()
        pre_mesh = bridge_hierarchy[0]
        fine_mesh = bridge_hierarchy[1]
    else:
        pre_mesh = PeriodicUnitSquareMesh(
            CONTINUATION_PREFINAL_BRIDGE_N,
            CONTINUATION_PREFINAL_BRIDGE_N,
        )
        fine_mesh = PeriodicUnitSquareMesh(BRIDGE_LEVELS[-1], BRIDGE_LEVELS[-1])

    pre_level = FASLevel(
        f"continuation_B{CONTINUATION_PREFINAL_BRIDGE_N}",
        0,
        pre_mesh,
        CONTINUATION_PREFINAL_TAU_PHYSICAL,
        V0,
        V1,
        mu0,
        mu1,
        coarse=None,
        bridge_n=CONTINUATION_PREFINAL_BRIDGE_N,
    )

    fine = FASLevel(
        f"continuation_B{BRIDGE_LEVELS[-1]}",
        1,
        fine_mesh,
        TAU_FINE_PHYSICAL,
        V0,
        V1,
        mu0,
        mu1,
        coarse=None,
        bridge_n=BRIDGE_LEVELS[-1],
    )

    return pre_level, fine, V0, V1


def fmg_level_triplet(fine):
    levels = hierarchy_level_objects(fine)
    if len(levels) != 3:
        raise RuntimeError(f"FMG requires exactly 3 levels, got {len(levels)}")
    if levels[-1] is not fine:
        raise RuntimeError("FMG hierarchy ordering error: finest level is not last")
    return levels[0], levels[1], levels[2]


def fmg_heat_action_count_row(fine, fmg_n, top_cycles):
    bottom, middle, top = fmg_level_triplet(fine)
    levels = [bottom, middle, top]
    actual = [level.heat_source_calls + level.heat_target_calls for level in levels]

    if fmg_n == 0:
        expected = [16 * top_cycles, 10 * top_cycles, 9 * top_cycles]
    else:
        expected = [
            30 * fmg_n + 16 * top_cycles,
            8 * fmg_n + 10 * top_cycles,
            1 + 9 * top_cycles,
        ]

    row = {
        "fmg_N": fmg_n,
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
            f"N={fmg_n}, top_cycles={top_cycles}"
        )

    return row

def main():
    result_dir = os.environ.get("RESULT_DIR", os.getcwd())
    os.makedirs(result_dir, exist_ok=True)

    COMM.Barrier()
    total_t0 = time.perf_counter()

    if RUN_MODE == "geometry_only":
        V0, V1, mu0, mu1 = build_support_spaces_and_measures()
        rows = support_geometry_rows(V0, V1, mu0, mu1)
        write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_geometry_only.csv"), rows)
        if COMM.rank == 0:
            print(f"EXPERIMENT_NAME={EXPERIMENT_NAME}")
            print("FAS_RUN_MODE=geometry_only")
            for row in rows:
                print("ANNULUS_GEOMETRY_ROW", row)
            print(f"RESULT_DIR={result_dir}")
        return

    continuation_prefinal = None
    if RUN_MODE == "fine_to_tol":
        fine, coarse, V0, V1 = build_fine_only_problem()
        hierarchy = [fine.hierarchy_row()]
    elif RUN_MODE == "bridge_continuation_to_tol":
        continuation_prefinal, fine, V0, V1 = build_bridge_continuation_problem()
        coarse = None
        hierarchy = [continuation_prefinal.hierarchy_row(), fine.hierarchy_row()]
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
        print(f"ANNULUS source_center={SOURCE_CENTER_PHYSICAL} target_center={TARGET_CENTER_PHYSICAL} source_r=({SOURCE_INNER_RADIUS_PHYSICAL},{SOURCE_OUTER_RADIUS_PHYSICAL}) target_r=({TARGET_INNER_RADIUS_PHYSICAL},{TARGET_OUTER_RADIUS_PHYSICAL})")
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



    if RUN_MODE == "bridge_continuation_to_tol":
        continuation_prefinal.reset_counters()
        fine.reset_counters()
        rows = []
        iteration_t0 = time.perf_counter()

        continuation_prefinal.smoother(
            p, q,
            b_p, b_q,
            CONTINUATION_PREFINAL_UPDATES,
            f"bridge_continuation_prefinal_B{CONTINUATION_PREFINAL_BRIDGE_N}",
        )
        continuation_prefinal.gauge_fix(p, q)

        prefinal_residual_p = continuation_prefinal.p_residual_norm(p, q, b_p)
        fine_before_transfer_p = fine.p_residual_norm(p, q, b_p)

        if CONTINUATION_TRANSFER_MODE == "potential":
            transfer_scale = CONTINUATION_PREFINAL_TAU_PHYSICAL / TAU_FINE_PHYSICAL
            p_transfer = Function(V0, name="p_potential_transferred_to_final_tau")
            q_transfer = Function(V1, name="q_potential_transferred_to_final_tau")
            p_transfer.interpolate(Constant(transfer_scale) * p)
            q_transfer.interpolate(Constant(transfer_scale) * q)
            p.assign(p_transfer)
            q.assign(q_transfer)
            transfer_gauge_shift = fine.gauge_fix(p, q)
        elif CONTINUATION_TRANSFER_MODE == "log_scaling":
            transfer_scale = 1.0
            transfer_gauge_shift = fine.gauge_fix(p, q)
        else:
            raise ValueError(f"Unknown FAS_CONTINUATION_TRANSFER_MODE={CONTINUATION_TRANSFER_MODE!r}")

        fine_after_transfer_gauge_p = fine.p_residual_norm(p, q, b_p)
        gauge_residual_delta = abs(fine_after_transfer_gauge_p - fine_before_transfer_p)
        residual_p = fine_after_transfer_gauge_p
        now = time.perf_counter()
        row = {
            "mode": "bridge_continuation_to_tol",
            "stage": "after_prefinal_transfer",
            "updates": 0,
            "prefinal_updates": CONTINUATION_PREFINAL_UPDATES,
            "final_updates": 0,
            "total_updates": CONTINUATION_PREFINAL_UPDATES,
            "prefinal_bridge_n": CONTINUATION_PREFINAL_BRIDGE_N,
            "final_bridge_n": BRIDGE_LEVELS[-1],
            "prefinal_tau_physical": CONTINUATION_PREFINAL_TAU_PHYSICAL,
            "final_tau_physical": TAU_FINE_PHYSICAL,
            "transfer_mode": CONTINUATION_TRANSFER_MODE,
            "transfer_scale": transfer_scale,
            "prefinal_residual_p": prefinal_residual_p,
            "fine_before_transfer_p": fine_before_transfer_p,
            "fine_after_transfer_gauge_p": fine_after_transfer_gauge_p,
            "transfer_gauge_shift": transfer_gauge_shift,
            "gauge_residual_delta": gauge_residual_delta,
            "residual_p": residual_p,
            "residual_q": "",
            "target_residual": TARGET_RESIDUAL,
            "reached_target": int(residual_p <= TARGET_RESIDUAL),
            "iteration_seconds": now - iteration_t0,
            "total_seconds": now - total_t0,
        }
        rows.append(row)
        if COMM.rank == 0:
            print("BRIDGE_CONTINUATION_TO_TOL_ROW", row)

        final_updates = 0
        while residual_p > TARGET_RESIDUAL and final_updates < MAX_FINE_ONLY_UPDATES:
            block = min(FINE_CHECK_BLOCK, MAX_FINE_ONLY_UPDATES - final_updates)
            fine.smoother(
                p, q,
                b_p, b_q,
                block,
                f"bridge_continuation_final_B{BRIDGE_LEVELS[-1]}_{final_updates + block}",
            )
            fine.gauge_fix(p, q)
            final_updates += block

            residual_p = fine.p_residual_norm(p, q, b_p)
            now = time.perf_counter()
            row = {
                "mode": "bridge_continuation_to_tol",
                "stage": "final_to_tol",
                "updates": final_updates,
                "prefinal_updates": CONTINUATION_PREFINAL_UPDATES,
                "final_updates": final_updates,
                "total_updates": CONTINUATION_PREFINAL_UPDATES + final_updates,
                "prefinal_bridge_n": CONTINUATION_PREFINAL_BRIDGE_N,
                "final_bridge_n": BRIDGE_LEVELS[-1],
                "prefinal_tau_physical": CONTINUATION_PREFINAL_TAU_PHYSICAL,
                "final_tau_physical": TAU_FINE_PHYSICAL,
                "transfer_mode": CONTINUATION_TRANSFER_MODE,
                "transfer_scale": transfer_scale,
                "residual_p": residual_p,
                "residual_q": "",
                "target_residual": TARGET_RESIDUAL,
                "reached_target": int(residual_p <= TARGET_RESIDUAL),
                "iteration_seconds": now - iteration_t0,
                "total_seconds": now - total_t0,
            }
            rows.append(row)
            if COMM.rank == 0:
                print("BRIDGE_CONTINUATION_TO_TOL_ROW", row)
            if residual_p <= TARGET_RESIDUAL:
                break

        write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_bridge_continuation_to_tol_trace.csv"), rows)
        profile_rows = continuation_prefinal.counter_rows() + fine.counter_rows()
        write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_heat_counts.csv"), profile_rows)
        write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_cost_profile.csv"), profile_rows)
        final_science_audit("bridge_continuation_to_tol", fine, p, q, b_p, b_q, rows[-1], result_dir)

        if COMM.rank == 0:
            print("BRIDGE_CONTINUATION_TO_TOL_FINAL", rows[-1])
            print(f"RESULT_DIR={result_dir}")
        return


    if RUN_MODE == "fmg_to_tol":
        fine.reset_counters()
        bottom, middle, top = fmg_level_triplet(fine)
        rows = []
        iteration_t0 = time.perf_counter()

        bottom_prep_seconds = 0.0
        middle_prep_seconds = 0.0
        entry_check_seconds = 0.0
        entry_residual_p = initial_p

        if FMG_N > 0:
            stage_t0 = time.perf_counter()
            for prep in range(1, FMG_N + 1):
                bottom.smoother(
                    p, q,
                    b_p, b_q,
                    COARSE_SMOOTH,
                    f"fmg_bottom_prep_N{prep}_B{bottom.bridge_n}",
                )
                bottom.gauge_fix(p, q)
            bottom_prep_seconds = COMM.allreduce(time.perf_counter() - stage_t0, op=MPI.MAX)

            stage_t0 = time.perf_counter()
            for prep in range(1, FMG_N + 1):
                middle.fas_cycle_perf(p, q, b_p, b_q)
                middle.gauge_fix(p, q)
            middle_prep_seconds = COMM.allreduce(time.perf_counter() - stage_t0, op=MPI.MAX)

            stage_t0 = time.perf_counter()
            entry_residual_p = fine.p_residual_norm(p, q, b_p)
            entry_check_seconds = COMM.allreduce(time.perf_counter() - stage_t0, op=MPI.MAX)

        prep_seconds = bottom_prep_seconds + middle_prep_seconds + entry_check_seconds
        top_t0 = time.perf_counter()
        top_cycles = 0

        if FMG_N > 0 and entry_residual_p <= TARGET_RESIDUAL:
            now = time.perf_counter()
            row = {
                "mode": "fmg_to_tol",
                "fmg_N": FMG_N,
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
        count_row = fmg_heat_action_count_row(fine, FMG_N, top_cycles)
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
