from firedrake import *
from mpi4py import MPI
import csv
import hashlib
import math
import os
import time
import numpy as np

EXPERIMENT_NAME = "run_t3_disk_to_annulus_stress_v49"

# T3 topology-change benchmark.  Source and target use approximately the same
# support complexity as T2: 131072 triangles each.  The source is a disk;
# the target is a genuine annulus built independently because the exact map
# changes topology at the measure-zero source centre.
SOURCE_GRID_N = int(os.environ.get("T3_SOURCE_GRID_N", "256"))
TARGET_RADIAL_LAYERS = int(os.environ.get("T3_TARGET_RADIAL_LAYERS", "64"))
TARGET_THETA_CELLS = int(os.environ.get("T3_TARGET_THETA_CELLS", "1024"))
DISK_CG_DEGREE = 2
BRIDGE_CG_DEGREE = 2
N_SUB = 16

BRIDGE_COARSE_N = int(os.environ.get("FAS_BRIDGE_COARSE_N", "230"))
N_BRIDGE_REFINEMENTS = int(os.environ.get("FAS_N_BRIDGE_REFINEMENTS", "2"))
BRIDGE_LEVELS = [BRIDGE_COARSE_N * 2**level for level in range(N_BRIDGE_REFINEMENTS + 1)]
BRIDGE_WIDTH_PHYSICAL = 4.0
TAU_FINE_PHYSICAL = float(os.environ.get("FAS_TAU_FINE_PHYSICAL", "2.373046875e-3"))
EPSILON_PHYSICAL = 2.0 * TAU_FINE_PHYSICAL

SOURCE_RADIUS_PHYSICAL = float(os.environ.get("T3_SOURCE_RADIUS_PHYSICAL", "0.6"))
TARGET_INNER_RADIUS_PHYSICAL = float(os.environ.get("T3_TARGET_INNER_RADIUS_PHYSICAL", "0.25"))
# Frozen equal-area choice: 0.65^2 - 0.25^2 = 0.60^2.
TARGET_OUTER_RADIUS_PHYSICAL = float(os.environ.get("T3_TARGET_OUTER_RADIUS_PHYSICAL", "0.65"))
RADIAL_AREA_FACTOR = (
    TARGET_OUTER_RADIUS_PHYSICAL**2 - TARGET_INNER_RADIUS_PHYSICAL**2
) / (SOURCE_RADIUS_PHYSICAL**2)
CENTER_CHARACTERISTIC_SCALE_PHYSICAL = (
    EPSILON_PHYSICAL / TARGET_INNER_RADIUS_PHYSICAL
)
CENTER_BINS_PHYSICAL = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, SOURCE_RADIUS_PHYSICAL + 1.0e-12]
DEEP_HOLE_BUFFER_PHYSICAL = min(2.0 * math.sqrt(TAU_FINE_PHYSICAL), 0.5 * TARGET_INNER_RADIUS_PHYSICAL)
DEEP_HOLE_RADIUS_PHYSICAL = max(TARGET_INNER_RADIUS_PHYSICAL - DEEP_HOLE_BUFFER_PHYSICAL, 0.0)

LOG_CLAMP = Constant(120.0)
TINY = Constant(1e-300)

FAS_CYCLES = int(os.environ.get("FAS_CYCLES", "8"))
PRE_SMOOTH = int(os.environ.get("FAS_PRE_SMOOTH", "2"))
COARSE_SMOOTH = int(os.environ.get("FAS_COARSE_SMOOTH", "7"))
POST_SMOOTH = int(os.environ.get("FAS_POST_SMOOTH", "1"))
RUN_MODE = os.environ.get("FAS_RUN_MODE", "benchmark_certificate")
CORRECTION_DAMPING = float(os.environ.get("FAS_CORRECTION_DAMPING", "1.0"))
COARSE_TAU_RATIO = float(os.environ.get("FAS_COARSE_TAU_RATIO", "1.0"))
TARGET_RESIDUAL = float(os.environ.get("FAS_TARGET_RESIDUAL", "4.31e-3"))
AUDIT_WRITE_VTK = int(os.environ.get("FAS_WRITE_VTK", "0"))

REF_CHECK_BLOCK = int(os.environ.get("FAS_REF_CHECK_BLOCK", "6"))
REF_MIN_FINE_UPDATES = int(os.environ.get("FAS_REF_MIN_FINE_UPDATES", "60"))
REF_MAX_FINE_UPDATES = int(os.environ.get("FAS_REF_MAX_FINE_UPDATES", "240"))
REF_RESIDUAL_TARGET = float(os.environ.get("FAS_REF_RESIDUAL_TARGET", "1e-6"))
REF_CHANGE_REL_TOL = float(os.environ.get("FAS_REF_CHANGE_REL_TOL", "1e-8"))
REF_STABLE_CHECKS = int(os.environ.get("FAS_REF_STABLE_CHECKS", "2"))

# Pre-solve certificate thresholds.  These are fixed before observing T3
# numerical behaviour and concern only validity of the manufactured problem.
CERT_MIN_SIGNED_TRIANGLE_AREA_PHYSICAL = 1.0e-12
CERT_MAX_AREA_REL_MISMATCH = 1.0e-10
CERT_MAX_CIRCLE_AREA_REL_ERROR = 2.0e-5
CERT_MAX_PUSHFORWARD_MOMENT_ERROR = 5.0e-5
CERT_MIN_BRIDGE_MARGIN_UNIT = 0.05
CERT_RADIUS_TOL_PHYSICAL = 5.0e-12

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


def exact_target_radius_physical(r):
    delta = Constant(TARGET_OUTER_RADIUS_PHYSICAL**2 - TARGET_INNER_RADIUS_PHYSICAL**2)
    R2 = Constant(SOURCE_RADIUS_PHYSICAL**2)
    return sqrt(Constant(TARGET_INNER_RADIUS_PHYSICAL**2) + delta * r * r / R2)


def exact_map_physical(x):
    r = sqrt(x[0] * x[0] + x[1] * x[1])
    r_safe = max_value(r, Constant(1.0e-14))
    s = exact_target_radius_physical(r_safe)
    scale = s / r_safe
    # At x=0 this returns (0,0), an arbitrary representative on the
    # measure-zero singular set.  All integral diagnostics use r>0 quadrature.
    return as_vector([scale * x[0], scale * x[1]])


def exact_map_unit_from_source_unit(x_unit):
    return physical_to_unit_vector(exact_map_physical(unit_to_physical_vector(x_unit)))


def exact_radial_eigenvalues_physical(r):
    r_safe = max_value(r, Constant(1.0e-14))
    s = exact_target_radius_physical(r_safe)
    delta = Constant(TARGET_OUTER_RADIUS_PHYSICAL**2 - TARGET_INNER_RADIUS_PHYSICAL**2)
    R2 = Constant(SOURCE_RADIUS_PHYSICAL**2)
    lam_r = delta * r_safe / (R2 * s)
    lam_t = s / r_safe
    return lam_r, lam_t


def _concentric_disk_xy_physical(i, j, n):
    """Shirley--Chiu concentric square-to-disk map used successfully in T2."""
    a = -1.0 + 2.0 * float(i) / float(n)
    b = -1.0 + 2.0 * float(j) / float(n)
    if abs(a) < 1.0e-15 and abs(b) < 1.0e-15:
        return 0.0, 0.0
    if abs(a) > abs(b):
        r = a
        theta = 0.25 * math.pi * (b / a)
    else:
        r = b
        theta = 0.5 * math.pi - 0.25 * math.pi * (a / b)
    return (
        SOURCE_RADIUS_PHYSICAL * r * math.cos(theta),
        SOURCE_RADIUS_PHYSICAL * r * math.sin(theta),
    )


def _grid_square_triangles(i, j, node):
    n00 = node(i, j)
    n10 = node(i + 1, j)
    n01 = node(i, j + 1)
    n11 = node(i + 1, j + 1)
    if (i + j) % 2 == 0:
        return [(n00, n10, n11), (n00, n11, n01)]
    return [(n00, n10, n01), (n10, n11, n01)]


def _signed_area_physical(a, b, c):
    return 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _write_gmsh_v2(path, coords_unit, triangles):
    with open(path, "w", newline="\n") as handle:
        handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        handle.write(f"$Nodes\n{len(coords_unit)}\n")
        for node_id, (x, y) in enumerate(coords_unit, start=1):
            handle.write(f"{node_id} {x:.17e} {y:.17e} 0.0\n")
        handle.write("$EndNodes\n")
        handle.write(f"$Elements\n{len(triangles)}\n")
        for elem_id, tri in enumerate(triangles, start=1):
            a, b, c = tri
            handle.write(f"{elem_id} 2 2 1 1 {a} {b} {c}\n")
        handle.write("$EndElements\n")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _generate_t3_support_meshes(mesh_dir):
    """Generate the connected source disk and genuine annular target.

    The source uses the same N=256 concentric disk mesh as T2.  The target is
    an independent polar annulus.  With 64 radial layers and 1024 angular
    sectors both supports contain exactly 131072 triangles.  The 1024 angular
    sectors also match the 4*N polygonal boundary segments of the source, so
    the two equal-area polygonal supports agree in total area to roundoff.
    """
    n = SOURCE_GRID_N
    nr = TARGET_RADIAL_LAYERS
    nt = TARGET_THETA_CELLS
    if n <= 0 or n % 2 != 0:
        raise ValueError("T3_SOURCE_GRID_N must be a positive even integer")
    if nr <= 0 or nt < 32:
        raise ValueError("T3 target annulus discretisation must have positive radial layers and >=32 angular cells")
    os.makedirs(mesh_dir, exist_ok=True)
    source_path = os.path.join(mesh_dir, f"t3_source_disk_N{n}.msh")
    target_path = os.path.join(mesh_dir, f"t3_target_annulus_nr{nr}_nt{nt}.msh")

    source_phys = []
    source_id = {}
    for j in range(n + 1):
        for i in range(n + 1):
            source_id[(i, j)] = len(source_phys) + 1
            source_phys.append(_concentric_disk_xy_physical(i, j, n))

    def sid(i, j):
        return source_id[(i, j)]

    source_triangles = []
    source_area = 0.0
    source_min_signed_area = math.inf
    source_max_signed_area = 0.0
    for j in range(n):
        for i in range(n):
            tris = _grid_square_triangles(i, j, sid)
            source_triangles.extend(tris)
            for tri in tris:
                pts = [source_phys[k - 1] for k in tri]
                area = _signed_area_physical(*pts)
                source_min_signed_area = min(source_min_signed_area, area)
                source_max_signed_area = max(source_max_signed_area, area)
                source_area += area
    if source_min_signed_area <= 0.0:
        raise RuntimeError(f"T3 generated a non-positive source triangle: {source_min_signed_area}")

    source_unit = [
        (0.5 + x / BRIDGE_WIDTH_PHYSICAL, 0.5 + y / BRIDGE_WIDTH_PHYSICAL)
        for x, y in source_phys
    ]

    target_phys = []
    for k in range(nr + 1):
        rr = TARGET_INNER_RADIUS_PHYSICAL + (
            TARGET_OUTER_RADIUS_PHYSICAL - TARGET_INNER_RADIUS_PHYSICAL
        ) * float(k) / float(nr)
        for j in range(nt):
            theta = 2.0 * math.pi * float(j) / float(nt)
            target_phys.append((rr * math.cos(theta), rr * math.sin(theta)))

    def tid(k, j):
        return k * nt + (j % nt) + 1

    target_triangles = []
    target_area = 0.0
    target_min_signed_area = math.inf
    target_max_signed_area = 0.0
    for k in range(nr):
        for j in range(nt):
            n00 = tid(k, j)
            n10 = tid(k + 1, j)
            n01 = tid(k, j + 1)
            n11 = tid(k + 1, j + 1)
            for tri in [(n00, n10, n11), (n00, n11, n01)]:
                target_triangles.append(tri)
                pts = [target_phys[idx - 1] for idx in tri]
                area = _signed_area_physical(*pts)
                target_min_signed_area = min(target_min_signed_area, area)
                target_max_signed_area = max(target_max_signed_area, area)
                target_area += area
    if target_min_signed_area <= 0.0:
        raise RuntimeError(f"T3 generated a non-positive target triangle: {target_min_signed_area}")

    target_unit = [
        (0.5 + x / BRIDGE_WIDTH_PHYSICAL, 0.5 + y / BRIDGE_WIDTH_PHYSICAL)
        for x, y in target_phys
    ]

    _write_gmsh_v2(source_path, source_unit, source_triangles)
    _write_gmsh_v2(target_path, target_unit, target_triangles)

    source_boundary_radius_error = 0.0
    for j in range(n + 1):
        for i in range(n + 1):
            if i in (0, n) or j in (0, n):
                x, y = source_phys[source_id[(i, j)] - 1]
                source_boundary_radius_error = max(
                    source_boundary_radius_error,
                    abs(math.hypot(x, y) - SOURCE_RADIUS_PHYSICAL),
                )
    target_inner_radius_error = max(
        abs(math.hypot(*target_phys[j]) - TARGET_INNER_RADIUS_PHYSICAL)
        for j in range(nt)
    )
    outer_offset = nr * nt
    target_outer_radius_error = max(
        abs(math.hypot(*target_phys[outer_offset + j]) - TARGET_OUTER_RADIUS_PHYSICAL)
        for j in range(nt)
    )
    source_center = source_phys[source_id[(n // 2, n // 2)] - 1]
    circle_area = math.pi * SOURCE_RADIUS_PHYSICAL**2
    continuum_annulus_area = math.pi * (
        TARGET_OUTER_RADIUS_PHYSICAL**2 - TARGET_INNER_RADIUS_PHYSICAL**2
    )

    metadata = {
        "source_grid_n": n,
        "target_radial_layers": nr,
        "target_theta_cells": nt,
        "source_nodes_generated": len(source_unit),
        "target_nodes_generated": len(target_unit),
        "source_triangles_generated": len(source_triangles),
        "target_triangles_generated": len(target_triangles),
        "source_min_signed_triangle_area_physical": source_min_signed_area,
        "source_max_signed_triangle_area_physical": source_max_signed_area,
        "target_min_signed_triangle_area_physical": target_min_signed_area,
        "target_max_signed_triangle_area_physical": target_max_signed_area,
        "source_area_generated_physical": source_area,
        "target_area_generated_physical": target_area,
        "source_target_area_relative_mismatch_generated": abs(source_area - target_area) / max(source_area, 1e-300),
        "source_circle_area_relative_error": abs(source_area - circle_area) / circle_area,
        "target_annulus_area_relative_error": abs(target_area - continuum_annulus_area) / continuum_annulus_area,
        "source_boundary_radius_max_error_physical": source_boundary_radius_error,
        "target_inner_radius_max_error_physical": target_inner_radius_error,
        "target_outer_radius_max_error_physical": target_outer_radius_error,
        "target_inner_polygon_inradius_physical": TARGET_INNER_RADIUS_PHYSICAL * math.cos(math.pi / nt),
        "target_outer_polygon_inradius_physical": TARGET_OUTER_RADIUS_PHYSICAL * math.cos(math.pi / nt),
        "source_center_radius_physical": math.hypot(*source_center),
        "source_mesh_sha256": _sha256_file(source_path),
        "target_mesh_sha256": _sha256_file(target_path),
        "source_mesh_path": source_path,
        "target_mesh_path": target_path,
    }
    return source_path, target_path, metadata


def _t3_mesh_paths_and_metadata():
    result_dir = os.environ.get("RESULT_DIR", os.getcwd())
    mesh_dir = os.path.join(result_dir, "t3_support_meshes")
    payload = None
    if COMM.rank == 0:
        payload = _generate_t3_support_meshes(mesh_dir)
    payload = COMM.bcast(payload, root=0)
    COMM.Barrier()
    return payload


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


def _global_mesh_radius_extrema_physical(mesh):
    data = mesh.coordinates.dat.data_ro
    if data.size:
        xp = BRIDGE_WIDTH_PHYSICAL * (data[:, 0] - 0.5)
        yp = BRIDGE_WIDTH_PHYSICAL * (data[:, 1] - 0.5)
        rr = np.sqrt(xp * xp + yp * yp)
        local_min = float(rr.min())
        local_max = float(rr.max())
    else:
        local_min, local_max = math.inf, -math.inf
    return (
        float(COMM.allreduce(local_min, op=MPI.MIN)),
        float(COMM.allreduce(local_max, op=MPI.MAX)),
    )


def global_nonfinite_count(function):
    data = function.dat.data_ro
    local = int(np.count_nonzero(~np.isfinite(data)))
    return int(COMM.allreduce(local, op=MPI.SUM))


def build_support_spaces_and_measures(return_metadata=False):
    source_path, target_path, metadata = _t3_mesh_paths_and_metadata()
    mesh0 = Mesh(source_path)
    mesh1 = Mesh(target_path)

    V0 = FunctionSpace(mesh0, "CG", DISK_CG_DEGREE)
    V1 = FunctionSpace(mesh1, "CG", DISK_CG_DEGREE)

    mu0 = Function(V0, name="mu0").interpolate(1.0)
    mu1 = Function(V1, name="mu1").interpolate(1.0)
    source_area_unit = float(assemble(mu0 * dx(domain=mesh0)))
    target_area_unit = float(assemble(mu1 * dx(domain=mesh1)))
    if source_area_unit <= 0.0 or target_area_unit <= 0.0:
        raise RuntimeError(f"T3 invalid support area: source={source_area_unit}, target={target_area_unit}")
    mu0.assign(mu0 / source_area_unit)
    mu1.assign(mu1 / target_area_unit)

    source_cells = int(COMM.allreduce(mesh0.cell_set.size, op=MPI.SUM))
    target_cells = int(COMM.allreduce(mesh1.cell_set.size, op=MPI.SUM))
    metadata = dict(metadata)
    metadata.update({
        "source_global_cells_firedrake": source_cells,
        "target_global_cells_firedrake": target_cells,
        "source_area_firedrake_physical": source_area_unit * BRIDGE_WIDTH_PHYSICAL**2,
        "target_area_firedrake_physical": target_area_unit * BRIDGE_WIDTH_PHYSICAL**2,
        "source_target_area_relative_mismatch_firedrake": abs(source_area_unit - target_area_unit) / max(source_area_unit, 1e-300),
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


def _global_masked_max(values, mask, default=float("nan")):
    v = values.dat.data_ro.reshape(-1)
    m = mask.dat.data_ro.reshape(-1) > 0.5
    local = float(np.max(v[m])) if np.any(m) else -math.inf
    value = float(COMM.allreduce(local, op=MPI.MAX))
    return default if not math.isfinite(value) else value


def t3_annulus_map_audit(fine, q, result_dir, method):
    """Post-production T3 audit.

    T3's exact map is singular only at x=0.  Global L2 errors are meaningful,
    but global derivative and Linf errors are dominated by the centre layer.
    We therefore report both global map metrics and centre-excluded diagnostics.
    """
    V0, V1 = fine.V0, fine.V1
    x = SpatialCoordinate(V0.mesh())
    y = SpatialCoordinate(V1.mesh())
    x_phys = unit_to_physical_vector(x)
    y_phys = unit_to_physical_vector(y)
    width = BRIDGE_WIDTH_PHYSICAL
    ri = Constant(TARGET_INNER_RADIUS_PHYSICAL)
    ro = Constant(TARGET_OUTER_RADIUS_PHYSICAL)

    q_shift = global_dof_max(q)
    q_shifted = Function(V1, name=f"{method}_q_shifted_for_map")
    exp_q_shifted = Function(V1, name=f"{method}_exp_q_shifted_for_map")
    q_shifted.interpolate(q - Constant(q_shift))
    exp_q_shifted.interpolate(exp(q_shifted))

    y_radius_phys = sqrt(y_phys[0] ** 2 + y_phys[1] ** 2)
    y_radius_safe = max_value(y_radius_phys, Constant(1.0e-14))

    Hden = Function(V0, name=f"{method}_map_denominator")
    fine.heat_scaling_to_source(exp_q_shifted, Hden)

    weights = {}
    for name, expr in {
        "y0": y[0],
        "y1": y[1],
        "ysq": y[0] ** 2 + y[1] ** 2,
        "radius": y_radius_phys,
        "radius_sq": y_radius_phys ** 2,
        "dir0": y_phys[0] / y_radius_safe,
        "dir1": y_phys[1] / y_radius_safe,
    }.items():
        w = Function(V1, name=f"{method}_{name}_weight")
        w.interpolate(expr * exp_q_shifted)
        weights[name] = w

    heats = {}
    for name, w in weights.items():
        h = Function(V0, name=f"{method}_H_{name}")
        fine.heat_scaling_to_source(w, h)
        heats[name] = h

    T0 = Function(V0, name=f"{method}_T0_unit")
    T1 = Function(V0, name=f"{method}_T1_unit")
    T0.interpolate(heats["y0"] / max_value(Hden, TINY))
    T1.interpolate(heats["y1"] / max_value(Hden, TINY))
    T_field = as_vector([T0, T1])

    Tx_phys = Constant(width) * (T0 - Constant(0.5))
    Ty_phys = Constant(width) * (T1 - Constant(0.5))
    map_radius = sqrt(Tx_phys ** 2 + Ty_phys ** 2)

    source_radius = sqrt(x_phys[0] ** 2 + x_phys[1] ** 2)
    source_radius_safe = max_value(source_radius, Constant(1.0e-14))
    er0 = x_phys[0] / source_radius_safe
    er1 = x_phys[1] / source_radius_safe
    et0 = -er1
    et1 = er0

    exact_radius = exact_target_radius_physical(source_radius_safe)
    T_exact_phys = exact_map_physical(x_phys)
    ex = Tx_phys - T_exact_phys[0]
    ey = Ty_phys - T_exact_phys[1]
    err_sq_phys = ex * ex + ey * ey
    radial_error = conditional(gt(source_radius, Constant(1.0e-13)), ex * er0 + ey * er1, Constant(0.0))
    tangential_error = conditional(gt(source_radius, Constant(1.0e-13)), ex * et0 + ey * et1, Constant(0.0))

    map_l2_physical = math.sqrt(max(float(assemble(err_sq_phys * fine.mu0 * dx)), 0.0))
    radial_l2_physical = math.sqrt(max(float(assemble(radial_error ** 2 * fine.mu0 * dx)), 0.0))
    tangential_l2_physical = math.sqrt(max(float(assemble(tangential_error ** 2 * fine.mu0 * dx)), 0.0))
    radius_l2_physical = math.sqrt(max(float(assemble((map_radius - exact_radius) ** 2 * fine.mu0 * dx)), 0.0))

    # Conditional second moment and total spread.
    E_y_sq_unit = Function(V0, name=f"{method}_E_y_sq_unit")
    variance_raw = Function(V0, name=f"{method}_variance_raw_unit")
    variance = Function(V0, name=f"{method}_variance_unit")
    E_y_sq_unit.interpolate(heats["ysq"] / max_value(Hden, TINY))
    variance_raw.interpolate(E_y_sq_unit - (T0 ** 2 + T1 ** 2))
    variance.interpolate(max_value(variance_raw, 0.0))
    monge_defect_unit = float(assemble(variance * fine.mu0 * dx))
    monge_defect_physical = width * width * monge_defect_unit
    rms_conditional_spread_physical = width * math.sqrt(max(monge_defect_unit, 0.0))

    # Separate radial and angular diffuseness of the conditional target law.
    E_radius = Function(V0, name=f"{method}_E_target_radius")
    E_radius_sq = Function(V0, name=f"{method}_E_target_radius_sq")
    radial_variance = Function(V0, name=f"{method}_target_radius_variance")
    E_dir0 = Function(V0, name=f"{method}_E_target_dir0")
    E_dir1 = Function(V0, name=f"{method}_E_target_dir1")
    angular_resultant = Function(V0, name=f"{method}_angular_resultant")
    angular_coherence = Function(V0, name=f"{method}_angular_coherence")
    E_radius.interpolate(heats["radius"] / max_value(Hden, TINY))
    E_radius_sq.interpolate(heats["radius_sq"] / max_value(Hden, TINY))
    radial_variance.interpolate(max_value(E_radius_sq - E_radius ** 2, Constant(0.0)))
    E_dir0.interpolate(heats["dir0"] / max_value(Hden, TINY))
    E_dir1.interpolate(heats["dir1"] / max_value(Hden, TINY))
    angular_resultant.interpolate(min_value(Constant(1.0), sqrt(E_dir0 ** 2 + E_dir1 ** 2)))
    angular_coherence.interpolate(
        conditional(
            gt(source_radius, Constant(1.0e-13)),
            er0 * E_dir0 + er1 * E_dir1,
            Constant(0.0),
        )
    )
    rms_radial_conditional_spread_physical = math.sqrt(max(float(assemble(radial_variance * fine.mu0 * dx)), 0.0))
    mean_angular_resultant = float(assemble(angular_resultant * fine.mu0 * dx))
    mean_angular_coherence = float(assemble(angular_coherence * fine.mu0 * dx))
    mean_circular_variance = 1.0 - mean_angular_resultant

    # Barycentric topology diagnostics.  The entropic coupling itself remains
    # supported on the annulus; these quantities diagnose only its barycentre.
    hole_indicator = conditional(lt(map_radius, ri), 1.0, 0.0)
    annulus_indicator = conditional(ge(map_radius, ri), 1.0, 0.0) * conditional(le(map_radius, ro), 1.0, 0.0)
    outer_escape_indicator = conditional(gt(map_radius, ro), 1.0, 0.0)
    deep_hole_indicator = conditional(lt(map_radius, Constant(DEEP_HOLE_RADIUS_PHYSICAL)), 1.0, 0.0)
    hole_penetration = max_value(ri - map_radius, Constant(0.0))

    barycentric_hole_mass = float(assemble(hole_indicator * fine.mu0 * dx))
    barycentric_annulus_mass = float(assemble(annulus_indicator * fine.mu0 * dx))
    barycentric_outer_escape_mass = float(assemble(outer_escape_indicator * fine.mu0 * dx))
    barycentric_deep_hole_mass = float(assemble(deep_hole_indicator * fine.mu0 * dx))
    mean_hole_penetration_physical = float(assemble(hole_penetration * fine.mu0 * dx))
    rms_hole_penetration_physical = math.sqrt(max(float(assemble(hole_penetration ** 2 * fine.mu0 * dx)), 0.0))

    # Numerical Jacobian.  Exact det(DT*) = (ro^2-ri^2)/R^2 = 1 here for r>0,
    # while the exact radial/tangential eigenvalues become singular separately.
    DT_num = grad(T_field)
    det_space = FunctionSpace(V0.mesh(), "DG", 0)
    det_T = Function(det_space, name=f"{method}_det_DT").interpolate(det(DT_num))
    det_min = global_dof_min(det_T)
    det_max = global_dof_max(det_T)
    tangled_area = float(assemble(conditional(le(det(DT_num), 0.0), 1.0, 0.0) * dx(domain=V0.mesh())))

    lam_r, lam_t = exact_radial_eigenvalues_physical(source_radius_safe)
    J00 = lam_r * er0 * er0 + lam_t * et0 * et0
    J01 = lam_r * er0 * er1 + lam_t * et0 * et1
    J10 = lam_r * er1 * er0 + lam_t * et1 * et0
    J11 = lam_r * er1 * er1 + lam_t * et1 * et1
    jac_departure_sq = (
        (DT_num[0,0] - J00) ** 2 + (DT_num[0,1] - J01) ** 2
        + (DT_num[1,0] - J10) ** 2 + (DT_num[1,1] - J11) ** 2
    )
    far_cut = Constant(4.0 * CENTER_CHARACTERISTIC_SCALE_PHYSICAL)
    far_indicator = conditional(ge(source_radius, far_cut), 1.0, 0.0)
    far_mass = float(assemble(far_indicator * fine.mu0 * dx))
    jacobian_error_l2_away = math.sqrt(max(float(assemble(far_indicator * jac_departure_sq * fine.mu0 * dx)), 0.0) / max(far_mass, 1e-300))
    det_error_l2_away = math.sqrt(max(float(assemble(far_indicator * (det(DT_num) - Constant(RADIAL_AREA_FACTOR)) ** 2 * fine.mu0 * dx)), 0.0) / max(far_mass, 1e-300))

    near_cut = Constant(2.0 * CENTER_CHARACTERISTIC_SCALE_PHYSICAL)
    near_indicator = conditional(le(source_radius, near_cut), 1.0, 0.0)
    near_mass = float(assemble(near_indicator * fine.mu0 * dx))
    near_spread = width * math.sqrt(max(float(assemble(near_indicator * variance * fine.mu0 * dx)), 0.0) / max(near_mass, 1e-300))
    near_radial_spread = math.sqrt(max(float(assemble(near_indicator * radial_variance * fine.mu0 * dx)), 0.0) / max(near_mass, 1e-300))
    near_angular_resultant = float(assemble(near_indicator * angular_resultant * fine.mu0 * dx)) / max(near_mass, 1e-300)
    near_angular_coherence = float(assemble(near_indicator * angular_coherence * fine.mu0 * dx)) / max(near_mass, 1e-300)

    # Cell-centre metrics avoid imposing a single exact direction at x=0.
    source_r_cell = Function(det_space, name=f"{method}_source_radius_cell").interpolate(source_radius)
    map_r_cell = Function(det_space, name=f"{method}_map_radius_cell").interpolate(map_radius)
    exact_r_cell = Function(det_space, name=f"{method}_exact_radius_cell").interpolate(exact_radius)
    err_cell = Function(det_space, name=f"{method}_map_error_cell_physical").interpolate(sqrt(err_sq_phys))
    ratio_cell = Function(det_space, name=f"{method}_radial_recovery_ratio_cell").interpolate(map_radius / max_value(exact_radius, Constant(1.0e-14)))

    rdat = source_r_cell.dat.data_ro.reshape(-1)
    mrdat = map_r_cell.dat.data_ro.reshape(-1)
    exdat = err_cell.dat.data_ro.reshape(-1)
    qdat = ratio_cell.dat.data_ro.reshape(-1)
    local_hole_r = float(np.max(rdat[mrdat < TARGET_INNER_RADIUS_PHYSICAL])) if np.any(mrdat < TARGET_INNER_RADIUS_PHYSICAL) else -math.inf
    local_deep_hole_r = float(np.max(rdat[mrdat < DEEP_HOLE_RADIUS_PHYSICAL])) if np.any(mrdat < DEEP_HOLE_RADIUS_PHYSICAL) else -math.inf
    local_r90 = float(np.max(rdat[qdat < 0.90])) if np.any(qdat < 0.90) else -math.inf
    local_r95 = float(np.max(rdat[qdat < 0.95])) if np.any(qdat < 0.95) else -math.inf
    max_source_radius_mapped_into_hole = float(COMM.allreduce(local_hole_r, op=MPI.MAX))
    max_source_radius_mapped_deep_hole = float(COMM.allreduce(local_deep_hole_r, op=MPI.MAX))
    radial_recovery_radius_90 = float(COMM.allreduce(local_r90, op=MPI.MAX))
    radial_recovery_radius_95 = float(COMM.allreduce(local_r95, op=MPI.MAX))
    raw_linf = float(COMM.allreduce(float(np.max(exdat)) if exdat.size else -math.inf, op=MPI.MAX))
    away2_mask = Function(det_space, name=f"{method}_away2_mask").interpolate(conditional(ge(source_radius, Constant(2.0 * CENTER_CHARACTERISTIC_SCALE_PHYSICAL)), 1.0, 0.0))
    away4_mask = Function(det_space, name=f"{method}_away4_mask").interpolate(conditional(ge(source_radius, Constant(4.0 * CENTER_CHARACTERISTIC_SCALE_PHYSICAL)), 1.0, 0.0))
    linf_away_2scale = _global_masked_max(err_cell, away2_mask)
    linf_away_4scale = _global_masked_max(err_cell, away4_mask)

    centre_rows = []
    for a, b in zip(CENTER_BINS_PHYSICAL[:-1], CENTER_BINS_PHYSICAL[1:]):
        ind = conditional(ge(source_radius, Constant(a)), 1.0, 0.0) * conditional(lt(source_radius, Constant(b)), 1.0, 0.0)
        mass = float(assemble(ind * fine.mu0 * dx))
        if mass > 0.0:
            row = {
                "source_radius_lower_physical": a,
                "source_radius_upper_physical": b,
                "source_mass": mass,
                "map_rms_error_physical": math.sqrt(max(float(assemble(ind * err_sq_phys * fine.mu0 * dx)), 0.0) / mass),
                "radial_rms_error_physical": math.sqrt(max(float(assemble(ind * radial_error**2 * fine.mu0 * dx)), 0.0) / mass),
                "tangential_rms_error_physical": math.sqrt(max(float(assemble(ind * tangential_error**2 * fine.mu0 * dx)), 0.0) / mass),
                "mean_map_radius_physical": float(assemble(ind * map_radius * fine.mu0 * dx)) / mass,
                "mean_exact_radius_physical": float(assemble(ind * exact_radius * fine.mu0 * dx)) / mass,
                "hole_fraction": float(assemble(ind * hole_indicator * fine.mu0 * dx)) / mass,
                "conditional_spread_rms_physical": width * math.sqrt(max(float(assemble(ind * variance * fine.mu0 * dx)), 0.0) / mass),
                "radial_conditional_spread_rms_physical": math.sqrt(max(float(assemble(ind * radial_variance * fine.mu0 * dx)), 0.0) / mass),
                "mean_angular_resultant": float(assemble(ind * angular_resultant * fine.mu0 * dx)) / mass,
                "mean_angular_coherence": float(assemble(ind * angular_coherence * fine.mu0 * dx)) / mass,
            }
        else:
            row = {
                "source_radius_lower_physical": a,
                "source_radius_upper_physical": b,
                "source_mass": mass,
            }
        centre_rows.append(row)
    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_{method}_centre_bins.csv"), centre_rows)

    denom_min = global_dof_min(Hden)
    denom_max = global_dof_max(Hden)
    denom_nonfinite = global_nonfinite_count(Hden)
    penetration_cell = Function(det_space, name=f"{method}_hole_penetration_cell").interpolate(hole_penetration)
    max_hole_penetration = global_dof_max(penetration_cell)

    row = {
        "map_l2_physical": map_l2_physical,
        "map_linf_cell_physical": raw_linf,
        "map_linf_away_2centre_scales_physical": linf_away_2scale,
        "map_linf_away_4centre_scales_physical": linf_away_4scale,
        "radial_l2_physical": radial_l2_physical,
        "tangential_l2_physical": tangential_l2_physical,
        "radius_l2_physical": radius_l2_physical,
        "exact_det_a.e.": RADIAL_AREA_FACTOR,
        "det_min": det_min,
        "det_max": det_max,
        "tangled_area": tangled_area,
        "jacobian_frobenius_l2_away_4centre_scales": jacobian_error_l2_away,
        "det_l2_error_away_4centre_scales": det_error_l2_away,
        "barycentric_hole_mass": barycentric_hole_mass,
        "barycentric_deep_hole_mass": barycentric_deep_hole_mass,
        "barycentric_annulus_mass": barycentric_annulus_mass,
        "barycentric_outer_escape_mass": barycentric_outer_escape_mass,
        "mean_hole_penetration_physical": mean_hole_penetration_physical,
        "rms_hole_penetration_physical": rms_hole_penetration_physical,
        "max_hole_penetration_cell_physical": max_hole_penetration,
        "max_source_radius_mapped_into_hole_physical": max_source_radius_mapped_into_hole if math.isfinite(max_source_radius_mapped_into_hole) else 0.0,
        "max_source_radius_mapped_deep_hole_physical": max_source_radius_mapped_deep_hole if math.isfinite(max_source_radius_mapped_deep_hole) else 0.0,
        "radial_recovery_radius_90_physical": radial_recovery_radius_90 if math.isfinite(radial_recovery_radius_90) else 0.0,
        "radial_recovery_radius_95_physical": radial_recovery_radius_95 if math.isfinite(radial_recovery_radius_95) else 0.0,
        "monge_defect_unit": monge_defect_unit,
        "monge_defect_physical": monge_defect_physical,
        "rms_conditional_spread_physical": rms_conditional_spread_physical,
        "rms_radial_conditional_spread_physical": rms_radial_conditional_spread_physical,
        "mean_angular_resultant": mean_angular_resultant,
        "mean_angular_coherence": mean_angular_coherence,
        "mean_circular_variance": mean_circular_variance,
        "centre_characteristic_scale_physical": CENTER_CHARACTERISTIC_SCALE_PHYSICAL,
        "near_centre_mass_2scale": near_mass,
        "near_centre_conditional_spread_physical": near_spread,
        "near_centre_radial_conditional_spread_physical": near_radial_spread,
        "near_centre_mean_angular_resultant": near_angular_resultant,
        "near_centre_mean_angular_coherence": near_angular_coherence,
        "map_denominator_min": denom_min,
        "map_denominator_max": denom_max,
        "map_denominator_nonfinite_count": denom_nonfinite,
        "target_inner_radius_physical": TARGET_INNER_RADIUS_PHYSICAL,
        "target_outer_radius_physical": TARGET_OUTER_RADIUS_PHYSICAL,
        "deep_hole_radius_physical": DEEP_HOLE_RADIUS_PHYSICAL,
    }
    return row


def _pushforward_moment_rows(V0, V1, mu0, mu1):
    x = SpatialCoordinate(V0.mesh())
    y = SpatialCoordinate(V1.mesh())
    xp = unit_to_physical_vector(x)
    yp = unit_to_physical_vector(y)
    T = exact_map_physical(xp)
    tests = [
        ("one", lambda z: Constant(1.0)),
        ("y0", lambda z: z[0]),
        ("y1", lambda z: z[1]),
        ("y0_sq", lambda z: z[0] ** 2),
        ("y0y1", lambda z: z[0] * z[1]),
        ("y1_sq", lambda z: z[1] ** 2),
        ("radius_sq", lambda z: z[0] ** 2 + z[1] ** 2),
        ("cos_2pi_y0", lambda z: cos(Constant(2.0 * math.pi) * z[0])),
        ("cos_2pi_y1", lambda z: cos(Constant(2.0 * math.pi) * z[1])),
    ]
    rows = []
    for name, f in tests:
        source_value = float(assemble(f(T) * mu0 * dx(domain=V0.mesh())))
        target_value = float(assemble(f(yp) * mu1 * dx(domain=V1.mesh())))
        rows.append({
            "moment": name,
            "source_pushforward_value": source_value,
            "target_value": target_value,
            "abs_error": abs(source_value - target_value),
        })
    return rows


def benchmark_certificate(V0, V1, mu0, mu1, metadata, result_dir):
    source_mesh, target_mesh = V0.mesh(), V1.mesh()
    source_mass = float(assemble(mu0 * dx(domain=source_mesh)))
    target_mass = float(assemble(mu1 * dx(domain=target_mesh)))
    source_bbox = _global_mesh_bbox(source_mesh)
    target_bbox = _global_mesh_bbox(target_mesh)
    target_rmin, target_rmax = _global_mesh_radius_extrema_physical(target_mesh)
    source_rmin, source_rmax = _global_mesh_radius_extrema_physical(source_mesh)
    bridge_margin = min(
        source_bbox[0], 1.0-source_bbox[1], source_bbox[2], 1.0-source_bbox[3],
        target_bbox[0], 1.0-target_bbox[1], target_bbox[2], 1.0-target_bbox[3],
    )
    moment_rows = _pushforward_moment_rows(V0, V1, mu0, mu1)
    max_moment_error = max(row["abs_error"] for row in moment_rows)

    row = {
        "source_radius_physical": SOURCE_RADIUS_PHYSICAL,
        "target_inner_radius_physical": TARGET_INNER_RADIUS_PHYSICAL,
        "target_outer_radius_physical": TARGET_OUTER_RADIUS_PHYSICAL,
        "continuum_area_factor": RADIAL_AREA_FACTOR,
        "centre_characteristic_scale_physical": CENTER_CHARACTERISTIC_SCALE_PHYSICAL,
        "source_mass": source_mass,
        "target_mass": target_mass,
        "source_density_min": global_dof_min(mu0),
        "target_density_min": global_dof_min(mu1),
        "source_density_nonfinite_count": global_nonfinite_count(mu0),
        "target_density_nonfinite_count": global_nonfinite_count(mu1),
        "source_radius_min_nodes_physical": source_rmin,
        "source_radius_max_nodes_physical": source_rmax,
        "target_radius_min_nodes_physical": target_rmin,
        "target_radius_max_nodes_physical": target_rmax,
        "bridge_margin_unit": bridge_margin,
        "max_pushforward_moment_abs_error": max_moment_error,
    }
    row.update(metadata)

    expected_source_cells = 2 * SOURCE_GRID_N * SOURCE_GRID_N
    expected_target_cells = 2 * TARGET_RADIAL_LAYERS * TARGET_THETA_CELLS
    checks = {
        "continuum_equal_area": abs(RADIAL_AREA_FACTOR - 1.0) <= 1.0e-14,
        "source_triangle_orientation_valid": metadata["source_min_signed_triangle_area_physical"] >= CERT_MIN_SIGNED_TRIANGLE_AREA_PHYSICAL,
        "target_triangle_orientation_valid": metadata["target_min_signed_triangle_area_physical"] >= CERT_MIN_SIGNED_TRIANGLE_AREA_PHYSICAL,
        "source_target_generated_area_match": metadata["source_target_area_relative_mismatch_generated"] <= CERT_MAX_AREA_REL_MISMATCH,
        "source_target_firedrake_area_match": metadata["source_target_area_relative_mismatch_firedrake"] <= CERT_MAX_AREA_REL_MISMATCH,
        "source_circle_geometry_resolution": metadata["source_circle_area_relative_error"] <= CERT_MAX_CIRCLE_AREA_REL_ERROR,
        "target_annulus_geometry_resolution": metadata["target_annulus_area_relative_error"] <= CERT_MAX_CIRCLE_AREA_REL_ERROR,
        "source_cell_count_expected": metadata["source_global_cells_firedrake"] == expected_source_cells,
        "target_cell_count_expected": metadata["target_global_cells_firedrake"] == expected_target_cells,
        "source_target_cell_counts_equal": expected_source_cells == expected_target_cells,
        "source_center_present": metadata["source_center_radius_physical"] <= CERT_RADIUS_TOL_PHYSICAL,
        "source_boundary_radius": metadata["source_boundary_radius_max_error_physical"] <= CERT_RADIUS_TOL_PHYSICAL,
        "target_inner_radius": metadata["target_inner_radius_max_error_physical"] <= CERT_RADIUS_TOL_PHYSICAL,
        "target_outer_radius": metadata["target_outer_radius_max_error_physical"] <= CERT_RADIUS_TOL_PHYSICAL,
        "target_hole_genuine": metadata["target_inner_polygon_inradius_physical"] >= TARGET_INNER_RADIUS_PHYSICAL - 2.0e-6,
        "source_density_positive": row["source_density_min"] > 0.0,
        "target_density_positive": row["target_density_min"] > 0.0,
        "densities_finite": row["source_density_nonfinite_count"] == 0 and row["target_density_nonfinite_count"] == 0,
        "unit_masses": abs(source_mass - 1.0) <= 1.0e-12 and abs(target_mass - 1.0) <= 1.0e-12,
        "pushforward_moments": max_moment_error <= CERT_MAX_PUSHFORWARD_MOMENT_ERROR,
        "bridge_margin": bridge_margin >= CERT_MIN_BRIDGE_MARGIN_UNIT,
    }
    row.update({f"check_{k}": int(v) for k, v in checks.items()})
    row["certificate_pass"] = int(all(checks.values()))
    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_benchmark_certificate.csv"), [row])
    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_pushforward_moments.csv"), moment_rows)
    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_mesh_metadata.csv"), [metadata])
    if COMM.rank == 0:
        print("T3_BENCHMARK_CERTIFICATE", row)
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"T3 benchmark certificate failed: {failed}")
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
    row.update(t3_annulus_map_audit(fine, q, result_dir, method))
    for name in ["log_marginal_p", "log_marginal_q", "marginal_p", "marginal_q", "logHq_on_0", "logHp_on_1", "Hq_shifted_on_0", "Hp_shifted_on_1"]:
        field = getattr(fine, name, None)
        if field is not None:
            row.update(fine.field_stats(field, name))
    row.update(fine.heat_stats(fine.Hq_shifted_on_0, "Hq_shifted_heat"))
    row.update(fine.heat_stats(fine.Hp_shifted_on_1, "Hp_shifted_heat"))
    row["audit_seconds"] = COMM.allreduce(time.perf_counter() - audit_t0, op=MPI.MAX)
    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_{method}_science_audit.csv"), [row])
    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_{method}_audit_heat_counts.csv"), fine.counter_rows())
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
            raise RuntimeError(f"Three-level FAS requested but B{bottom.bridge_n} performed zero heat actions.")


def build_problem(coarse_tau_ratio=1.0):
    V0, V1, mu0, mu1 = build_support_spaces_and_measures()
    bridge_hierarchy = build_bridge_hierarchy()
    levels = []
    coarse = None
    for level, mesh in enumerate(bridge_hierarchy):
        tau_physical = TAU_FINE_PHYSICAL * coarse_tau_ratio if level == 0 else TAU_FINE_PHYSICAL
        current = FASLevel(
            f"level{level}_B{BRIDGE_LEVELS[level]}", level, mesh, tau_physical,
            V0, V1, mu0, mu1, coarse=coarse,
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
        f"level{fine_index}_B{BRIDGE_LEVELS[fine_index]}", fine_index,
        bridge_meshes[fine_index], TAU_FINE_PHYSICAL, V0, V1, mu0, mu1, coarse=None,
    )
    return fine, None, V0, V1


def gauge_aligned_distance(level, p, q, p_ref, q_ref):
    dp = Function(p.function_space(), name="t3_dp")
    dq = Function(q.function_space(), name="t3_dq")
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
    return math.sqrt(max(float(assemble(p * p * level.mu0 * dx)) + float(assemble(q * q * level.mu1 * dx)), 0.0))


def run_t3_reference_fas_validation(result_dir, fine, V0, V1, p, q, b_p, b_q, total_t0):
    if len(hierarchy_level_objects(fine)) != 3:
        raise RuntimeError("T3 production reference/FAS validation requires exactly three levels")

    # Phase A: independent fine-only reference on B920.
    p.assign(0.0); q.assign(0.0); b_p.assign(0.0); b_q.assign(0.0)
    fine.reset_counters()
    previous_p = Function(V0, name="t3_ref_previous_p").assign(p)
    previous_q = Function(V1, name="t3_ref_previous_q").assign(q)
    previous_change_abs = float("nan")
    stable_checks = 0
    reference_certified = False
    ref_rows = []
    ref_t0 = time.perf_counter()
    updates = 0

    while updates < REF_MAX_FINE_UPDATES:
        block = min(REF_CHECK_BLOCK, REF_MAX_FINE_UPDATES - updates)
        fine.smoother(p, q, b_p, b_q, block, f"t3_reference_block_{updates+block}")
        fine.gauge_fix(p, q)
        updates += block
        residual_p, residual_q = fine.residual_norms(p, q, b_p, b_q)
        change_p, change_q, change_abs, change_shift = gauge_aligned_distance(fine, p, q, previous_p, previous_q)
        norm_now = reference_norm(fine, p, q)
        change_rel = change_abs / max(norm_now, 1.0e-300)
        change_contraction = change_abs / previous_change_abs if math.isfinite(previous_change_abs) and previous_change_abs > 0.0 else float("nan")
        tail_est = float("nan")
        if math.isfinite(change_contraction) and 0.0 <= change_contraction < 1.0:
            tail_est = change_contraction / max(1.0 - change_contraction, 1.0e-30) * change_abs
        criteria = updates >= REF_MIN_FINE_UPDATES and residual_p <= REF_RESIDUAL_TARGET and change_rel <= REF_CHANGE_REL_TOL
        stable_checks = stable_checks + 1 if criteria else 0
        reference_certified = stable_checks >= REF_STABLE_CHECKS
        ref_rows.append({
            "fine_bridge_n": fine.bridge_n,
            "updates": updates,
            "residual_p": residual_p,
            "residual_q": residual_q,
            "change_p": change_p,
            "change_q": change_q,
            "change_abs": change_abs,
            "change_rel": change_rel,
            "change_contraction": change_contraction,
            "estimated_remaining_tail": tail_est,
            "change_gauge_shift": change_shift,
            "stable_checks": stable_checks,
            "reference_certified": int(reference_certified),
        })
        previous_p.assign(p); previous_q.assign(q); previous_change_abs = change_abs
        if COMM.rank == 0:
            print("T3_REFERENCE_ROW", ref_rows[-1])
        if reference_certified:
            break

    reference_seconds = COMM.allreduce(time.perf_counter() - ref_t0, op=MPI.MAX)
    if COMM.rank == 0 and not reference_certified:
        print("T3_REFERENCE_WARNING reference not formally certified; FAS stress result remains valid and reference comparison is provisional.")
    p_ref = Function(V0, name="t3_p_ref").assign(p)
    q_ref = Function(V1, name="t3_q_ref").assign(q)
    reference_heat_rows = fine.counter_rows()
    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_reference_trace.csv"), ref_rows)
    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_reference_heat_counts.csv"), reference_heat_rows)

    # Phase B: the single frozen cold FAS production attempt.
    p.assign(0.0); q.assign(0.0); b_p.assign(0.0); b_q.assign(0.0)
    fine.reset_counters()
    initial_residual = fine.p_residual_norm(p, q, b_p)
    ep0, eq0, error0, shift0 = gauge_aligned_distance(fine, p, q, p_ref, q_ref)
    trajectory = [{
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
    fine.reset_counters()  # exclude cycle-zero diagnostics from production counts
    production_t0 = time.perf_counter()
    previous_residual, previous_error = initial_residual, error0
    for cycle in range(1, FAS_CYCLES + 1):
        fine.fas_cycle_perf(p, q, b_p, b_q)
        residual_p = fine.p_residual_norm(p, q, b_p)
        ep, eq, error, shift = gauge_aligned_distance(fine, p, q, p_ref, q_ref)
        now = time.perf_counter()
        trajectory.append({
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
            "iteration_seconds": COMM.allreduce(now - production_t0, op=MPI.MAX),
        })
        previous_residual, previous_error = residual_p, error
        if COMM.rank == 0:
            print("T3_REFERENCE_FAS_ROW", trajectory[-1])
        if residual_p <= TARGET_RESIDUAL:
            break

    cycles = len(trajectory) - 1
    assert_requested_hierarchy_is_active(fine)
    levels = hierarchy_level_objects(fine)
    observed = [level.heat_source_calls + level.heat_target_calls for level in levels]
    expected = [16 * cycles, 10 * cycles, 9 * cycles]
    if observed != expected:
        raise RuntimeError(f"T3 production heat-count failure: observed={observed}, expected={expected}")
    production_counter_rows = fine.counter_rows()
    production_seconds = COMM.allreduce(time.perf_counter() - production_t0, op=MPI.MAX)
    final_error = trajectory[-1]["combined_error_to_reference"]
    last_ref = ref_rows[-1]
    tail_est = last_ref.get("estimated_remaining_tail", float("nan"))
    summary = {
        "cycles": cycles,
        "reached_target": trajectory[-1]["reached_target"],
        "initial_residual_p": initial_residual,
        "final_residual_p": trajectory[-1]["residual_p"],
        "initial_discrete_reference_error": error0,
        "final_discrete_reference_error": final_error,
        "geomean_residual_contraction": (trajectory[-1]["residual_p"] / initial_residual) ** (1.0 / max(cycles, 1)),
        "geomean_discrete_error_contraction": (final_error / error0) ** (1.0 / max(cycles, 1)),
        "reference_updates": updates,
        "reference_residual_p": last_ref["residual_p"],
        "reference_residual_q": last_ref["residual_q"],
        "reference_change_abs": last_ref["change_abs"],
        "reference_change_rel": last_ref["change_rel"],
        "reference_change_contraction": last_ref["change_contraction"],
        "reference_estimated_remaining_tail": tail_est,
        "reference_certified": int(reference_certified),
        "reference_uncertainty_over_final_error": last_ref["change_abs"] / max(final_error, 1e-300),
        "reference_tail_over_final_error": tail_est / max(final_error, 1e-300) if math.isfinite(tail_est) else float("nan"),
        "reference_seconds": reference_seconds,
        "production_iteration_seconds": production_seconds,
        "job_seconds_before_final_audit": COMM.allreduce(time.perf_counter() - total_t0, op=MPI.MAX),
        "observed_heat_actions": str(observed),
        "expected_heat_actions": str(expected),
        "heat_count_match": 1,
    }
    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_fas_discrete_error_trajectory.csv"), trajectory)
    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_production_heat_counts.csv"), production_counter_rows)
    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_reference_fas_summary.csv"), [summary])

    perf_row = {
        "cycle": cycles,
        "residual_p": trajectory[-1]["residual_p"],
        "reached_target": trajectory[-1]["reached_target"],
        "iteration_seconds": production_seconds,
        "total_seconds": production_seconds,
    }
    science_row = final_science_audit("reference_fas_to_tol", fine, p, q, b_p, b_q, perf_row, result_dir)
    summary.update({
        "map_l2_physical": science_row.get("map_l2_physical", ""),
        "barycentric_hole_mass": science_row.get("barycentric_hole_mass", ""),
        "max_source_radius_mapped_into_hole_physical": science_row.get("max_source_radius_mapped_into_hole_physical", ""),
        "tangled_area": science_row.get("tangled_area", ""),
        "det_min": science_row.get("det_min", ""),
    })
    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_reference_fas_summary_with_map.csv"), [summary])
    if COMM.rank == 0:
        print("T3_REFERENCE_FAS_FINAL", summary)
        print(f"RESULT_DIR={result_dir}")


def run_fas_to_tol(result_dir, fine, p, q, b_p, b_q, total_t0):
    # Used only for the one-cycle plumbing smoke.  Production uses the
    # reference_fas_to_tol mode below.
    fine.reset_counters()
    rows = []
    iteration_t0 = time.perf_counter()
    for cycle in range(1, FAS_CYCLES + 1):
        fine.fas_cycle_perf(p, q, b_p, b_q)
        residual_p = fine.p_residual_norm(p, q, b_p)
        now = time.perf_counter()
        rows.append({
            "mode": "fas_to_tol",
            "cycle": cycle,
            "residual_p": residual_p,
            "target_residual": TARGET_RESIDUAL,
            "reached_target": int(residual_p <= TARGET_RESIDUAL),
            "iteration_seconds": COMM.allreduce(now - iteration_t0, op=MPI.MAX),
            "total_seconds": COMM.allreduce(now - total_t0, op=MPI.MAX),
        })
        if COMM.rank == 0:
            print("T3_FAS_TO_TOL_ROW", rows[-1])
        if residual_p <= TARGET_RESIDUAL:
            break
    assert_requested_hierarchy_is_active(fine)
    levels = hierarchy_level_objects(fine)
    observed = [level.heat_source_calls + level.heat_target_calls for level in levels]
    cycles = len(rows)
    expected = [16 * cycles, 9 * cycles] if len(levels) == 2 else [16 * cycles, 10 * cycles, 9 * cycles]
    if observed != expected:
        raise RuntimeError(f"T3 smoke heat-count failure: observed={observed}, expected={expected}")
    rows[-1].update({"observed_heat_actions": str(observed), "expected_heat_actions": str(expected), "heat_count_match": 1})
    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_fas_to_tol_trace.csv"), rows)
    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_heat_counts.csv"), fine.counter_rows())
    final_science_audit("fas_to_tol", fine, p, q, b_p, b_q, rows[-1], result_dir)
    if COMM.rank == 0:
        print("T3_FAS_TO_TOL_FINAL", rows[-1])
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

    fine, coarse, V0, V1 = build_problem(COARSE_TAU_RATIO)
    hierarchy = hierarchy_rows(fine)
    write_rows(os.path.join(result_dir, f"{EXPERIMENT_NAME}_hierarchy.csv"), hierarchy)
    p = Function(V0, name="p").assign(0.0)
    q = Function(V1, name="q").assign(0.0)
    b_p = Function(V0, name="b_p_finest").assign(0.0)
    b_q = Function(V1, name="b_q_finest").assign(0.0)

    if COMM.rank == 0:
        print(f"EXPERIMENT_NAME={EXPERIMENT_NAME}")
        print(
            "T3_CONFIGURATION",
            f"source_R={SOURCE_RADIUS_PHYSICAL}",
            f"target_ri={TARGET_INNER_RADIUS_PHYSICAL}",
            f"target_ro={TARGET_OUTER_RADIUS_PHYSICAL}",
            f"area_factor={RADIAL_AREA_FACTOR}",
            f"centre_scale={CENTER_CHARACTERISTIC_SCALE_PHYSICAL}",
            f"source_N={SOURCE_GRID_N}",
            f"target_nr={TARGET_RADIAL_LAYERS}",
            f"target_nt={TARGET_THETA_CELLS}",
        )
        print(f"FAS_CYCLES={FAS_CYCLES} PRE={PRE_SMOOTH} COARSE={COARSE_SMOOTH} POST={POST_SMOOTH} COARSE_TAU_RATIO={COARSE_TAU_RATIO}")
        for row in hierarchy:
            print("FAS_HIERARCHY", row)

    if RUN_MODE == "fas_to_tol":
        run_fas_to_tol(result_dir, fine, p, q, b_p, b_q, total_t0)
        return
    if RUN_MODE == "reference_fas_to_tol":
        run_t3_reference_fas_validation(result_dir, fine, V0, V1, p, q, b_p, b_q, total_t0)
        return
    raise ValueError(f"Unknown T3 FAS_RUN_MODE={RUN_MODE!r}")


if __name__ == "__main__":
    main()
