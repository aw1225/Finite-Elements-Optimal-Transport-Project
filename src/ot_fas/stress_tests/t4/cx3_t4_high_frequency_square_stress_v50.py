from firedrake import *
from mpi4py import MPI
import csv
import hashlib
import math
import os
import time
import numpy as np

EXPERIMENT_NAME = "run_t4_high_frequency_square_stress_v50"

# T4 spectral-resolution stress benchmark.
#
# The source and target supports are the same square [-1/2,1/2]^2.  The exact
# Brenier potential contains an explicitly manufactured oscillatory component.
# The target density is uniform and the source density is det(DT_*), so T_*
# is the exact quadratic-cost transport.  This isolates fine-scale structure
# without adding difficult support geometry.
#
# The q-family is a frequency-raised variant of the classical explicit
# square-to-square benchmark used by Lindsey & Rubinstein (SIAM J. Math. Anal.
# 49 (2017), Section 7.2).  The original uses mode m=4 and beta=4.  T4 freezes
# m=6 and beta=8 to create a stronger but still uniformly convex benchmark.

SUPPORT_GRID_N = int(os.environ.get("T4_SUPPORT_GRID_N", "256"))
SUPPORT_HALF_WIDTH_PHYSICAL = float(os.environ.get("T4_SUPPORT_HALF_WIDTH_PHYSICAL", "0.5"))
OSC_MODE = int(os.environ.get("T4_OSC_MODE", "6"))
OSC_BETA = float(os.environ.get("T4_OSC_BETA", "8.0"))
OSC_OMEGA = 2.0 * math.pi * OSC_MODE
OSC_WAVELENGTH_PHYSICAL = (2.0 * SUPPORT_HALF_WIDTH_PHYSICAL) / OSC_MODE

DISK_CG_DEGREE = 2  # historical name retained because FASLevel is frozen
BRIDGE_CG_DEGREE = 2
N_SUB = 16

BRIDGE_COARSE_N = int(os.environ.get("FAS_BRIDGE_COARSE_N", "230"))
N_BRIDGE_REFINEMENTS = int(os.environ.get("FAS_N_BRIDGE_REFINEMENTS", "2"))
BRIDGE_LEVELS = [BRIDGE_COARSE_N * 2**level for level in range(N_BRIDGE_REFINEMENTS + 1)]
BRIDGE_WIDTH_PHYSICAL = 4.0
TAU_FINE_PHYSICAL = float(os.environ.get("FAS_TAU_FINE_PHYSICAL", "2.373046875e-3"))
EPSILON_PHYSICAL = 2.0 * TAU_FINE_PHYSICAL

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

# Pre-solve benchmark-certificate thresholds.  They certify only that the
# manufactured T4 problem has the intended mathematical properties; they are
# not tuned to the FAS result.
CERT_MIN_SIGNED_TRIANGLE_AREA_PHYSICAL = 1.0e-12
CERT_MAX_AREA_REL_MISMATCH = 1.0e-12
CERT_MIN_EXACT_JACOBIAN_EIG = 0.45
CERT_MIN_EXACT_DET = 0.20
CERT_MAX_MAP_BOX_OVERSHOOT_PHYSICAL = 2.0e-6
CERT_MAX_PUSHFORWARD_MOMENT_ERROR = 5.0e-4
CERT_MIN_BRIDGE_MARGIN_UNIT = 0.05
CERT_MIN_SOURCE_DENSITY_CONTRAST = 5.0
CERT_MAX_COARSE_CELLS_PER_WAVELENGTH = 10.0
CERT_MIN_FINE_CELLS_PER_WAVELENGTH = 35.0

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


def osc_q_physical(z):
    """Generalized Lindsey--Rubinstein q_m on [-1/2,1/2]."""
    w = Constant(OSC_OMEGA)
    return (
        (
            -z * z / w
            + Constant(2.0 / OSC_OMEGA**3)
            + Constant(1.0 / (4.0 * OSC_OMEGA))
        ) * cos(w * z)
        + Constant(2.0 / OSC_OMEGA**2) * z * sin(w * z)
    )


def osc_q_prime_physical(z):
    # Exact derivative of osc_q_physical:
    # q'(z) = (z^2 - 1/4) sin(omega z).
    return (z * z - Constant(0.25)) * sin(Constant(OSC_OMEGA) * z)


def osc_q_second_physical(z):
    # Exact second derivative:
    # q''(z) = omega(z^2 - 1/4) cos(omega z) + 2z sin(omega z).
    return (
        Constant(OSC_OMEGA) * (z * z - Constant(0.25)) * cos(Constant(OSC_OMEGA) * z)
        + Constant(2.0) * z * sin(Constant(OSC_OMEGA) * z)
    )


def exact_potential_physical(x):
    return (
        Constant(0.5) * (x[0] * x[0] + x[1] * x[1])
        + Constant(OSC_BETA) * osc_q_physical(x[0]) * osc_q_physical(x[1])
    )


def exact_map_physical(x):
    return as_vector([
        x[0] + Constant(OSC_BETA) * osc_q_prime_physical(x[0]) * osc_q_physical(x[1]),
        x[1] + Constant(OSC_BETA) * osc_q_physical(x[0]) * osc_q_prime_physical(x[1]),
    ])


def exact_DT_physical(x):
    return as_matrix([
        [
            Constant(1.0) + Constant(OSC_BETA) * osc_q_second_physical(x[0]) * osc_q_physical(x[1]),
            Constant(OSC_BETA) * osc_q_prime_physical(x[0]) * osc_q_prime_physical(x[1]),
        ],
        [
            Constant(OSC_BETA) * osc_q_prime_physical(x[0]) * osc_q_prime_physical(x[1]),
            Constant(1.0) + Constant(OSC_BETA) * osc_q_physical(x[0]) * osc_q_second_physical(x[1]),
        ],
    ])


def exact_map_unit_from_source_unit(x_unit):
    return physical_to_unit_vector(exact_map_physical(unit_to_physical_vector(x_unit)))


def _signed_area_physical(a, b, c):
    return 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _grid_square_triangles(i, j, node):
    n00 = node(i, j)
    n10 = node(i + 1, j)
    n01 = node(i, j + 1)
    n11 = node(i + 1, j + 1)
    if (i + j) % 2 == 0:
        return [(n00, n10, n11), (n00, n11, n01)]
    return [(n00, n10, n01), (n10, n11, n01)]


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


def _q_numpy(z):
    w = OSC_OMEGA
    return (
        (-z*z/w + 2.0/w**3 + 1.0/(4.0*w)) * np.cos(w*z)
        + (2.0*z/w**2) * np.sin(w*z)
    )


def _qp_numpy(z):
    return (z*z - 0.25) * np.sin(OSC_OMEGA*z)


def _qpp_numpy(z):
    return (
        OSC_OMEGA * (z*z - 0.25) * np.cos(OSC_OMEGA*z)
        + 2.0*z*np.sin(OSC_OMEGA*z)
    )


def _dense_analytic_preflight(sample_n=1025):
    """Independent dense NumPy check of T4 convexity and map range."""
    z = np.linspace(-SUPPORT_HALF_WIDTH_PHYSICAL, SUPPORT_HALF_WIDTH_PHYSICAL, sample_n)
    q = _q_numpy(z)
    qp = _qp_numpy(z)
    qpp = _qpp_numpy(z)
    qx, qy = q[:, None], q[None, :]
    qpx, qpy = qp[:, None], qp[None, :]
    qppx, qppy = qpp[:, None], qpp[None, :]
    X, Y = z[:, None], z[None, :]

    a = 1.0 + OSC_BETA*qppx*qy
    d = 1.0 + OSC_BETA*qx*qppy
    b = OSC_BETA*qpx*qpy
    disc = np.sqrt((a-d)**2 + 4.0*b*b)
    lam_min = 0.5*(a+d-disc)
    lam_max = 0.5*(a+d+disc)
    detv = a*d-b*b
    T0 = X + OSC_BETA*qpx*qy
    T1 = Y + OSC_BETA*qx*qpy
    disp = np.sqrt((T0-X)**2 + (T1-Y)**2)

    return {
        "dense_sample_n": sample_n,
        "dense_exact_lambda_min": float(lam_min.min()),
        "dense_exact_lambda_max": float(lam_max.max()),
        "dense_exact_det_min": float(detv.min()),
        "dense_exact_det_max": float(detv.max()),
        "dense_exact_map_xmin": float(T0.min()),
        "dense_exact_map_xmax": float(T0.max()),
        "dense_exact_map_ymin": float(T1.min()),
        "dense_exact_map_ymax": float(T1.max()),
        "dense_exact_displacement_max_physical": float(disp.max()),
        "dense_source_density_contrast": float(detv.max()/detv.min()),
    }


def _generate_t4_square_meshes(mesh_dir):
    n = SUPPORT_GRID_N
    if n <= 0:
        raise ValueError("T4_SUPPORT_GRID_N must be positive")
    if abs(SUPPORT_HALF_WIDTH_PHYSICAL - 0.5) > 1.0e-14:
        raise ValueError("T4 v50 freezes the support square to [-0.5,0.5]^2")
    if OSC_MODE <= 0:
        raise ValueError("T4_OSC_MODE must be positive")
    if OSC_MODE % 2 != 0:
        raise ValueError("T4 v50 freezes an even oscillation mode so the boundary phase is symmetric")

    os.makedirs(mesh_dir, exist_ok=True)
    source_path = os.path.join(mesh_dir, f"t4_source_square_N{n}.msh")
    target_path = os.path.join(mesh_dir, f"t4_target_square_N{n}.msh")

    side = 2.0 * SUPPORT_HALF_WIDTH_PHYSICAL
    coords_phys = []
    ids = {}
    for j in range(n + 1):
        y = -SUPPORT_HALF_WIDTH_PHYSICAL + side * float(j) / float(n)
        for i in range(n + 1):
            x = -SUPPORT_HALF_WIDTH_PHYSICAL + side * float(i) / float(n)
            ids[(i, j)] = len(coords_phys) + 1
            coords_phys.append((x, y))

    def node(i, j):
        return ids[(i, j)]

    triangles = []
    area = 0.0
    amin = math.inf
    amax = 0.0
    for j in range(n):
        for i in range(n):
            for tri in _grid_square_triangles(i, j, node):
                pts = [coords_phys[k-1] for k in tri]
                signed = _signed_area_physical(*pts)
                amin = min(amin, signed)
                amax = max(amax, signed)
                area += signed
                triangles.append(tri)
    if amin <= 0.0:
        raise RuntimeError(f"T4 generated a non-positive square triangle: {amin}")

    coords_unit = [
        (0.5 + x/BRIDGE_WIDTH_PHYSICAL, 0.5 + y/BRIDGE_WIDTH_PHYSICAL)
        for x, y in coords_phys
    ]
    _write_gmsh_v2(source_path, coords_unit, triangles)
    _write_gmsh_v2(target_path, coords_unit, triangles)

    preflight = _dense_analytic_preflight()
    h_support = side / n
    h_b230 = BRIDGE_WIDTH_PHYSICAL / 230.0
    h_b460 = BRIDGE_WIDTH_PHYSICAL / 460.0
    h_b920 = BRIDGE_WIDTH_PHYSICAL / 920.0
    heat_attenuation = math.exp(-TAU_FINE_PHYSICAL * OSC_OMEGA**2)

    metadata = {
        "support_grid_n": n,
        "source_nodes_generated": len(coords_unit),
        "target_nodes_generated": len(coords_unit),
        "source_triangles_generated": len(triangles),
        "target_triangles_generated": len(triangles),
        "source_min_signed_triangle_area_physical": amin,
        "source_max_signed_triangle_area_physical": amax,
        "target_min_signed_triangle_area_physical": amin,
        "target_max_signed_triangle_area_physical": amax,
        "source_area_generated_physical": area,
        "target_area_generated_physical": area,
        "source_target_area_relative_mismatch_generated": 0.0,
        "support_h_physical": h_support,
        "osc_mode": OSC_MODE,
        "osc_beta": OSC_BETA,
        "osc_omega": OSC_OMEGA,
        "osc_wavelength_physical": OSC_WAVELENGTH_PHYSICAL,
        "support_cells_per_wavelength": OSC_WAVELENGTH_PHYSICAL / h_support,
        "B230_cells_per_wavelength": OSC_WAVELENGTH_PHYSICAL / h_b230,
        "B460_cells_per_wavelength": OSC_WAVELENGTH_PHYSICAL / h_b460,
        "B920_cells_per_wavelength": OSC_WAVELENGTH_PHYSICAL / h_b920,
        "continuum_heat_attenuation_exp_minus_tau_omega2": heat_attenuation,
        "source_mesh_sha256": _sha256_file(source_path),
        "target_mesh_sha256": _sha256_file(target_path),
        "source_mesh_path": source_path,
        "target_mesh_path": target_path,
    }
    metadata.update(preflight)
    return source_path, target_path, metadata


def _t4_mesh_paths_and_metadata():
    result_dir = os.environ.get("RESULT_DIR", os.getcwd())
    mesh_dir = os.path.join(result_dir, "t4_support_meshes")
    payload = None
    if COMM.rank == 0:
        payload = _generate_t4_square_meshes(mesh_dir)
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


def global_nonfinite_count(function):
    data = function.dat.data_ro
    local = int(np.count_nonzero(~np.isfinite(data)))
    return int(COMM.allreduce(local, op=MPI.SUM))


def build_support_spaces_and_measures(return_metadata=False):
    source_path, target_path, metadata = _t4_mesh_paths_and_metadata()
    mesh0 = Mesh(source_path)
    mesh1 = Mesh(target_path)

    V0 = FunctionSpace(mesh0, "CG", DISK_CG_DEGREE)
    V1 = FunctionSpace(mesh1, "CG", DISK_CG_DEGREE)

    x = SpatialCoordinate(mesh0)
    xp = unit_to_physical_vector(x)
    det_exact = det(exact_DT_physical(xp))

    mu0 = Function(V0, name="mu0").interpolate(det_exact)
    mu1 = Function(V1, name="mu1").interpolate(1.0)

    source_raw = float(assemble(mu0 * dx(domain=mesh0)))
    target_raw = float(assemble(mu1 * dx(domain=mesh1)))
    if source_raw <= 0.0 or target_raw <= 0.0:
        raise RuntimeError(f"T4 invalid raw mass/area: source={source_raw}, target={target_raw}")
    mu0.assign(mu0/source_raw)
    mu1.assign(mu1/target_raw)

    source_cells = int(COMM.allreduce(mesh0.cell_set.size, op=MPI.SUM))
    target_cells = int(COMM.allreduce(mesh1.cell_set.size, op=MPI.SUM))
    metadata = dict(metadata)
    metadata.update({
        "source_global_cells_firedrake": source_cells,
        "target_global_cells_firedrake": target_cells,
        "source_det_integral_unit": source_raw,
        "target_area_unit": target_raw,
        "source_target_det_area_relative_mismatch_firedrake": abs(source_raw-target_raw)/max(target_raw, 1e-300),
        "source_area_firedrake_physical": float(assemble(Constant(1.0)*dx(domain=mesh0))) * BRIDGE_WIDTH_PHYSICAL**2,
        "target_area_firedrake_physical": float(assemble(Constant(1.0)*dx(domain=mesh1))) * BRIDGE_WIDTH_PHYSICAL**2,
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



def t4_oscillatory_map_audit(fine, q, result_dir, method):
    """Post-production audit of map accuracy and oscillatory-mode recovery."""
    V0, V1 = fine.V0, fine.V1
    x = SpatialCoordinate(V0.mesh())
    y = SpatialCoordinate(V1.mesh())
    xp = unit_to_physical_vector(x)
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
    ysq_weight.interpolate((y[0]**2 + y[1]**2) * exp_q_shifted)

    Hnum0 = Function(V0, name=f"{method}_map_num0")
    Hnum1 = Function(V0, name=f"{method}_map_num1")
    Hnumsq = Function(V0, name=f"{method}_map_numsq")
    fine.heat_scaling_to_source(y0_weight, Hnum0)
    fine.heat_scaling_to_source(y1_weight, Hnum1)
    fine.heat_scaling_to_source(ysq_weight, Hnumsq)

    T0 = Function(V0, name=f"{method}_T0_unit")
    T1 = Function(V0, name=f"{method}_T1_unit")
    T0.interpolate(Hnum0/max_value(Hden, TINY))
    T1.interpolate(Hnum1/max_value(Hden, TINY))
    T_unit = as_vector([T0, T1])
    T_phys = unit_to_physical_vector(T_unit)
    T_exact_phys = exact_map_physical(xp)

    ex = T_phys[0]-T_exact_phys[0]
    ey = T_phys[1]-T_exact_phys[1]
    err_sq = ex*ex + ey*ey
    map_l2 = math.sqrt(max(float(assemble(err_sq*fine.mu0*dx)), 0.0))

    Verr = FunctionSpace(V0.mesh(), "CG", DISK_CG_DEGREE)
    err_mag = Function(Verr, name=f"{method}_map_error_physical")
    err_mag.interpolate(sqrt(err_sq))
    map_linf = global_dof_max(err_mag)

    DT_num = grad(T_unit)  # physical and unit Jacobians are identical under common affine scaling
    DT_exact = exact_DT_physical(xp)
    Vdg0 = FunctionSpace(V0.mesh(), "DG", 0)
    det_num = Function(Vdg0, name=f"{method}_det_DT").interpolate(det(DT_num))
    det_exact = Function(Vdg0, name=f"{method}_det_DT_exact").interpolate(det(DT_exact))
    jac_diff_sq = sum((DT_num[i,j]-DT_exact[i,j])**2 for i in range(2) for j in range(2))
    jac_l2 = math.sqrt(max(float(assemble(jac_diff_sq*fine.mu0*dx)), 0.0))
    det_l2 = math.sqrt(max(float(assemble((det(DT_num)-det(DT_exact))**2*fine.mu0*dx)), 0.0))
    det_min = global_dof_min(det_num)
    det_max = global_dof_max(det_num)
    exact_det_min = global_dof_min(det_exact)
    exact_det_max = global_dof_max(det_exact)
    tangled_area = float(assemble(conditional(le(det(DT_num), 0.0), 1.0, 0.0)*dx(domain=V0.mesh())))

    # Total conditional spread / Monge defect.
    E_y_sq = Function(V0, name=f"{method}_E_y_sq_unit")
    variance = Function(V0, name=f"{method}_conditional_variance_unit")
    E_y_sq.interpolate(Hnumsq/max_value(Hden, TINY))
    variance.interpolate(max_value(E_y_sq-(T0*T0+T1*T1), Constant(0.0)))
    monge_defect_unit = float(assemble(variance*fine.mu0*dx))
    monge_defect_physical = width*width*monge_defect_unit
    rms_conditional_spread = width*math.sqrt(max(monge_defect_unit,0.0))

    # Exact oscillatory displacement recovery.  This is more informative than
    # raw map L2 because the identity part of T_* is much larger than the
    # deliberately small, high-frequency perturbation.
    dstar0 = T_exact_phys[0]-xp[0]
    dstar1 = T_exact_phys[1]-xp[1]
    dnum0 = T_phys[0]-xp[0]
    dnum1 = T_phys[1]-xp[1]
    exact_disp_sq = float(assemble((dstar0*dstar0+dstar1*dstar1)*fine.mu0*dx))
    num_disp_sq = float(assemble((dnum0*dnum0+dnum1*dnum1)*fine.mu0*dx))
    disp_inner = float(assemble((dnum0*dstar0+dnum1*dstar1)*fine.mu0*dx))
    exact_disp_rms = math.sqrt(max(exact_disp_sq,0.0))
    numerical_disp_rms = math.sqrt(max(num_disp_sq,0.0))
    osc_gain = disp_inner/max(exact_disp_sq,1e-300)
    osc_cosine = disp_inner/max(exact_disp_rms*numerical_disp_rms,1e-300)

    exact_d0_sq = float(assemble(dstar0*dstar0*fine.mu0*dx))
    exact_d1_sq = float(assemble(dstar1*dstar1*fine.mu0*dx))
    component0_gain = float(assemble(dnum0*dstar0*fine.mu0*dx))/max(exact_d0_sq,1e-300)
    component1_gain = float(assemble(dnum1*dstar1*fine.mu0*dx))/max(exact_d1_sq,1e-300)

    mean_disp0 = float(assemble(dnum0*fine.mu0*dx))
    mean_disp1 = float(assemble(dnum1*fine.mu0*dx))

    # Support escape is a barycentric-map diagnostic.  Exact T_* remains in
    # the square.  Finite entropy can create small boundary bias, so report but
    # do not use it as a production acceptance condition.
    a = Constant(SUPPORT_HALF_WIDTH_PHYSICAL)
    inside = (
        conditional(ge(T_phys[0], -a), 1.0, 0.0)
        * conditional(le(T_phys[0], a), 1.0, 0.0)
        * conditional(ge(T_phys[1], -a), 1.0, 0.0)
        * conditional(le(T_phys[1], a), 1.0, 0.0)
    )
    barycentric_support_escape_mass = float(assemble((Constant(1.0)-inside)*fine.mu0*dx))

    row = {
        "map_q_shift": q_shift,
        "map_denominator_min": global_dof_min(Hden),
        "map_denominator_max": global_dof_max(Hden),
        "map_denominator_nonfinite_count": global_nonfinite_count(Hden),
        "map_l2_physical": map_l2,
        "map_linf_physical": map_linf,
        "jacobian_frobenius_l2": jac_l2,
        "det_l2_error": det_l2,
        "det_min": det_min,
        "det_max": det_max,
        "exact_det_min": exact_det_min,
        "exact_det_max": exact_det_max,
        "tangled_area": tangled_area,
        "monge_defect_unit": monge_defect_unit,
        "monge_defect_physical": monge_defect_physical,
        "rms_conditional_spread_physical": rms_conditional_spread,
        "exact_oscillatory_displacement_rms_physical": exact_disp_rms,
        "numerical_displacement_rms_physical": numerical_disp_rms,
        "oscillatory_displacement_gain": osc_gain,
        "oscillatory_displacement_cosine_alignment": osc_cosine,
        "oscillatory_component0_gain": component0_gain,
        "oscillatory_component1_gain": component1_gain,
        "mean_displacement_x_physical": mean_disp0,
        "mean_displacement_y_physical": mean_disp1,
        "barycentric_support_escape_mass": barycentric_support_escape_mass,
        "osc_mode": OSC_MODE,
        "osc_beta": OSC_BETA,
        "osc_wavelength_physical": OSC_WAVELENGTH_PHYSICAL,
    }

    if AUDIT_WRITE_VTK:
        Vvec = VectorFunctionSpace(V0.mesh(), "CG", DISK_CG_DEGREE)
        Tvec = Function(Vvec, name=f"{method}_T_unit").interpolate(T_unit)
        Texact = Function(Vvec, name=f"{method}_T_exact_unit").interpolate(exact_map_unit_from_source_unit(x))
        VTKFile(os.path.join(result_dir, f"{EXPERIMENT_NAME}_{method}_map.pvd")).write(
            Tvec, Texact, err_mag, det_num, det_exact, variance
        )
    if COMM.rank == 0:
        print("T4_OSCILLATORY_MAP_AUDIT", row)
    return row


def _pushforward_moment_rows(V0, V1, mu0, mu1):
    x = SpatialCoordinate(V0.mesh())
    y = SpatialCoordinate(V1.mesh())
    xp = unit_to_physical_vector(x)
    yp = unit_to_physical_vector(y)
    T = exact_map_physical(xp)
    w = Constant(OSC_OMEGA)
    tests = [
        ("one", lambda z: Constant(1.0)),
        ("y0", lambda z: z[0]),
        ("y1", lambda z: z[1]),
        ("y0_sq", lambda z: z[0]**2),
        ("y0y1", lambda z: z[0]*z[1]),
        ("y1_sq", lambda z: z[1]**2),
        ("cos_omega_y0", lambda z: cos(w*z[0])),
        ("sin_omega_y0", lambda z: sin(w*z[0])),
        ("cos_omega_y1", lambda z: cos(w*z[1])),
        ("sin_omega_y1", lambda z: sin(w*z[1])),
        ("coscos_omega", lambda z: cos(w*z[0])*cos(w*z[1])),
        ("sinsin_omega", lambda z: sin(w*z[0])*sin(w*z[1])),
    ]
    rows=[]
    for name,f in tests:
        source_value=float(assemble(f(T)*mu0*dx(domain=V0.mesh())))
        target_value=float(assemble(f(yp)*mu1*dx(domain=V1.mesh())))
        rows.append({
            "moment":name,
            "source_pushforward_value":source_value,
            "target_value":target_value,
            "abs_error":abs(source_value-target_value),
        })
    return rows


def _spectral_design_rows():
    rows=[]
    for bridge_n in BRIDGE_LEVELS:
        h=BRIDGE_WIDTH_PHYSICAL/float(bridge_n)
        rows.append({
            "bridge_n": bridge_n,
            "bridge_h_physical": h,
            "osc_mode": OSC_MODE,
            "osc_omega": OSC_OMEGA,
            "osc_wavelength_physical": OSC_WAVELENGTH_PHYSICAL,
            "bridge_cells_per_wavelength": OSC_WAVELENGTH_PHYSICAL/h,
            "continuum_heat_attenuation_exp_minus_tau_omega2": math.exp(-TAU_FINE_PHYSICAL*OSC_OMEGA**2),
            "tau_physical": TAU_FINE_PHYSICAL,
        })
    return rows


def benchmark_certificate(V0, V1, mu0, mu1, metadata, result_dir):
    mesh0, mesh1 = V0.mesh(), V1.mesh()
    x = SpatialCoordinate(mesh0)
    xp = unit_to_physical_vector(x)
    H = exact_DT_physical(xp)
    trH = H[0,0]+H[1,1]
    disc = sqrt((H[0,0]-H[1,1])**2 + Constant(4.0)*H[0,1]**2)
    Vdg = FunctionSpace(mesh0, "DG", 0)
    lam_min = Function(Vdg, name="t4_exact_lambda_min").interpolate(Constant(0.5)*(trH-disc))
    lam_max = Function(Vdg, name="t4_exact_lambda_max").interpolate(Constant(0.5)*(trH+disc))
    det_exact = Function(Vdg, name="t4_exact_det").interpolate(det(H))

    Texact = exact_map_physical(xp)
    V0s = FunctionSpace(mesh0, "CG", DISK_CG_DEGREE)
    T0s = Function(V0s, name="t4_exact_T0_sample").interpolate(Texact[0])
    T1s = Function(V0s, name="t4_exact_T1_sample").interpolate(Texact[1])
    dmag = Function(V0s, name="t4_exact_displacement_sample").interpolate(
        sqrt((Texact[0]-xp[0])**2+(Texact[1]-xp[1])**2)
    )

    source_mass=float(assemble(mu0*dx(domain=mesh0)))
    target_mass=float(assemble(mu1*dx(domain=mesh1)))
    source_bbox=_global_mesh_bbox(mesh0)
    target_bbox=_global_mesh_bbox(mesh1)
    bridge_margin=min(
        source_bbox[0],1.0-source_bbox[1],source_bbox[2],1.0-source_bbox[3],
        target_bbox[0],1.0-target_bbox[1],target_bbox[2],1.0-target_bbox[3],
    )
    moment_rows=_pushforward_moment_rows(V0,V1,mu0,mu1)
    max_moment_error=max(r["abs_error"] for r in moment_rows)
    density_min=global_dof_min(mu0)
    density_max=global_dof_max(mu0)
    density_contrast=density_max/max(density_min,1e-300)

    box_a=SUPPORT_HALF_WIDTH_PHYSICAL
    map_overshoot=max(
        max(0.0, -box_a-global_dof_min(T0s)),
        max(0.0, global_dof_max(T0s)-box_a),
        max(0.0, -box_a-global_dof_min(T1s)),
        max(0.0, global_dof_max(T1s)-box_a),
    )

    row={
        "benchmark":"T4_high_frequency_square",
        "support_grid_n":SUPPORT_GRID_N,
        "support_half_width_physical":SUPPORT_HALF_WIDTH_PHYSICAL,
        "osc_mode":OSC_MODE,
        "osc_beta":OSC_BETA,
        "osc_omega":OSC_OMEGA,
        "osc_wavelength_physical":OSC_WAVELENGTH_PHYSICAL,
        "continuum_heat_attenuation_exp_minus_tau_omega2":math.exp(-TAU_FINE_PHYSICAL*OSC_OMEGA**2),
        "source_mass":source_mass,
        "target_mass":target_mass,
        "source_density_min":density_min,
        "source_density_max":density_max,
        "source_density_contrast":density_contrast,
        "target_density_min":global_dof_min(mu1),
        "target_density_max":global_dof_max(mu1),
        "source_density_nonfinite_count":global_nonfinite_count(mu0),
        "target_density_nonfinite_count":global_nonfinite_count(mu1),
        "exact_lambda_min_dg0":global_dof_min(lam_min),
        "exact_lambda_max_dg0":global_dof_max(lam_max),
        "exact_det_min_dg0":global_dof_min(det_exact),
        "exact_det_max_dg0":global_dof_max(det_exact),
        "exact_map_xmin_sample":global_dof_min(T0s),
        "exact_map_xmax_sample":global_dof_max(T0s),
        "exact_map_ymin_sample":global_dof_min(T1s),
        "exact_map_ymax_sample":global_dof_max(T1s),
        "exact_map_box_overshoot_physical":map_overshoot,
        "exact_displacement_max_sample_physical":global_dof_max(dmag),
        "bridge_margin_unit":bridge_margin,
        "max_pushforward_moment_abs_error":max_moment_error,
    }
    row.update(metadata)

    expected_cells=2*SUPPORT_GRID_N*SUPPORT_GRID_N
    coarse_cpw=OSC_WAVELENGTH_PHYSICAL/(BRIDGE_WIDTH_PHYSICAL/230.0)
    fine_cpw=OSC_WAVELENGTH_PHYSICAL/(BRIDGE_WIDTH_PHYSICAL/920.0)
    checks={
        "source_triangle_orientation_valid":metadata["source_min_signed_triangle_area_physical"]>=CERT_MIN_SIGNED_TRIANGLE_AREA_PHYSICAL,
        "target_triangle_orientation_valid":metadata["target_min_signed_triangle_area_physical"]>=CERT_MIN_SIGNED_TRIANGLE_AREA_PHYSICAL,
        "source_target_generated_area_match":metadata["source_target_area_relative_mismatch_generated"]<=CERT_MAX_AREA_REL_MISMATCH,
        "source_target_det_area_match":metadata["source_target_det_area_relative_mismatch_firedrake"]<=5.0e-5,
        "source_cell_count_expected":metadata["source_global_cells_firedrake"]==expected_cells,
        "target_cell_count_expected":metadata["target_global_cells_firedrake"]==expected_cells,
        "exact_map_strongly_convex":min(global_dof_min(lam_min), metadata["dense_exact_lambda_min"])>=CERT_MIN_EXACT_JACOBIAN_EIG,
        "exact_det_positive":min(global_dof_min(det_exact), metadata["dense_exact_det_min"])>=CERT_MIN_EXACT_DET,
        "exact_map_inside_square":max(map_overshoot,
            max(0.0,-box_a-metadata["dense_exact_map_xmin"]),
            max(0.0,metadata["dense_exact_map_xmax"]-box_a),
            max(0.0,-box_a-metadata["dense_exact_map_ymin"]),
            max(0.0,metadata["dense_exact_map_ymax"]-box_a))<=CERT_MAX_MAP_BOX_OVERSHOOT_PHYSICAL,
        "source_density_positive":density_min>0.0,
        "target_density_positive":global_dof_min(mu1)>0.0,
        "densities_finite":global_nonfinite_count(mu0)==0 and global_nonfinite_count(mu1)==0,
        "unit_masses":abs(source_mass-1.0)<=1.0e-12 and abs(target_mass-1.0)<=1.0e-12,
        "source_density_contrast_nontrivial":density_contrast>=CERT_MIN_SOURCE_DENSITY_CONTRAST,
        "pushforward_moments":max_moment_error<=CERT_MAX_PUSHFORWARD_MOMENT_ERROR,
        "bridge_margin":bridge_margin>=CERT_MIN_BRIDGE_MARGIN_UNIT,
        "coarse_frequency_is_stressful":coarse_cpw<=CERT_MAX_COARSE_CELLS_PER_WAVELENGTH,
        "fine_frequency_is_well_resolved":fine_cpw>=CERT_MIN_FINE_CELLS_PER_WAVELENGTH,
    }
    row.update({f"check_{k}":int(v) for k,v in checks.items()})
    row["certificate_pass"]=int(all(checks.values()))

    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_benchmark_certificate.csv"),[row])
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_pushforward_moments.csv"),moment_rows)
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_mesh_metadata.csv"),[metadata])
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_spectral_design.csv"),_spectral_design_rows())
    if COMM.rank==0:
        print("T4_BENCHMARK_CERTIFICATE",row)
    if not all(checks.values()):
        failed=[k for k,v in checks.items() if not v]
        raise RuntimeError(f"T4 benchmark certificate failed: {failed}; row={row}")
    return row


def final_science_audit(method, fine, p, q, b_p, b_q, performance_row, result_dir):
    audit_t0=time.perf_counter()
    fine.reset_counters()
    full_rp,full_rq=fine.residual_norms(p,q,b_p,b_q)
    marginal_p_unit,marginal_q_unit=fine.marginal_residual_norms(p,q)
    width=BRIDGE_WIDTH_PHYSICAL
    row={
        "method":method,
        "target_residual":TARGET_RESIDUAL,
        "performance_total_seconds":performance_row.get("total_seconds",""),
        "performance_iteration_seconds":performance_row.get("iteration_seconds",""),
        "performance_reached_target":performance_row.get("reached_target",""),
        "performance_cycles":performance_row.get("cycle",""),
        "performance_updates":performance_row.get("updates",""),
        "performance_residual_p":performance_row.get("residual_p",""),
        "full_residual_p":full_rp,
        "full_residual_q":full_rq,
        "source_marginal_l2_unit":marginal_p_unit,
        "target_marginal_l2_unit":marginal_q_unit,
        "source_marginal_l2_physical":marginal_p_unit/width,
        "target_marginal_l2_physical":marginal_q_unit/width,
        "audit_write_vtk":AUDIT_WRITE_VTK,
    }
    row.update(fine.field_stats(p,"p"))
    row.update(fine.field_stats(q,"q"))
    row.update(t4_oscillatory_map_audit(fine,q,result_dir,method))
    for name in ["log_marginal_p","log_marginal_q","marginal_p","marginal_q",
                 "logHq_on_0","logHp_on_1","Hq_shifted_on_0","Hp_shifted_on_1"]:
        field=getattr(fine,name,None)
        if field is not None:
            row.update(fine.field_stats(field,name))
    row.update(fine.heat_stats(fine.Hq_shifted_on_0,"Hq_shifted_heat"))
    row.update(fine.heat_stats(fine.Hp_shifted_on_1,"Hp_shifted_heat"))
    row["audit_seconds"]=COMM.allreduce(time.perf_counter()-audit_t0,op=MPI.MAX)
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_{method}_science_audit.csv"),[row])
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_{method}_audit_heat_counts.csv"),fine.counter_rows())
    if COMM.rank==0:
        print("FAS_SCIENCE_AUDIT",row)
    return row


def hierarchy_level_objects(level):
    levels=[]
    if level.coarse is not None:
        levels.extend(hierarchy_level_objects(level.coarse))
    levels.append(level)
    return levels


def hierarchy_rows(level):
    return [item.hierarchy_row() for item in hierarchy_level_objects(level)]


def assert_requested_hierarchy_is_active(fine):
    levels=hierarchy_level_objects(fine)
    observed=[level.bridge_n for level in levels]
    expected=list(BRIDGE_LEVELS)
    if observed!=expected:
        raise RuntimeError(f"Expected hierarchy {expected}, got {observed}")
    if len(expected)>=3:
        bottom=levels[0]
        bottom_calls=bottom.heat_source_calls+bottom.heat_target_calls
        if bottom_calls<=0:
            raise RuntimeError(f"Three-level FAS requested but B{bottom.bridge_n} performed zero heat actions.")


def build_problem(coarse_tau_ratio=1.0):
    V0,V1,mu0,mu1=build_support_spaces_and_measures()
    bridge_hierarchy=build_bridge_hierarchy()
    levels=[]
    coarse=None
    for level,mesh in enumerate(bridge_hierarchy):
        tau_physical=TAU_FINE_PHYSICAL*coarse_tau_ratio if level==0 else TAU_FINE_PHYSICAL
        current=FASLevel(
            f"level{level}_B{BRIDGE_LEVELS[level]}",level,mesh,tau_physical,
            V0,V1,mu0,mu1,coarse=coarse,
        )
        levels.append(current)
        coarse=current
    fine=levels[-1]
    immediate_coarse=levels[-2] if len(levels)>=2 else None
    return fine,immediate_coarse,V0,V1


def gauge_aligned_fields(level,p,q,p_ref,q_ref):
    dp=Function(p.function_space(),name="t4_dp")
    dq=Function(q.function_space(),name="t4_dq")
    dp.interpolate(p-p_ref)
    dq.interpolate(q-q_ref)
    a=float(assemble(dp*level.mu0*dx))
    b=float(assemble(dq*level.mu1*dx))
    shift=0.5*(a-b)
    dp.interpolate(p-p_ref-Constant(shift))
    dq.interpolate(q-q_ref+Constant(shift))
    ep=math.sqrt(max(float(assemble(dp*dp*level.mu0*dx)),0.0))
    eq=math.sqrt(max(float(assemble(dq*dq*level.mu1*dx)),0.0))
    return dp,dq,ep,eq,math.sqrt(ep*ep+eq*eq),shift


def gauge_aligned_distance(level,p,q,p_ref,q_ref):
    _,_,ep,eq,e,shift=gauge_aligned_fields(level,p,q,p_ref,q_ref)
    return ep,eq,e,shift


def _fourier_probe_rss(field, mu, mode):
    mesh=field.function_space().mesh()
    xp=unit_to_physical_vector(SpatialCoordinate(mesh))
    w=Constant(2.0*math.pi*mode)
    probes=[
        cos(w*xp[0])*cos(w*xp[1]),
        sin(w*xp[0])*sin(w*xp[1]),
        cos(w*xp[0])*sin(w*xp[1]),
        sin(w*xp[0])*cos(w*xp[1]),
    ]
    coeffs=[]
    for psi in probes:
        norm=math.sqrt(max(float(assemble(psi*psi*mu*dx(domain=mesh))),1e-300))
        coeffs.append(float(assemble(field*psi*mu*dx(domain=mesh)))/norm)
    return coeffs, math.sqrt(sum(c*c for c in coeffs))


def spectral_error_probes(level,dp,dq):
    hi_p_coeffs,hi_p=_fourier_probe_rss(dp,level.mu0,OSC_MODE)
    hi_q_coeffs,hi_q=_fourier_probe_rss(dq,level.mu1,OSC_MODE)
    low_mode=max(1,OSC_MODE//2)
    lo_p_coeffs,lo_p=_fourier_probe_rss(dp,level.mu0,low_mode)
    lo_q_coeffs,lo_q=_fourier_probe_rss(dq,level.mu1,low_mode)
    row={
        "high_probe_mode":OSC_MODE,
        "low_probe_mode":low_mode,
        "p_high_fourier_probe_rss":hi_p,
        "q_high_fourier_probe_rss":hi_q,
        "combined_high_fourier_probe_rss":math.sqrt(hi_p*hi_p+hi_q*hi_q),
        "p_low_fourier_probe_rss":lo_p,
        "q_low_fourier_probe_rss":lo_q,
        "combined_low_fourier_probe_rss":math.sqrt(lo_p*lo_p+lo_q*lo_q),
    }
    for i,c in enumerate(hi_p_coeffs): row[f"p_high_probe_c{i}"]=c
    for i,c in enumerate(hi_q_coeffs): row[f"q_high_probe_c{i}"]=c
    for i,c in enumerate(lo_p_coeffs): row[f"p_low_probe_c{i}"]=c
    for i,c in enumerate(lo_q_coeffs): row[f"q_low_probe_c{i}"]=c
    return row


def reference_norm(level,p,q):
    return math.sqrt(max(
        float(assemble(p*p*level.mu0*dx))
        +float(assemble(q*q*level.mu1*dx)),0.0))


def run_t4_reference_fas_validation(result_dir,fine,V0,V1,p,q,b_p,b_q,total_t0):
    if len(hierarchy_level_objects(fine))!=3:
        raise RuntimeError("T4 production reference/FAS validation requires exactly three levels")

    # Phase A: independent B920 fine-only Sinkhorn reference.
    fine.reset_counters()
    p.assign(0.0); q.assign(0.0); b_p.assign(0.0); b_q.assign(0.0)
    previous_p=Function(V0,name="t4_reference_previous_p").assign(p)
    previous_q=Function(V1,name="t4_reference_previous_q").assign(q)
    previous_change_abs=float("nan")
    ref_rows=[]
    updates=0
    stable_checks=0
    reference_certified=False
    ref_t0=time.perf_counter()

    while updates<REF_MAX_FINE_UPDATES:
        block=min(REF_CHECK_BLOCK,REF_MAX_FINE_UPDATES-updates)
        fine.smoother(p,q,b_p,b_q,block,f"t4_reference_block_{updates+block}")
        updates+=block
        fine.gauge_fix(p,q)
        residual_p,residual_q=fine.residual_norms(p,q,b_p,b_q)

        change_p=Function(V0,name="t4_reference_change_p")
        change_q=Function(V1,name="t4_reference_change_q")
        change_p.interpolate(p-previous_p)
        change_q.interpolate(q-previous_q)
        a=float(assemble(change_p*fine.mu0*dx))
        b=float(assemble(change_q*fine.mu1*dx))
        change_shift=0.5*(a-b)
        change_p.interpolate(p-previous_p-Constant(change_shift))
        change_q.interpolate(q-previous_q+Constant(change_shift))
        change_p_norm=math.sqrt(max(float(assemble(change_p*change_p*fine.mu0*dx)),0.0))
        change_q_norm=math.sqrt(max(float(assemble(change_q*change_q*fine.mu1*dx)),0.0))
        change_abs=math.sqrt(change_p_norm**2+change_q_norm**2)
        norm_now=reference_norm(fine,p,q)
        change_rel=change_abs/max(norm_now,1e-300)
        change_contraction=change_abs/previous_change_abs if math.isfinite(previous_change_abs) and previous_change_abs>0 else float("nan")
        tail_est=float("nan")
        if math.isfinite(change_contraction) and 0.0<=change_contraction<1.0:
            tail_est=change_contraction/max(1.0-change_contraction,1e-30)*change_abs
        criteria=(updates>=REF_MIN_FINE_UPDATES and residual_p<=REF_RESIDUAL_TARGET and change_rel<=REF_CHANGE_REL_TOL)
        stable_checks=stable_checks+1 if criteria else 0
        reference_certified=stable_checks>=REF_STABLE_CHECKS
        ref_rows.append({
            "fine_bridge_n":fine.bridge_n,
            "updates":updates,
            "residual_p":residual_p,
            "residual_q":residual_q,
            "change_p":change_p_norm,
            "change_q":change_q_norm,
            "change_abs":change_abs,
            "change_rel":change_rel,
            "change_contraction":change_contraction,
            "estimated_remaining_tail":tail_est,
            "change_gauge_shift":change_shift,
            "stable_checks":stable_checks,
            "reference_certified":int(reference_certified),
        })
        previous_p.assign(p); previous_q.assign(q); previous_change_abs=change_abs
        if COMM.rank==0:
            print("T4_REFERENCE_ROW",ref_rows[-1])
        if reference_certified:
            break

    reference_seconds=COMM.allreduce(time.perf_counter()-ref_t0,op=MPI.MAX)
    if COMM.rank==0 and not reference_certified:
        print("T4_REFERENCE_WARNING reference not formally certified; FAS stress result remains valid and reference comparison is provisional.")
    p_ref=Function(V0,name="t4_p_ref").assign(p)
    q_ref=Function(V1,name="t4_q_ref").assign(q)
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_reference_trace.csv"),ref_rows)
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_reference_heat_counts.csv"),fine.counter_rows())

    # Phase B: single frozen cold FAS attempt.
    p.assign(0.0);q.assign(0.0);b_p.assign(0.0);b_q.assign(0.0)
    fine.reset_counters()
    initial_residual=fine.p_residual_norm(p,q,b_p)
    dp,dq,ep0,eq0,error0,shift0=gauge_aligned_fields(fine,p,q,p_ref,q_ref)
    spectral0=spectral_error_probes(fine,dp,dq)
    trajectory=[{
        "cycle":0,
        "residual_p":initial_residual,
        "residual_ratio_to_initial":1.0,
        "residual_contraction_from_previous":"",
        "p_error_to_reference":ep0,
        "q_error_to_reference":eq0,
        "combined_error_to_reference":error0,
        "error_ratio_to_initial":1.0,
        "error_contraction_from_previous":"",
        "gauge_alignment_shift":shift0,
        "target_residual":TARGET_RESIDUAL,
        "reached_target":int(initial_residual<=TARGET_RESIDUAL),
        **spectral0,
    }]
    fine.reset_counters()
    production_t0=time.perf_counter()
    previous_residual=initial_residual
    previous_error=error0
    previous_high=spectral0["combined_high_fourier_probe_rss"]
    previous_low=spectral0["combined_low_fourier_probe_rss"]

    for cycle in range(1,FAS_CYCLES+1):
        fine.fas_cycle_perf(p,q,b_p,b_q)
        residual_p=fine.p_residual_norm(p,q,b_p)
        dp,dq,ep,eq,error,shift=gauge_aligned_fields(fine,p,q,p_ref,q_ref)
        spectral=spectral_error_probes(fine,dp,dq)
        high=spectral["combined_high_fourier_probe_rss"]
        low=spectral["combined_low_fourier_probe_rss"]
        now=time.perf_counter()
        trajectory.append({
            "cycle":cycle,
            "residual_p":residual_p,
            "residual_ratio_to_initial":residual_p/initial_residual,
            "residual_contraction_from_previous":residual_p/previous_residual,
            "p_error_to_reference":ep,
            "q_error_to_reference":eq,
            "combined_error_to_reference":error,
            "error_ratio_to_initial":error/error0,
            "error_contraction_from_previous":error/previous_error,
            "gauge_alignment_shift":shift,
            "target_residual":TARGET_RESIDUAL,
            "reached_target":int(residual_p<=TARGET_RESIDUAL),
            "iteration_seconds":COMM.allreduce(now-production_t0,op=MPI.MAX),
            "high_probe_contraction_from_previous":high/max(previous_high,1e-300),
            "low_probe_contraction_from_previous":low/max(previous_low,1e-300),
            **spectral,
        })
        previous_residual=residual_p
        previous_error=error
        previous_high=high
        previous_low=low
        if COMM.rank==0:
            print("T4_REFERENCE_FAS_ROW",trajectory[-1])
        if residual_p<=TARGET_RESIDUAL:
            break

    cycles=len(trajectory)-1
    assert_requested_hierarchy_is_active(fine)
    levels=hierarchy_level_objects(fine)
    observed=[level.heat_source_calls+level.heat_target_calls for level in levels]
    expected=[16*cycles,10*cycles,9*cycles]
    if observed!=expected:
        raise RuntimeError(f"T4 production heat-count failure: observed={observed}, expected={expected}")

    production_seconds=COMM.allreduce(time.perf_counter()-production_t0,op=MPI.MAX)
    final_error=trajectory[-1]["combined_error_to_reference"]
    last_ref=ref_rows[-1]
    tail_est=last_ref.get("estimated_remaining_tail",float("nan"))
    summary={
        "cycles":cycles,
        "reached_target":trajectory[-1]["reached_target"],
        "initial_residual_p":initial_residual,
        "final_residual_p":trajectory[-1]["residual_p"],
        "initial_discrete_reference_error":error0,
        "final_discrete_reference_error":final_error,
        "geomean_residual_contraction":(trajectory[-1]["residual_p"]/initial_residual)**(1.0/max(cycles,1)),
        "geomean_discrete_error_contraction":(final_error/error0)**(1.0/max(cycles,1)),
        "initial_high_fourier_probe_rss":trajectory[0]["combined_high_fourier_probe_rss"],
        "final_high_fourier_probe_rss":trajectory[-1]["combined_high_fourier_probe_rss"],
        "initial_low_fourier_probe_rss":trajectory[0]["combined_low_fourier_probe_rss"],
        "final_low_fourier_probe_rss":trajectory[-1]["combined_low_fourier_probe_rss"],
        "reference_updates":updates,
        "reference_residual_p":last_ref["residual_p"],
        "reference_residual_q":last_ref["residual_q"],
        "reference_change_abs":last_ref["change_abs"],
        "reference_change_rel":last_ref["change_rel"],
        "reference_change_contraction":last_ref["change_contraction"],
        "reference_estimated_remaining_tail":tail_est,
        "reference_certified":int(reference_certified),
        "reference_uncertainty_over_final_error":last_ref["change_abs"]/max(final_error,1e-300),
        "reference_tail_over_final_error":tail_est/max(final_error,1e-300) if math.isfinite(tail_est) else float("nan"),
        "reference_seconds":reference_seconds,
        "production_iteration_seconds":production_seconds,
        "job_seconds_before_final_audit":COMM.allreduce(time.perf_counter()-total_t0,op=MPI.MAX),
        "observed_heat_actions":str(observed),
        "expected_heat_actions":str(expected),
        "heat_count_match":1,
    }
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_fas_discrete_error_trajectory.csv"),trajectory)
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_production_heat_counts.csv"),fine.counter_rows())
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_reference_fas_summary.csv"),[summary])

    perf_row={
        "cycle":cycles,
        "residual_p":trajectory[-1]["residual_p"],
        "reached_target":trajectory[-1]["reached_target"],
        "iteration_seconds":production_seconds,
        "total_seconds":production_seconds,
    }
    science_row=final_science_audit("reference_fas_to_tol",fine,p,q,b_p,b_q,perf_row,result_dir)
    summary.update({
        "map_l2_physical":science_row.get("map_l2_physical",""),
        "oscillatory_displacement_gain":science_row.get("oscillatory_displacement_gain",""),
        "oscillatory_displacement_cosine_alignment":science_row.get("oscillatory_displacement_cosine_alignment",""),
        "barycentric_support_escape_mass":science_row.get("barycentric_support_escape_mass",""),
        "tangled_area":science_row.get("tangled_area",""),
        "det_min":science_row.get("det_min",""),
    })
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_reference_fas_summary_with_map.csv"),[summary])
    if COMM.rank==0:
        print("T4_REFERENCE_FAS_FINAL",summary)
        print(f"RESULT_DIR={result_dir}")


def run_fas_to_tol(result_dir,fine,p,q,b_p,b_q,total_t0):
    # One-cycle plumbing smoke only.
    fine.reset_counters()
    rows=[]
    iteration_t0=time.perf_counter()
    for cycle in range(1,FAS_CYCLES+1):
        fine.fas_cycle_perf(p,q,b_p,b_q)
        residual_p=fine.p_residual_norm(p,q,b_p)
        now=time.perf_counter()
        rows.append({
            "mode":"fas_to_tol",
            "cycle":cycle,
            "residual_p":residual_p,
            "target_residual":TARGET_RESIDUAL,
            "reached_target":int(residual_p<=TARGET_RESIDUAL),
            "iteration_seconds":COMM.allreduce(now-iteration_t0,op=MPI.MAX),
            "total_seconds":COMM.allreduce(now-total_t0,op=MPI.MAX),
        })
        if COMM.rank==0:
            print("T4_FAS_TO_TOL_ROW",rows[-1])
        if residual_p<=TARGET_RESIDUAL:
            break

    assert_requested_hierarchy_is_active(fine)
    levels=hierarchy_level_objects(fine)
    observed=[level.heat_source_calls+level.heat_target_calls for level in levels]
    cycles=len(rows)
    expected=[16*cycles,9*cycles] if len(levels)==2 else [16*cycles,10*cycles,9*cycles]
    if observed!=expected:
        raise RuntimeError(f"T4 smoke heat-count failure: observed={observed}, expected={expected}")
    rows[-1].update({"observed_heat_actions":str(observed),"expected_heat_actions":str(expected),"heat_count_match":1})
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_fas_to_tol_trace.csv"),rows)
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_heat_counts.csv"),fine.counter_rows())
    final_science_audit("fas_to_tol",fine,p,q,b_p,b_q,rows[-1],result_dir)
    if COMM.rank==0:
        print("T4_FAS_TO_TOL_FINAL",rows[-1])
        print(f"RESULT_DIR={result_dir}")


def main():
    result_dir=os.environ.get("RESULT_DIR",os.getcwd())
    os.makedirs(result_dir,exist_ok=True)
    COMM.Barrier()
    total_t0=time.perf_counter()

    if RUN_MODE=="benchmark_certificate":
        V0,V1,mu0,mu1,metadata=build_support_spaces_and_measures(return_metadata=True)
        benchmark_certificate(V0,V1,mu0,mu1,metadata,result_dir)
        if COMM.rank==0:
            print(f"RESULT_DIR={result_dir}")
        return

    fine,coarse,V0,V1=build_problem(COARSE_TAU_RATIO)
    hierarchy=hierarchy_rows(fine)
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_hierarchy.csv"),hierarchy)
    p=Function(V0,name="p").assign(0.0)
    q=Function(V1,name="q").assign(0.0)
    b_p=Function(V0,name="b_p_finest").assign(0.0)
    b_q=Function(V1,name="b_q_finest").assign(0.0)

    if COMM.rank==0:
        print(f"EXPERIMENT_NAME={EXPERIMENT_NAME}")
        print(
            "T4_CONFIGURATION",
            f"support_N={SUPPORT_GRID_N}",
            f"mode={OSC_MODE}",
            f"beta={OSC_BETA}",
            f"omega={OSC_OMEGA}",
            f"wavelength={OSC_WAVELENGTH_PHYSICAL}",
            f"heat_attenuation={math.exp(-TAU_FINE_PHYSICAL*OSC_OMEGA**2)}",
        )
        print(f"FAS_CYCLES={FAS_CYCLES} PRE={PRE_SMOOTH} COARSE={COARSE_SMOOTH} POST={POST_SMOOTH} COARSE_TAU_RATIO={COARSE_TAU_RATIO}")
        for row in hierarchy:
            print("FAS_HIERARCHY",row)

    if RUN_MODE=="fas_to_tol":
        run_fas_to_tol(result_dir,fine,p,q,b_p,b_q,total_t0)
        return
    if RUN_MODE=="reference_fas_to_tol":
        run_t4_reference_fas_validation(result_dir,fine,V0,V1,p,q,b_p,b_q,total_t0)
        return
    raise ValueError(f"Unknown T4 FAS_RUN_MODE={RUN_MODE!r}")


if __name__=="__main__":
    main()
