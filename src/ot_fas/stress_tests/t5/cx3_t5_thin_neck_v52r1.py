from firedrake import *
from mpi4py import MPI
import csv
import hashlib
import json
import math
import os
import time
import numpy as np

EXPERIMENT_NAME = "run_t5_thin_neck_v52r1_exploratory"

# T5: connected thin-neck target robustness benchmark.
#
# Scientific design:
#   * source: uniform unit disk in physical coordinates;
#   * target: uniform connected smooth two-lobed support with a narrow neck;
#   * solver core: frozen from the successful T4 v50 implementation;
#   * target regularisation, state spaces, FAS cycle and heat hierarchy are fixed;
#   * no exact unregularised Brenier map is assumed or manufactured;
#   * solver correctness is judged against an independent fine-only B920
#     discrete entropic reference;
#   * post-production diagnostics quantify barycentric-map support escape,
#     conditional spread and left/neck/right lobe mixing.
#
# The target geometry is strongly non-convex but the frozen geometry is NOT
# claimed to satisfy a theorem-level sufficient condition for discontinuity of
# the unregularised Brenier map.  The numerical curvature diagnostic is
# recorded only as benchmark-design context.

SOURCE_RADIUS_PHYSICAL = 1.0
TARGET_L = 1.20
TARGET_S0 = 0.060
TARGET_S1 = 0.80
TARGET_BETA = 0.28
TARGET_P = 4
TARGET_NECK_WIDTH_PHYSICAL = 2.0 * TARGET_S0
TARGET_NECK_XI_CUT = TARGET_BETA * (math.log(2.0) ** (1.0 / TARGET_P))
TARGET_NECK_X_CUT_PHYSICAL = TARGET_L * TARGET_NECK_XI_CUT
TARGET_SUPPORT_MEMBERSHIP_TOL_PHYSICAL = 2.0e-4

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
REF_RESIDUAL_TARGET = float(os.environ.get("FAS_REF_RESIDUAL_TARGET", "2.2e-3"))
REF_CHANGE_REL_TOL = float(os.environ.get("FAS_REF_CHANGE_REL_TOL", "1e-6"))
REF_STABLE_CHECKS = int(os.environ.get("FAS_REF_STABLE_CHECKS", "2"))
REQUIRE_REFERENCE_CERTIFIED = int(os.environ.get("T5_REQUIRE_REFERENCE_CERTIFIED", "1"))

# Frozen asset hashes generated before nonlinear output is observed.
EXPECTED_SOURCE_MESH_SHA256 = "b8f020e59538a42f17feb82824550b1c8daad71d06ecc76612ea8acd5ee9eb81"
EXPECTED_TARGET_MESH_SHA256 = "1269c95b276059f2f4938088820654e58bf4f8a3227ed18a614b5aece6522e56"

# Geometry-certificate thresholds, all fixed before solver output.
CERT_MIN_SIGNED_TRIANGLE_AREA_PHYSICAL = 1.0e-7
CERT_MIN_MESH_ANGLE_DEGREES = 15.0
CERT_MAX_MESH_ASPECT = 5.0
CERT_MIN_NECK_CELLS_ACROSS_BY_MAX_DIAMETER = 10.0
CERT_MIN_BRIDGE_MARGIN_UNIT = 0.15
CERT_MAX_AREA_REL_MISMATCH = 2.0e-8
CERT_MAX_SOURCE_AREA_REL_ERROR_TO_PI = 5.0e-5
CERT_MIN_NECK_OVER_HEAT_LENGTH = 1.0
CERT_MAX_NECK_OVER_HEAT_LENGTH = 2.5

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


def target_scale_physical(xi):
    # TARGET_P is frozen even, so no absolute value is needed.
    return Constant(TARGET_S0) + Constant(TARGET_S1 - TARGET_S0) * (
        Constant(1.0) - exp(-((xi / Constant(TARGET_BETA)) ** TARGET_P))
    )


def target_half_height_physical(x_phys):
    xi = x_phys / Constant(TARGET_L)
    return target_scale_physical(xi) * sqrt(max_value(Constant(1.0) - xi * xi, Constant(0.0)))


def target_membership_indicator_from_physical(z):
    h = target_half_height_physical(z[0])
    tol = Constant(TARGET_SUPPORT_MEMBERSHIP_TOL_PHYSICAL)
    return (
        conditional(ge(z[0], Constant(-TARGET_L)-tol), 1.0, 0.0)
        * conditional(le(z[0], Constant(TARGET_L)+tol), 1.0, 0.0)
        * conditional(ge(z[1], -h-tol), 1.0, 0.0)
        * conditional(le(z[1], h+tol), 1.0, 0.0)
    )


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _asset_root():
    return os.environ.get(
        "T5_ASSET_ROOT",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "t5_assets_v52"),
    )


def _asset_paths_and_metadata():
    root = _asset_root()
    source_path = os.path.join(root, "t5_source_disk_v52.msh")
    target_path = os.path.join(root, "t5_target_thin_neck_v52.msh")
    metadata_path = os.path.join(root, "t5_mesh_assets_v52_metadata.json")
    payload = None
    if COMM.rank == 0:
        for path in (source_path, target_path, metadata_path):
            if not os.path.isfile(path):
                raise RuntimeError(f"Missing frozen T5 asset: {path}")
        source_sha = _sha256_file(source_path)
        target_sha = _sha256_file(target_path)
        if source_sha != EXPECTED_SOURCE_MESH_SHA256:
            raise RuntimeError(f"T5 source mesh SHA mismatch: {source_sha} != {EXPECTED_SOURCE_MESH_SHA256}")
        if target_sha != EXPECTED_TARGET_MESH_SHA256:
            raise RuntimeError(f"T5 target mesh SHA mismatch: {target_sha} != {EXPECTED_TARGET_MESH_SHA256}")
        with open(metadata_path) as handle:
            metadata = json.load(handle)
        if metadata.get("source_mesh_sha256") != source_sha or metadata.get("target_mesh_sha256") != target_sha:
            raise RuntimeError("T5 mesh metadata hashes disagree with frozen assets")
        payload = (source_path, target_path, metadata_path, metadata)
    payload = COMM.bcast(payload, root=0)
    COMM.Barrier()
    return payload


def _read_gmsh_v2_triangles_physical(path):
    nodes = {}
    triangles = []
    with open(path) as handle:
        lines = iter(handle)
        for raw in lines:
            tag = raw.strip()
            if tag == "$Nodes":
                n = int(next(lines).strip())
                for _ in range(n):
                    parts = next(lines).split()
                    node_id = int(parts[0])
                    xu, yu = float(parts[1]), float(parts[2])
                    nodes[node_id] = (
                        BRIDGE_WIDTH_PHYSICAL * (xu - 0.5),
                        BRIDGE_WIDTH_PHYSICAL * (yu - 0.5),
                    )
                end = next(lines).strip()
                if end != "$EndNodes":
                    raise RuntimeError(f"Malformed Gmsh node section in {path}")
            elif tag == "$Elements":
                m = int(next(lines).strip())
                for _ in range(m):
                    parts = next(lines).split()
                    element_type = int(parts[1])
                    ntags = int(parts[2])
                    if element_type == 2:
                        tri = tuple(int(v) for v in parts[3 + ntags: 6 + ntags])
                        triangles.append(tri)
                end = next(lines).strip()
                if end != "$EndElements":
                    raise RuntimeError(f"Malformed Gmsh element section in {path}")
    if not nodes or not triangles:
        raise RuntimeError(f"No triangle mesh found in {path}")
    ids = sorted(nodes)
    index = {node_id: i for i, node_id in enumerate(ids)}
    points = np.asarray([nodes[node_id] for node_id in ids], dtype=float)
    tri = np.asarray([[index[a], index[b], index[c]] for a, b, c in triangles], dtype=np.int64)
    return points, tri


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _triangle_components(triangles):
    parent = np.arange(len(triangles), dtype=np.int64)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    edge_owner = {}
    edge_counts = {}
    for i, tri in enumerate(triangles):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = tuple(sorted((int(a), int(b))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            if edge in edge_owner:
                union(i, edge_owner[edge])
            else:
                edge_owner[edge] = i
    components = len({find(i) for i in range(len(triangles))})
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    return components, boundary_edges


def _mesh_asset_stats(path, is_target=False):
    points, triangles = _read_gmsh_v2_triangles_physical(path)
    xy = points[triangles]
    signed = 0.5 * _cross2(xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0])
    l01 = np.linalg.norm(xy[:, 1] - xy[:, 0], axis=1)
    l12 = np.linalg.norm(xy[:, 2] - xy[:, 1], axis=1)
    l20 = np.linalg.norm(xy[:, 0] - xy[:, 2], axis=1)
    lengths = np.stack((l01, l12, l20), axis=1)
    max_edge = lengths.max(axis=1)
    min_alt = 2.0 * signed / np.maximum(max_edge, 1.0e-300)
    aspect = max_edge / np.maximum(min_alt, 1.0e-300)
    angles = []
    for A, B, C in ((l01, l20, l12), (l01, l12, l20), (l12, l20, l01)):
        cosine = (A*A + B*B - C*C) / np.maximum(2.0*A*B, 1.0e-300)
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    min_angle = np.stack(angles, axis=1).min(axis=1)
    components, boundary_edges = _triangle_components(triangles)
    bbox = (float(points[:,0].min()), float(points[:,0].max()), float(points[:,1].min()), float(points[:,1].max()))
    row = {
        "parsed_vertex_count": int(len(points)),
        "parsed_triangle_count": int(len(triangles)),
        "parsed_connected_triangle_components": int(components),
        "parsed_boundary_edge_count": int(len(boundary_edges)),
        "parsed_area_physical": float(signed.sum()),
        "parsed_min_signed_triangle_area_physical": float(signed.min()),
        "parsed_max_signed_triangle_area_physical": float(signed.max()),
        "parsed_min_angle_degrees": float(min_angle.min()),
        "parsed_max_aspect_edge_over_altitude": float(aspect.max()),
        "parsed_min_edge_physical": float(lengths.min()),
        "parsed_max_edge_physical": float(lengths.max()),
        "parsed_bbox_xmin_physical": bbox[0],
        "parsed_bbox_xmax_physical": bbox[1],
        "parsed_bbox_ymin_physical": bbox[2],
        "parsed_bbox_ymax_physical": bbox[3],
    }
    if is_target:
        centroids = xy.mean(axis=1)
        neck_mask = np.abs(centroids[:,0]) <= TARGET_NECK_X_CUT_PHYSICAL
        row.update({
            "parsed_neck_area_fraction": float(signed[neck_mask].sum()/signed.sum()),
            "parsed_max_triangle_diameter_in_neck_physical": float(max_edge[neck_mask].max()),
            "parsed_neck_width_over_max_neck_triangle_diameter": float(TARGET_NECK_WIDTH_PHYSICAL/max_edge[neck_mask].max()),
        })
        boundary_ids = sorted(set(v for edge in boundary_edges for v in edge))
        boundary_points = points[boundary_ids]
        central = boundary_points[np.abs(boundary_points[:,0]) <= 0.02]
        top = central[central[:,1] > 0.0, 1]
        bottom = central[central[:,1] < 0.0, 1]
        measured_width = float(top.min() - bottom.max()) if len(top) and len(bottom) else float("nan")
        row["parsed_measured_neck_width_physical"] = measured_width
    return row


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


def _distributed_mesh_metrics(mesh, prefix):
    # CellSize is a Firedrake-supported UFL cell metric.  The extrema are
    # reduced globally rather than reported from one MPI rank.  Exact signed
    # areas and quality extrema are independently certified by parsing the
    # frozen mesh asset on rank 0.
    Vdg = FunctionSpace(mesh, "DG", 0)
    cell_size = Function(Vdg, name=f"{prefix}_cell_size_unit").interpolate(CellSize(mesh))
    cells = int(COMM.allreduce(mesh.cell_set.size, op=MPI.SUM))
    return {
        f"{prefix}_global_cells_firedrake": cells,
        f"{prefix}_cell_size_min_physical": global_dof_min(cell_size) * BRIDGE_WIDTH_PHYSICAL,
        f"{prefix}_cell_size_max_physical": global_dof_max(cell_size) * BRIDGE_WIDTH_PHYSICAL,
    }


def build_support_spaces_and_measures(return_metadata=False):
    source_path, target_path, metadata_path, asset_metadata = _asset_paths_and_metadata()
    mesh0 = Mesh(source_path)
    mesh1 = Mesh(target_path)
    V0 = FunctionSpace(mesh0, "CG", DISK_CG_DEGREE)
    V1 = FunctionSpace(mesh1, "CG", DISK_CG_DEGREE)
    mu0 = Function(V0, name="mu0").interpolate(1.0)
    mu1 = Function(V1, name="mu1").interpolate(1.0)
    source_raw = float(assemble(mu0 * dx(domain=mesh0)))
    target_raw = float(assemble(mu1 * dx(domain=mesh1)))
    if source_raw <= 0.0 or target_raw <= 0.0:
        raise RuntimeError(f"T5 invalid raw support masses: source={source_raw}, target={target_raw}")
    mu0.assign(mu0/source_raw)
    mu1.assign(mu1/target_raw)
    metadata = {
        "source_mesh_path": source_path,
        "target_mesh_path": target_path,
        "metadata_path": metadata_path,
        "source_mesh_sha256": _sha256_file(source_path),
        "target_mesh_sha256": _sha256_file(target_path),
        "source_area_unit_firedrake": source_raw,
        "target_area_unit_firedrake": target_raw,
        "source_area_firedrake_physical": float(assemble(Constant(1.0)*dx(domain=mesh0))) * BRIDGE_WIDTH_PHYSICAL**2,
        "target_area_firedrake_physical": float(assemble(Constant(1.0)*dx(domain=mesh1))) * BRIDGE_WIDTH_PHYSICAL**2,
        "asset_metadata": asset_metadata,
    }
    metadata.update(_distributed_mesh_metrics(mesh0, "source"))
    metadata.update(_distributed_mesh_metrics(mesh1, "target"))
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


def benchmark_certificate(V0, V1, mu0, mu1, metadata, result_dir):
    mesh0, mesh1 = V0.mesh(), V1.mesh()
    source_bbox = _global_mesh_bbox(mesh0)
    target_bbox = _global_mesh_bbox(mesh1)
    bridge_margin = min(
        source_bbox[0], 1.0-source_bbox[1], source_bbox[2], 1.0-source_bbox[3],
        target_bbox[0], 1.0-target_bbox[1], target_bbox[2], 1.0-target_bbox[3],
    )
    parsed_source = parsed_target = None
    if COMM.rank == 0:
        parsed_source = _mesh_asset_stats(metadata["source_mesh_path"], is_target=False)
        parsed_target = _mesh_asset_stats(metadata["target_mesh_path"], is_target=True)
    parsed_source = COMM.bcast(parsed_source, root=0)
    parsed_target = COMM.bcast(parsed_target, root=0)

    source_mass = float(assemble(mu0*dx(domain=mesh0)))
    target_mass = float(assemble(mu1*dx(domain=mesh1)))
    heat_length = math.sqrt(2.0*TAU_FINE_PHYSICAL)
    neck_over_heat = TARGET_NECK_WIDTH_PHYSICAL/heat_length
    asset_meta = metadata["asset_metadata"]
    curvature_min = float(asset_meta["target"]["minimum_connected_arc_curvature_integral_numerical"])
    source_area_phys = metadata["source_area_firedrake_physical"]
    target_area_phys = metadata["target_area_firedrake_physical"]

    row = {
        "benchmark":"T5_connected_thin_neck",
        "source_radius_physical":SOURCE_RADIUS_PHYSICAL,
        "target_L":TARGET_L,
        "target_s0":TARGET_S0,
        "target_s1":TARGET_S1,
        "target_beta":TARGET_BETA,
        "target_p":TARGET_P,
        "neck_width_physical":TARGET_NECK_WIDTH_PHYSICAL,
        "neck_partition_x_cut_physical":TARGET_NECK_X_CUT_PHYSICAL,
        "heat_length_sqrt_2tau_physical":heat_length,
        "neck_width_over_sqrt_2tau":neck_over_heat,
        "source_mass":source_mass,
        "target_mass":target_mass,
        "source_density_min":global_dof_min(mu0),
        "source_density_max":global_dof_max(mu0),
        "target_density_min":global_dof_min(mu1),
        "target_density_max":global_dof_max(mu1),
        "source_density_nonfinite_count":global_nonfinite_count(mu0),
        "target_density_nonfinite_count":global_nonfinite_count(mu1),
        "source_bbox_unit":str(source_bbox),
        "target_bbox_unit":str(target_bbox),
        "bridge_margin_unit":bridge_margin,
        "source_area_firedrake_physical":source_area_phys,
        "target_area_firedrake_physical":target_area_phys,
        "curvature_min_connected_arc_numerical":curvature_min,
        "minus_pi":-math.pi,
        "theorem_discontinuity_threshold_crossed":int(curvature_min < -math.pi),
        "theorem_discontinuity_claimed":0,
    }
    for key, value in metadata.items():
        if key != "asset_metadata":
            row[key] = value
    row.update({f"source_{k}":v for k,v in parsed_source.items()})
    row.update({f"target_{k}":v for k,v in parsed_target.items()})

    source_area_rel = abs(source_area_phys-parsed_source["parsed_area_physical"])/max(parsed_source["parsed_area_physical"],1e-300)
    target_area_rel = abs(target_area_phys-parsed_target["parsed_area_physical"])/max(parsed_target["parsed_area_physical"],1e-300)
    source_pi_rel = abs(parsed_source["parsed_area_physical"]-math.pi)/math.pi
    measured_neck = parsed_target["parsed_measured_neck_width_physical"]
    checks = {
        "asset_hashes": metadata["source_mesh_sha256"]==EXPECTED_SOURCE_MESH_SHA256 and metadata["target_mesh_sha256"]==EXPECTED_TARGET_MESH_SHA256,
        "source_connected": parsed_source["parsed_connected_triangle_components"]==1,
        "target_connected": parsed_target["parsed_connected_triangle_components"]==1,
        "source_orientation": parsed_source["parsed_min_signed_triangle_area_physical"]>=CERT_MIN_SIGNED_TRIANGLE_AREA_PHYSICAL,
        "target_orientation": parsed_target["parsed_min_signed_triangle_area_physical"]>=CERT_MIN_SIGNED_TRIANGLE_AREA_PHYSICAL,
        "source_quality_angle": parsed_source["parsed_min_angle_degrees"]>=CERT_MIN_MESH_ANGLE_DEGREES,
        "target_quality_angle": parsed_target["parsed_min_angle_degrees"]>=CERT_MIN_MESH_ANGLE_DEGREES,
        "source_quality_aspect": parsed_source["parsed_max_aspect_edge_over_altitude"]<=CERT_MAX_MESH_ASPECT,
        "target_quality_aspect": parsed_target["parsed_max_aspect_edge_over_altitude"]<=CERT_MAX_MESH_ASPECT,
        "source_cell_count_global": metadata["source_global_cells_firedrake"]==parsed_source["parsed_triangle_count"],
        "target_cell_count_global": metadata["target_global_cells_firedrake"]==parsed_target["parsed_triangle_count"],
        "source_area_consistency": source_area_rel<=CERT_MAX_AREA_REL_MISMATCH,
        "target_area_consistency": target_area_rel<=CERT_MAX_AREA_REL_MISMATCH,
        "source_disk_area": source_pi_rel<=CERT_MAX_SOURCE_AREA_REL_ERROR_TO_PI,
        "unit_masses": abs(source_mass-1.0)<=1e-12 and abs(target_mass-1.0)<=1e-12,
        "densities_positive": global_dof_min(mu0)>0.0 and global_dof_min(mu1)>0.0,
        "densities_finite": global_nonfinite_count(mu0)==0 and global_nonfinite_count(mu1)==0,
        "bridge_margin": bridge_margin>=CERT_MIN_BRIDGE_MARGIN_UNIT,
        "neck_resolution": parsed_target["parsed_neck_width_over_max_neck_triangle_diameter"]>=CERT_MIN_NECK_CELLS_ACROSS_BY_MAX_DIAMETER,
        "neck_width_asset_match": math.isfinite(measured_neck) and abs(measured_neck-TARGET_NECK_WIDTH_PHYSICAL)<=0.01*TARGET_NECK_WIDTH_PHYSICAL,
        "neck_entropic_stress_scale": CERT_MIN_NECK_OVER_HEAT_LENGTH<=neck_over_heat<=CERT_MAX_NECK_OVER_HEAT_LENGTH,
        "target_not_theorem_certified_by_curvature_scan": curvature_min>=-math.pi,
    }
    row["source_area_relative_mismatch_parser_vs_firedrake"] = source_area_rel
    row["target_area_relative_mismatch_parser_vs_firedrake"] = target_area_rel
    row["source_area_relative_error_to_pi"] = source_pi_rel
    row.update({f"check_{k}":int(v) for k,v in checks.items()})
    row["certificate_pass"] = int(all(checks.values()))

    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_benchmark_certificate.csv"),[row])
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_mesh_metadata.csv"),[row])
    if COMM.rank==0:
        print("T5_BENCHMARK_CERTIFICATE",row)
    if not all(checks.values()):
        failed=[k for k,v in checks.items() if not v]
        raise RuntimeError(f"T5 benchmark certificate failed: {failed}; row={row}")
    return row


def _reconstruct_t5_conditional_fields(fine, q, prefix):
    V0,V1=fine.V0,fine.V1
    y=SpatialCoordinate(V1.mesh())
    yp=unit_to_physical_vector(y)
    q_shift=global_dof_max(q)
    q_shifted=Function(V1,name=f"{prefix}_q_shifted")
    exp_q=Function(V1,name=f"{prefix}_exp_q_shifted")
    q_shifted.interpolate(q-Constant(q_shift))
    exp_q.interpolate(exp(q_shifted))

    Hden=Function(V0,name=f"{prefix}_Hden")
    fine.heat_scaling_to_source(exp_q,Hden)
    y0w=Function(V1,name=f"{prefix}_y0w").interpolate(y[0]*exp_q)
    y1w=Function(V1,name=f"{prefix}_y1w").interpolate(y[1]*exp_q)
    ysqw=Function(V1,name=f"{prefix}_ysqw").interpolate((y[0]**2+y[1]**2)*exp_q)
    H0=Function(V0,name=f"{prefix}_H0"); H1=Function(V0,name=f"{prefix}_H1"); Hsq=Function(V0,name=f"{prefix}_Hsq")
    fine.heat_scaling_to_source(y0w,H0); fine.heat_scaling_to_source(y1w,H1); fine.heat_scaling_to_source(ysqw,Hsq)
    T0=Function(V0,name=f"{prefix}_T0_unit").interpolate(H0/max_value(Hden,TINY))
    T1=Function(V0,name=f"{prefix}_T1_unit").interpolate(H1/max_value(Hden,TINY))
    Eysq=Function(V0,name=f"{prefix}_Eysq_unit").interpolate(Hsq/max_value(Hden,TINY))
    variance=Function(V0,name=f"{prefix}_conditional_variance_unit").interpolate(max_value(Eysq-(T0*T0+T1*T1),Constant(0.0)))

    left=Function(V1,name=f"{prefix}_left_indicator")
    right=Function(V1,name=f"{prefix}_right_indicator")
    neck=Function(V1,name=f"{prefix}_neck_indicator")
    left.interpolate(conditional(lt(yp[0],Constant(-TARGET_NECK_X_CUT_PHYSICAL)),1.0,0.0))
    right.interpolate(conditional(gt(yp[0],Constant(TARGET_NECK_X_CUT_PHYSICAL)),1.0,0.0))
    neck.interpolate(Constant(1.0)-left-right)
    lw=Function(V1,name=f"{prefix}_left_weight").interpolate(left*exp_q)
    rw=Function(V1,name=f"{prefix}_right_weight").interpolate(right*exp_q)
    Hleft=Function(V0,name=f"{prefix}_Hleft"); Hright=Function(V0,name=f"{prefix}_Hright")
    fine.heat_scaling_to_source(lw,Hleft); fine.heat_scaling_to_source(rw,Hright)
    pL=Function(V0,name=f"{prefix}_p_left").interpolate(Hleft/max_value(Hden,TINY))
    pR=Function(V0,name=f"{prefix}_p_right").interpolate(Hright/max_value(Hden,TINY))
    pN=Function(V0,name=f"{prefix}_p_neck").interpolate(Constant(1.0)-pL-pR)
    mix=Function(V0,name=f"{prefix}_lobe_mixing").interpolate(Constant(4.0)*pL*pR)
    side_mix=Function(V0,name=f"{prefix}_side_normalized_lobe_mixing").interpolate(
        Constant(4.0)*pL*pR/max_value((pL+pR)*(pL+pR),Constant(1.0e-30))
    )
    return {
        "q_shift":q_shift,"Hden":Hden,"T0":T0,"T1":T1,"variance":variance,
        "pL":pL,"pN":pN,"pR":pR,"mix":mix,"side_mix":side_mix,
        "target_left_indicator":left,"target_neck_indicator":neck,"target_right_indicator":right,
    }


def _conditional_summary(fine, fields, prefix):
    V0=fine.V0
    x=SpatialCoordinate(V0.mesh())
    xp=unit_to_physical_vector(x)
    T_unit=as_vector([fields["T0"],fields["T1"]])
    T_phys=unit_to_physical_vector(T_unit)
    width=BRIDGE_WIDTH_PHYSICAL
    variance=fields["variance"]
    monge_unit=float(assemble(variance*fine.mu0*dx))
    inside=target_membership_indicator_from_physical(T_phys)
    escape=float(assemble((Constant(1.0)-inside)*fine.mu0*dx))
    mapped_neck=(
        conditional(ge(T_phys[0],Constant(-TARGET_NECK_X_CUT_PHYSICAL)),1.0,0.0)
        * conditional(le(T_phys[0],Constant(TARGET_NECK_X_CUT_PHYSICAL)),1.0,0.0)
        * inside
    )
    central_slab=(
        conditional(ge(T_phys[0],Constant(-TARGET_NECK_X_CUT_PHYSICAL)),1.0,0.0)
        * conditional(le(T_phys[0],Constant(TARGET_NECK_X_CUT_PHYSICAL)),1.0,0.0)
    )
    Vdg=FunctionSpace(V0.mesh(),"DG",0)
    detmap=Function(Vdg,name=f"{prefix}_det_DT").interpolate(det(grad(T_unit)))
    orientation_unit=float(assemble(conditional(le(det(grad(T_unit)),0.0),1.0,0.0)*dx(domain=V0.mesh())))
    mix=fields["mix"]; side_mix=fields["side_mix"]
    pL,pN,pR=fields["pL"],fields["pN"],fields["pR"]
    sumerr=Function(V0,name=f"{prefix}_partition_sum_error").interpolate(pL+pN+pR-Constant(1.0))
    return {
        f"{prefix}_q_shift":fields["q_shift"],
        f"{prefix}_map_denominator_min":global_dof_min(fields["Hden"]),
        f"{prefix}_map_denominator_max":global_dof_max(fields["Hden"]),
        f"{prefix}_map_denominator_nonfinite_count":global_nonfinite_count(fields["Hden"]),
        f"{prefix}_monge_defect_unit":monge_unit,
        f"{prefix}_monge_defect_physical":width*width*monge_unit,
        f"{prefix}_rms_conditional_spread_physical":width*math.sqrt(max(monge_unit,0.0)),
        f"{prefix}_mean_left_probability":float(assemble(pL*fine.mu0*dx)),
        f"{prefix}_mean_neck_probability":float(assemble(pN*fine.mu0*dx)),
        f"{prefix}_mean_right_probability":float(assemble(pR*fine.mu0*dx)),
        f"{prefix}_lobe_mixing_mean":float(assemble(mix*fine.mu0*dx)),
        f"{prefix}_side_normalized_lobe_mixing_mean":float(assemble(side_mix*fine.mu0*dx)),
        f"{prefix}_lobe_mixing_source_mass_gt_0p25":float(assemble(conditional(gt(mix,0.25),1.0,0.0)*fine.mu0*dx)),
        f"{prefix}_lobe_mixing_source_mass_gt_0p50":float(assemble(conditional(gt(mix,0.50),1.0,0.0)*fine.mu0*dx)),
        f"{prefix}_left_right_ambiguous_source_mass":float(assemble(conditional(gt(pL,0.1),1.0,0.0)*conditional(gt(pR,0.1),1.0,0.0)*fine.mu0*dx)),
        f"{prefix}_partition_sum_linf_error":linf_norm(sumerr),
        f"{prefix}_barycentric_support_escape_mass":escape,
        f"{prefix}_barycentric_mapped_into_neck_mass":float(assemble(mapped_neck*fine.mu0*dx)),
        f"{prefix}_barycentric_central_slab_mass":float(assemble(central_slab*fine.mu0*dx)),
        f"{prefix}_det_min":global_dof_min(detmap),
        f"{prefix}_det_max":global_dof_max(detmap),
        f"{prefix}_orientation_reversing_area_unit":orientation_unit,
        f"{prefix}_orientation_reversing_area_physical":orientation_unit*width*width,
    }


def t5_map_and_lobe_audit(fine,q,result_dir,method,q_ref=None):
    fas_fields=_reconstruct_t5_conditional_fields(fine,q,f"{method}_fas")
    row=_conditional_summary(fine,fas_fields,"fas")
    target_neck_mass=float(assemble(fas_fields["target_neck_indicator"]*fine.mu1*dx(domain=fine.V1.mesh())))
    target_left_mass=float(assemble(fas_fields["target_left_indicator"]*fine.mu1*dx(domain=fine.V1.mesh())))
    target_right_mass=float(assemble(fas_fields["target_right_indicator"]*fine.mu1*dx(domain=fine.V1.mesh())))
    row.update({
        "target_left_mass_partition":target_left_mass,
        "target_neck_mass_partition":target_neck_mass,
        "target_right_mass_partition":target_right_mass,
        "fas_conditional_neck_mass_balance_error":abs(row["fas_mean_neck_probability"]-target_neck_mass),
    })
    ref_fields=None
    if q_ref is not None:
        ref_fields=_reconstruct_t5_conditional_fields(fine,q_ref,f"{method}_ref")
        row.update(_conditional_summary(fine,ref_fields,"ref"))
        Tfas=unit_to_physical_vector(as_vector([fas_fields["T0"],fas_fields["T1"]]))
        Tref=unit_to_physical_vector(as_vector([ref_fields["T0"],ref_fields["T1"]]))
        map_diff_sq=(Tfas[0]-Tref[0])**2+(Tfas[1]-Tref[1])**2
        row["fas_vs_reference_barycentric_map_l2_physical"]=math.sqrt(max(float(assemble(map_diff_sq*fine.mu0*dx)),0.0))
        prob_diff_sq=(fas_fields["pL"]-ref_fields["pL"])**2+(fas_fields["pN"]-ref_fields["pN"])**2+(fas_fields["pR"]-ref_fields["pR"])**2
        row["fas_vs_reference_lobe_probability_l2"]=math.sqrt(max(float(assemble(prob_diff_sq*fine.mu0*dx)),0.0))
        row["ref_conditional_neck_mass_balance_error"]=abs(row["ref_mean_neck_probability"]-target_neck_mass)

    if AUDIT_WRITE_VTK:
        Vvec=VectorFunctionSpace(fine.V0.mesh(),"CG",DISK_CG_DEGREE)
        Tfasvec=Function(Vvec,name=f"{method}_Tfas_unit").interpolate(as_vector([fas_fields["T0"],fas_fields["T1"]]))
        outputs=[Tfasvec,fas_fields["variance"],fas_fields["pL"],fas_fields["pN"],fas_fields["pR"],fas_fields["mix"]]
        if ref_fields is not None:
            Trefvec=Function(Vvec,name=f"{method}_Tref_unit").interpolate(as_vector([ref_fields["T0"],ref_fields["T1"]]))
            outputs.append(Trefvec)
        VTKFile(os.path.join(result_dir,f"{EXPERIMENT_NAME}_{method}_t5_map_lobes.pvd")).write(*outputs)
    if COMM.rank==0:
        print("T5_MAP_LOBE_AUDIT",row)
    return row


def final_science_audit(method,fine,p,q,b_p,b_q,performance_row,result_dir,q_ref=None):
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
    row.update(fine.field_stats(p,"p")); row.update(fine.field_stats(q,"q"))
    row.update(t5_map_and_lobe_audit(fine,q,result_dir,method,q_ref=q_ref))
    for name in ["log_marginal_p","log_marginal_q","marginal_p","marginal_q","logHq_on_0","logHp_on_1","Hq_shifted_on_0","Hp_shifted_on_1"]:
        field=getattr(fine,name,None)
        if field is not None:
            row.update(fine.field_stats(field,name))
    row.update(fine.heat_stats(fine.Hq_shifted_on_0,"Hq_shifted_heat"))
    row.update(fine.heat_stats(fine.Hp_shifted_on_1,"Hp_shifted_heat"))
    row["audit_seconds"]=COMM.allreduce(time.perf_counter()-audit_t0,op=MPI.MAX)
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_{method}_science_audit.csv"),[row])
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_{method}_audit_heat_counts.csv"),fine.counter_rows())
    if COMM.rank==0: print("FAS_SCIENCE_AUDIT",row)
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



def reference_norm(level,p,q):
    return math.sqrt(max(
        float(assemble(p*p*level.mu0*dx))
        +float(assemble(q*q*level.mu1*dx)),0.0))



def run_t5_reference_fas_validation(result_dir,fine,V0,V1,p,q,b_p,b_q,total_t0):
    # Phase A: independent fine-only B920 reference.  No FAS correction enters.
    p.assign(0.0); q.assign(0.0); b_p.assign(0.0); b_q.assign(0.0)
    fine.reset_counters()
    ref_rows=[]
    previous_p=Function(V0,name="t5_reference_previous_p").assign(p)
    previous_q=Function(V1,name="t5_reference_previous_q").assign(q)
    change_p=Function(V0,name="t5_reference_change_p")
    change_q=Function(V1,name="t5_reference_change_q")
    dp_raw=Function(V0,name="t5_reference_dp_raw")
    dq_raw=Function(V1,name="t5_reference_dq_raw")
    trace_path=os.path.join(result_dir,f"{EXPERIMENT_NAME}_reference_trace.csv")
    previous_change_abs=float("nan")
    stable_checks=0; reference_certified=False; updates=0
    ref_t0=time.perf_counter()
    while updates<REF_MAX_FINE_UPDATES:
        block=min(REF_CHECK_BLOCK,REF_MAX_FINE_UPDATES-updates)
        fine.smoother(p,q,b_p,b_q,block,f"t5_reference_block_{updates+block}")
        fine.gauge_fix(p,q)
        updates+=block
        residual_p,residual_q=fine.residual_norms(p,q,b_p,b_q)
        dp_raw.interpolate(p-previous_p); dq_raw.interpolate(q-previous_q)
        a=float(assemble(dp_raw*fine.mu0*dx)); b=float(assemble(dq_raw*fine.mu1*dx)); change_shift=0.5*(a-b)
        change_p.interpolate(p-previous_p-Constant(change_shift)); change_q.interpolate(q-previous_q+Constant(change_shift))
        change_p_norm=math.sqrt(max(float(assemble(change_p*change_p*fine.mu0*dx)),0.0))
        change_q_norm=math.sqrt(max(float(assemble(change_q*change_q*fine.mu1*dx)),0.0))
        change_abs=math.sqrt(change_p_norm**2+change_q_norm**2)
        norm_now=reference_norm(fine,p,q); change_rel=change_abs/max(norm_now,1e-300)
        change_contraction=change_abs/previous_change_abs if math.isfinite(previous_change_abs) and previous_change_abs>0 else float("nan")
        tail_est=float("nan")
        if math.isfinite(change_contraction) and 0.0<=change_contraction<1.0:
            tail_est=change_contraction/max(1.0-change_contraction,1e-30)*change_abs
        criteria=(updates>=REF_MIN_FINE_UPDATES and residual_p<=REF_RESIDUAL_TARGET and change_rel<=REF_CHANGE_REL_TOL)
        stable_checks=stable_checks+1 if criteria else 0; reference_certified=stable_checks>=REF_STABLE_CHECKS
        ref_rows.append({
            "fine_bridge_n":fine.bridge_n,"updates":updates,"residual_p":residual_p,"residual_q":residual_q,
            "change_p":change_p_norm,"change_q":change_q_norm,"change_abs":change_abs,"change_rel":change_rel,
            "change_contraction":change_contraction,"estimated_remaining_tail":tail_est,"change_gauge_shift":change_shift,
            "stable_checks":stable_checks,"reference_certified":int(reference_certified),
        })
        write_rows(trace_path,ref_rows)
        previous_p.assign(p); previous_q.assign(q); previous_change_abs=change_abs
        if COMM.rank==0: print("T5_REFERENCE_ROW",ref_rows[-1])
        if reference_certified: break

    reference_seconds=COMM.allreduce(time.perf_counter()-ref_t0,op=MPI.MAX)
    write_rows(trace_path,ref_rows)
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_reference_heat_counts.csv"),fine.counter_rows())
    if not reference_certified and REQUIRE_REFERENCE_CERTIFIED:
        raise RuntimeError(f"T5 B920 reference failed preregistered certification after {updates} fine-only updates; no production FAS run performed")
    if COMM.rank==0 and not reference_certified:
        print("T5_REFERENCE_WARNING reference not formally certified; subsequent reference comparisons are provisional.")
    p_ref=Function(V0,name="t5_p_ref").assign(p); q_ref=Function(V1,name="t5_q_ref").assign(q)

    # Phase B: single frozen cold FAS attempt.
    p.assign(0.0); q.assign(0.0); b_p.assign(0.0); b_q.assign(0.0); fine.reset_counters()
    initial_residual=fine.p_residual_norm(p,q,b_p)
    _,_,ep0,eq0,error0,shift0=gauge_aligned_fields(fine,p,q,p_ref,q_ref)
    trajectory=[{
        "cycle":0,"residual_p":initial_residual,"residual_ratio_to_initial":1.0,"residual_contraction_from_previous":"",
        "p_error_to_reference":ep0,"q_error_to_reference":eq0,"combined_error_to_reference":error0,"error_ratio_to_initial":1.0,
        "error_contraction_from_previous":"","gauge_alignment_shift":shift0,"target_residual":TARGET_RESIDUAL,
        "reached_target":int(initial_residual<=TARGET_RESIDUAL),
    }]
    fine.reset_counters(); production_t0=time.perf_counter(); previous_residual=initial_residual; previous_error=error0
    for cycle in range(1,FAS_CYCLES+1):
        fine.fas_cycle_perf(p,q,b_p,b_q)
        residual_p=fine.p_residual_norm(p,q,b_p)
        _,_,ep,eq,error,shift=gauge_aligned_fields(fine,p,q,p_ref,q_ref)
        now=time.perf_counter()
        trajectory.append({
            "cycle":cycle,"residual_p":residual_p,"residual_ratio_to_initial":residual_p/initial_residual,
            "residual_contraction_from_previous":residual_p/previous_residual,"p_error_to_reference":ep,"q_error_to_reference":eq,
            "combined_error_to_reference":error,"error_ratio_to_initial":error/error0,"error_contraction_from_previous":error/previous_error,
            "gauge_alignment_shift":shift,"target_residual":TARGET_RESIDUAL,"reached_target":int(residual_p<=TARGET_RESIDUAL),
            "iteration_seconds":COMM.allreduce(now-production_t0,op=MPI.MAX),
        })
        previous_residual=residual_p; previous_error=error
        if COMM.rank==0: print("T5_REFERENCE_FAS_ROW",trajectory[-1])
        if residual_p<=TARGET_RESIDUAL: break

    cycles=len(trajectory)-1; assert_requested_hierarchy_is_active(fine)
    levels=hierarchy_level_objects(fine); observed=[level.heat_source_calls+level.heat_target_calls for level in levels]
    expected=[16*cycles,10*cycles,9*cycles]
    if observed!=expected:
        raise RuntimeError(f"T5 production heat-count failure: observed={observed}, expected={expected}")
    production_seconds=COMM.allreduce(time.perf_counter()-production_t0,op=MPI.MAX)
    final_error=trajectory[-1]["combined_error_to_reference"]; last_ref=ref_rows[-1]; tail_est=last_ref.get("estimated_remaining_tail",float("nan"))
    summary={
        "cycles":cycles,"reached_target":trajectory[-1]["reached_target"],"initial_residual_p":initial_residual,
        "final_residual_p":trajectory[-1]["residual_p"],"initial_discrete_reference_error":error0,"final_discrete_reference_error":final_error,
        "geomean_residual_contraction":(trajectory[-1]["residual_p"]/initial_residual)**(1.0/max(cycles,1)),
        "geomean_discrete_error_contraction":(final_error/error0)**(1.0/max(cycles,1)),
        "reference_updates":updates,"reference_residual_p":last_ref["residual_p"],"reference_residual_q":last_ref["residual_q"],
        "reference_change_abs":last_ref["change_abs"],"reference_change_rel":last_ref["change_rel"],
        "reference_change_contraction":last_ref["change_contraction"],"reference_estimated_remaining_tail":tail_est,
        "reference_certified":int(reference_certified),"reference_uncertainty_over_final_error":last_ref["change_abs"]/max(final_error,1e-300),
        "reference_tail_over_final_error":tail_est/max(final_error,1e-300) if math.isfinite(tail_est) else float("nan"),
        "reference_seconds":reference_seconds,"production_iteration_seconds":production_seconds,
        "job_seconds_before_final_audit":COMM.allreduce(time.perf_counter()-total_t0,op=MPI.MAX),
        "observed_heat_actions":str(observed),"expected_heat_actions":str(expected),"heat_count_match":1,
    }
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_fas_discrete_error_trajectory.csv"),trajectory)
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_production_heat_counts.csv"),fine.counter_rows())
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_reference_fas_summary.csv"),[summary])
    perf_row={"cycle":cycles,"residual_p":trajectory[-1]["residual_p"],"reached_target":trajectory[-1]["reached_target"],"iteration_seconds":production_seconds,"total_seconds":production_seconds}
    science_row=final_science_audit("reference_fas_to_tol",fine,p,q,b_p,b_q,perf_row,result_dir,q_ref=q_ref)
    for key in [
        "fas_vs_reference_barycentric_map_l2_physical","fas_vs_reference_lobe_probability_l2",
        "fas_barycentric_support_escape_mass","ref_barycentric_support_escape_mass",
        "fas_rms_conditional_spread_physical","ref_rms_conditional_spread_physical",
        "fas_lobe_mixing_mean","ref_lobe_mixing_mean","fas_det_min","ref_det_min",
        "fas_orientation_reversing_area_physical","ref_orientation_reversing_area_physical",
        "target_neck_mass_partition",
    ]:
        summary[key]=science_row.get(key,"")
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_reference_fas_summary_with_map.csv"),[summary])
    if COMM.rank==0:
        print("T5_REFERENCE_FAS_FINAL",summary); print(f"RESULT_DIR={result_dir}")


def run_fas_to_tol(result_dir,fine,p,q,b_p,b_q,total_t0):
    # One-cycle plumbing smoke only; not a science result.
    fine.reset_counters(); rows=[]; iteration_t0=time.perf_counter()
    for cycle in range(1,FAS_CYCLES+1):
        fine.fas_cycle_perf(p,q,b_p,b_q); residual_p=fine.p_residual_norm(p,q,b_p); now=time.perf_counter()
        rows.append({"mode":"fas_to_tol","cycle":cycle,"residual_p":residual_p,"target_residual":TARGET_RESIDUAL,
                     "reached_target":int(residual_p<=TARGET_RESIDUAL),"iteration_seconds":COMM.allreduce(now-iteration_t0,op=MPI.MAX),
                     "total_seconds":COMM.allreduce(now-total_t0,op=MPI.MAX)})
        if COMM.rank==0: print("T5_FAS_TO_TOL_ROW",rows[-1])
        if residual_p<=TARGET_RESIDUAL: break
    assert_requested_hierarchy_is_active(fine); levels=hierarchy_level_objects(fine)
    observed=[level.heat_source_calls+level.heat_target_calls for level in levels]; cycles=len(rows)
    expected=[16*cycles,9*cycles] if len(levels)==2 else [16*cycles,10*cycles,9*cycles]
    if observed!=expected: raise RuntimeError(f"T5 smoke heat-count failure: observed={observed}, expected={expected}")
    rows[-1].update({"observed_heat_actions":str(observed),"expected_heat_actions":str(expected),"heat_count_match":1})
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_fas_to_tol_trace.csv"),rows)
    write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_heat_counts.csv"),fine.counter_rows())
    final_science_audit("fas_to_tol",fine,p,q,b_p,b_q,rows[-1],result_dir,q_ref=None)
    if COMM.rank==0: print("T5_FAS_TO_TOL_FINAL",rows[-1]); print(f"RESULT_DIR={result_dir}")


def main():
    result_dir=os.environ.get("RESULT_DIR",os.getcwd()); os.makedirs(result_dir,exist_ok=True); COMM.Barrier(); total_t0=time.perf_counter()
    if RUN_MODE=="benchmark_certificate":
        V0,V1,mu0,mu1,metadata=build_support_spaces_and_measures(return_metadata=True)
        benchmark_certificate(V0,V1,mu0,mu1,metadata,result_dir)
        if COMM.rank==0: print(f"RESULT_DIR={result_dir}")
        return
    fine,coarse,V0,V1=build_problem(COARSE_TAU_RATIO)
    hierarchy=hierarchy_rows(fine); write_rows(os.path.join(result_dir,f"{EXPERIMENT_NAME}_hierarchy.csv"),hierarchy)
    p=Function(V0,name="p").assign(0.0); q=Function(V1,name="q").assign(0.0); b_p=Function(V0,name="b_p_finest").assign(0.0); b_q=Function(V1,name="b_q_finest").assign(0.0)
    if COMM.rank==0:
        print(f"EXPERIMENT_NAME={EXPERIMENT_NAME}")
        print("T5_CONFIGURATION",f"source_radius={SOURCE_RADIUS_PHYSICAL}",f"target_L={TARGET_L}",f"s0={TARGET_S0}",f"s1={TARGET_S1}",f"beta={TARGET_BETA}",f"p={TARGET_P}",f"neck_width={TARGET_NECK_WIDTH_PHYSICAL}",f"neck_over_sqrt_2tau={TARGET_NECK_WIDTH_PHYSICAL/math.sqrt(2.0*TAU_FINE_PHYSICAL)}")
        print(f"FAS_CYCLES={FAS_CYCLES} PRE={PRE_SMOOTH} COARSE={COARSE_SMOOTH} POST={POST_SMOOTH} COARSE_TAU_RATIO={COARSE_TAU_RATIO}")
        for row in hierarchy: print("FAS_HIERARCHY",row)
    if RUN_MODE=="fas_to_tol": run_fas_to_tol(result_dir,fine,p,q,b_p,b_q,total_t0); return
    if RUN_MODE=="reference_fas_to_tol": run_t5_reference_fas_validation(result_dir,fine,V0,V1,p,q,b_p,b_q,total_t0); return
    raise ValueError(f"Unknown T5 FAS_RUN_MODE={RUN_MODE!r}")

if __name__=="__main__":
    main()

