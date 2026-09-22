#!/usr/bin/env python3
"""Offline deterministic support-mesh generator for T5 thin-neck experiment v52.

This file is provenance material, not a CX3 runtime dependency.  It requires
NumPy, SciPy and Matplotlib.  The generated Gmsh-v2 meshes are frozen assets;
the runtime jobs load them directly and verify SHA256 hashes.
"""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
from matplotlib.path import Path as MplPath

BRIDGE_WIDTH_PHYSICAL = 4.0
SOURCE_RADIUS_PHYSICAL = 1.0
TARGET_L = 1.20
TARGET_S0 = 0.060
TARGET_S1 = 0.80
TARGET_BETA = 0.28
TARGET_P = 4
SOURCE_H = 0.0075
TARGET_H = 0.0070
BOUNDARY_DENSE_N = 50000
BOUNDARY_SHRINK_RADIUS_FACTOR = 0.65
TAU_FINE_PHYSICAL = 2.373046875e-3

ROOT = Path(__file__).resolve().parent
MESH_DIR = ROOT / "t5_assets_v52"
SOURCE_PATH = MESH_DIR / "t5_source_disk_v52.msh"
TARGET_PATH = MESH_DIR / "t5_target_thin_neck_v52.msh"
METADATA_PATH = MESH_DIR / "t5_mesh_assets_v52_metadata.json"


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def physical_to_unit(points):
    return 0.5 + points / BRIDGE_WIDTH_PHYSICAL


def target_scale(xi):
    xi = np.asarray(xi)
    return TARGET_S0 + (TARGET_S1 - TARGET_S0) * (1.0 - np.exp(-((xi / TARGET_BETA) ** TARGET_P)))


def source_boundary_dense():
    theta = np.linspace(0.0, 2.0 * math.pi, BOUNDARY_DENSE_N, endpoint=False)
    return np.column_stack((SOURCE_RADIUS_PHYSICAL * np.cos(theta), SOURCE_RADIUS_PHYSICAL * np.sin(theta)))


def target_boundary_dense():
    theta = np.linspace(0.0, 2.0 * math.pi, BOUNDARY_DENSE_N, endpoint=False)
    xi = np.cos(theta)
    return np.column_stack((TARGET_L * xi, target_scale(xi) * np.sin(theta)))


def resample_closed_curve(curve, spacing):
    closed = np.vstack((curve, curve[0]))
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arclen = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(arclen[-1])
    n = max(32, int(round(total / spacing)))
    samples = np.linspace(0.0, total, n, endpoint=False)
    idx = np.searchsorted(arclen, samples, side="right") - 1
    idx = np.minimum(idx, len(seg) - 1)
    t = (samples - arclen[idx]) / np.maximum(arclen[idx + 1] - arclen[idx], 1.0e-300)
    return closed[idx] * (1.0 - t[:, None]) + closed[idx + 1] * t[:, None]


def triangle_components(triangles):
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
    for i, tri in enumerate(triangles):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = tuple(sorted((int(a), int(b))))
            if edge in edge_owner:
                union(i, edge_owner[edge])
            else:
                edge_owner[edge] = i
    return len({find(i) for i in range(len(triangles))})


def boundary_edge_count(triangles):
    counts = {}
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = tuple(sorted((int(a), int(b))))
            counts[edge] = counts.get(edge, 0) + 1
    return sum(v == 1 for v in counts.values())


def quality_metrics(points, triangles):
    xy = points[triangles]
    signed = 0.5 * cross2(xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0])
    if float(signed.min()) <= 0.0:
        raise RuntimeError(f"non-positive triangle orientation: {signed.min()}")
    l01 = np.linalg.norm(xy[:, 1] - xy[:, 0], axis=1)
    l12 = np.linalg.norm(xy[:, 2] - xy[:, 1], axis=1)
    l20 = np.linalg.norm(xy[:, 0] - xy[:, 2], axis=1)
    lengths = np.stack((l01, l12, l20), axis=1)
    max_edge = lengths.max(axis=1)
    min_alt = 2.0 * signed / np.maximum(max_edge, 1.0e-300)
    aspect = max_edge / np.maximum(min_alt, 1.0e-300)
    angles = []
    for A, B, C in ((l01, l20, l12), (l01, l12, l20), (l12, l20, l01)):
        cosine = (A * A + B * B - C * C) / np.maximum(2.0 * A * B, 1.0e-300)
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    min_angle = np.stack(angles, axis=1).min(axis=1)
    return {
        "vertex_count": int(len(points)),
        "triangle_count": int(len(triangles)),
        "area_physical": float(signed.sum()),
        "min_signed_triangle_area_physical": float(signed.min()),
        "max_signed_triangle_area_physical": float(signed.max()),
        "min_angle_degrees": float(min_angle.min()),
        "max_aspect_edge_over_altitude": float(aspect.max()),
        "min_edge_physical": float(lengths.min()),
        "max_edge_physical": float(lengths.max()),
    }


def mesh_support(boundary_dense, h, label):
    boundary = resample_closed_curve(boundary_dense, h)
    path = MplPath(boundary, closed=True)
    xmin, ymin = boundary.min(axis=0)
    xmax, ymax = boundary.max(axis=0)
    dy = math.sqrt(3.0) * h / 2.0
    rows = []
    for j, y in enumerate(np.arange(ymin + 0.5 * dy, ymax, dy)):
        xs = np.arange(xmin + 0.5 * h + (0.5 * h if j % 2 else 0.0), xmax, h)
        pts = np.column_stack((xs, np.full_like(xs, y)))
        # Negative radius erodes the polygon slightly, preventing boundary slivers.
        mask = path.contains_points(pts, radius=-BOUNDARY_SHRINK_RADIUS_FACTOR * h)
        if np.any(mask):
            rows.append(pts[mask])
    interior = np.vstack(rows)
    points = np.vstack((interior, boundary))
    rounded = np.round(points, 13)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    points = points[np.sort(unique_idx)]

    simplices = Delaunay(points).simplices
    xy = points[simplices]
    probes = [
        xy.mean(axis=1),
        0.5 * (xy[:, 0] + xy[:, 1]),
        0.5 * (xy[:, 1] + xy[:, 2]),
        0.5 * (xy[:, 2] + xy[:, 0]),
    ]
    keep = np.ones(len(simplices), dtype=bool)
    for probe in probes:
        keep &= path.contains_points(probe, radius=1.0e-10)
    triangles = simplices[keep].copy()
    xy = points[triangles]
    signed = 0.5 * cross2(xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0])
    flip = signed < 0.0
    triangles[flip] = triangles[flip][:, [0, 2, 1]]

    used = np.unique(triangles.ravel())
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    points = points[used]
    triangles = remap[triangles]

    metrics = quality_metrics(points, triangles)
    # Polygon area by shoelace on the frozen resampled boundary.
    polygon_area = 0.5 * abs(float(np.sum(cross2(boundary, np.roll(boundary, -1, axis=0)))))
    metrics.update({
        "label": label,
        "requested_h_physical": h,
        "boundary_vertex_count_requested": int(len(boundary)),
        "polygon_area_physical": polygon_area,
        "triangle_area_to_polygon_area_ratio": metrics["area_physical"] / polygon_area,
        "connected_triangle_components": triangle_components(triangles),
        "boundary_edge_count": boundary_edge_count(triangles),
        "bbox_physical": [float(xmin), float(ymin), float(xmax), float(ymax)],
    })
    return points, triangles, metrics


def write_gmsh_v2(path, points_physical, triangles):
    points_unit = physical_to_unit(points_physical)
    with open(path, "w", newline="\n") as handle:
        handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        handle.write(f"$Nodes\n{len(points_unit)}\n")
        for node_id, (x, y) in enumerate(points_unit, start=1):
            handle.write(f"{node_id} {x:.17e} {y:.17e} 0.0\n")
        handle.write("$EndNodes\n")
        handle.write(f"$Elements\n{len(triangles)}\n")
        for elem_id, tri in enumerate(triangles, start=1):
            a, b, c = (int(v) + 1 for v in tri)
            handle.write(f"{elem_id} 2 2 1 1 {a} {b} {c}\n")
        handle.write("$EndElements\n")


def target_neck_metrics(points, triangles):
    neck_width = 2.0 * TARGET_S0
    xi_cut = TARGET_BETA * (math.log(2.0) ** (1.0 / TARGET_P))
    x_cut = TARGET_L * xi_cut
    xy = points[triangles]
    centroids = xy.mean(axis=1)
    areas = 0.5 * np.abs(cross2(xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]))
    diameters = np.stack((
        np.linalg.norm(xy[:, 1] - xy[:, 0], axis=1),
        np.linalg.norm(xy[:, 2] - xy[:, 1], axis=1),
        np.linalg.norm(xy[:, 0] - xy[:, 2], axis=1),
    ), axis=1).max(axis=1)
    mask = np.abs(centroids[:, 0]) <= x_cut
    return {
        "neck_width_physical": neck_width,
        "neck_partition_x_cut_physical": x_cut,
        "neck_partition_xi_cut": xi_cut,
        "neck_area_fraction_centroid_partition": float(areas[mask].sum() / areas.sum()),
        "max_triangle_diameter_in_neck_physical": float(diameters[mask].max()),
        "area_weighted_mean_triangle_diameter_in_neck_physical": float(np.average(diameters[mask], weights=areas[mask])),
        "neck_width_over_max_neck_triangle_diameter": float(neck_width / diameters[mask].max()),
        "neck_width_over_mean_neck_triangle_diameter": float(neck_width / np.average(diameters[mask], weights=areas[mask])),
    }


def curvature_scan():
    theta = np.linspace(0.0, 2.0 * math.pi, 200001, endpoint=False)
    xi = np.cos(theta)
    x = TARGET_L * xi
    y = target_scale(xi) * np.sin(theta)
    dtheta = float(theta[1] - theta[0])
    dx = np.gradient(x, dtheta, edge_order=2)
    dy = np.gradient(y, dtheta, edge_order=2)
    tangent = np.unwrap(np.arctan2(dy, dx))
    increments = np.diff(tangent)
    # Kadane minimum contiguous sum.  With theta=0 at the right-most convex
    # point, the most-negative connected arc does not cross the array seam.
    best = math.inf
    current = 0.0
    for value in increments:
        value = float(value)
        current = value if current > 0.0 else current + value
        best = min(best, current)
    return {
        "minimum_connected_arc_curvature_integral_numerical": float(best),
        "minus_pi_threshold": -math.pi,
        "theorem_discontinuity_threshold_crossed": int(best < -math.pi),
        "curvature_scan_disclaimer": "numerical design diagnostic only; T5 does not claim theorem-certified discontinuity",
    }


def main():
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    source_points, source_triangles, source_meta = mesh_support(source_boundary_dense(), SOURCE_H, "source_disk")
    target_points, target_triangles, target_meta = mesh_support(target_boundary_dense(), TARGET_H, "target_thin_neck")
    write_gmsh_v2(SOURCE_PATH, source_points, source_triangles)
    write_gmsh_v2(TARGET_PATH, target_points, target_triangles)
    target_meta.update(target_neck_metrics(target_points, target_triangles))
    target_meta.update(curvature_scan())
    heat_length = math.sqrt(2.0 * TAU_FINE_PHYSICAL)
    metadata = {
        "generator": "generate_t5_mesh_assets_v52.py",
        "bridge_width_physical": BRIDGE_WIDTH_PHYSICAL,
        "source_radius_physical": SOURCE_RADIUS_PHYSICAL,
        "target_parameters": {"L": TARGET_L, "s0": TARGET_S0, "s1": TARGET_S1, "beta": TARGET_BETA, "p": TARGET_P},
        "source": source_meta,
        "target": target_meta,
        "tau_fine_physical": TAU_FINE_PHYSICAL,
        "heat_length_sqrt_2tau_physical": heat_length,
        "neck_width_over_sqrt_2tau": 2.0 * TARGET_S0 / heat_length,
        "source_mesh_sha256": sha256_file(SOURCE_PATH),
        "target_mesh_sha256": sha256_file(TARGET_PATH),
    }
    with open(METADATA_PATH, "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
