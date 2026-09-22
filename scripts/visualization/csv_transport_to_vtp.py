#!/usr/bin/env python3
"""Convert rank-split transport-map CSV exports into ParaView VTP point clouds."""

from __future__ import annotations

import argparse
import csv
import re
import xml.sax.saxutils as xml_escape
from pathlib import Path

import numpy as np

RANK_RE = re.compile(r"^(?P<stem>.+)_rank\d{4}\.csv$")
FIELD_RE = re.compile(r"__\d+_(?P<field>.+)$")


def field_name_from_stem(stem: str) -> str:
    match = FIELD_RE.search(stem)
    return match.group("field") if match else stem


def find_groups(base: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in sorted(base.glob("*rank*.csv")):
        match = RANK_RE.match(path.name)
        if not match:
            continue
        groups.setdefault(match.group("stem"), []).append(path)
    return groups


def read_group(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    coords_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            rows = [[float(x) for x in row] for row in reader]
        if len(header) < 3:
            raise ValueError(f"{path} has too few columns: {header}")
        arr = np.asarray(rows, dtype=float)
        coords_parts.append(arr[:, :2])
        value_parts.append(arr[:, 2:])

    coords = np.vstack(coords_parts)
    values = np.vstack(value_parts)
    key = np.round(coords, decimals=13)
    _, unique_idx = np.unique(key, axis=0, return_index=True)
    unique_idx.sort()
    return coords[unique_idx], values[unique_idx]


def write_vtp(path: Path, coords: np.ndarray, values: np.ndarray, field_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = coords.shape[0]
    points = np.column_stack([coords, np.zeros(n)])
    connectivity = np.arange(n, dtype=int)
    offsets = np.arange(1, n + 1, dtype=int)

    if values.shape[1] == 1:
        value_text = " ".join(f"{v:.17g}" for v in values[:, 0])
        components = 1
    else:
        padded = np.zeros((n, 3), dtype=float)
        padded[:, : values.shape[1]] = values
        value_text = "\n".join(" ".join(f"{v:.17g}" for v in row) for row in padded)
        components = 3

    point_text = "\n".join(" ".join(f"{v:.17g}" for v in row) for row in points)
    escaped_name = xml_escape.escape(field_name)
    path.write_text(
        f"""<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
  <PolyData>
    <Piece NumberOfPoints="{n}" NumberOfVerts="{n}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">
      <PointData>
        <DataArray type="Float64" Name="{escaped_name}" NumberOfComponents="{components}" format="ascii">
          {value_text}
        </DataArray>
      </PointData>
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          {point_text}
        </DataArray>
      </Points>
      <Verts>
        <DataArray type="Int64" Name="connectivity" format="ascii">
          {" ".join(str(i) for i in connectivity)}
        </DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii">
          {" ".join(str(i) for i in offsets)}
        </DataArray>
      </Verts>
    </Piece>
  </PolyData>
</VTKFile>
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or (args.input / "vtp_point_clouds")
    groups = find_groups(args.input)
    if not groups:
        raise SystemExit(f"No rank CSV groups found in {args.input}")

    summary_rows = []
    for stem, paths in sorted(groups.items()):
        field = field_name_from_stem(stem)
        coords, values = read_group(paths)
        out_path = out_dir / f"{field}.vtp"
        write_vtp(out_path, coords, values, field)
        flat = values.reshape(-1)
        summary_rows.append({
            "field": field,
            "rank_files": len(paths),
            "points": len(coords),
            "components": values.shape[1],
            "min": float(np.nanmin(flat)),
            "max": float(np.nanmax(flat)),
            "mean": float(np.nanmean(flat)),
            "vtp": str(out_path),
        })
        print(f"wrote {out_path} ({len(coords)} points, {values.shape[1]} components)")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "csv_to_vtp_summary.csv"
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["field", "rank_files", "points", "components", "min", "max", "mean", "vtp"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"summary {summary}")


if __name__ == "__main__":
    main()
