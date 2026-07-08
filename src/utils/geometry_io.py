import os
from typing import List, Optional, Tuple

import numpy as np

from .data_utils import load_obj


def load_ply(path: str) -> Tuple[np.ndarray, List[List[int]], Optional[np.ndarray]]:
    """Load a simple ASCII PLY file.

    Supports vertex positions, optional vertex colors, and polygon faces.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"{path} is not a PLY file")

    format_line = None
    header_end_idx = None
    vertex_count = 0
    face_count = 0
    vertex_props: List[str] = []
    current_element = None

    for idx, line in enumerate(lines[1:], start=1):
        stripped = line.strip()
        if stripped.startswith("format "):
            format_line = stripped
        elif stripped.startswith("element "):
            _, element_name, element_count = stripped.split()
            current_element = element_name
            if element_name == "vertex":
                vertex_count = int(element_count)
            elif element_name == "face":
                face_count = int(element_count)
        elif stripped.startswith("property ") and current_element == "vertex":
            parts = stripped.split()
            vertex_props.append(parts[-1])
        elif stripped == "end_header":
            header_end_idx = idx
            break

    if header_end_idx is None:
        raise ValueError(f"PLY file {path} is missing an end_header line")
    if format_line != "format ascii 1.0":
        raise ValueError(f"Only ASCII PLY files are supported, got: {format_line!r}")

    data_lines = lines[header_end_idx + 1 :]
    if len(data_lines) < vertex_count + face_count:
        raise ValueError(f"PLY file {path} ended before all elements were read")

    prop_index = {name: i for i, name in enumerate(vertex_props)}
    if not {"x", "y", "z"}.issubset(prop_index):
        raise ValueError(f"PLY file {path} does not define x/y/z vertex properties")

    vertices = []
    colors = []
    has_color = {"red", "green", "blue"}.issubset(prop_index)

    for line in data_lines[:vertex_count]:
        values = line.strip().split()
        vertices.append(
            [
                float(values[prop_index["x"]]),
                float(values[prop_index["y"]]),
                float(values[prop_index["z"]]),
            ]
        )
        if has_color:
            colors.append(
                [
                    int(values[prop_index["red"]]),
                    int(values[prop_index["green"]]),
                    int(values[prop_index["blue"]]),
                ]
            )

    faces: List[List[int]] = []
    for line in data_lines[vertex_count : vertex_count + face_count]:
        values = line.strip().split()
        if not values:
            continue
        count = int(values[0])
        indices = [int(v) for v in values[1 : 1 + count]]
        if len(indices) >= 3:
            faces.append(indices)

    vertices_array = np.asarray(vertices, dtype=np.float32)
    colors_array = np.asarray(colors, dtype=np.uint8) if has_color else None
    return vertices_array, faces, colors_array


def load_geometry(path: str) -> Tuple[np.ndarray, List[List[int]], Optional[np.ndarray]]:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".obj":
        vertices, faces = load_obj(path)
        return np.asarray(vertices, dtype=np.float32), faces, None
    if suffix == ".ply":
        return load_ply(path)
    raise ValueError(f"Unsupported geometry file type: {suffix}")


def triangulate_faces(faces: List[List[int]]) -> List[List[int]]:
    triangles: List[List[int]] = []
    for face in faces:
        if len(face) < 3:
            continue
        if len(face) == 3:
            triangles.append(face)
            continue
        root = face[0]
        for idx in range(1, len(face) - 1):
            triangles.append([root, face[idx], face[idx + 1]])
    return triangles
