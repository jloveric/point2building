import importlib.util
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from src.utils import data_utils
from src.utils.geometry_io import load_geometry, triangulate_faces


st.set_page_config(page_title="Point2Building Viewer", layout="wide")

DATA_ROOT = Path("/data2/point-cloud-datasets/MunichWF")
POINTCLOUD_DIR = DATA_ROOT / "pc_part"
GEOMETRY_DIR = DATA_ROOT / "objs"
TEST_LIST_PATH = DATA_ROOT / "test_list.txt"


@st.cache_data(show_spinner=False)
def load_test_ids(test_list_path: str) -> list[str]:
    with open(test_list_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@st.cache_data(show_spinner=False)
def load_reference_geometry(path: str):
    return load_geometry(path)


@st.cache_resource(show_spinner=False)
def load_models(vertex_checkpoint_path: str, face_checkpoint_path: str, device: str):
    from train_face_model import load_f_models
    from train_vertex_model import load_v_models

    vertex_model = load_v_models(device=device, split="test").to(device)
    face_model = load_f_models(device=device, split="test").to(device)

    vertex_checkpoint = torch.load(vertex_checkpoint_path, map_location=device)
    face_checkpoint = torch.load(face_checkpoint_path, map_location=device)
    vertex_model.load_state_dict(vertex_checkpoint["state_dict"])
    face_model.load_state_dict(face_checkpoint["state_dict"])

    vertex_model.eval()
    face_model.eval()
    return vertex_model, face_model


def build_figure(vertices: np.ndarray, faces, name: str, colors=None, show_points=True):
    figure = go.Figure()
    triangles = triangulate_faces(faces)

    if triangles:
        i = [face[0] for face in triangles]
        j = [face[1] for face in triangles]
        k = [face[2] for face in triangles]
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=i,
                j=j,
                k=k,
                name=name,
                opacity=0.55,
                color="#4c78a8",
                flatshading=True,
                showscale=False,
            )
        )

    if show_points:
        marker_color = None
        if colors is not None and len(colors) == len(vertices):
            marker_color = [f"rgb({r}, {g}, {b})" for r, g, b in colors]
        figure.add_trace(
            go.Scatter3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                mode="markers",
                name=f"{name} points",
                marker=dict(
                    size=3,
                    color=marker_color if marker_color is not None else "#f58518",
                    opacity=0.85,
                ),
            )
        )

    if triangles:
        edge_x = []
        edge_y = []
        edge_z = []
        seen_edges = set()
        for face in triangles:
            for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge = tuple(sorted((start, end)))
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                edge_x.extend([vertices[start, 0], vertices[end, 0], None])
                edge_y.extend([vertices[start, 1], vertices[end, 1], None])
                edge_z.extend([vertices[start, 2], vertices[end, 2], None])
        figure.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                name=f"{name} wireframe",
                line=dict(color="#1f2937", width=3),
                opacity=0.9,
            )
        )

    figure.update_layout(
        title=name,
        scene=dict(
            aspectmode="data",
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant"),
    )
    return figure


def make_overlay(input_geom, output_geom):
    input_vertices, input_faces, input_colors = input_geom
    output_vertices, output_faces, output_colors = output_geom
    figure = go.Figure()

    if triangulate_faces(input_faces):
        triangles = triangulate_faces(input_faces)
        figure.add_trace(
            go.Mesh3d(
                x=input_vertices[:, 0],
                y=input_vertices[:, 1],
                z=input_vertices[:, 2],
                i=[face[0] for face in triangles],
                j=[face[1] for face in triangles],
                k=[face[2] for face in triangles],
                name="input mesh",
                opacity=0.25,
                color="#54a24b",
                flatshading=True,
            )
        )

    figure.add_trace(
        go.Scatter3d(
            x=input_vertices[:, 0],
            y=input_vertices[:, 1],
            z=input_vertices[:, 2],
            mode="markers",
            name="input ply",
            marker=dict(
                size=3,
                color=(
                    [f"rgb({r}, {g}, {b})" for r, g, b in input_colors]
                    if input_colors is not None and len(input_colors) == len(input_vertices)
                    else "#54a24b"
                ),
                opacity=0.75,
            ),
        )
    )

    if triangulate_faces(output_faces):
        triangles = triangulate_faces(output_faces)
        figure.add_trace(
            go.Mesh3d(
                x=output_vertices[:, 0],
                y=output_vertices[:, 1],
                z=output_vertices[:, 2],
                i=[face[0] for face in triangles],
                j=[face[1] for face in triangles],
                k=[face[2] for face in triangles],
                name="output mesh",
                opacity=0.5,
                color="#4c78a8",
                flatshading=True,
            )
        )

        edge_x = []
        edge_y = []
        edge_z = []
        seen_edges = set()
        for face in triangles:
            for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge = tuple(sorted((start, end)))
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                edge_x.extend([output_vertices[start, 0], output_vertices[end, 0], None])
                edge_y.extend([output_vertices[start, 1], output_vertices[end, 1], None])
                edge_z.extend([output_vertices[start, 2], output_vertices[end, 2], None])
        figure.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                name="output wireframe",
                line=dict(color="#111827", width=3),
                opacity=0.95,
            )
        )

    figure.add_trace(
        go.Scatter3d(
            x=output_vertices[:, 0],
            y=output_vertices[:, 1],
            z=output_vertices[:, 2],
            mode="markers",
            name="output geometry",
            marker=dict(
                size=3,
                color=(
                    [f"rgb({r}, {g}, {b})" for r, g, b in output_colors]
                    if output_colors is not None and len(output_colors) == len(output_vertices)
                    else "#4c78a8"
                ),
                opacity=0.8,
            ),
        )
    )

    figure.update_layout(
        title="Overlay",
        scene=dict(
            aspectmode="data",
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return figure


def build_pointcloud_batch(points: np.ndarray, device: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, np.ndarray]]:
    import MinkowskiEngine as ME

    centered_points, center = data_utils.center_vertices_np(points, return_center=True)
    normalized_points, scale = data_utils.normalize_vertices_scale_np(centered_points, return_scale=True)

    floor_points = normalized_points.copy()
    floor_points[:, 2] = -np.max(normalized_points[:, 2])
    normalized_points = np.vstack([normalized_points, floor_points]).astype(np.float32)

    pts_tensor = torch.from_numpy(normalized_points).float().to(device)
    coords = data_utils.quantize_verts(pts_tensor)

    voxel_dict = {}
    for idx in range(coords.shape[0]):
        coord_tuple = (coords[idx, 0].item(), coords[idx, 1].item(), coords[idx, 2].item())
        if coord_tuple not in voxel_dict:
            voxel_dict[coord_tuple] = torch.cat([pts_tensor[idx], torch.tensor([1.0], device=device)], dim=-1)
        else:
            voxel_dict[coord_tuple] += torch.cat([pts_tensor[idx], torch.tensor([1.0], device=device)], dim=-1)

    locations = torch.tensor(list(voxel_dict.keys()))
    features = torch.stack(list(voxel_dict.values()))
    features = features / features[:, 3:]
    features = torch.cat([data_utils.quantize_verts(features[:, :3]), features[:, -1:]], dim=-1)

    pc_coords, pc_feats = ME.utils.sparse_collate([locations], [features])
    batch = {
        "vertices_flat": torch.zeros((1, 1), dtype=torch.float32, device=device),
        "pc_coords": pc_coords.to(device),
        "pc_feats": pc_feats.to(device),
    }
    return batch, {"center": center, "scale": scale}


def run_vertex_model(vertex_model, points: np.ndarray, device: str):
    vertex_batch, normalization = build_pointcloud_batch(points, device)

    with torch.no_grad():
        prediction = vertex_model.sample_mask(
            num_samples=1,
            max_sample_length=100,
            context=vertex_batch,
            top_p=0.9,
            recenter_verts=False,
            only_return_complete=False,
        )

    num_vertices = int(prediction["num_vertices"][0].item())
    predicted_vertices = prediction["vertices"][0][:num_vertices].detach().cpu()
    predicted_vertices = data_utils.quantize_verts(predicted_vertices)
    predicted_vertices, _ = torch.unique(predicted_vertices, dim=0, return_inverse=True)
    predicted_vertices = predicted_vertices[data_utils.torch_lexsort(predicted_vertices.T)]
    predicted_vertices = data_utils.dequantize_verts(predicted_vertices).cpu().numpy()
    return predicted_vertices, normalization


def run_face_model(face_model, vertices: np.ndarray, normalization: Dict[str, np.ndarray], device: str, top_p: float):
    quantized_vertices = data_utils.quantize_verts(torch.from_numpy(vertices).float().to(device)).to(torch.int32)
    assert int(quantized_vertices.min().item()) >= 0
    assert int(quantized_vertices.max().item()) < 2**8
    face_vertices = data_utils.dequantize_verts(quantized_vertices).unsqueeze(0)
    face_batch = {
        "vertices": face_vertices.to(device),
        "vertices_mask": torch.ones_like(face_vertices[..., 0], dtype=torch.float32).to(device),
        "files_list": ["prediction.obj"],
    }

    with torch.no_grad():
        prediction = face_model.sample_mask(
            context=face_batch,
            max_sample_length=500,
            top_p=top_p,
            only_return_complete=False,
        )

    num_face_indices = int(prediction["num_face_indices"][0].item())
    sampled_faces = prediction["faces"][0]
    faces = data_utils.unflatten_faces(sampled_faces[:num_face_indices].detach().cpu().numpy())
    predicted_vertices = face_vertices[0].detach().cpu().numpy()
    predicted_vertices = predicted_vertices * normalization["scale"] + normalization["center"]
    return predicted_vertices, faces


st.title("Point2Building Viewer")
st.write("Run the Point2Building models on MunichWF test samples and compare the predicted geometry to the input point cloud and optional reference mesh.")

if not TEST_LIST_PATH.exists():
    st.error(f"Test list not found: {TEST_LIST_PATH}")
    st.stop()
if not POINTCLOUD_DIR.exists():
    st.error(f"Point cloud directory not found: {POINTCLOUD_DIR}")
    st.stop()
if not GEOMETRY_DIR.exists():
    st.error(f"Geometry directory not found: {GEOMETRY_DIR}")
    st.stop()

device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    st.error("Model sampling in this repo currently requires CUDA because the sampling code calls `.cuda()` internally.")
    st.stop()
can_run_inference = importlib.util.find_spec("MinkowskiEngine") is not None

st.sidebar.header("Inference")
st.sidebar.write(f"Device: `{device}`")
if not can_run_inference:
    st.sidebar.warning("`MinkowskiEngine` is not installed, so model inference is disabled.")
vertex_checkpoint_path = st.sidebar.text_input(
    "Vertex checkpoint",
    value=str(Path("./saved_model/vertex_model/checkpoint_v.pth")),
)
face_checkpoint_path = st.sidebar.text_input(
    "Face checkpoint",
    value=str(Path("./saved_model/face_model/checkpoint_f.pth")),
)
top_p = st.sidebar.slider("Face sampling top-p", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
show_reference = st.sidebar.checkbox("Show reference OBJ", value=True)
render_points = st.sidebar.checkbox("Show points", value=True)

if can_run_inference:
    if not Path(vertex_checkpoint_path).exists():
        st.error(f"Vertex checkpoint not found: {vertex_checkpoint_path}")
        st.stop()
    if not Path(face_checkpoint_path).exists():
        st.error(f"Face checkpoint not found: {face_checkpoint_path}")
        st.stop()

test_ids = load_test_ids(str(TEST_LIST_PATH))
sample_id = st.selectbox("Test sample", test_ids)

ply_path = POINTCLOUD_DIR / f"{sample_id}.ply"
reference_path = GEOMETRY_DIR / f"{sample_id}.obj"

if not ply_path.exists():
    st.error(f"Missing point cloud file: {ply_path}")
    st.stop()

input_geom = load_reference_geometry(str(ply_path))
reference_geom = load_reference_geometry(str(reference_path)) if show_reference and reference_path.exists() else None

run_inference = st.button("Run model inference", type="primary", disabled=not can_run_inference)
if not can_run_inference:
    st.info("Install `MinkowskiEngine` to enable model inference.")
elif run_inference or "prediction" in st.session_state:
    if run_inference:
        vertex_model, face_model = load_models(vertex_checkpoint_path, face_checkpoint_path, device)
        predicted_vertices, normalization = run_vertex_model(vertex_model, input_geom[0], device)
        predicted_vertices, predicted_faces = run_face_model(
            face_model,
            predicted_vertices,
            normalization,
            device,
            top_p=top_p,
        )
        st.session_state["prediction"] = {
            "sample_id": sample_id,
            "vertices": predicted_vertices,
            "faces": predicted_faces,
        }

    prediction = st.session_state.get("prediction")
    if prediction is not None and prediction.get("sample_id") == sample_id:
        prediction_geom = (prediction["vertices"], prediction["faces"], None)

        st.subheader("Summary")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Input vertices", len(input_geom[0]))
            st.metric("Input faces", len(input_geom[1]))
        with c2:
            st.metric("Predicted vertices", len(prediction_geom[0]))
            st.metric("Predicted faces", len(prediction_geom[1]))

        view_left, view_right = st.columns(2)
        with view_left:
            st.plotly_chart(
                build_figure(input_geom[0], input_geom[1], "Input PLY", input_geom[2], show_points=render_points),
                use_container_width=True,
            )
        with view_right:
            st.plotly_chart(
                build_figure(prediction_geom[0], prediction_geom[1], "Predicted Geometry", prediction_geom[2], show_points=render_points),
                use_container_width=True,
            )

        st.plotly_chart(make_overlay(input_geom, prediction_geom), use_container_width=True)

        if reference_geom is not None:
            st.subheader("Reference")
            ref_left, ref_right = st.columns(2)
            with ref_left:
                st.plotly_chart(
                    build_figure(reference_geom[0], reference_geom[1], "Reference OBJ", reference_geom[2], show_points=render_points),
                    use_container_width=True,
                )
            with ref_right:
                st.plotly_chart(make_overlay(reference_geom, prediction_geom), use_container_width=True)
    elif run_inference:
        st.session_state.pop("prediction", None)
