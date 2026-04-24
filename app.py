from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.model import ScaffoldedPointPredictor


st.set_page_config(page_title="3D Hand Detection Demo", layout="wide")

DEFAULT_EVAL_NPZ = REPO_ROOT / "data" / "eval_data_600_verts.npz"
DEFAULT_CHECKPOINT = REPO_ROOT / "scripts" / "checkpoints" / "model_600_verts_15_vectors.pth"
DEFAULT_ANGLE_CLUSTER = REPO_ROOT / "clustering" / "train_camera_clusters_k8.npz"
DEFAULT_POSE_CLUSTER = REPO_ROOT / "clustering" / "train_pose_clusters_k10.npz"

SKEL_CONNECTIONS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]


def _list_npz_files() -> list[Path]:
    return sorted(REPO_ROOT.glob("**/*.npz"))


def _list_cluster_files() -> list[Path]:
    return sorted(REPO_ROOT.glob("clustering/**/*clusters*.npz"))


def _coerce_scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str, num_verts: int = 600, num_vectors: int = 15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScaffoldedPointPredictor(
        num_joints=21,
        num_verts=num_verts,
        num_vectors=num_vectors,
        pretrained_backbone=False,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


@st.cache_data(show_spinner=False)
def load_eval_dataset(npz_path: str):
    with np.load(npz_path) as data:
        return {
            "images": data["x_test"],
            "gt_joints": data["y_test_joints"],
            "gt_verts": data["y_test_verts"],
        }


def run_inference(model, device, image_np: np.ndarray):
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(Image.fromarray(image_np)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_joints, pred_vectors, pred_verts = model(input_tensor)

    return (
        pred_joints.cpu().numpy().reshape(21, 3),
        pred_vectors.cpu().numpy().reshape(15, 3),
        pred_verts.cpu().numpy().reshape(-1, 3),
    )


def render_eval_figure(
    image: np.ndarray,
):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image)
    ax.set_title("Input image")
    ax.axis("off")
    fig.tight_layout()
    return fig


def render_interactive_eval_plot(
    pred_joints: np.ndarray,
    pred_vectors: np.ndarray,
    pred_verts: np.ndarray,
):
    tip_ids = [4, 8, 12, 16, 20]
    mcp_ids = [2, 5, 9, 13, 17]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=pred_verts[:, 0],
            y=pred_verts[:, 1],
            z=pred_verts[:, 2],
            mode="markers",
            marker={"size": 2.4, "color": "rgba(220,70,60,0.45)"},
            name="Pred verts",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=pred_joints[:, 0],
            y=pred_joints[:, 1],
            z=pred_joints[:, 2],
            mode="markers",
            marker={"size": 4.5, "color": "black", "symbol": "x"},
            name="Pred joints",
        )
    )

    for finger_idx, finger in enumerate(SKEL_CONNECTIONS):
        fig.add_trace(
            go.Scatter3d(
                x=pred_joints[finger, 0],
                y=pred_joints[finger, 1],
                z=pred_joints[finger, 2],
                mode="lines",
                line={"color": "royalblue", "width": 6},
                name="Skeleton" if finger_idx == 0 else None,
                showlegend=finger_idx == 0,
            )
        )

    pairs = []
    for t1 in range(5):
        for t2 in range(t1 + 1, 5):
            pairs.append((t1, t2))

    for n in range(10):
        start_idx = tip_ids[pairs[n][0]]
        start_pos = pred_joints[start_idx]
        vec = pred_vectors[n]
        end_pos = start_pos + vec
        fig.add_trace(
            go.Scatter3d(
                x=[start_pos[0], end_pos[0]],
                y=[start_pos[1], end_pos[1]],
                z=[start_pos[2], end_pos[2]],
                mode="lines",
                line={"color": "goldenrod", "width": 5},
                name="Inter-tip vectors" if n == 0 else None,
                showlegend=n == 0,
            )
        )

    for n in range(5):
        start_idx = mcp_ids[n]
        start_pos = pred_joints[start_idx]
        vec = pred_vectors[10 + n]
        end_pos = start_pos + vec
        fig.add_trace(
            go.Scatter3d(
                x=[start_pos[0], end_pos[0]],
                y=[start_pos[1], end_pos[1]],
                z=[start_pos[2], end_pos[2]],
                mode="lines",
                line={"color": "purple", "width": 5},
                name="Bone vectors" if n == 0 else None,
                showlegend=n == 0,
            )
        )

    all_points = np.concatenate([pred_verts, pred_joints], axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.25)

    fig.update_layout(
        title="Prediction preview",
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
        scene={
            "xaxis": {"visible": False, "range": [center[0] - radius, center[0] + radius]},
            "yaxis": {"visible": False, "range": [center[1] - radius, center[1] + radius]},
            "zaxis": {"visible": False, "range": [center[2] - radius, center[2] + radius]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.35, "y": -1.35, "z": 1.0}},
        },
    )
    return fig


def safe_normalize(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, eps, None)


def rank_cluster_members(features: np.ndarray, labels: np.ndarray, centers: np.ndarray):
    normalized_features = safe_normalize(features.astype(np.float32))
    normalized_centers = safe_normalize(centers.astype(np.float32))
    ranked_members_by_cluster: dict[int, np.ndarray] = {}
    ranked_scores_by_cluster: dict[int, np.ndarray] = {}

    for cluster_id in np.unique(labels).tolist():
        members = np.flatnonzero(labels == cluster_id)
        member_scores = normalized_features[members] @ normalized_centers[cluster_id]
        rank_order = np.argsort(-member_scores, kind="stable")
        ranked_members_by_cluster[cluster_id] = members[rank_order]
        ranked_scores_by_cluster[cluster_id] = member_scores[rank_order].astype(np.float32)

    return ranked_members_by_cluster, ranked_scores_by_cluster


@st.cache_data(show_spinner=False)
def load_cluster_artifact(npz_path: str):
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_images_from_dataset(npz_path: Path, split: str, progress_label: str = "Loading dataset images...") -> np.ndarray:
    split_to_key = {"train": "x_train", "val": "x_val", "test": "x_test"}
    member_name = f"{split_to_key[split]}.npy"
    progress = st.progress(0, text=progress_label)

    with zipfile.ZipFile(npz_path) as archive:
        member_info = archive.getinfo(member_name)
        with archive.open(member_name) as member_file:
            total = max(member_info.file_size, 1)
            buffer = io.BytesIO()
            while True:
                chunk = member_file.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)
                progress.progress(min(buffer.tell() / total, 1.0), text=progress_label)

    buffer.seek(0)
    images = np.load(buffer, allow_pickle=False)
    progress.empty()
    return images


def load_images_cached(npz_path: str, split: str, progress_label: str) -> np.ndarray:
    cache_key = f"{npz_path}::{split}"
    image_cache = st.session_state.setdefault("_image_cache", {})
    if cache_key not in image_cache:
        image_cache[cache_key] = load_images_from_dataset(Path(npz_path), split, progress_label=progress_label)
    return image_cache[cache_key]


def infer_split_from_name(path: Path) -> str:
    stem = path.stem
    for split in ("train", "val", "test"):
        if stem.startswith(f"{split}_"):
            return split
    return "train"


def render_cluster_section(
    title: str,
    artifact_path: Path,
    dataset_path: Path,
    samples_per_cluster: int,
    feature_key: str,
    guidance_text: str,
    drop_last_two: bool = False,
    preferred_first_cluster: int | None = None,
):
    artifact = load_cluster_artifact(str(artifact_path))
    labels = artifact["labels"].astype(np.int32)
    centers = artifact["centers"].astype(np.float32)
    features = artifact[feature_key].astype(np.float32)
    split = infer_split_from_name(artifact_path)

    images = load_images_cached(str(dataset_path), split, progress_label=f"Loading {title.lower()} dataset images...")
    ranked_members, ranked_scores = rank_cluster_members(features, labels, centers)
    cluster_ids = sorted(ranked_members)
    if drop_last_two and len(cluster_ids) > 2:
        cluster_ids = cluster_ids[:-2]
    if preferred_first_cluster is not None and preferred_first_cluster in cluster_ids:
        cluster_ids = [preferred_first_cluster] + [cluster_id for cluster_id in cluster_ids if cluster_id != preferred_first_cluster]

    st.subheader(title)
    st.caption(f"{artifact_path.name} • split `{split}` • {len(cluster_ids)} clusters shown")
    st.info(guidance_text)

    lookup_state_key = f"{title}_lookup_index"
    if lookup_state_key not in st.session_state:
        st.session_state[lookup_state_key] = 0

    lookup_col1, lookup_col2 = st.columns([1, 2], gap="medium")
    with lookup_col1:
        if st.button("Random dataset index", key=f"{title}_random_lookup", use_container_width=True):
            st.session_state[lookup_state_key] = int(np.random.randint(0, len(labels)))
    with lookup_col2:
        selected_index = st.slider(
            "Find cluster for dataset index",
            min_value=0,
            max_value=max(len(labels) - 1, 0),
            value=int(st.session_state[lookup_state_key]),
            key=lookup_state_key,
        )

    selected_index = int(selected_index)
    selected_cluster = int(labels[selected_index])
    cluster_visible = selected_cluster in cluster_ids
    lookup_members = ranked_members[selected_cluster]
    lookup_rank_matches = np.flatnonzero(lookup_members == selected_index)
    lookup_rank = int(lookup_rank_matches[0] + 1) if len(lookup_rank_matches) > 0 else None

    ordered_cluster_ids = list(cluster_ids)
    if selected_cluster in ordered_cluster_ids:
        ordered_cluster_ids = [selected_cluster] + [
            cluster_id for cluster_id in ordered_cluster_ids if cluster_id != selected_cluster
        ]
        if preferred_first_cluster is not None and preferred_first_cluster in ordered_cluster_ids[1:]:
            ordered_cluster_ids = (
                [ordered_cluster_ids[0], preferred_first_cluster]
                + [
                    cluster_id
                    for cluster_id in ordered_cluster_ids[1:]
                    if cluster_id != preferred_first_cluster
                ]
            )

    preview_col1, preview_col2 = st.columns([1, 2], gap="medium")
    with preview_col1:
        st.image(images[selected_index], use_container_width=True)
    with preview_col2:
        st.markdown(f"**Selected dataset index:** {selected_index}")
        st.markdown(f"**Cluster:** {selected_cluster}")
        if lookup_rank is not None:
            st.markdown(f"**Rank within cluster:** {lookup_rank}")
        if cluster_visible:
            st.success(f"Cluster {selected_cluster} is shown below.")
        else:
            st.warning(f"Cluster {selected_cluster} is not currently displayed below.")

    for cluster_id in ordered_cluster_ids:
        members = ranked_members[cluster_id]
        scores = ranked_scores[cluster_id]
        count = min(samples_per_cluster, len(members))
        header_suffix = " ← selected sample belongs here" if cluster_id == selected_cluster and cluster_visible else ""
        if cluster_id == selected_cluster and cluster_visible:
            st.markdown(
                f"<div style='padding:0.4rem 0.6rem;border:2px solid #ff4b4b;border-radius:0.5rem;'>"
                f"<strong>Cluster {cluster_id}</strong>  Top {count} representative samples{header_suffix}"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"**Cluster {cluster_id}**  Top {count} representative samples{header_suffix}")
        cols = st.columns(samples_per_cluster, gap="small")
        for i in range(samples_per_cluster):
            with cols[i]:
                if i < count:
                    sample_idx = int(members[i])
                    similarity = float(scores[i])
                    st.image(images[sample_idx], use_container_width=True)
                    caption = f"#{i + 1} • sim {similarity:.3f}"
                    if sample_idx == selected_index:
                        caption += " • selected"
                    st.caption(caption)
                else:
                    st.empty()
        st.markdown("")


def render_model_eval_tab():
    st.subheader("Model Prediction Demo")
    dataset = load_eval_dataset(str(DEFAULT_EVAL_NPZ))
    num_samples = len(dataset["images"])
    default_index = int(st.session_state.get("eval_sample_index", 0))

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Random sample", use_container_width=True):
            st.session_state["eval_sample_index"] = int(np.random.randint(0, num_samples))
    with c2:
        if st.button("Next sample", use_container_width=True):
            st.session_state["eval_sample_index"] = (int(st.session_state.get("eval_sample_index", 0)) + 1) % num_samples
    with c3:
        sample_index = st.slider(
            "Sample index",
            min_value=0,
            max_value=max(num_samples - 1, 0),
            value=int(st.session_state.get("eval_sample_index", default_index)),
        )
        st.session_state["eval_sample_index"] = sample_index

    with st.spinner("Running model inference..."):
        model, device = load_model(str(DEFAULT_CHECKPOINT))
        pred_joints, pred_vectors, pred_verts = run_inference(model, device, dataset["images"][sample_index])

    gt_joints = dataset["gt_joints"][sample_index].reshape(21, 3)
    gt_verts = dataset["gt_verts"][sample_index].reshape(-1, 3)
    left_col, right_col = st.columns([1, 1.35], gap="medium")
    with left_col:
        fig = render_eval_figure(dataset["images"][sample_index])
        st.pyplot(fig, clear_figure=True, use_container_width=True)
    with right_col:
        plotly_fig = render_interactive_eval_plot(pred_joints, pred_vectors, pred_verts)
        st.plotly_chart(plotly_fig, use_container_width=True)


def render_cluster_tab(
    title: str,
    artifact_path: Path,
    dataset_path: Path,
    feature_key: str,
    guidance_text: str,
    drop_last_two: bool = False,
    preferred_first_cluster: int | None = None,
):
    samples_per_cluster = st.sidebar.slider(
        f"{title} samples/cluster",
        min_value=1,
        max_value=8,
        value=6,
        key=f"{title}_samples",
    )

    render_cluster_section(
        title=title,
        artifact_path=artifact_path,
        dataset_path=dataset_path,
        samples_per_cluster=samples_per_cluster,
        feature_key=feature_key,
        guidance_text=guidance_text,
        drop_last_two=drop_last_two,
        preferred_first_cluster=preferred_first_cluster,
    )


st.title("3D Hand Detection Demo")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Model Eval",
        "Angle Clustering",
        "Pose Clustering",
        "Future Demo 1",
        "Future Demo 2",
    ]
)

with tab1:
    render_model_eval_tab()

with tab2:
    render_cluster_tab(
        "Angle Clustering",
        DEFAULT_ANGLE_CLUSTER,
        REPO_ROOT / "data" / "train_data_600_verts.npz",
        "coord_frames",
        guidance_text=(
            "Use the target image below to sanity-check the clustering. The hands in the matching cluster "
            "should face the same direction as the target image."
        ),
    )

with tab3:
    render_cluster_tab(
        "Pose Clustering",
        DEFAULT_POSE_CLUSTER,
        REPO_ROOT / "data" / "train_data_600_verts.npz",
        "pose_features",
        guidance_text=(
            "Use the target image below to sanity-check the clustering. The hands in the matching cluster "
            "should have a similar hand pose to the target image."
        ),
        preferred_first_cluster=4,
    )

with tab4:
    st.info("Reserved for the next demo module.")

with tab5:
    st.info("Reserved for another upcoming demo module.")
