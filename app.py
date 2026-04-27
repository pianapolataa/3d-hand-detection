from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image, ImageDraw
from streamlit_webrtc import WebRtcMode, webrtc_streamer
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
DEFAULT_RETRIEVAL_INDEX = REPO_ROOT / "data" / "retrieval_index_600_verts.npz"
DEFAULT_LANDMARKER_PATH = REPO_ROOT / "knn" / "hand_landmarker.task"
DEFAULT_REFERENCE_POSES = REPO_ROOT / "knn" / "reference_poses.npz"

SKEL_CONNECTIONS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20),
]

CLOUD_SIZE = 180
MATCH_THRESHOLD = 0.92
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17
FINGERTIPS = [4, 8, 12, 16, 20]
POSE_HINTS = {
    "FIST": "all fingers curled into a fist",
    "OPEN_PALM": "all five fingers fully extended",
    "PEACE": "index + middle up, others curled",
    "THUMBS_UP": "thumb extended up, other fingers curled",
    "POINT": "index finger extended, others curled",
}


def _list_npz_files() -> list[Path]:
    return sorted(REPO_ROOT.glob("**/*.npz"))


def _list_cluster_files() -> list[Path]:
    return sorted(REPO_ROOT.glob("clustering/**/*clusters*.npz"))


def _coerce_scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _safe_normalize(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, eps, None)


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
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image)
    ax.set_title("Input image",fontsize=18, fontweight="bold", pad=12)
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
        title={
        "text": "Prediction Preview",
        "font": {"size": 18, "weight": "bold"},
        "x":0.5,
        "xanchor": "center",
    },
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


@st.cache_data(show_spinner=False)
def load_retrieval_index(index_path: str):
    with np.load(index_path, allow_pickle=False) as data:
        return {
            "images": data["images"],
            "embeddings": data["embeddings"].astype(np.float32),
            "gt_joints": data["gt_joints"].astype(np.float32),
            "gt_verts": data["gt_verts"].astype(np.float32),
        }


@st.cache_resource(show_spinner=False)
def load_hand_landmarker(model_path: str):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


@st.cache_data(show_spinner=False)
def load_reference_poses(reference_path: str):
    with np.load(reference_path, allow_pickle=False) as data:
        return list(data["names"]), data["embeddings"].astype(np.float32)


def detect_hand_landmarks(image_rgb: np.ndarray):
    import mediapipe as mp

    landmarker = load_hand_landmarker(str(DEFAULT_LANDMARKER_PATH))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection = landmarker.detect(mp_image)
    if not detection.hand_landmarks:
        return None, None
    landmarks = detection.hand_landmarks[0]
    joints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    return landmarks, joints


def landmarks_to_embedding(landmarks) -> np.ndarray | None:
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    vec = pts.flatten()
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return None
    return vec / norm


def hand_frame_axes(joints: np.ndarray):
    wrist = joints[WRIST]
    index_mcp = joints[INDEX_MCP] - wrist
    pinky_mcp = joints[PINKY_MCP] - wrist
    middle_mcp = joints[MIDDLE_MCP] - wrist

    v1 = _safe_normalize(np.cross(index_mcp, pinky_mcp))
    v2 = _safe_normalize(middle_mcp - np.dot(middle_mcp, v1) * v1)
    v3 = _safe_normalize(np.cross(v1, v2))
    return v1.astype(np.float32), v2.astype(np.float32), v3.astype(np.float32)


def to_hand_frame(joints: np.ndarray):
    centered = joints - joints[WRIST]
    v1, v2, v3 = hand_frame_axes(joints)
    rotation = np.stack([v1, v2, v3], axis=0)
    return (centered @ rotation.T).astype(np.float32)


def fingertip_embedding(joints: np.ndarray) -> np.ndarray | None:
    hand_frame = to_hand_frame(joints)
    tips = hand_frame[FINGERTIPS]
    dists = np.linalg.norm(tips, axis=1)
    feat = np.concatenate([tips.reshape(-1), dists])
    norm = np.linalg.norm(feat)
    if norm < 1e-6:
        return None
    return (feat / norm).astype(np.float32)


def match_pose(query_emb: np.ndarray, ref_names: list[str], ref_embs: np.ndarray):
    sims = ref_embs @ query_emb
    best = int(np.argmax(sims))
    return ref_names[best], float(sims[best])


def draw_skeleton(frame_bgr: np.ndarray, joints: np.ndarray):
    height, width = frame_bgr.shape[:2]
    points = [(int(j[0] * width), int(j[1] * height)) for j in joints]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame_bgr, points[a], points[b], (0, 220, 0), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame_bgr, point, 4, (0, 255, 255), -1, cv2.LINE_AA)


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.8,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
):
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


class PoseMatchProcessor:
    def __init__(self):
        self.landmarker = load_hand_landmarker(str(DEFAULT_LANDMARKER_PATH))
        self.ref_names, self.ref_embs = load_reference_poses(str(DEFAULT_REFERENCE_POSES))

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmarks, joints = detect_hand_landmarks(frame_rgb)

        if joints is None:
            put_text(image, "No hand detected", (30, 50), scale=0.9, color=(0, 220, 220))
        else:
            draw_skeleton(image, joints)
            emb = fingertip_embedding(joints)
            if emb is not None:
                name, sim = match_pose(emb, self.ref_names, self.ref_embs)
                if sim >= MATCH_THRESHOLD:
                    label, color = name, (80, 220, 80)
                else:
                    label, color = "(no confident match)", (80, 200, 220)
                put_text(image, label, (30, 60), scale=1.4, color=color, thickness=3)
                put_text(image, f"sim = {sim:.3f}", (30, 110), scale=0.8, color=(255, 255, 255))

        put_text(image, "Live hand pose demo", (30, image.shape[0] - 50), scale=0.7, color=(180, 180, 180))
        put_text(image, "Move your hand to match a reference pose", (30, image.shape[0] - 20), scale=0.6, color=(180, 180, 180))
        return av.VideoFrame.from_ndarray(image, format="bgr24")


def cosine_search(query_emb: np.ndarray, embeddings: np.ndarray, k: int):
    sims = embeddings @ query_emb
    top_k = np.argsort(sims)[::-1][:k]
    return top_k, sims[top_k]


def draw_hand_overlay(image_rgb: np.ndarray, landmarks) -> Image.Image:
    image = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(image)
    width, height = image.size
    points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

    for a, b in HAND_CONNECTIONS:
        draw.line([points[a], points[b]], fill=(0, 220, 0), width=3)
    for x, y in points:
        r = 4
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 220, 0), outline=(0, 160, 0))

    return image


def render_cloud_2d(verts: np.ndarray, joints: np.ndarray, size: int = CLOUD_SIZE) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    pts2d = np.vstack([verts[:, :2], joints[:, :2]])
    mins, maxs = pts2d.min(axis=0), pts2d.max(axis=0)
    span = (maxs - mins).max()
    if span < 1e-6:
        return img

    scale = (size * 0.82) / span
    center = (mins + maxs) / 2.0

    def to_px(points: np.ndarray):
        px = ((points[:, :2] - center) * scale + size / 2).astype(int)
        px[:, 1] = size - 1 - px[:, 1]
        return px

    verts_px = to_px(verts)
    joints_px = to_px(joints)

    for px in verts_px:
        if 0 <= px[0] < size and 0 <= px[1] < size:
            cv2.circle(img, (int(px[0]), int(px[1])), 2, (237, 149, 100), -1)

    for finger in SKEL_CONNECTIONS:
        for idx in range(len(finger) - 1):
            cv2.line(
                img,
                tuple(joints_px[finger[idx]]),
                tuple(joints_px[finger[idx + 1]]),
                (80, 80, 220),
                1,
                cv2.LINE_AA,
            )

    for px in joints_px:
        if 0 <= px[0] < size and 0 <= px[1] < size:
            cv2.circle(img, (int(px[0]), int(px[1])), 3, (0, 165, 255), -1)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False)
def prerender_clouds(index_path: str):
    index = load_retrieval_index(index_path)
    return np.stack(
        [
            render_cloud_2d(index["gt_verts"][i].reshape(-1, 3), index["gt_joints"][i].reshape(21, 3))
            for i in range(len(index["images"]))
        ],
        axis=0,
    )


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
    st.write("")

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
                f"<div style='padding:0.4rem 0.6rem;border:2px solid #ff4b4b;border-radius:0.5rem;margin-bottom: 1rem;'>"
                f"<strong>🔴Cluster {cluster_id}</strong>  Top {count} representative samples{header_suffix}"
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
                    caption = f"#{i + 1} • similarity {similarity:.3f}"
                    if sample_idx == selected_index:
                        caption += " • selected"
                    st.caption(caption)
                else:
                    st.empty()
        st.divider()
        st.markdown("")


def render_model_eval_tab():
    st.subheader("Model Prediction Demo")
    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)
    dataset = load_eval_dataset(str(DEFAULT_EVAL_NPZ))
    num_samples = len(dataset["images"])
    default_index = int(st.session_state.get("eval_sample_index", 0))

    c1, c2, c3 = st.columns([0.7, 0.7, 1.5])
    with c1:
        st.write("")
        if st.button("Random sample", use_container_width=True):
            st.session_state["eval_sample_index"] = int(np.random.randint(0, num_samples))
    with c2:
        st.write("")
        if st.button("Next sample", use_container_width=True):
            st.session_state["eval_sample_index"] = (int(st.session_state.get("eval_sample_index", 0)) + 1) % num_samples
    with c3:
        st.markdown("Sample index")
        sample_index = st.slider(
            "",
            min_value=0,
            max_value=max(num_samples - 1, 0),
            value=int(st.session_state.get("eval_sample_index", default_index)),
            label_visibility="collapsed",
        )
        st.session_state["eval_sample_index"] = sample_index

    with st.spinner("Running model inference..."):
        model, device = load_model(str(DEFAULT_CHECKPOINT))
        pred_joints, pred_vectors, pred_verts = run_inference(model, device, dataset["images"][sample_index])
    st.markdown("<div style='margin-bottom: 1.8rem;'></div>", unsafe_allow_html=True)

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


def render_live_pose_demo_tab():
    st.subheader("Live Hand Pose Demo")
    st.info(
        "This is the Streamlit version of `knn/demo.py`. It runs a live webcam stream, detects a single hand "
        "with MediaPipe, builds the same hand-frame fingertip embedding, and overlays the best matching pose label."
    )
    ref_names, _ = load_reference_poses(str(DEFAULT_REFERENCE_POSES))
    st.caption(f"Loaded reference poses: {', '.join(ref_names)}")
    st.caption("Allow camera access in your browser, then press Start.")

    webrtc_streamer(
        key="live-pose-demo",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=PoseMatchProcessor,
        async_processing=True,
    )


st.title("3D Hand Detection Demo")
st.sidebar.markdown("## ⚙️ Display Settings")
st.sidebar.caption("Adjust how many samples are shown per cluster.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Model Eval",
        "Angle Clustering",
        "Pose Clustering",
        "Live Pose Demo",
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
    render_live_pose_demo_tab()

with tab5:
    st.info("Reserved for another upcoming demo module.")
