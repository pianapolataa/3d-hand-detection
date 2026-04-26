from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "train_data_600_verts.npz"

SPLIT_TO_KEYS = {
    "train": ("x_train", "y_train_verts"),
    "val": ("x_val", "y_val_verts"),
    "test": ("x_test", "y_test_verts"),
}

WRIST_ID = 0
INDEX_MCP_ID = 5
MIDDLE_MCP_ID = 9
PINKY_MCP_ID = 17


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reload a saved clustering run and regenerate cluster preview grids.",
    )
    parser.add_argument(
        "--clusters-path",
        type=Path,
        required=True,
        help="Path to a saved *_clusters_k*.npz artifact from run_camera_angle_clustering.py",
    )
    parser.add_argument(
        "--npz-path",
        type=Path,
        default=None,
        help="Optional dataset override. If omitted, use the path saved in the clustering artifact.",
    )
    parser.add_argument(
        "--split",
        choices=sorted(SPLIT_TO_KEYS.keys()),
        default=None,
        help="Optional split override. If omitted, infer from the artifact filename when possible.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Deprecated. Preview is displayed interactively instead of being written to disk.",
    )
    parser.add_argument("--samples-per-cluster", type=int, default=6)
    return parser.parse_args()


def _coerce_scalar(value: np.ndarray | str | int) -> str | int:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _infer_split(clusters_path: Path) -> str | None:
    stem = clusters_path.stem
    for split in SPLIT_TO_KEYS:
        if stem.startswith(f"{split}_"):
            return split
    return None


def _default_output_path(clusters_path: Path) -> Path:
    stem = clusters_path.stem
    preview_stem = stem.replace("_clusters_", "_cluster_previews_eval_")
    if preview_stem == stem:
        preview_stem = f"{stem}_preview_eval"
    return clusters_path.with_name(f"{preview_stem}.png")


class _TqdmReader:
    def __init__(self, file_obj, progress_bar: tqdm) -> None:
        self._file_obj = file_obj
        self._progress_bar = progress_bar

    def read(self, size: int = -1):
        chunk = self._file_obj.read(size)
        self._progress_bar.update(len(chunk))
        return chunk

    def seek(self, offset: int, whence: int = 0):
        return self._file_obj.seek(offset, whence)

    def tell(self) -> int:
        return self._file_obj.tell()

    def close(self) -> None:
        self._file_obj.close()

    def seekable(self) -> bool:
        return self._file_obj.seekable()

    def readable(self) -> bool:
        return self._file_obj.readable()

    def __getattr__(self, name: str):
        return getattr(self._file_obj, name)


def safe_normalize(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, eps, None)


def compute_hand_frame_components(pred_joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wrist = pred_joints[:, WRIST_ID, :]
    index_mcp = pred_joints[:, INDEX_MCP_ID, :] - wrist
    pinky_mcp = pred_joints[:, PINKY_MCP_ID, :] - wrist
    middle_mcp = pred_joints[:, MIDDLE_MCP_ID, :] - wrist

    palm_normal = np.cross(index_mcp, pinky_mcp)
    palm_normal = safe_normalize(palm_normal)
    middle_axis = safe_normalize(middle_mcp)
    return palm_normal.astype(np.float32), middle_axis.astype(np.float32)


def rank_cluster_members(
    coord_frames: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    normalized_features = safe_normalize(coord_frames.astype(np.float32))
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


def _set_axes_equal(ax, joints: np.ndarray) -> None:
    mins = joints.min(axis=0)
    maxs = joints.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.25)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_cluster_preview_grid(
    images: np.ndarray,
    sample_indices: np.ndarray,
    ranked_members_by_cluster: dict[int, np.ndarray],
    ranked_scores_by_cluster: dict[int, np.ndarray],
    samples_per_cluster: int,
) -> None:
    cluster_ids = np.array(sorted(ranked_members_by_cluster))
    num_rows = len(cluster_ids)
    fig_width = min(max(1.7 * samples_per_cluster, 5.2), 9.0)
    fig_height = min(max(1.45 * num_rows, 2.6), 6.2)
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    subfigs = fig.subfigures(num_rows, 1, hspace=0.28)

    if num_rows == 1:
        subfigs = [subfigs]

    for row_idx, cluster_id in enumerate(cluster_ids.tolist()):
        row_fig = subfigs[row_idx]
        ranked_members = ranked_members_by_cluster[cluster_id]
        ranked_scores = ranked_scores_by_cluster[cluster_id]
        sample_size = min(samples_per_cluster, len(ranked_members))
        row_fig.suptitle(
            f"Cluster {cluster_id}   Top {sample_size} representative sample{'s' if sample_size != 1 else ''}",
            fontsize=9,
            fontweight="semibold",
            y=1.2,
        )
        row_axes = row_fig.subplots(
            1,
            samples_per_cluster,
            squeeze=False,
            gridspec_kw={"wspace": 0.01},
        )

        for sample_slot in range(samples_per_cluster):
            ax_img = row_axes[0, sample_slot]

            ax_img.axis("off")

            if sample_slot >= sample_size:
                continue

            sample_idx = int(ranked_members[sample_slot])
            similarity = float(ranked_scores[sample_slot])

            ax_img.imshow(images[sample_idx])
            ax_img.set_title(
                f"#{sample_slot + 1}  sim {similarity:.3f}",
                fontsize=7,
                pad=1,
            )

    fig.suptitle("Camera-Angle Cluster Previews", fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.99, top=0.9, bottom=0.04, hspace=0.04)
    plt.show()


def load_cluster_artifact(
    clusters_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path | None]:
    with tqdm(total=6, desc="Loading cluster artifact", unit="step") as pbar:
        with np.load(clusters_path, allow_pickle=False) as data:
            pbar.update(1)
            labels = data["labels"].astype(np.int32)
            pbar.update(1)
            centers = data["centers"].astype(np.float32)
            pbar.update(1)
            coord_frames = data["coord_frames"].astype(np.float32)
            pbar.update(1)
            pred_joints = data["pred_joints"].astype(np.float32)
            sample_indices = data["sample_indices"].astype(np.int32)
            pbar.update(1)
            source_npz_path = None
            if "source_npz_path" in data.files:
                source_npz_value = _coerce_scalar(data["source_npz_path"])
                source_npz_path = Path(str(source_npz_value))
            pbar.update(1)

    return labels, centers, coord_frames, pred_joints, sample_indices, source_npz_path


def load_images_with_progress(npz_path: Path, split: str) -> np.ndarray:
    image_key, _ = SPLIT_TO_KEYS[split]
    member_name = f"{image_key}.npy"
    with zipfile.ZipFile(npz_path) as archive:
        member_info = archive.getinfo(member_name)
        with archive.open(member_name) as member_file, tqdm(
            total=member_info.file_size,
            desc=f"Loading dataset images",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            images = np.load(_TqdmReader(member_file, pbar), allow_pickle=False)
    return images


def main() -> None:
    args = parse_args()

    labels, centers, coord_frames, pred_joints, sample_indices, saved_npz_path = load_cluster_artifact(
        args.clusters_path
    )

    split = args.split or _infer_split(args.clusters_path) or "train"
    npz_path = args.npz_path or saved_npz_path or DEFAULT_DATA_PATH
    if split not in SPLIT_TO_KEYS:
        raise ValueError(f"Unsupported split '{split}'. Expected one of {sorted(SPLIT_TO_KEYS)}")
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {npz_path}. Pass --npz-path to point at the local processed dataset."
        )

    print(f"Cluster artifact: {args.clusters_path}")
    print(f"Dataset: {npz_path}")
    print(f"Split: {split}")

    images = load_images_with_progress(npz_path=npz_path, split=split)

    if len(images) != len(labels):
        raise ValueError(
            f"Image count mismatch: dataset split has {len(images)} images, "
            f"but clustering artifact has {len(labels)} labels."
        )
    if len(pred_joints) != len(labels):
        raise ValueError(
            f"Joint count mismatch: clustering artifact has {len(pred_joints)} predicted joint sets "
            f"for {len(labels)} labels."
        )
    if len(sample_indices) != len(labels):
        raise ValueError(
            f"Sample-index mismatch: clustering artifact has {len(sample_indices)} sample indices "
            f"for {len(labels)} labels."
        )

    ranked_members_by_cluster, ranked_scores_by_cluster = rank_cluster_members(coord_frames, labels, centers)

    save_cluster_preview_grid(
        images=images,
        sample_indices=sample_indices,
        ranked_members_by_cluster=ranked_members_by_cluster,
        ranked_scores_by_cluster=ranked_scores_by_cluster,
        samples_per_cluster=args.samples_per_cluster,
    )
    print("Displayed cluster previews.")


if __name__ == "__main__":
    main()

