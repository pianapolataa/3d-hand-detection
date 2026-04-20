from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from cosine_kmeans import CosineKMeans
from frame_utils import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_CLUSTERS,
    DEFAULT_NUM_VECTORS,
    DEFAULT_NUM_VERTS,
    DEFAULT_RANDOM_SEED,
    WRIST_ID,
    NUM_JOINTS,
    build_inference_model,
    compute_hand_frame_components,
    compute_hand_frame_features,
    default_image_transform,
    infer_model_dims_from_checkpoint,
    preprocess_image_batch,
    resolve_existing_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "train_data_600_verts.npz"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "clustering" / "outputs"
DEFAULT_CHECKPOINT_PATH = resolve_existing_path(
    [
        REPO_ROOT / "scripts" / "checkpoints" / "model_600_verts_15_vectors.pth",
        REPO_ROOT / "checkpoints" / "model_600_verts_15_vectors.pth",
    ]
)

SPLIT_TO_KEYS = {
    "train": ("x_train", "y_train_verts"),
    "val": ("x_val", "y_val_verts"),
    "test": ("x_test", "y_test_verts"),
}

SPLIT_TO_SAMPLE_INDEX_KEYS = {
    "train": "sample_indices_train",
    "val": "sample_indices_val",
    "test": "sample_indices_test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster hand camera angles with cosine k-means.")
    parser.add_argument("--npz-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--split", choices=sorted(SPLIT_TO_KEYS.keys()), default="train")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-verts", type=int, default=DEFAULT_NUM_VERTS)
    parser.add_argument("--num-vectors", type=int, default=DEFAULT_NUM_VECTORS)
    parser.add_argument("--num-clusters", type=int, default=DEFAULT_NUM_CLUSTERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--samples-per-cluster", type=int, default=6)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_images(npz_path: Path, split: str) -> tuple[np.ndarray, int, np.ndarray]:
    image_key, verts_key = SPLIT_TO_KEYS[split]
    sample_index_key = SPLIT_TO_SAMPLE_INDEX_KEYS[split]
    with np.load(npz_path) as data:
        images = data[image_key]
        num_verts = int(data[verts_key].shape[1] // 3)
        if sample_index_key in data:
            sample_indices = data[sample_index_key].astype(np.int32)
        else:
            sample_indices = np.arange(len(images), dtype=np.int32)
    return images, num_verts, sample_indices


def extract_predicted_hand_frames(
    images: np.ndarray,
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    transform = default_image_transform()
    feature_batches = []
    joint_batches = []

    for start_idx in tqdm(range(0, len(images), batch_size), desc="Extracting hand frames"):
        batch_np = images[start_idx : start_idx + batch_size]
        input_batch = preprocess_image_batch(batch_np, transform).to(device)

        with torch.no_grad():
            pred_joints, _, _ = model(input_batch)

        pred_joints_np = pred_joints.cpu().numpy().reshape(-1, NUM_JOINTS, 3)
        hand_frames = compute_hand_frame_features(pred_joints_np)

        feature_batches.append(hand_frames)
        joint_batches.append(pred_joints_np.astype(np.float32))

    return np.concatenate(feature_batches, axis=0), np.concatenate(joint_batches, axis=0)


def save_frame_dataset(
    output_path: Path,
    coord_frames: np.ndarray,
    sample_indices: np.ndarray,
    split: str,
    source_npz_path: Path,
) -> None:
    np.savez_compressed(
        output_path,
        coord_frames=coord_frames.astype(np.float32),
        sample_indices=sample_indices.astype(np.int32),
        split=np.array(split),
        source_npz_path=np.array(str(source_npz_path)),
    )


def save_cluster_summary_csv(output_path: Path, labels: np.ndarray) -> None:
    unique_labels, counts = np.unique(labels, return_counts=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cluster_id", "count"])
        for cluster_id, count in zip(unique_labels.tolist(), counts.tolist()):
            writer.writerow([cluster_id, count])


def save_cluster_preview_grid(
    images: np.ndarray,
    pred_joints: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
    output_path: Path,
    samples_per_cluster: int,
    seed: int,
) -> None:
    cluster_ids = np.unique(labels)
    rng = np.random.default_rng(seed)
    num_rows = len(cluster_ids)
    num_cols = samples_per_cluster * 2
    fig = plt.figure(figsize=(4.6 * samples_per_cluster, 3.6 * num_rows))

    for row_idx, cluster_id in enumerate(cluster_ids.tolist()):
        members = np.flatnonzero(labels == cluster_id)
        sample_size = min(samples_per_cluster, len(members))
        chosen = rng.choice(members, size=sample_size, replace=False)

        for sample_slot in range(samples_per_cluster):
            image_subplot_idx = row_idx * num_cols + (2 * sample_slot) + 1
            axes_subplot_idx = image_subplot_idx + 1

            ax_img = fig.add_subplot(num_rows, num_cols, image_subplot_idx)
            ax_axes = fig.add_subplot(num_rows, num_cols, axes_subplot_idx, projection="3d")

            ax_img.axis("off")

            if sample_slot >= sample_size:
                ax_axes.axis("off")
                continue

            sample_idx = int(chosen[sample_slot])
            original_idx = int(sample_indices[sample_idx])
            joints = pred_joints[sample_idx]
            wrist = joints[WRIST_ID]
            palm_normal, middle_axis = compute_hand_frame_components(joints[None, :, :])
            palm_normal = palm_normal[0]
            middle_axis = middle_axis[0]

            ax_img.imshow(images[sample_idx])
            ax_img.set_title(f"c{cluster_id} | ds {sample_idx} | src {original_idx}", fontsize=9)

            ax_axes.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=14, c="black", alpha=0.65)
            ax_axes.scatter(wrist[0], wrist[1], wrist[2], s=40, c="orange", marker="o")
            ax_axes.quiver(
                wrist[0], wrist[1], wrist[2],
                palm_normal[0], palm_normal[1], palm_normal[2],
                color="crimson", linewidth=2.0, arrow_length_ratio=0.18,
            )
            ax_axes.quiver(
                wrist[0], wrist[1], wrist[2],
                middle_axis[0], middle_axis[1], middle_axis[2],
                color="royalblue", linewidth=2.0, arrow_length_ratio=0.18,
            )
            _set_axes_equal(ax_axes, joints)
            ax_axes.view_init(elev=20, azim=-60)
            ax_axes.set_xticks([])
            ax_axes.set_yticks([])
            ax_axes.set_zticks([])
            ax_axes.set_title("axes", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _set_axes_equal(ax, joints: np.ndarray) -> None:
    mins = joints.min(axis=0)
    maxs = joints.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.25)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_cluster_axes_grid(
    pred_joints: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
    output_path: Path,
    samples_per_cluster: int,
    seed: int,
) -> None:
    cluster_ids = np.unique(labels)
    rng = np.random.default_rng(seed)
    num_rows = len(cluster_ids)
    num_cols = samples_per_cluster
    fig = plt.figure(figsize=(4.5 * num_cols, 4.0 * num_rows))

    for row_idx, cluster_id in enumerate(cluster_ids.tolist()):
        members = np.flatnonzero(labels == cluster_id)
        sample_size = min(samples_per_cluster, len(members))
        chosen = rng.choice(members, size=sample_size, replace=False)

        for col_idx in range(num_cols):
            subplot_idx = row_idx * num_cols + col_idx + 1
            ax = fig.add_subplot(num_rows, num_cols, subplot_idx, projection="3d")
            if col_idx >= sample_size:
                ax.axis("off")
                continue

            sample_idx = int(chosen[col_idx])
            original_idx = int(sample_indices[sample_idx])
            joints = pred_joints[sample_idx]
            wrist = joints[WRIST_ID]
            palm_normal, middle_axis = compute_hand_frame_components(joints[None, :, :])
            palm_normal = palm_normal[0]
            middle_axis = middle_axis[0]

            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=14, c="black", alpha=0.65)
            ax.scatter(wrist[0], wrist[1], wrist[2], s=40, c="orange", marker="o")
            ax.quiver(
                wrist[0], wrist[1], wrist[2],
                palm_normal[0], palm_normal[1], palm_normal[2],
                color="crimson", linewidth=2.0, arrow_length_ratio=0.18,
            )
            ax.quiver(
                wrist[0], wrist[1], wrist[2],
                middle_axis[0], middle_axis[1], middle_axis[2],
                color="royalblue", linewidth=2.0, arrow_length_ratio=0.18,
            )
            _set_axes_equal(ax, joints)
            ax.view_init(elev=20, azim=-60)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_title(f"c{cluster_id} | ds {sample_idx} | src {original_idx}", fontsize=9)

    fig.suptitle("Predicted Hand Coordinate Axes: red=palm normal, blue=wrist->middle MCP", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    images, detected_num_verts, sample_indices = load_images(args.npz_path, args.split)
    checkpoint_dims = infer_model_dims_from_checkpoint(args.checkpoint_path)

    if args.num_verts != detected_num_verts:
        raise ValueError(
            f"--num-verts={args.num_verts} does not match dataset verts={detected_num_verts} "
            f"for {args.npz_path}"
        )
    if args.num_verts != checkpoint_dims["num_verts"]:
        raise ValueError(
            f"--num-verts={args.num_verts} does not match checkpoint verts={checkpoint_dims['num_verts']} "
            f"for {args.checkpoint_path}"
        )
    if args.num_vectors != checkpoint_dims["num_vectors"]:
        raise ValueError(
            f"--num-vectors={args.num_vectors} does not match checkpoint vectors={checkpoint_dims['num_vectors']} "
            f"for {args.checkpoint_path}"
        )

    print(f"Using device: {device}")
    print(f"Loaded {len(images)} images from split '{args.split}'")
    print(f"NUM_VERTS={args.num_verts} | NUM_VECTORS={args.num_vectors}")

    model = build_inference_model(
        checkpoint_path=args.checkpoint_path,
        num_joints=checkpoint_dims["num_joints"],
        num_verts=args.num_verts,
        num_vectors=args.num_vectors,
        device=device,
    )

    coord_frames, pred_joints = extract_predicted_hand_frames(
        images=images,
        model=model,
        batch_size=args.batch_size,
        device=device,
    )

    frame_dataset_path = args.output_dir / f"{args.split}_coord_frames.npz"
    save_frame_dataset(
        output_path=frame_dataset_path,
        coord_frames=coord_frames,
        sample_indices=sample_indices,
        split=args.split,
        source_npz_path=args.npz_path,
    )
    print(f"Saved frame dataset to {frame_dataset_path}")

    clusterer = CosineKMeans(
        n_clusters=args.num_clusters,
        max_iters=args.max_iters,
        n_init=args.n_init,
        random_state=args.seed,
        verbose=True,
    ).fit(coord_frames)

    labels = clusterer.labels_
    assert labels is not None
    centers = clusterer.centers_
    assert centers is not None

    assignments_path = args.output_dir / f"{args.split}_clusters_k{args.num_clusters}.npz"
    np.savez_compressed(
        assignments_path,
        labels=labels.astype(np.int32),
        centers=centers.astype(np.float32),
        coord_frames=coord_frames.astype(np.float32),
        pred_joints=pred_joints.astype(np.float32),
        sample_indices=sample_indices.astype(np.int32),
        objective=np.array(clusterer.objective_, dtype=np.float32),
        num_iters=np.array(clusterer.n_iter_, dtype=np.int32),
        num_clusters=np.array(args.num_clusters, dtype=np.int32),
        source_npz_path=np.array(str(args.npz_path)),
        checkpoint_path=np.array(str(args.checkpoint_path)),
    )
    print(f"Saved clustering outputs to {assignments_path}")

    summary_csv_path = args.output_dir / f"{args.split}_cluster_summary_k{args.num_clusters}.csv"
    save_cluster_summary_csv(summary_csv_path, labels)
    print(f"Saved cluster summary to {summary_csv_path}")

    preview_path = args.output_dir / f"{args.split}_cluster_previews_k{args.num_clusters}.png"
    save_cluster_preview_grid(
        images=images,
        pred_joints=pred_joints,
        labels=labels,
        sample_indices=sample_indices,
        output_path=preview_path,
        samples_per_cluster=args.samples_per_cluster,
        seed=args.seed,
    )
    print(f"Saved cluster previews to {preview_path}")

    axes_path = args.output_dir / f"{args.split}_cluster_axes_k{args.num_clusters}.png"
    save_cluster_axes_grid(
        pred_joints=pred_joints,
        labels=labels,
        sample_indices=sample_indices,
        output_path=axes_path,
        samples_per_cluster=args.samples_per_cluster,
        seed=args.seed,
    )
    print(f"Saved cluster axes visualization to {axes_path}")


if __name__ == "__main__":
    main()
