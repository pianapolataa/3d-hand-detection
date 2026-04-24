import numpy as np
import torch
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

from cosine_kmeans import CosineKMeans

from frame_utils import (
    NUM_JOINTS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_CLUSTERS,
    DEFAULT_RANDOM_SEED,
    build_inference_model,
    infer_model_dims_from_checkpoint,
    preprocess_image_batch,
    default_image_transform,
    compute_pose_features,
    resolve_existing_path,
    safe_normalize,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "train_data_600_verts.npz"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "clustering" / "pose_cluster_outputs"
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
    parser.add_argument("--num-verts", type=int, default=None)
    parser.add_argument("--num-vectors", type=int, default=None)
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

def extract_predicted_pose_features(images, model, batch_size, device, num_vectors):
        transform = default_image_transform()
        feature_batches = []
        joint_batches = []

        for start_idx in tqdm(range(0, len(images), batch_size), desc="Extracting pose features"):
            batch_np = images[start_idx : start_idx + batch_size]
            input_batch = preprocess_image_batch(batch_np, transform).to(device)

            with torch.no_grad():
                pred_joints, pred_vectors, _ = model(input_batch)

            pred_joints_np = pred_joints.cpu().numpy().reshape(-1, NUM_JOINTS, 3)
            pred_vectors_np = pred_vectors.cpu().numpy().reshape(-1, num_vectors, 3)

            pose_features = compute_pose_features(pred_joints_np, pred_vectors_np)
            feature_batches.append(pose_features)
            joint_batches.append(pred_joints_np.astype(np.float32))

        return np.concatenate(feature_batches, axis=0), np.concatenate(joint_batches, axis=0)

def save_pose_dataset(
    output_path: Path,
    pose_features: np.ndarray,
    sample_indices: np.ndarray,
    split: str,
    source_npz_path: Path,
) -> None:
    np.savez_compressed(
        output_path,
        pose_features=pose_features.astype(np.float32),
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


def rank_cluster_members(
    pose_features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    normalized_features = safe_normalize(pose_features.astype(np.float32))
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


def save_cluster_rankings_csv(
    output_path: Path,
    ranked_members_by_cluster: dict[int, np.ndarray],
    ranked_scores_by_cluster: dict[int, np.ndarray],
    sample_indices: np.ndarray,
) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cluster_id", "rank", "dataset_index", "source_sample_index", "cosine_similarity"])
        for cluster_id in sorted(ranked_members_by_cluster):
            ranked_members = ranked_members_by_cluster[cluster_id]
            ranked_scores = ranked_scores_by_cluster[cluster_id]
            for rank, (sample_idx, score) in enumerate(zip(ranked_members.tolist(), ranked_scores.tolist()), start=1):
                writer.writerow([cluster_id, rank, sample_idx, int(sample_indices[sample_idx]), f"{score:.6f}"])


def save_cluster_preview_grid(
    images: np.ndarray,
    sample_indices: np.ndarray,
    ranked_members_by_cluster: dict[int, np.ndarray],
    ranked_scores_by_cluster: dict[int, np.ndarray],
    output_path: Path,
    samples_per_cluster: int,
) -> None:
    cluster_ids = np.array(sorted(ranked_members_by_cluster))
    num_rows = len(cluster_ids)
    num_cols = samples_per_cluster

    fig = plt.figure(figsize=(3.6 * num_cols, 3.4 * num_rows))

    for row_idx, cluster_id in enumerate(cluster_ids.tolist()):
        ranked_members = ranked_members_by_cluster[cluster_id]
        ranked_scores = ranked_scores_by_cluster[cluster_id]
        sample_size = min(samples_per_cluster, len(ranked_members))

        for col_idx in range(samples_per_cluster):
            subplot_idx = row_idx * num_cols + col_idx + 1
            ax = fig.add_subplot(num_rows, num_cols, subplot_idx)
            ax.axis("off")

            if col_idx >= sample_size:
                continue

            sample_idx = int(ranked_members[col_idx])
            similarity = float(ranked_scores[col_idx])
            original_idx = int(sample_indices[sample_idx])

            ax.imshow(images[sample_idx])
            ax.set_title(
                f"c{cluster_id} #{col_idx + 1} | sim {similarity:.3f}\nds {sample_idx} | src {original_idx}",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    images, detected_num_verts, sample_indices = load_images(args.npz_path, args.split)
    checkpoint_dims = infer_model_dims_from_checkpoint(args.checkpoint_path)
    resolved_num_verts = checkpoint_dims["num_verts"] if args.num_verts is None else args.num_verts
    resolved_num_vectors = checkpoint_dims["num_vectors"] if args.num_vectors is None else args.num_vectors

    if resolved_num_verts != detected_num_verts:
        raise ValueError(
            f"--num-verts={resolved_num_verts} does not match dataset verts={detected_num_verts} "
            f"for {args.npz_path}"
        )
    if resolved_num_verts != checkpoint_dims["num_verts"]:
        raise ValueError(
            f"--num-verts={resolved_num_verts} does not match checkpoint verts={checkpoint_dims['num_verts']} "
            f"for {args.checkpoint_path}"
        )
    if resolved_num_vectors != checkpoint_dims["num_vectors"]:
        raise ValueError(
            f"--num-vectors={resolved_num_vectors} does not match checkpoint vectors={checkpoint_dims['num_vectors']} "
            f"for {args.checkpoint_path}"
        )

    print(f"Using device: {device}")
    print(f"Loaded {len(images)} images from split '{args.split}'")
    print(f"NUM_VERTS={resolved_num_verts} | NUM_VECTORS={resolved_num_vectors}")

    model = build_inference_model(
        checkpoint_path=args.checkpoint_path,
        num_joints=checkpoint_dims["num_joints"],
        num_verts=resolved_num_verts,
        num_vectors=resolved_num_vectors,
        device=device,
    )

    pose_features, pred_joints = extract_predicted_pose_features(
        images=images,
        model=model,
        batch_size=args.batch_size,
        device=device,
        num_vectors=resolved_num_vectors,
    )

    print("pose_features shape:", pose_features.shape)
    print("pred_joints shape:", pred_joints.shape)

    pose_dataset_path = args.output_dir / f"{args.split}_pose_features.npz"
    save_pose_dataset(
        output_path=pose_dataset_path,
        pose_features=pose_features,
        sample_indices=sample_indices,
        split=args.split,
        source_npz_path=args.npz_path,
    )
    print(f"Saved pose dataset to {pose_dataset_path}")

    clusterer = CosineKMeans(
        n_clusters=args.num_clusters,
        max_iters = args.max_iters,
        n_init = args.n_init,
        random_state=args.seed,
        verbose=True)
    clusterer.fit(pose_features)

    labels = clusterer.labels_
    centers = clusterer.centers_

    assignments_path = args.output_dir / f"{args.split}_clusters_k{args.num_clusters}.npz"
    np.savez_compressed(
        assignments_path,
        labels=labels.astype(np.int32),
        centers=centers.astype(np.float32),
        pose_features=pose_features.astype(np.float32),
        pred_joints=pred_joints.astype(np.float32),
        sample_indices=sample_indices.astype(np.int32),
        objective=np.array(clusterer.objective_, dtype=np.float32),
        num_iters=np.array(clusterer.n_iter_, dtype=np.int32),
        num_clusters=np.array(args.num_clusters, dtype=np.int32),
        source_npz_path=np.array(str(args.npz_path)),
        checkpoint_path=np.array(str(args.checkpoint_path)),
        num_verts=np.array(resolved_num_verts, dtype=np.int32),
        num_vectors=np.array(resolved_num_vectors, dtype=np.int32),
    )
    print(f"Saved clustering outputs to {assignments_path}")

    print("cluster labels:", labels)
    print("cluster distribution:", np.bincount(labels))

    ranked_members_by_cluster, ranked_scores_by_cluster = rank_cluster_members(pose_features, labels, centers)

    summary_csv_path = args.output_dir / f"{args.split}_cluster_summary_k{args.num_clusters}.csv"
    save_cluster_summary_csv(summary_csv_path, labels)
    print(f"Saved cluster summary to {summary_csv_path}")

    rankings_csv_path = args.output_dir / f"{args.split}_cluster_rankings_k{args.num_clusters}.csv"
    save_cluster_rankings_csv(
        output_path=rankings_csv_path,
        ranked_members_by_cluster=ranked_members_by_cluster,
        ranked_scores_by_cluster=ranked_scores_by_cluster,
        sample_indices=sample_indices,
    )
    print(f"Saved cluster rankings to {rankings_csv_path}")

    preview_path = args.output_dir / f"{args.split}_cluster_previews_k{args.num_clusters}.png"
    save_cluster_preview_grid(
        images=images,
        sample_indices=sample_indices,
        ranked_members_by_cluster=ranked_members_by_cluster,
        ranked_scores_by_cluster=ranked_scores_by_cluster,
        output_path=preview_path,
        samples_per_cluster=args.samples_per_cluster,
    )
    print(f"Saved cluster previews to {preview_path}")

if __name__ == "__main__":
    main()

