"""
Build kNN retrieval index from FreiHAND eval set.

Uses GT joint positions (already wrist-centered and scale-normalized in the
.npz) as embeddings. No model inference needed — the index is purely geometric.

At query time (live_demo.py), MediaPipe landmarks are centered on the wrist
and L2-normalized the same way, giving a consistent cosine-similarity space.
"""

import argparse
import os

import numpy as np

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_NPZ = os.path.join(REPO_ROOT, "data", "eval_data_600_verts.npz")
DEFAULT_OUT = os.path.join(REPO_ROOT, "data", "retrieval_index_600_verts.npz")
NUM_VERTS   = 600


def parse_args():
    p = argparse.ArgumentParser(description="Build kNN retrieval index from GT joints.")
    p.add_argument("--npz",       default=DEFAULT_NPZ, help="Path to eval .npz")
    p.add_argument("--out",       default=DEFAULT_OUT, help="Output index path")
    p.add_argument("--num-verts", type=int, default=NUM_VERTS)
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.npz):
        print(f"Dataset not found: {args.npz}")
        raise SystemExit(1)

    data      = np.load(args.npz)
    images    = data["x_test"]         # (N, 224, 224, 3)  uint8
    gt_joints = data["y_test_joints"]  # (N, 63)  wrist-centered, scale-normalized
    gt_verts  = data["y_test_verts"]   # (N, V*3)
    N         = len(images)
    print(f"Dataset: {N} samples  ({args.npz})")

    # L2-normalise joint vectors → unit sphere for cosine search
    norms      = np.linalg.norm(gt_joints, axis=1, keepdims=True)
    embeddings = gt_joints / np.maximum(norms, 1e-8)   # (N, 63)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        images=images,
        embeddings=embeddings,  # (N, 63) L2-normalised
        gt_joints=gt_joints,    # (N, 63) for display
        gt_verts=gt_verts,      # (N, V*3) for display
    )
    print(f"Index saved → {args.out}")
    print(f"  embeddings : {embeddings.shape}  (L2-normalised, 63-dim joint vectors)")
    print(f"  gt_joints  : {gt_joints.shape}")
    print(f"  gt_verts   : {gt_verts.shape}")


if __name__ == "__main__":
    main()
