"""
Offline kNN pose retrieval over the prebuilt index.

Picks a query by dataset index (or random), computes cosine similarity against
all stored joint embeddings, and displays the query + top-k matches.

"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INDEX = os.path.join(REPO_ROOT, "data", "retrieval_index_600_verts.npz")
NUM_VERTS     = 600

SKEL_CONNECTIONS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]


def parse_args():
    p = argparse.ArgumentParser(description="kNN pose retrieval.")
    p.add_argument("--index",    default=DEFAULT_INDEX)
    p.add_argument("--query",    type=int, default=None, help="Dataset index (default: random)")
    p.add_argument("--k",        type=int, default=5)
    p.add_argument("--num-verts",type=int, default=NUM_VERTS)
    p.add_argument("--elev",     type=float, default=-90)
    p.add_argument("--azim",     type=float, default=-90)
    return p.parse_args()


def cosine_search(query_emb, embeddings, k, exclude=-1):
    sims = embeddings @ query_emb
    if exclude >= 0:
        sims[exclude] = -np.inf
    top_k = np.argsort(sims)[::-1][:k]
    return top_k, sims[top_k]


def _draw_cloud(ax, verts, joints, title="", elev=-90, azim=-90):
    verts  = verts.reshape(-1, 3)
    joints = joints.reshape(21, 3)
    ax.scatter(verts[:, 0],  verts[:, 1],  verts[:, 2],  s=6,  c="steelblue", alpha=0.35)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=30, c="orange",    zorder=5)
    for finger in SKEL_CONNECTIONS:
        ax.plot(joints[finger, 0], joints[finger, 1], joints[finger, 2],
                color="tomato", linewidth=1.5)
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def main():
    args = parse_args()

    if not os.path.exists(args.index):
        print(f"Index not found: {args.index}")
        print("Run  python3 knn/build_index.py  first.")
        raise SystemExit(1)

    d          = np.load(args.index)
    images     = d["images"]      # (N, 224, 224, 3)
    embeddings = d["embeddings"]  # (N, 63) L2-normalised
    gt_joints  = d["gt_joints"]   # (N, 63)
    gt_verts   = d["gt_verts"]    # (N, V*3)
    N          = len(images)

    query_idx = int(np.clip(
        args.query if args.query is not None else np.random.randint(0, N),
        0, N - 1,
    ))
    print(f"Query index: {query_idx}  (dataset size: {N})")

    top_k_idx, top_k_sims = cosine_search(embeddings[query_idx], embeddings,
                                           k=args.k, exclude=query_idx)
    for rank, (i, s) in enumerate(zip(top_k_idx, top_k_sims), 1):
        print(f"  Rank {rank}: index={i:4d}  similarity={s:.4f}")

    ncols = args.k + 1
    fig = plt.figure(figsize=(4.2 * ncols, 8))
    fig.suptitle(
        f"Query #{query_idx}  —  Top-{args.k} retrievals  (cosine sim on GT joint vectors)",
        fontsize=13, y=1.01,
    )

    ax = fig.add_subplot(2, ncols, 1)
    ax.imshow(images[query_idx]); ax.set_title("Query image"); ax.axis("off")

    for col, (db_idx, sim) in enumerate(zip(top_k_idx, top_k_sims)):
        ax = fig.add_subplot(2, ncols, col + 2)
        ax.imshow(images[db_idx])
        ax.set_title(f"#{db_idx}  sim={sim:.3f}", fontsize=9)
        ax.axis("off")

    ax = fig.add_subplot(2, ncols, ncols + 1, projection="3d")
    _draw_cloud(ax, gt_verts[query_idx], gt_joints[query_idx],
                title="Query cloud", elev=args.elev, azim=args.azim)

    for col, (db_idx, sim) in enumerate(zip(top_k_idx, top_k_sims)):
        ax = fig.add_subplot(2, ncols, ncols + col + 2, projection="3d")
        _draw_cloud(ax, gt_verts[db_idx], gt_joints[db_idx],
                    title=f"Rank {col + 1}", elev=args.elev, azim=args.azim)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
