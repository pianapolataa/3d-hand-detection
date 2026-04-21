"""
retrieve.py — kNN pose retrieval over the prebuilt index.

Given a query image (by dataset index, or picked at random), computes its
cosine similarity against every stored backbone embedding and displays the
query alongside the top-k most similar samples.

Layout (matplotlib):
  Row 0: query image  |  retrieval 1 image  |  …  |  retrieval k image
  Row 1: query cloud  |  retrieval 1 cloud  |  …  |  retrieval k cloud

Usage (from scripts/):
    python retrieve.py                          # random query, k=5
    python retrieve.py --query 42               # specific sample
    python retrieve.py --query 7 --k 3          # top-3
    python retrieve.py --use-gt-verts           # show GT point clouds instead of predicted
    python retrieve.py --index ../data/retrieval_index_778_verts.npz --num-verts 778
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from model import ScaffoldedPointPredictor

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_INDEX = "../data/retrieval_index_600_verts.npz"
DEFAULT_MODEL = "checkpoints/model_600_verts_15_vectors.pth"
NUM_VERTS     = 600
NUM_VECTORS   = 15

PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

SKEL_CONNECTIONS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="kNN pose retrieval.")
    p.add_argument("--index",      default=DEFAULT_INDEX, help="Path to retrieval index .npz")
    p.add_argument("--model",      default=DEFAULT_MODEL, help="Path to .pth checkpoint (only needed for external images)")
    p.add_argument("--query",      type=int, default=None, help="Dataset index to use as query (default: random)")
    p.add_argument("--k",          type=int, default=5,    help="Number of nearest neighbours to retrieve")
    p.add_argument("--num-verts",  type=int, default=NUM_VERTS)
    p.add_argument("--num-vectors",type=int, default=NUM_VECTORS)
    p.add_argument("--use-gt-verts", action="store_true", help="Display GT point clouds instead of model predictions")
    p.add_argument("--elev",       type=float, default=-90, help="3-D plot elevation angle")
    p.add_argument("--azim",       type=float, default=-90, help="3-D plot azimuth angle")
    return p.parse_args()


# ── Model + hook (used only when embedding a query not in the index) ──────────

def load_model_with_hook(model_path, num_verts, num_vectors, device):
    """Load model and attach a hook to capture backbone features."""
    model = ScaffoldedPointPredictor(
        num_joints=21, num_verts=num_verts, num_vectors=num_vectors,
        pretrained_backbone=False,  # checkpoint already contains fine-tuned backbone weights
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    captured = {}
    def _hook(module, inp, out):
        captured["feat"] = out
    handle = model.backbone_layers.register_forward_hook(_hook)
    return model, captured, handle


def embed_image(pil_img, model, captured, device):
    """Run a PIL image through the model and return (embedding, pred_joints, pred_verts)."""
    tensor = PREPROCESS(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        p_joints, _, p_verts = model(tensor)
    feat = captured["feat"].squeeze().cpu().numpy()  # (512,)
    feat /= max(np.linalg.norm(feat), 1e-8)
    return feat, p_joints.cpu().numpy().flatten(), p_verts.cpu().numpy().flatten()


# ── Similarity search ─────────────────────────────────────────────────────────

def cosine_search(query_emb: np.ndarray, embeddings: np.ndarray, k: int, exclude: int = -1):
    """Return (indices, similarities) of top-k most similar entries."""
    sims = embeddings @ query_emb          # (N,)  — embeddings are already L2-normalised
    if exclude >= 0:
        sims[exclude] = -np.inf            # mask self so it's never returned
    top_k = np.argsort(sims)[::-1][:k]
    return top_k, sims[top_k]


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _draw_point_cloud(ax, verts, joints, title="", elev=-90, azim=-90):
    verts  = verts.reshape(-1, 3)
    joints = joints.reshape(21, 3)

    ax.scatter(verts[:, 0],  verts[:, 1],  verts[:, 2],
               s=6, c="steelblue", alpha=0.35, rasterized=True)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               s=30, c="orange", zorder=5)
    for finger in SKEL_CONNECTIONS:
        ax.plot(joints[finger, 0], joints[finger, 1], joints[finger, 2],
                color="tomato", linewidth=1.5)

    ax.set_title(title, fontsize=9)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def visualise(
    query_img, query_verts, query_joints,
    db_images, db_verts, db_joints,
    db_indices, similarities,
    query_label="Query",
    elev=-90, azim=-90,
):
    k = len(db_indices)
    ncols = k + 1  # query col + k retrieval cols
    fig = plt.figure(figsize=(4.2 * ncols, 8))
    fig.suptitle(
        f"{query_label}  —  Top-{k} retrievals  (cosine similarity on ResNet18 backbone features)",
        fontsize=13, y=1.01,
    )

    # ── Row 0: images ─────────────────────────────────────────────────────────
    ax_qi = fig.add_subplot(2, ncols, 1)
    ax_qi.imshow(query_img)
    ax_qi.set_title("Query image", fontsize=10)
    ax_qi.axis("off")

    for col, (db_idx, sim) in enumerate(zip(db_indices, similarities)):
        ax = fig.add_subplot(2, ncols, col + 2)
        ax.imshow(db_images[db_idx])
        ax.set_title(f"#{db_idx}  sim={sim:.3f}", fontsize=9)
        ax.axis("off")

    # ── Row 1: 3-D point clouds ───────────────────────────────────────────────
    ax_qp = fig.add_subplot(2, ncols, ncols + 1, projection="3d")
    _draw_point_cloud(ax_qp, query_verts, query_joints,
                      title="Query cloud", elev=elev, azim=azim)

    for col, (db_idx, sim) in enumerate(zip(db_indices, similarities)):
        ax = fig.add_subplot(2, ncols, ncols + col + 2, projection="3d")
        _draw_point_cloud(ax, db_verts[db_idx], db_joints[db_idx],
                          title=f"Rank {col + 1}", elev=elev, azim=azim)

    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load index
    if not os.path.exists(args.index):
        print(f"Index not found: {args.index}")
        print("Run  python build_index.py  first.")
        sys.exit(1)

    idx_data = np.load(args.index)
    images     = idx_data["images"]       # (N, 224, 224, 3)
    embeddings = idx_data["embeddings"]   # (N, 512)  L2-normalised
    pred_joints = idx_data["pred_joints"] # (N, 63)
    pred_verts  = idx_data["pred_verts"]  # (N, V*3)
    gt_joints   = idx_data["gt_joints"]
    gt_verts    = idx_data["gt_verts"]

    db_joints = gt_joints  if args.use_gt_verts else pred_joints
    db_verts  = gt_verts   if args.use_gt_verts else pred_verts

    N = len(images)
    query_idx = args.query if args.query is not None else np.random.randint(0, N)
    query_idx = int(np.clip(query_idx, 0, N - 1))
    print(f"Query index: {query_idx}  (dataset size: {N})")

    # Query embedding: pull directly from index (no re-inference needed)
    query_emb    = embeddings[query_idx]
    query_verts  = db_verts[query_idx]
    query_joints = db_joints[query_idx]

    # Retrieval
    top_k_indices, top_k_sims = cosine_search(
        query_emb, embeddings, k=args.k, exclude=query_idx
    )
    print(f"\nTop-{args.k} matches:")
    for rank, (i, s) in enumerate(zip(top_k_indices, top_k_sims), 1):
        print(f"  Rank {rank}: index={i:4d}  similarity={s:.4f}")

    visualise(
        query_img=images[query_idx],
        query_verts=query_verts,
        query_joints=query_joints,
        db_images=images,
        db_verts=db_verts,
        db_joints=db_joints,
        db_indices=top_k_indices,
        similarities=top_k_sims,
        query_label=f"Query #{query_idx}",
        elev=args.elev,
        azim=args.azim,
    )


if __name__ == "__main__":
    main()
