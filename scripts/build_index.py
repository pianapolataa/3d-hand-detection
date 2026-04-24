import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from model import ScaffoldedPointPredictor

DEFAULT_NPZ   = "../data/eval_data_600_verts.npz"
DEFAULT_MODEL = "checkpoints/model_600_verts_15_vectors.pth"
DEFAULT_OUT   = "../data/retrieval_index_600_verts.npz"
NUM_VERTS     = 600
NUM_VECTORS   = 15

PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def parse_args():
    p = argparse.ArgumentParser(description="Build kNN retrieval index.")
    p.add_argument("--npz",        default=DEFAULT_NPZ,   help="Path to eval .npz")
    p.add_argument("--model",      default=DEFAULT_MODEL, help="Path to .pth checkpoint")
    p.add_argument("--out",        default=DEFAULT_OUT,   help="Output index path")
    p.add_argument("--num-verts",  type=int, default=NUM_VERTS)
    p.add_argument("--num-vectors",type=int, default=NUM_VECTORS)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = ScaffoldedPointPredictor(
        num_joints=21, num_verts=args.num_verts, num_vectors=args.num_vectors,
        pretrained_backbone=False,  # checkpoint already contains fine-tuned backbone weights
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Loaded model: {args.model}")

    # ── Hook to capture backbone output (shape: [1, 512, 1, 1]) ───────────────
    captured = {}
    def _hook(module, inp, out):
        captured["feat"] = out

    handle = model.backbone_layers.register_forward_hook(_hook)

    # ── Load dataset ──────────────────────────────────────────────────────────
    data = np.load(args.npz)
    images     = data["x_test"]         # (N, 224, 224, 3)  uint8
    gt_joints  = data["y_test_joints"]  # (N, 63)            float32
    gt_verts   = data["y_test_verts"]   # (N, V*3)           float32
    N = len(images)
    V = args.num_verts
    print(f"Dataset: {N} samples, {V} verts each  ({args.npz})")

    # ── Run inference ─────────────────────────────────────────────────────────
    embeddings   = np.zeros((N, 512),    dtype=np.float32)
    pred_joints  = np.zeros((N, 63),     dtype=np.float32)
    pred_verts   = np.zeros((N, V * 3),  dtype=np.float32)

    with torch.no_grad():
        for i in tqdm(range(N), desc="Extracting embeddings"):
            pil = Image.fromarray(images[i])
            tensor = PREPROCESS(pil).unsqueeze(0).to(device)

            p_joints, _, p_verts = model(tensor)

            # backbone feature captured by hook: [1, 512, 1, 1]
            embeddings[i]  = captured["feat"].squeeze().cpu().numpy()
            pred_joints[i] = p_joints.cpu().numpy().flatten()
            pred_verts[i]  = p_verts.cpu().numpy().flatten()

    handle.remove()

    # ── L2-normalise for cosine similarity (dot product at query time) ────────
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / np.maximum(norms, 1e-8)

    # ── Save index ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        images=images,                  # raw uint8 for display
        embeddings=embeddings_norm,     # L2-normalised 512-dim features
        pred_joints=pred_joints,        # (N, 63)  model predictions
        pred_verts=pred_verts,          # (N, V*3) model predictions
        gt_joints=gt_joints,            # (N, 63)  ground truth
        gt_verts=gt_verts,              # (N, V*3) ground truth
    )
    print(f"\nIndex saved → {args.out}")
    print(f"  embeddings : {embeddings_norm.shape}  (L2-normalised)")
    print(f"  pred_joints: {pred_joints.shape}")
    print(f"  pred_verts : {pred_verts.shape}")


if __name__ == "__main__":
    main()
