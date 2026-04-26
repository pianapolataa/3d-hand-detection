import argparse
import json
import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DATA_DIR = "FreiHAND_pub_v2"
NUM_SAMPLES = 32560
VAL_SIZE = 0.1
EVAL_SAMPLES = 100
IMG_SIZE = 224
DEFAULT_TARGET_VERTS = 600


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess FreiHAND into train/val/eval NPZ files.")
    parser.add_argument("--target-verts", type=int, default=DEFAULT_TARGET_VERTS)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--val-size", type=float, default=VAL_SIZE)
    parser.add_argument("--eval-samples", type=int, default=EVAL_SAMPLES)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    return parser.parse_args()


def preprocess_data(args):
    target_verts = args.target_verts
    out_file = f"data/train_data_{target_verts}_verts.npz"
    eval_file = f"data/eval_data_{target_verts}_verts.npz"

    print(f"🚀 Starting Preprocessing | Target Verts: {target_verts} | Eval Samples: {args.eval_samples}")

    try:
        with open(os.path.join(args.data_dir, "training_verts.json"), "r") as handle:
            verts_all = np.array(json.load(handle))
        with open(os.path.join(args.data_dir, "training_xyz.json"), "r") as handle:
            joints_all = np.array(json.load(handle))
        with open(os.path.join(args.data_dir, "training_scale.json"), "r") as handle:
            scales_all = np.array(json.load(handle))
    except FileNotFoundError as exc:
        print(f"❌ Error: Files not found.\n{exc}")
        return

    vert_indices = np.linspace(0, 777, target_verts, dtype=int)

    processed_images = []
    processed_joints = []
    processed_verts = []
    processed_indices = []

    print(f"🛠 Processing {args.num_samples} samples...")
    for sample_idx in tqdm(range(args.num_samples)):
        img_name = f"{sample_idx:08d}.jpg"
        img_path = os.path.join(args.data_dir, "training", "rgb", img_name)
        mask_path = os.path.join(args.data_dir, "training", "mask", img_name)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        wrist = joints_all[sample_idx][0]
        scale = scales_all[sample_idx]

        verts_selected = verts_all[sample_idx][vert_indices]
        verts_norm = (verts_selected - wrist) / scale
        joints_norm = (joints_all[sample_idx] - wrist) / scale

        img = cv2.resize(img, (args.img_size, args.img_size))
        mask = cv2.resize(mask, (args.img_size, args.img_size))
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        img_black_bg = cv2.bitwise_and(img, img, mask=binary_mask)

        processed_images.append(cv2.cvtColor(img_black_bg, cv2.COLOR_BGR2RGB))
        processed_joints.append(joints_norm.astype(np.float32).flatten())
        processed_verts.append(verts_norm.astype(np.float32).flatten())
        processed_indices.append(sample_idx)

    x_all = np.array(processed_images, dtype=np.uint8)
    y_joints_all = np.array(processed_joints, dtype=np.float32)
    y_verts_all = np.array(processed_verts, dtype=np.float32)
    sample_indices_all = np.array(processed_indices, dtype=np.int32)

    x_eval = x_all[-args.eval_samples :]
    y_joints_eval = y_joints_all[-args.eval_samples :]
    y_verts_eval = y_verts_all[-args.eval_samples :]
    sample_indices_eval = sample_indices_all[-args.eval_samples :]

    x_remainder = x_all[: -args.eval_samples]
    y_joints_remainder = y_joints_all[: -args.eval_samples]
    y_verts_remainder = y_verts_all[: -args.eval_samples]
    sample_indices_remainder = sample_indices_all[: -args.eval_samples]

    idx_train, idx_val = train_test_split(
        np.arange(len(x_remainder)),
        test_size=args.val_size,
        random_state=42,
    )

    print(f"💾 Saving Eval set ({args.eval_samples} samples) to {eval_file}...")
    np.savez_compressed(
        eval_file,
        x_test=x_eval,
        y_test_joints=y_joints_eval,
        y_test_verts=y_verts_eval,
        sample_indices_test=sample_indices_eval,
    )

    print(f"💾 Saving Train/Val set to {out_file}...")
    np.savez_compressed(
        out_file,
        x_train=x_remainder[idx_train],
        y_train_joints=y_joints_remainder[idx_train],
        y_train_verts=y_verts_remainder[idx_train],
        sample_indices_train=sample_indices_remainder[idx_train],
        x_val=x_remainder[idx_val],
        y_val_joints=y_joints_remainder[idx_val],
        y_val_verts=y_verts_remainder[idx_val],
        sample_indices_val=sample_indices_remainder[idx_val],
    )

    print("✅ Preprocessing Complete!")
    print(f"Train: {len(idx_train)} | Val: {len(idx_val)} | Eval: {args.eval_samples}")
    print(f"Vertex Count per sample: {target_verts} (Flattened: {target_verts * 3})")
    print("Original FreiHAND sample ids saved as sample_indices_train/sample_indices_val/sample_indices_test")


if __name__ == "__main__":
    preprocess_data(parse_args())
