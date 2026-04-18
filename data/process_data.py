import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# --- CONFIGURATION ---
DATA_DIR = '../FreiHAND_pub_v2'
OUT_FILE = './train_data_600_verts.npz'
EVAL_FILE = './eval_data_600_verts.npz' 


NUM_SAMPLES = 32560
VAL_SIZE = 0.1      # 10% of (NUM_SAMPLES - 100)
EVAL_SAMPLES = 100  # Exactly 100 samples for the eval file
IMG_SIZE = 224     


# --- MESH RESOLUTION CONFIG ---
# Set this to 778 for full, 389 for half, or 194 for quarter res.
# 778 is not easily divisible, so we use linear spacing to get an exact count.
TARGET_VERTS = 600


def preprocess_data():
   print(f"🚀 Starting Preprocessing | Target Verts: {TARGET_VERTS} | Eval Samples: {EVAL_SAMPLES}")


   # 1. Load Annotation Files
   try:
       with open(os.path.join(DATA_DIR, 'training_verts.json'), 'r') as f:
           verts_all = np.array(json.load(f))
       with open(os.path.join(DATA_DIR, 'training_xyz.json'), 'r') as f:
           joints_all = np.array(json.load(f))
       with open(os.path.join(DATA_DIR, 'training_scale.json'), 'r') as f:
           scales_all = np.array(json.load(f))
   except FileNotFoundError as e:
       print(f"❌ Error: Files not found.\n{e}")
       return


   # Create indices for downsampling the vertices
   # This picks TARGET_VERTS evenly spaced points from the 778 available
   vert_indices = np.linspace(0, 777, TARGET_VERTS, dtype=int)


   processed_images = []
   processed_joints = []
   processed_verts = []


   print(f"🛠 Processing {NUM_SAMPLES} samples...")
   for i in tqdm(range(NUM_SAMPLES)):
      
       # --- PART A: GEOMETRIC NORMALIZATION ---
       wrist = joints_all[i][0]  
       scale = scales_all[i]     


       # 1. Downsample and Normalize Vertices
       # We grab only the indices we want
       verts_selected = verts_all[i][vert_indices]
       verts_norm = (verts_selected - wrist) / scale
      
       # 2. Normalize 21 Joints
       joints_norm = (joints_all[i] - wrist) / scale


       processed_verts.append(verts_norm.astype(np.float32).flatten())
       processed_joints.append(joints_norm.astype(np.float32).flatten())


       # --- PART B: BLACK BACKGROUND MASKING ---
       img_name = f"{i:08d}.jpg"
       img_path = os.path.join(DATA_DIR, 'training', 'rgb', img_name)
       mask_path = os.path.join(DATA_DIR, 'training', 'mask', img_name)
      
       img = cv2.imread(img_path)
       mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
      
       if img is None or mask is None:
           continue


       img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
       mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
       _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
       img_black_bg = cv2.bitwise_and(img, img, mask=binary_mask)
      
       processed_images.append(cv2.cvtColor(img_black_bg, cv2.COLOR_BGR2RGB))


   # 2. Convert to NumPy Arrays
   X = np.array(processed_images, dtype=np.uint8)
   Y_j = np.array(processed_joints, dtype=np.float32)
   Y_v = np.array(processed_verts, dtype=np.float32)


   # 3. TRIPLE SPLIT LOGIC
   # First, peel off the 100 Eval samples from the end
   X_eval = X[-EVAL_SAMPLES:]
   Y_j_eval = Y_j[-EVAL_SAMPLES:]
   Y_v_eval = Y_v[-EVAL_SAMPLES:]


   # Remaining data for Train/Val
   X_remainder = X[:-EVAL_SAMPLES]
   Y_j_remainder = Y_j[:-EVAL_SAMPLES]
   Y_v_remainder = Y_v[:-EVAL_SAMPLES]


   # Split the remainder into Train and Val
   idx_train, idx_val = train_test_split(
       np.arange(len(X_remainder)),
       test_size=VAL_SIZE,
       random_state=42
   )


   # 4. SAVE EVAL DATA
   print(f"💾 Saving Eval set (100 samples) to {EVAL_FILE}...")
   np.savez_compressed(EVAL_FILE,
                       x_test=X_eval,
                       y_test_joints=Y_j_eval,
                       y_test_verts=Y_v_eval)


   # 5. SAVE TRAIN/VAL DATA
   print(f"💾 Saving Train/Val set to {OUT_FILE}...")
   np.savez_compressed(OUT_FILE,
                       x_train=X_remainder[idx_train],
                       y_train_joints=Y_j_remainder[idx_train],
                       y_train_verts=Y_v_remainder[idx_train],
                       x_val=X_remainder[idx_val],
                       y_val_joints=Y_j_remainder[idx_val],
                       y_val_verts=Y_v_remainder[idx_val])
  
   print(f"✅ Preprocessing Complete!")
   print(f"Train: {len(idx_train)} | Val: {len(idx_val)} | Eval: {EVAL_SAMPLES}")
   print(f"Vertex Count per sample: {TARGET_VERTS} (Flattened: {TARGET_VERTS * 3})")


if __name__ == "__main__":
   preprocess_data()