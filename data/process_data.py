import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATA_DIR = '../FreiHAND_pub_v2' 
OUT_FILE = './train_data_778_full_with_joints.npz' # Changed name to reflect black bg
NUM_SAMPLES = 32560 # You can set this to 32560 for the full dataset
VAL_SIZE = 0.1      
IMG_SIZE = 224      

def preprocess_data():
    print("🚀 Starting Clean Preprocessing (Black Background + Multi-Task)...")

    # 1. Load Annotation Files
    try:
        with open(os.path.join(DATA_DIR, 'training_verts.json'), 'r') as f:
            verts_all = np.array(json.load(f))
        with open(os.path.join(DATA_DIR, 'training_xyz.json'), 'r') as f:
            joints_all = np.array(json.load(f))
        with open(os.path.join(DATA_DIR, 'training_scale.json'), 'r') as f:
            scales_all = np.array(json.load(f))
    except FileNotFoundError as e:
        print(f"❌ Error: Files not found. Check your DATA_DIR path.\n{e}")
        return

    processed_images = []
    processed_joints = []
    processed_verts = []

    print(f"🛠 Isurating {NUM_SAMPLES} hand samples onto black backgrounds...")
    for i in tqdm(range(NUM_SAMPLES)):
        
        # --- PART A: SYNCED GEOMETRIC NORMALIZATION ---
        # Both Skeleton (21) and Mesh (778) must use the same Anchor and Scale
        wrist = joints_all[i][0]   
        scale = scales_all[i]      

        # 1. Normalize 778 Vertices
        verts_norm = (verts_all[i] - wrist) / scale
        
        # 2. Normalize 21 Joints
        joints_norm = (joints_all[i] - wrist) / scale

        processed_verts.append(verts_norm.astype(np.float32).flatten())
        processed_joints.append(joints_norm.astype(np.float32).flatten())

        # --- PART B: PURE BLACK BACKGROUND MASKING ---
        img_name = f"{i:08d}.jpg"
        img_path = os.path.join(DATA_DIR, 'training', 'rgb', img_name)
        mask_path = os.path.join(DATA_DIR, 'training', 'mask', img_name)
        
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            continue

        # Resize to model input size
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        # Create binary mask (Thresholding ensures pure 0 or 255)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Apply mask: keeps the hand pixels, turns everything else to (0, 0, 0)
        img_black_bg = cv2.bitwise_and(img, img, mask=binary_mask)
        
        # Convert BGR to RGB for PyTorch
        processed_images.append(cv2.cvtColor(img_black_bg, cv2.COLOR_BGR2RGB))

    # 2. Convert to NumPy Arrays
    X = np.array(processed_images, dtype=np.uint8)
    Y_j = np.array(processed_joints, dtype=np.float32)
    Y_v = np.array(processed_verts, dtype=np.float32)

    # 3. Split while keeping indices synchronized
    indices = np.arange(len(X))
    idx_train, idx_val = train_test_split(indices, test_size=VAL_SIZE, random_state=42)

    # 4. Save with Keys for PointDataset
    print(f"💾 Saving clean dataset to {OUT_FILE}...")
    np.savez_compressed(OUT_FILE, 
                        x_train=X[idx_train], 
                        y_train_joints=Y_j[idx_train], 
                        y_train_verts=Y_v[idx_train],
                        x_val=X[idx_val], 
                        y_val_joints=Y_j[idx_val], 
                        y_val_verts=Y_v[idx_val])
    
    print(f"✅ Done! Training: {len(idx_train)} | Val: {len(idx_val)}")

if __name__ == "__main__":
    preprocess_data()