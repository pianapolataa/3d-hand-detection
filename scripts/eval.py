# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from model import ScaffoldedPointPredictor
# from torchvision import transforms
# from PIL import Image

# # 1. Setup & Device
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NPZ_PATH = '../data/train_data_778_full_with_joints.npz' 
# MODEL_PATH = 'checkpoints/best_point_model.pth'

# # 2. Load Model
# print(f"🔄 Loading Scaffolded Model from {MODEL_PATH}...")
# model = ScaffoldedPointPredictor(num_joints=21, num_verts=778).to(DEVICE)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
# model.eval()

# # 3. Load Validation Data
# print(f"📂 Loading validation data...")
# data = np.load(NPZ_PATH)
# x_val = data['x_val']
# y_val_joints = data['y_val_joints']
# y_val_verts = data['y_val_verts']

# # 4. Skeletal Connection Mapping
# SKEL_CONNECTIONS = [
#     [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], 
#     [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]
# ]

# # 5. Inference Function
# def run_inference(index=None):
#     if index is None:
#         index = np.random.randint(0, len(x_val))
    
#     img_np = x_val[index]
#     pil_img = Image.fromarray(img_np)
    
#     preprocess = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    
#     with torch.no_grad():
#         p_joints, p_verts = model(input_tensor)
    
#     pred_j = p_joints.cpu().numpy().reshape(21, 3)
#     pred_v = p_verts.cpu().numpy().reshape(778, 3)
#     gt_j = y_val_joints[index].reshape(21, 3)
#     gt_v = y_val_verts[index].reshape(778, 3)
    
#     return img_np, pred_j, pred_v, gt_j, gt_v, index

# # 6. Looping Visualization
# NUM_SAMPLES = 20
# print(f"🖼️ Displaying {NUM_SAMPLES} random samples. Close the window to see the next one...")

# for i in range(NUM_SAMPLES):
#     img, p_j, p_v, gt_j, gt_v, idx = run_inference()

#     fig = plt.figure(figsize=(16, 8))
#     fig.suptitle(f"Sample {i+1}/{NUM_SAMPLES} (Dataset Index: {idx})", fontsize=16)

#     # Subplot 1: Input Image
#     ax1 = fig.add_subplot(1, 2, 1)
#     ax1.imshow(img)
#     ax1.set_title("Input RGB")
#     ax1.axis('off')

#     # Subplot 2: 3D Comparison
#     ax2 = fig.add_subplot(1, 2, 2, projection='3d')

#     # --- 1. Plot GROUND TRUTH ---
#     ax2.scatter(gt_j[:, 0], gt_j[:, 1], gt_j[:, 2], s=50, c='green', marker='o', alpha=0.3, label='GT Joints')
#     for finger in SKEL_CONNECTIONS:
#         ax2.plot(gt_j[finger, 0], gt_j[finger, 1], gt_j[finger, 2], color='green', linestyle='--', linewidth=1, alpha=0.3)
#     # ax2.scatter(gt_v[:, 0], gt_v[:, 1], gt_v[:, 2], s=1, c='green', alpha=0.05)

#     # --- 2. Plot PREDICTIONS ---
#     # Mesh (Red cloud)
#     ax2.scatter(p_v[:, 0], p_v[:, 1], p_v[:, 2], s=2, c='red', alpha=0.4, label='Pred Mesh')
#     # Joints (Black dots)
#     ax2.scatter(p_j[:, 0], p_j[:, 1], p_j[:, 2], s=40, c='black', marker='x', label='Pred Joints')
#     # Skeleton (Blue lines)
#     for finger in SKEL_CONNECTIONS:
#         label = 'Pred Skeleton' if finger == SKEL_CONNECTIONS[0] else ""
#         ax2.plot(p_j[finger, 0], p_j[finger, 1], p_j[finger, 2], color='blue', linewidth=2, label=label)

#     # --- 3. Formatting ---
#     ax2.set_title("3D Scaffold: Pred (Blue/Red) vs GT (Green)")
#     ax2.view_init(elev=-90, azim=-90)
#     ax2.legend(loc='upper right')

#     plt.tight_layout()
#     plt.show() # Script will pause here until you close the plot window

import torch
import numpy as np
import matplotlib.pyplot as plt
from model import ScaffoldedPointPredictor
from torchvision import transforms
from PIL import Image

# 1. Setup & Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NPZ_PATH = '../data/train_data_778_full_with_joints.npz' 
MODEL_PATH = 'checkpoints/best_point_model_newest_12560.pth'
NUM_VERTS = 778  # Must match the NUM_VERTS used in training

# 2. Load Model
print(f"🔄 Loading Scaffolded Model (Vertices: {NUM_VERTS}) from {MODEL_PATH}...")
# Note: num_joints=21, num_verts=178 matches your updated model.py
model = ScaffoldedPointPredictor(num_joints=21, num_verts=NUM_VERTS).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 3. Load Validation Data (Always 778 from the NPZ)
print(f"📂 Loading validation data...")
data = np.load(NPZ_PATH)
x_val = data['x_val']
y_val_joints = data['y_val_joints']
y_val_verts = data['y_val_verts'] 

# 4. Skeletal Connection Mapping
SKEL_CONNECTIONS = [
    [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], 
    [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]
]

# 5. Inference Function
def run_inference(index=None):
    if index is None:
        index = np.random.randint(0, len(x_val))
    
    img_np = x_val[index]
    pil_img = Image.fromarray(img_np)
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # UNPACK THREE VALUES: Joints, Vectors (unused in plot), and Vertices
        p_joints, p_vectors, p_verts = model(input_tensor)
    
    # Reshape predicted mesh to 178, Ground Truth remains 778
    pred_j = p_joints.cpu().numpy().reshape(21, 3)
    pred_v = p_verts.cpu().numpy().reshape(NUM_VERTS, 3)
    
    gt_j = y_val_joints[index].reshape(21, 3)
    gt_v = y_val_verts[index].reshape(778, 3)
    
    return img_np, pred_j, pred_v, gt_j, gt_v, index

# 6. Looping Visualization
NUM_SAMPLES = 20
print(f"🖼️ Displaying {NUM_SAMPLES} samples. Close window for next...")

for i in range(NUM_SAMPLES):
    img, p_j, p_v, gt_j, gt_v, idx = run_inference()

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f"Sample {i+1}/{NUM_SAMPLES} | Index: {idx} | Res: {NUM_VERTS}pts", fontsize=16)

    # Subplot 1: Input Image
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img)
    ax1.set_title("Input RGB")
    ax1.axis('off')

    # Subplot 2: 3D Comparison
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # --- 1. Plot GROUND TRUTH (Ghost View) ---
    ax2.scatter(gt_j[:, 0], gt_j[:, 1], gt_j[:, 2], s=50, c='green', marker='o', alpha=0.2, label='GT Joints')
    for finger in SKEL_CONNECTIONS:
        ax2.plot(gt_j[finger, 0], gt_j[finger, 1], gt_j[finger, 2], color='green', linestyle='--', linewidth=1, alpha=0.2)
    # The 778 dense points act as a "ghost" background
    ax2.scatter(gt_v[:, 0], gt_v[:, 1], gt_v[:, 2], s=1, c='green', alpha=0.03)

    # --- 2. Plot PREDICTIONS (Focus View) ---
    # The 178 sparse points should clearly outline the hand shape
    ax2.scatter(p_v[:, 0], p_v[:, 1], p_v[:, 2], s=8, c='red', alpha=0.6, label=f'Pred Mesh ({NUM_VERTS} pts)')
    
    # Predicted Joints (Black dots)
    ax2.scatter(p_j[:, 0], p_j[:, 1], p_j[:, 2], s=40, c='black', marker='x', label='Pred Joints')
    
    # Predicted Skeleton (Blue lines)
    for finger in SKEL_CONNECTIONS:
        label = 'Pred Skeleton' if finger == SKEL_CONNECTIONS[0] else ""
        ax2.plot(p_j[finger, 0], p_j[finger, 1], p_j[finger, 2], color='blue', linewidth=2, label=label)

    # --- 3. Formatting ---
    ax2.set_title("3D Distribution Fitting")
    # Top-down perspective
    ax2.view_init(elev=-90, azim=-90)
    
    # Fixed limits to prevent auto-zooming jitter
    limit = 0.5
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()