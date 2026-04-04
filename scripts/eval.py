import torch
import numpy as np
import matplotlib.pyplot as plt
from model import ScaffoldedPointPredictor
from torchvision import transforms
from PIL import Image

# 1. Setup & Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NPZ_PATH = '../data/train_data_778_full_with_joints.npz' # Use your new robust dataset
MODEL_PATH = 'checkpoints/best_point_model.pth'

# 2. Load Model
print(f"🔄 Loading Scaffolded Model from {MODEL_PATH}...")
# Ensure parameters match your model.py (21 joints, 778 verts)
model = ScaffoldedPointPredictor(num_joints=21, num_verts=778).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 3. Load Validation Data
print(f"📂 Loading validation data...")
data = np.load(NPZ_PATH)
x_val = data['x_val']
y_val_joints = data['y_val_joints'] # Ground Truth Skeleton
y_val_verts = data['y_val_verts']   # Ground Truth Mesh

# 4. Skeletal Connection Mapping (Standard MANO/FreiHAND)
# This connects the 21 joints into a readable hand skeleton
SKEL_CONNECTIONS = [
    [0, 1, 2, 3, 4],    # Thumb
    [0, 5, 6, 7, 8],    # Index
    [0, 9, 10, 11, 12],  # Middle
    [0, 13, 14, 15, 16], # Ring
    [0, 17, 18, 19, 20]  # Pinky
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
        # Unpack the two-headed output
        p_joints, p_verts = model(input_tensor)
    
    # Reshape results
    pred_j = p_joints.cpu().numpy().reshape(21, 3)
    pred_v = p_verts.cpu().numpy().reshape(778, 3)
    
    gt_j = y_val_joints[index].reshape(21, 3)
    gt_v = y_val_verts[index].reshape(778, 3)
    
    return img_np, pred_j, pred_v, gt_j, gt_v, index

# 6. Plotting
img, p_j, p_v, gt_j, gt_v, idx = run_inference()

fig = plt.figure(figsize=(16, 8))

# Subplot 1: Input Image
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(img)
ax1.set_title(f"Sample #{idx}")
ax1.axis('off')

# Subplot 2: 3D Scaffold Comparison
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# --- 1. Plot GROUND TRUTH (The Target) ---
# GT Joints in Green
ax2.scatter(gt_j[:, 0], gt_j[:, 1], gt_j[:, 2], s=50, c='green', marker='o', alpha=0.5, label='GT Joints')

# GT Skeleton Lines (Green, Dashed)
for finger in SKEL_CONNECTIONS:
    ax2.plot(gt_j[finger, 0], gt_j[finger, 1], gt_j[finger, 2], color='green', linestyle='--', linewidth=1, alpha=0.4)

# GT Mesh (Very faint green cloud)
ax2.scatter(gt_v[:, 0], gt_v[:, 1], gt_v[:, 2], s=1, c='green')


# --- 2. Plot PREDICTIONS (The Model Output) ---
# Predicted Mesh (Red cloud)
ax2.scatter(p_v[:, 0], p_v[:, 1], p_v[:, 2], s=2, c='red', label='Pred Mesh')

# Predicted Joints (Black dots)
ax2.scatter(p_j[:, 0], p_j[:, 1], p_j[:, 2], s=40, c='black', marker='x', label='Pred Joints')

# Predicted Skeleton Lines (Blue, Solid)
for finger in SKEL_CONNECTIONS:
    ax2.plot(p_j[finger, 0], p_j[finger, 1], p_j[finger, 2], color='blue', linewidth=2, label='Pred Skeleton' if finger == SKEL_CONNECTIONS[0] else "")


# --- 3. Formatting ---
ax2.set_title("3D Comparison: Predicted vs. Ground Truth")
# Consistent limits help you see the scale
# limit = 0.5
# ax2.set_xlim(-limit, limit)
# ax2.set_ylim(-limit, limit)
# ax2.set_zlim(-limit, limit)

# Perspective: Top-down to match image orientation
ax2.view_init(elev=-90, azim=-90)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()