import torch
import numpy as np
import matplotlib.pyplot as plt
from model import ScaffoldedPointPredictor
from torchvision import transforms
from PIL import Image

# 1. Setup & Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NPZ_PATH = '../data/eval_data_600_verts.npz' 
MODEL_PATH = 'checkpoints/model_600_verts_15_vectors.pth'
NUM_VERTS = 600  
NUM_VECTORS = 15

# MANO Indices for plotting
TIP_IDS = [4, 8, 12, 16, 20] 
MCP_IDS = [2, 5, 9, 13, 17]

# 2. Load Model
model = ScaffoldedPointPredictor(num_joints=21, num_verts=NUM_VERTS, num_vectors=NUM_VECTORS).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 3. Load Validation Data
data = np.load(NPZ_PATH)
x_val = data['x_test']
y_val_joints = data['y_test_joints']
y_val_verts = data['y_test_verts'] 

# 4. Skeletal Connection Mapping
SKEL_CONNECTIONS = [
    [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], 
    [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]
]

# 5. Inference Function (UPDATED TO RETURN VECTORS)
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
        p_joints, p_vectors, p_verts = model(input_tensor)
    
    pred_j = p_joints.cpu().numpy().reshape(21, 3)
    pred_v = p_verts.cpu().numpy().reshape(NUM_VERTS, 3)
    pred_vec = p_vectors.cpu().numpy().reshape(15, 3) # Unpacking predicted vectors
    
    gt_j = y_val_joints[index].reshape(21, 3)
    gt_v = y_val_verts[index].reshape(NUM_VERTS, 3)
    
    return img_np, pred_j, pred_v, pred_vec, gt_j, gt_v, index

# 6. Looping Visualization
NUM_SAMPLES = 20

for i in range(NUM_SAMPLES):
    img, p_j, p_v, p_vec, gt_j, gt_v, idx = run_inference()

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f"Sample {i+1} | Res: {NUM_VERTS}pts | Vec: {NUM_VECTORS}", fontsize=16)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img)
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # --- 1. Plot GROUND TRUTH (Ghost View) ---
    ax2.scatter(gt_v[:, 0], gt_v[:, 1], gt_v[:, 2], s=1, c='green', alpha=0.05)

    # --- 2. Plot PREDICTIONS ---
    ax2.scatter(p_v[:, 0], p_v[:, 1], p_v[:, 2], s=10, c='red', alpha=0.4)
    ax2.scatter(p_j[:, 0], p_j[:, 1], p_j[:, 2], s=40, c='black', marker='x', label='Pred Joints')
    
    for finger in SKEL_CONNECTIONS:
        ax2.plot(p_j[finger, 0], p_j[finger, 1], p_j[finger, 2], color='blue', linewidth=2)

    # --- 3. NEW: PLOT PREDICTED VECTORS ---
    # Inter-tip pairs (0-9)
    pairs = []
    for t1 in range(5):
        for t2 in range(t1 + 1, 5):
            pairs.append((t1, t2))
    
    # 

    # Plot Inter-tip Vectors (Gold)
    for n in range(10):
        start_idx = TIP_IDS[pairs[n][0]]
        start_pos = p_j[start_idx]
        vec = p_vec[n]
        ax2.quiver(start_pos[0], start_pos[1], start_pos[2], 
                  vec[0], vec[1], vec[2], 
                  color='gold', length=1.0, alpha=0.8, arrow_length_ratio=0.2)

    # Plot Bone Vectors (Purple) - MCP to TIP
    for n in range(5):
        start_idx = MCP_IDS[n]
        start_pos = p_j[start_idx]
        vec = p_vec[10 + n]
        ax2.quiver(start_pos[0], start_pos[1], start_pos[2], 
                  vec[0], vec[1], vec[2], 
                  color='purple', length=1.0, alpha=0.8, arrow_length_ratio=0.2)

    # Formatting
    ax2.set_title("Scaffold: Skeleton + Geometric Vectors")
    ax2.view_init(elev=-90, azim=-90)
    limit = 0.5

    plt.tight_layout()
    plt.show()