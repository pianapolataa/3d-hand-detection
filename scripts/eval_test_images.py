import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import ScaffoldedPointPredictor

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'checkpoints/best_point_model_newest_12560.pth'
# MODEL_PATH = 'checkpoints/model_70_verts.pth' 
# MODEL_PATH = 'checkpoints/model_70_verts_15_vectors.pth'
IMAGE_FOLDER = 'eval_images' 
IMG_SIZE = 224
# NUM_VERTS = 778 # Matches your "original distance" training setup
NUM_VERTS = 778 # Matches your "original distance" training setup
NUM_VECTORS = 9

# Skeletal Connection Mapping (MANO Standard)
SKEL_CONNECTIONS = [
    [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], 
    [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]
]

# --- 1. LOAD MODEL ---
print(f"🔄 Loading Super-Fusion model from {MODEL_PATH}...")
# Note: num_vectors defaults to 9 in our model.py
model = ScaffoldedPointPredictor(num_joints=21, num_verts=NUM_VERTS, num_vectors=NUM_VECTORS).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- 2. PREPROCESSING ---
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_on_image(img_path):
    raw_img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(raw_img).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        # FIX: Unpack 3 values (Joints, Vectors, Vertices)
        p_joints, p_vectors, p_verts = model(input_tensor)
    
    # Reshape results back to 3D coordinates
    pred_j = p_joints.cpu().numpy().reshape(21, 3)
    pred_v = p_verts.cpu().numpy().reshape(NUM_VERTS, 3)
    
    return np.array(raw_img.resize((IMG_SIZE, IMG_SIZE))), pred_j, pred_v

# --- 3. MAIN LOOP ---
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print(f"❌ No images found in '{IMAGE_FOLDER}' folder.")
else:
    print(f"🖼️ Found {len(image_files)} images. Starting inference...")

for filename in image_files:
    img_path = os.path.join(IMAGE_FOLDER, filename)
    print(f"🔍 Processing: {filename}")
    
    img_display, p_j, p_v = predict_on_image(img_path)
    
    fig = plt.figure(figsize=(15, 7))
    fig.suptitle(f"Inference: {filename}", fontsize=14)
    
    # Left: Original Image
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img_display)
    ax1.set_title("Input (Resized)")
    ax1.axis('off')
    
    # Right: 3D Reconstruction
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # 1. Plot the Mesh (Red cloud)
    ax2.scatter(p_v[:, 0], p_v[:, 1], p_v[:, 2], s=2, c='red', alpha=0.4, label='Pred Mesh')
    
    # 2. Plot the Joints (Black crosses)
    ax2.scatter(p_j[:, 0], p_j[:, 1], p_j[:, 2], s=40, c='black', marker='x', label='Pred Joints')
    
    # 3. Draw the Skeleton (Blue lines)
    for finger in SKEL_CONNECTIONS:
        ax2.plot(p_j[finger, 0], p_j[finger, 1], p_j[finger, 2], color='blue', linewidth=2)

    # 4. Formatting
    ax2.set_title("3D Scaffold Prediction")
    ax2.view_init(elev=-90, azim=-90) # Match FreiHAND camera view
    
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()