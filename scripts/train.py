# # import torch
# # import torch.nn as nn
# # import os
# # from torch.utils.data import DataLoader
# # from torchvision import transforms
# # from model import ScaffoldedPointPredictor
# # from dataset import PointDataset
# # from tqdm import tqdm 

# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # EPOCHS = 60
# # BATCH_SIZE = 64 
# # LEARNING_RATE = 1e-4
# # NPZ_PATH = '../data/train_data_778_full_with_joints.npz' 

# # # Loss Component Weights
# # W_JOINT = 2.0 
# # W_MESH = 1.0
# # W_VOLUME = 0.5  # Our new volume-proxy weight
# # W_BONE = 0.3    # Anatomical constraint

# # # Tip indices for the 21-joint skeleton
# # TIP_IDS = [4, 8, 12, 16, 20] 

# # def hand_reconstruction_loss(pred_joints, gt_joints, pred_verts, gt_verts):
# #     l1 = nn.L1Loss()
# #     mse = nn.MSELoss()
    
# #     # 1. Weighted Joint Loss (Focus on Fingertips)
# #     # pred_joints shape: [Batch, 63] -> reshape to [Batch, 21, 3]
# #     pj = pred_joints.view(-1, 21, 3)
# #     gj = gt_joints.view(-1, 21, 3)
    
# #     joint_err = torch.abs(pj - gj) # L1 distance
# #     # Create weight map: 3.0 for tips, 1.0 for everything else
# #     weights = torch.ones(21).to(DEVICE)
# #     weights[TIP_IDS] = 3.0
# #     # Apply weights across the (Batch, 21, 3) dimensions
# #     loss_j = torch.mean(joint_err * weights.unsqueeze(-1))
    
# #     # 2. Mesh Loss (Standard Skin Detail)
# #     loss_v = l1(pred_verts, gt_verts)
    
# #     # 3. VOLUME PROXY LOSS (Variance)
# #     # If the index is up, variance is high. If down, variance is low.
# #     # We compare the 'spread' of predicted points vs ground truth points.
# #     pv = pred_verts.view(-1, 778, 3)
# #     gv = gt_verts.view(-1, 778, 3)
    
# #     pred_var = torch.var(pv, dim=1) # Variance across the 778 points
# #     gt_var = torch.var(gv, dim=1)
# #     loss_volume = mse(pred_var, gt_var)
    
# #     # 4. BONE LENGTH CONSISTENCY
# #     # Prevents 'Mean Shape Bias' by ensuring finger segments don't stretch
# #     # We check the distance between Knuckle (e.g., 5) and Mid-joint (6)
# #     def get_bone_lengths(j):
# #         # Sample specific finger segments: [5-6, 9-10, 13-14, 17-18]
# #         p1 = j[:, [5, 9, 13, 17], :]
# #         p2 = j[:, [6, 10, 14, 18], :]
# #         return torch.norm(p1 - p2, dim=-1)
    
# #     loss_bone = mse(get_bone_lengths(pj), get_bone_lengths(gj))

# #     return (W_JOINT * loss_j) + (W_MESH * loss_v) + (W_VOLUME * loss_volume) + (W_BONE * loss_bone)

# # # --- STANDARD SETUP ---
# # os.makedirs("checkpoints", exist_ok=True)
# # transform = transforms.Compose([
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# # ])

# # train_loader = DataLoader(PointDataset(NPZ_PATH, 'train', transform), batch_size=BATCH_SIZE, shuffle=True)
# # val_loader = DataLoader(PointDataset(NPZ_PATH, 'val', transform), batch_size=BATCH_SIZE)

# # model = ScaffoldedPointPredictor(num_joints=21, num_verts=778).to(DEVICE)
# # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# # # --- TRAINING LOOP ---
# # for epoch in range(EPOCHS):
# #     model.train()
# #     total_train_loss = 0
# #     pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
# #     for imgs, gt_j, gt_v in pbar:
# #         imgs, gt_j, gt_v = imgs.to(DEVICE), gt_j.to(DEVICE), gt_v.to(DEVICE)
        
# #         pred_j, pred_v = model(imgs)
        
# #         loss = hand_reconstruction_loss(pred_j, gt_j, pred_v, gt_v)
        
# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()
        
# #         total_train_loss += loss.item()
# #         pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

# #     # Validation (Simplified for brevity)
# #     model.eval()
# #     # ... (Standard validation logic here) ...
# #     torch.save(model.state_dict(), "checkpoints/best_point_model_full.pth")

# # print("✅ Training complete with Volume-Aware constraints.")

# import torch
# import torch.nn as nn
# import os
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from model import ScaffoldedPointPredictor
# from dataset import PointDataset
# from tqdm import tqdm 

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# EPOCHS = 60
# BATCH_SIZE = 4 
# LEARNING_RATE = 1e-4
# NPZ_PATH = '../data/train_data_778_2560_with_joints.npz' 

# # Loss Component Weights
# W_JOINT = 2.0 
# W_MESH = 1.0
# W_VOLUME = 0.5 
# W_GEOM = 1.5    # Increased weight for our new pose geometry

# # Joint Index Constants
# TIP_IDS = [4, 8, 12, 16, 20] 
# MCP_IDS = [2, 5, 9, 13, 17]
# THUMB_TIP = 4
# OTHER_TIPS = [8, 12, 16, 20]

# def hand_reconstruction_loss(pred_joints, gt_joints, pred_verts, gt_verts):
#     l1 = nn.L1Loss()
#     mse = nn.MSELoss()
    
#     # Reshape: [Batch, 21, 3]
#     pj = pred_joints.view(-1, 21, 3)
#     gj = gt_joints.view(-1, 21, 3)
    
#     # 1. Base Joint Loss (Standard L1 on all points)
#     weights = torch.ones(21).to(DEVICE)
#     weights[TIP_IDS] = 3.0 
#     loss_j = torch.mean(torch.abs(pj - gj) * weights.unsqueeze(-1))
    
#     # 2. Mesh Loss (Surface Detail)
#     loss_v = l1(pred_verts, gt_verts)
    
#     # 3. Volume Proxy (Variance)
#     pv = pred_verts.view(-1, 778, 3)
#     gv = gt_verts.view(-1, 778, 3)
#     loss_volume = mse(torch.var(pv, dim=1), torch.var(gv, dim=1))
    
#     # 4. VECTOR POSE CONSTRAINTS (Strong Geometric Matching)
#     def get_pose_vectors(j):
#         # A. Vectors: Thumb Tip -> Other Tips (4 vectors)
#         thumb_tip = j[:, THUMB_TIP, :].unsqueeze(1) # [Batch, 1, 3]
#         other_tips = j[:, OTHER_TIPS, :]            # [Batch, 4, 3]
#         v_to_thumb = other_tips - thumb_tip         # [Batch, 4, 3]
        
#         # B. Vectors: MCP -> Fingertip (5 vectors)
#         # This defines the orientation of each individual finger
#         tips = j[:, TIP_IDS, :] # [Batch, 5, 3]
#         mcps = j[:, MCP_IDS, :] # [Batch, 5, 3]
#         v_mcp_tip = tips - mcps  # [Batch, 5, 3]
        
#         return torch.cat([v_to_thumb, v_mcp_tip], dim=1) # [Batch, 9, 3]

#     pred_vectors = get_pose_vectors(pj)
#     gt_vectors = get_pose_vectors(gj)
    
#     # Vector matching loss (matching XYZ directions)
#     loss_geom = l1(pred_vectors, gt_vectors)

#     return (W_JOINT * loss_j) + (W_MESH * loss_v) + (W_VOLUME * loss_volume) + (W_GEOM * loss_geom)
# # --- Standard Training Setup (Unchanged) ---
# os.makedirs("checkpoints", exist_ok=True)
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# train_loader = DataLoader(PointDataset(NPZ_PATH, 'train', transform), batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(PointDataset(NPZ_PATH, 'val', transform), batch_size=BATCH_SIZE)

# model = ScaffoldedPointPredictor(num_joints=21, num_verts=778).to(DEVICE)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# for epoch in range(EPOCHS):
#     model.train()
#     total_train_loss = 0
#     pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
#     for imgs, gt_j, gt_v in pbar:
#         imgs, gt_j, gt_v = imgs.to(DEVICE), gt_j.to(DEVICE), gt_v.to(DEVICE)
        
#         pred_j, pred_v = model(imgs)
#         loss = hand_reconstruction_loss(pred_j, gt_j, pred_v, gt_v)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         total_train_loss += loss.item()
#         pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

#     # Save model every epoch
#     torch.save(model.state_dict(), "checkpoints/best_point_model_newest.pth")

# print("✅ Training complete with Pose-Geometry constraints.")

import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from model import ScaffoldedPointPredictor
from dataset import PointDataset
from tqdm import tqdm 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 300
BATCH_SIZE = 64 
LEARNING_RATE = 1e-4
NPZ_PATH = '../data/train_data_778_12560_with_joints.npz' 
NUM_VERTS = 778 # Adjustable parameter for the mesh resolution

# Loss Weights
W_JOINT = 2.0 
W_MESH = 1.0
W_GEOM = 2.0    # Weight for the explicit Vector Head

# Joint Index Constants (MANO)
TIP_IDS = [4, 8, 12, 16, 20] 
MCP_IDS = [2, 5, 9, 13, 17]
THUMB_TIP = 4
OTHER_TIPS = [8, 12, 16, 20]

# We need to calculate the GROUND TRUTH vectors from the GT joints
def get_gt_vectors(j):
    # Thumb Tip -> Other Tips
    thumb_tip = j[:, THUMB_TIP, :].unsqueeze(1)
    other_tips = j[:, OTHER_TIPS, :]
    v_to_thumb = other_tips - thumb_tip
    
    # MCP -> Fingertip
    tips = j[:, TIP_IDS, :]
    mcps = j[:, MCP_IDS, :]
    v_mcp_tip = tips - mcps
    
    return torch.cat([v_to_thumb, v_mcp_tip], dim=1) # [Batch, 9, 3]

def hand_reconstruction_loss(pred_joints, pred_vectors, gt_joints, pred_verts, gt_verts):
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    
    pj = pred_joints.view(-1, 21, 3)
    gj = gt_joints.view(-1, 21, 3)
    pv = pred_verts.view(-1, NUM_VERTS, 3)
    gv = gt_verts.view(-1, NUM_VERTS, 3)

    # 1. Weighted Joint Loss 
    # This ensures the model cares 3x more about the tips than the palm
    joint_err = torch.abs(pj - gj) 
    weights = torch.ones(21).to(DEVICE)
    weights[TIP_IDS] = 3.0
    loss_j = torch.mean(joint_err * weights.unsqueeze(-1))

    # 2. Mesh Loss (Standard L1 Skin Detail)
    # We use the reshaped versions to ensure dimensions are perfect
    loss_v = l1(pv, gv)
    
    # 3. Vector Loss (Orientation Head Supervision)
    gt_vectors = get_gt_vectors(gj)
    loss_geom = l1(pred_vectors.view(-1, 9, 3), gt_vectors)

    # Combined Weighted Loss
    return (W_JOINT * loss_j) + (W_MESH * loss_v) + (W_GEOM * loss_geom)

os.makedirs("checkpoints", exist_ok=True)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(PointDataset(NPZ_PATH, 'train', transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(PointDataset(NPZ_PATH, 'val', transform), batch_size=BATCH_SIZE)

# Initialize model with adjustable NUM_VERTS
model = ScaffoldedPointPredictor(num_joints=21, num_verts=NUM_VERTS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- EARLY STOPPING CONFIG ---
patience = 10          # How many epochs to wait before stopping
patience_counter = 0  # Tracker
best_val_loss = float('inf')

# --- TRAINING LOOP ---
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for imgs, gt_j, gt_v in pbar:
        imgs, gt_j, gt_v = imgs.to(DEVICE), gt_j.to(DEVICE), gt_v.to(DEVICE)
        
        # Unpack THREE values now
        pred_j, pred_vec, pred_v = model(imgs)
        
        # Pass predicted vectors into loss
        loss = hand_reconstruction_loss(pred_j, pred_vec, gt_j, pred_v, gt_v)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, gt_j, gt_v in val_loader:
            imgs, gt_j, gt_v = imgs.to(DEVICE), gt_j.to(DEVICE), gt_v.to(DEVICE)
            pj, pv_vec, pv = model(imgs)
            val_loss += hand_reconstruction_loss(pj, pv_vec, gt_j, pv, gt_v).item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"End of Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f}")

    # 2. EARLY STOPPING LOGIC
    if avg_val_loss < best_val_loss:
        # If the model improved, save it and reset the counter
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "checkpoints/best_point_model_newest_12560.pth")
        print(f"🌟 New Best Model Saved!")
    else:
        # If the model didn't improve, increment the counter
        patience_counter += 1
        print(f"⚠️ No improvement. Patience: {patience_counter}/{patience}")

    if patience_counter >= patience:
        print(f"🛑 Early stopping triggered. Training finished at epoch {epoch+1}.")
        break