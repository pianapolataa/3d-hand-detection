import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from model import ScaffoldedPointPredictor
from dataset import PointDataset
from tqdm import tqdm 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 60
BATCH_SIZE = 64 
LEARNING_RATE = 1e-4
NPZ_PATH = '../data/train_data_778_12560_with_joints.npz' 

# Loss Component Weights
W_JOINT = 2.0 
W_MESH = 1.0
W_VOLUME = 0.5  # Our new volume-proxy weight
W_BONE = 0.3    # Anatomical constraint

# Tip indices for the 21-joint skeleton
TIP_IDS = [4, 8, 12, 16, 20] 

def hand_reconstruction_loss(pred_joints, gt_joints, pred_verts, gt_verts):
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    
    # 1. Weighted Joint Loss (Focus on Fingertips)
    # pred_joints shape: [Batch, 63] -> reshape to [Batch, 21, 3]
    pj = pred_joints.view(-1, 21, 3)
    gj = gt_joints.view(-1, 21, 3)
    
    joint_err = torch.abs(pj - gj) # L1 distance
    # Create weight map: 3.0 for tips, 1.0 for everything else
    weights = torch.ones(21).to(DEVICE)
    weights[TIP_IDS] = 3.0
    # Apply weights across the (Batch, 21, 3) dimensions
    loss_j = torch.mean(joint_err * weights.unsqueeze(-1))
    
    # 2. Mesh Loss (Standard Skin Detail)
    loss_v = l1(pred_verts, gt_verts)
    
    # 3. VOLUME PROXY LOSS (Variance)
    # If the index is up, variance is high. If down, variance is low.
    # We compare the 'spread' of predicted points vs ground truth points.
    pv = pred_verts.view(-1, 778, 3)
    gv = gt_verts.view(-1, 778, 3)
    
    pred_var = torch.var(pv, dim=1) # Variance across the 778 points
    gt_var = torch.var(gv, dim=1)
    loss_volume = mse(pred_var, gt_var)
    
    # 4. BONE LENGTH CONSISTENCY
    # Prevents 'Mean Shape Bias' by ensuring finger segments don't stretch
    # We check the distance between Knuckle (e.g., 5) and Mid-joint (6)
    def get_bone_lengths(j):
        # Sample specific finger segments: [5-6, 9-10, 13-14, 17-18]
        p1 = j[:, [5, 9, 13, 17], :]
        p2 = j[:, [6, 10, 14, 18], :]
        return torch.norm(p1 - p2, dim=-1)
    
    loss_bone = mse(get_bone_lengths(pj), get_bone_lengths(gj))

    return (W_JOINT * loss_j) + (W_MESH * loss_v) + (W_VOLUME * loss_volume) + (W_BONE * loss_bone)

# --- STANDARD SETUP ---
os.makedirs("checkpoints", exist_ok=True)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(PointDataset(NPZ_PATH, 'train', transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(PointDataset(NPZ_PATH, 'val', transform), batch_size=BATCH_SIZE)

model = ScaffoldedPointPredictor(num_joints=21, num_verts=778).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- TRAINING LOOP ---
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for imgs, gt_j, gt_v in pbar:
        imgs, gt_j, gt_v = imgs.to(DEVICE), gt_j.to(DEVICE), gt_v.to(DEVICE)
        
        pred_j, pred_v = model(imgs)
        
        loss = hand_reconstruction_loss(pred_j, gt_j, pred_v, gt_v)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

    # Validation (Simplified for brevity)
    model.eval()
    # ... (Standard validation logic here) ...
    torch.save(model.state_dict(), "checkpoints/best_point_model_test.pth")

print("✅ Training complete with Volume-Aware constraints.")