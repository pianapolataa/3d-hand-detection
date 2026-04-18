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
NPZ_PATH = '../data/train_data_70_verts.npz' 
NUM_VERTS = 70 # Adjustable parameter for the mesh resolution

# Loss Weights
W_JOINT = 3.0 
W_MESH = 1.0
W_GEOM = 3.0    # Weight for the explicit Vector Head

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        torch.save(model.state_dict(), "checkpoints/model_70_verts.pth")
        print(f"🌟 New Best Model Saved!")
    else:
        # If the model didn't improve, increment the counter
        patience_counter += 1
        print(f"⚠️ No improvement. Patience: {patience_counter}/{patience}")

    if patience_counter >= patience:
        print(f"🛑 Early stopping triggered. Training finished at epoch {epoch+1}.")
        break