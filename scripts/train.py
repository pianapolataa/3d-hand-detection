import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from model import PointPredictor
from dataset import PointDataset

# 1. Hyperparameters & Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 64 # Increased batch size for better stability
LEARNING_RATE = 1e-4
NPZ_PATH = '../data/train_data_778.npz'

# Ensure checkpoints directory exists
os.makedirs("checkpoints", exist_ok=True)

# 2. Data Augmentation & Normalization
# Since images were already resized in preprocess.py, we just normalize
transform = transforms.Compose([
    transforms.ToTensor(), # Converts 0-255 to 0.0-1.0
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Training and Validation sets separately
train_dataset = PointDataset(npz_path=NPZ_PATH, mode='train', transform=transform)
val_dataset = PointDataset(npz_path=NPZ_PATH, mode='val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Model, Loss, and Optimizer
# Note: num_points=778 results in 2334 output neurons (x, y, z)
model = PointPredictor(num_points=778).to(DEVICE)
criterion = nn.L1Loss() # Laplace distribution bias for robustness
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 4. Training & Validation Loop
print(f"🚀 Starting training on {DEVICE}...")

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    # --- TRAINING PHASE ---
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # --- VALIDATION PHASE ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train:.5f} | Val Loss: {avg_val:.5f}")

    # Save the BEST model based on validation performance
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), "checkpoints/best_point_model.pth")
        print("⭐ New best model saved!")

print("✅ Training complete.")