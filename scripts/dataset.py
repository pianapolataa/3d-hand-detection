import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class PointDataset(Dataset):
    def __init__(self, npz_path, mode='train', transform=None):
        """
        Args:
            npz_path (str): Path to the robust_train_data.npz file.
            mode (str): Either 'train' or 'val'.
            transform (callable): Torchvision transforms (Normalize, ToTensor).
        """
        # Load the compressed data
        # Note: Ensure your preprocessing script saved 'y_joints' and 'y_verts'
        data = np.load(npz_path)
        
        if mode == 'train':
            self.images = data['x_train']
            self.joints = data['y_train_joints'] # The 21 Skeleton points (63 values)
            self.verts = data['y_train_verts']   # The 778 Mesh points (2334 values)
        elif mode == 'val':
            self.images = data['x_val']
            self.joints = data['y_val_joints']
            self.verts = data['y_val_verts']
        else:
            raise ValueError("Mode must be 'train' or 'val'")
            
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Get the image (NumPy [224, 224, 3])
        image = self.images[idx]
        
        # 2. Get the labels
        # Joints: (63,) | Verts: (2334,)
        joints = torch.from_numpy(self.joints[idx]).float()
        verts = torch.from_numpy(self.verts[idx]).float()
        
        # 3. Apply transformations
        if self.transform:
            # Convert NumPy HWC to PIL for torchvision compatibility
            image = Image.fromarray(image)
            image = self.transform(image)
            
        # Return all three for the Multi-Task Training Loop
        return image, joints, verts