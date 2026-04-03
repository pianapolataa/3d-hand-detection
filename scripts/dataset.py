import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class PointDataset(Dataset):
    def __init__(self, npz_path, mode='train', transform=None):
        """
        Args:
            npz_path (str): Path to the train_data_778.npz file.
            mode (str): Either 'train' or 'val' to load the respective split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Load the compressed data
        data = np.load(npz_path)
        
        # Select the correct split based on mode
        if mode == 'train':
            self.images = data['x_train']
            self.verts = data['y_train']
        elif mode == 'val':
            self.images = data['x_val']
            self.verts = data['y_val']
        else:
            raise ValueError("Mode must be 'train' or 'val'")
            
        self.transform = transform

    def __len__(self):
        # Returns the number of samples in the selected split
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Get the image (currently a NumPy array [224, 224, 3])
        image = self.images[idx]
        
        # 2. Get the label (the flattened 2334 vertex coordinates)
        labels = torch.from_numpy(self.verts[idx]).float()
        
        # 3. Apply transformations
        if self.transform:
            # We convert the NumPy array to a PIL Image first because 
            # torchvision transforms usually expect PIL or Tensors.
            image = Image.fromarray(image)
            image = self.transform(image)
            
        return image, labels