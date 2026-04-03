import torch
import torch.nn as nn
from torchvision import models

class PointPredictor(nn.Module):
    def __init__(self, num_points=778):
        super(PointPredictor, self).__init__()
        
        # 1. Backbone: Feature Extraction (The "Eye")
        # Pretrained ResNet18 extracts spatial features from the RGB image
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # We strip the final classification layer, keeping up to the Global Average Pool
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # 2. Deep Rectifier Head (The "Brain" from your notes)
        # Total output is num_points * 3 (X, Y, Z for each vertex)
        num_outputs = num_points * 3 
        
        self.rectifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),        # Layer U0
            nn.ReLU(),                   # max(0, ...)
            nn.Dropout(0.1),             # Regularization for "Most Accurate" results
            nn.Linear(1024, 1024),       # Layer U1
            nn.ReLU(),                   # max(0, ...)
            nn.Linear(1024, num_outputs) # Final Linear Output UL
        )

    def forward(self, x):
        # x shape: [Batch, 3, 224, 224]
        x = self.features(x)
        # x shape after features: [Batch, 512, 1, 1]
        
        # Output shape: [Batch, 2334]
        out = self.rectifier_head(x)
        return out