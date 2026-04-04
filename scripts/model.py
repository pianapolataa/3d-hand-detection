import torch
import torch.nn as nn
from torchvision import models

class ScaffoldedPointPredictor(nn.Module):
    def __init__(self, num_joints=21, num_verts=778):
        super(ScaffoldedPointPredictor, self).__init__()
        
        # 1. Shared Backbone: ResNet18 Feature Extraction
        # Extracts 512 high-level spatial features from the 224x224 image
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1]) # Output: [Batch, 512, 1, 1]
        
        # 2. Skeletal Scaffold Head (The "MediaPipe-style" Anchor)
        # This head learns the 21 fundamental joints (63 values)
        # It provides the global structural 'truth' for the hand
        self.joint_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_joints * 3) # Output: 63
        )
        
        # 3. Dense Mesh Head (The 778 Vertices)
        # Crucially, this head receives BOTH the 512 image features AND the 63 joint coords.
        # This 'scaffolding' tells the model exactly where the hand structure is 
        # before it tries to guess the skin surface.
        combined_input_dim = 512 + (num_joints * 3) # 512 + 63 = 575
        
        self.mesh_head = nn.Sequential(
            nn.Linear(combined_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_verts * 3) # Output: 2334
        )

    def forward(self, x):
        # x: [Batch, 3, 224, 224]
        
        # Stage 1: Feature Extraction
        feat = self.features(x)
        feat_flat = torch.flatten(feat, 1) # [Batch, 512]
        
        # Stage 2: Predict Skeletal Joints (Stage 1 output)
        # These joints act as the 'scaffold' for the next layer
        pred_joints = self.joint_head(feat_flat) # [Batch, 63]
        
        # Stage 3: Feature Fusion
        # Concatenate image features with predicted joints
        # This forces the mesh head to be 'spatially aware'
        fused_features = torch.cat((feat_flat, pred_joints), dim=1) # [Batch, 575]
        
        # Stage 4: Predict Dense Mesh (778 vertices)
        pred_vertices = self.mesh_head(fused_features) # [Batch, 2334]
        
        # We return both so the Training script can calculate separate losses
        return pred_joints, pred_vertices