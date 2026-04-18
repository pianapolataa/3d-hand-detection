import torch
import torch.nn as nn
from torchvision import models

class ScaffoldedPointPredictor(nn.Module):
    def __init__(self, num_joints=21, num_verts=778, num_vectors=9):
        super(ScaffoldedPointPredictor, self).__init__()
        
        # 1. Shared Backbone: ResNet18 Feature Extraction (512 features)
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # We strip the final FC layer to get raw feature maps
        self.backbone_layers = nn.Sequential(*list(backbone.children())[:-1]) 
        
        # 2. Joint Scaffold Head (21 Joints * 3 = 63 values)
        self.joint_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_joints * 3) 
        )

        # 3. Vector Orientation Head (9 Vectors * 3 = 27 values)
        # This head explicitly learns 'Pinch' and 'Flexion' directions
        self.vector_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_vectors * 3) 
        )
        
        # 4. Dense Mesh Head (778 Vertices * 3 = 2334 values)
        # INPUT: Image Features (512) + Predicted Joints (63) + Predicted Vectors (27)
        # TOTAL INPUT DIM: 602
        combined_input_dim = 512 + (num_joints * 3) + (num_vectors * 3)
        
        self.mesh_head = nn.Sequential(
            nn.Linear(combined_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_verts * 3) 
        )

    def forward(self, x):
        # x: [Batch, 3, 224, 224]
        
        # --- STAGE 1: FEATURE EXTRACTION ---
        feat = self.backbone_layers(x) # [Batch, 512, 1, 1]
        feat_flat = torch.flatten(feat, 1) # [Batch, 512]
        
        # --- STAGE 2: SCAFFOLD PREDICTION ---
        # We predict BOTH the absolute positions and relative orientations
        pred_joints = self.joint_head(feat_flat)   # [Batch, 63]
        pred_vectors = self.vector_head(feat_flat) # [Batch, 27]
        
        # --- STAGE 3: THE SUPER-FUSION ---
        # We concatenate raw image 'intuition' with explicit geometric 'facts'
        # feat_flat (512) + pred_joints (63) + pred_vectors (27) = 602
        fused_features = torch.cat((feat_flat, pred_joints, pred_vectors), dim=1) 
        
        # --- STAGE 4: DENSE SURFACE RECONSTRUCTION ---
        pred_vertices = self.mesh_head(fused_features) # [Batch, 2334]
        
        # Return all three so we can calculate individual losses for each head
        return pred_joints, pred_vectors, pred_vertices