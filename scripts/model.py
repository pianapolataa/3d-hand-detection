# import torch
# import torch.nn as nn
# from torchvision import models

# class ScaffoldedPointPredictor(nn.Module):
#     def __init__(self, num_joints=21, num_verts=778):
#         super(ScaffoldedPointPredictor, self).__init__()
        
#         # 1. Shared Backbone: ResNet18 Feature Extraction
#         # Extracts 512 high-level spatial features from the 224x224 image
#         backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#         self.features = nn.Sequential(*list(backbone.children())[:-1]) # Output: [Batch, 512, 1, 1]
        
#         # 2. Skeletal Scaffold Head (The "MediaPipe-style" Anchor)
#         # This head learns the 21 fundamental joints (63 values)
#         # It provides the global structural 'truth' for the hand
#         self.joint_head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(512, num_joints * 3) # Output: 63
#         )
        
#         # 3. Dense Mesh Head (The 778 Vertices)
#         # Crucially, this head receives BOTH the 512 image features AND the 63 joint coords.
#         # This 'scaffolding' tells the model exactly where the hand structure is 
#         # before it tries to guess the skin surface.
#         combined_input_dim = 512 + (num_joints * 3) # 512 + 63 = 575
        
#         self.mesh_head = nn.Sequential(
#             nn.Linear(combined_input_dim, 1024),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(1024, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, num_verts * 3) # Output: 2334
#         )

#     def forward(self, x):
#         # x: [Batch, 3, 224, 224]
        
#         # Stage 1: Feature Extraction
#         feat = self.features(x)
#         feat_flat = torch.flatten(feat, 1) # [Batch, 512]
        
#         # Stage 2: Predict Skeletal Joints (Stage 1 output)
#         # These joints act as the 'scaffold' for the next layer
#         pred_joints = self.joint_head(feat_flat) # [Batch, 63]
        
#         # Stage 3: Feature Fusion
#         # Concatenate image features with predicted joints
#         # This forces the mesh head to be 'spatially aware'
#         fused_features = torch.cat((feat_flat, pred_joints), dim=1) # [Batch, 575]
        
#         # Stage 4: Predict Dense Mesh (778 vertices)
#         pred_vertices = self.mesh_head(fused_features) # [Batch, 2334]
        
#         # We return both so the Training script can calculate separate losses
#         return pred_joints, pred_vertices

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