from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    from torchvision import models, transforms
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "torchvision is required for clustering inference. "
        "Activate one environment only, then install repo dependencies with "
        "`python -m pip install -e .` from the 3d-hand-detection repo root."
    ) from exc


NUM_JOINTS = 21
DEFAULT_NUM_VERTS = 600
DEFAULT_NUM_VECTORS = 15
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_CLUSTERS = 4
DEFAULT_RANDOM_SEED = 42

WRIST_ID = 0
TIP_IDS = [4, 8, 12, 16, 20]
MCP_IDS = [2, 5, 9, 13, 17]
THUMB_TIP = 4
OTHER_TIPS = [8, 12, 16, 20]

INDEX_MCP_ID = 5
MIDDLE_MCP_ID = 9
PINKY_MCP_ID = 17


class ScaffoldedPointPredictor(nn.Module):
    """Checkpoint-compatible copy of the training model without weight downloads."""

    def __init__(
        self,
        num_joints: int = NUM_JOINTS,
        num_verts: int = DEFAULT_NUM_VERTS,
        num_vectors: int = DEFAULT_NUM_VECTORS,
    ) -> None:
        super().__init__()

        backbone = models.resnet18(weights=None)
        self.backbone_layers = nn.Sequential(*list(backbone.children())[:-1])

        self.joint_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_joints * 3),
        )

        self.vector_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_vectors * 3),
        )

        combined_input_dim = 512 + (num_joints * 3) + (num_vectors * 3)
        self.mesh_head = nn.Sequential(
            nn.Linear(combined_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_verts * 3),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone_layers(x)
        feat_flat = torch.flatten(feat, 1)

        pred_joints = self.joint_head(feat_flat)
        pred_vectors = self.vector_head(feat_flat)

        fused_features = torch.cat((feat_flat, pred_joints, pred_vectors), dim=1)
        pred_vertices = self.mesh_head(fused_features)
        return pred_joints, pred_vectors, pred_vertices


def resolve_existing_path(candidates: Iterable[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find any of these paths:\n{joined}")


def infer_num_verts_from_npz(npz_path: Path, split: str = "train") -> int:
    split_map = {
        "train": "y_train_verts",
        "val": "y_val_verts",
        "test": "y_test_verts",
    }
    target_key = split_map[split]
    with np.load(npz_path) as data:
        return int(data[target_key].shape[1] // 3)


def infer_model_dims_from_checkpoint(checkpoint_path: Path) -> Dict[str, int]:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    num_joints = int(state_dict["joint_head.3.weight"].shape[0] // 3)
    num_vectors = int(state_dict["vector_head.3.weight"].shape[0] // 3)
    num_verts = int(state_dict["mesh_head.5.weight"].shape[0] // 3)
    return {
        "num_joints": num_joints,
        "num_vectors": num_vectors,
        "num_verts": num_verts,
    }


def build_inference_model(
    checkpoint_path: Path,
    num_joints: int,
    num_verts: int,
    num_vectors: int,
    device: torch.device,
) -> nn.Module:
    model = ScaffoldedPointPredictor(
        num_joints=num_joints,
        num_verts=num_verts,
        num_vectors=num_vectors,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def default_image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def preprocess_image_batch(
    images: np.ndarray,
    transform: transforms.Compose,
) -> torch.Tensor:
    tensors = [transform(Image.fromarray(image)) for image in images]
    return torch.stack(tensors, dim=0)


def safe_normalize(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, eps, None)


def compute_hand_frame_features(pred_joints: np.ndarray) -> np.ndarray:
    """
    Convert predicted joints of shape [N, 21, 3] into [N, 6] hand-frame features.
    """
    palm_normal, middle_axis = compute_hand_frame_components(pred_joints)
    return np.concatenate([palm_normal, middle_axis], axis=1).astype(np.float32)


def compute_hand_frame_components(pred_joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return normalized palm normal and wrist->middle-mcp axis for joints [N, 21, 3].
    """
    wrist = pred_joints[:, WRIST_ID, :]
    index_mcp = pred_joints[:, INDEX_MCP_ID, :] - wrist
    pinky_mcp = pred_joints[:, PINKY_MCP_ID, :] - wrist
    middle_mcp = pred_joints[:, MIDDLE_MCP_ID, :] - wrist

    palm_normal = np.cross(index_mcp, pinky_mcp)
    palm_normal = safe_normalize(palm_normal)
    middle_axis = safe_normalize(middle_mcp)
    return palm_normal.astype(np.float32), middle_axis.astype(np.float32)
