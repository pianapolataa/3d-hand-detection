import os
import io
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms
from model import ScaffoldedPointPredictor
from rembg import remove as rembg_remove

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'scripts/checkpoints/model_600_verts_15_vectors.pth'
IMAGE_FOLDER = 'eval_images'  # Folder containing test images
IMG_SIZE = 224
NUM_JOINTS = 21

# Set this manually to match the checkpoint you want to evaluate.
# Examples: 70, 194, 400, 600, 778
NUM_VERTS = 600

# Optional: set this manually if you know the checkpoint's scaffold/joint output format.
# If left as None, it will be inferred from the checkpoint.
NUM_VECTORS = None

MASK_THRESHOLD = 10
TARGET_RATIO = 0.75
BBOX_PAD_RATIO = 0.15

print("✅ Using rembg + non-black-pixel bounding box (no MediaPipe).")

# Skeletal Connection Mapping (MANO Standard)
SKEL_CONNECTIONS = [
    [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]
]

# --- 1. LOAD MODEL ---
print(f"🔄 Loading Super-Fusion model from {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

vector_head_bias = checkpoint.get('vector_head.3.bias')
if vector_head_bias is None:
    raise KeyError("Missing 'vector_head.3.bias' in checkpoint; cannot infer joint/vector output size.")
mesh_head_bias = checkpoint.get('mesh_head.5.bias')
if mesh_head_bias is None:
    raise KeyError("Missing 'mesh_head.5.bias' in checkpoint; cannot infer vertex output size.")

JOINT_OUTPUT_DIMS = int(vector_head_bias.numel())
NUM_CHECKPOINT_VECTORS = JOINT_OUTPUT_DIMS // 3
NUM_CHECKPOINT_VERTS = int(mesh_head_bias.numel() // 3)

if NUM_VECTORS is None:
    NUM_VECTORS = NUM_CHECKPOINT_VECTORS

print(f"📌 Config NUM_VERTS: {NUM_VERTS}")
print(f"📌 Checkpoint NUM_VERTS: {NUM_CHECKPOINT_VERTS}")
print(f"📌 Config NUM_VECTORS: {NUM_VECTORS}")
print(f"📌 Checkpoint NUM_VECTORS: {NUM_CHECKPOINT_VECTORS}")
print(f"📌 Joint branch output dims: {JOINT_OUTPUT_DIMS}")

if NUM_VERTS != NUM_CHECKPOINT_VERTS:
    raise ValueError(
        f"NUM_VERTS mismatch: config says {NUM_VERTS}, but checkpoint outputs {NUM_CHECKPOINT_VERTS}. "
        f"Please update NUM_VERTS at the top of eval_test_images.py to match the selected model."
    )

if NUM_VECTORS != NUM_CHECKPOINT_VECTORS:
    raise ValueError(
        f"NUM_VECTORS mismatch: config says {NUM_VECTORS}, but checkpoint outputs {NUM_CHECKPOINT_VECTORS}. "
        f"Please update NUM_VECTORS (or set it to None) to match the selected model."
    )

HAS_TRUE_21_JOINTS = (JOINT_OUTPUT_DIMS == NUM_JOINTS * 3)
NUM_CHECKPOINT_JOINTS = JOINT_OUTPUT_DIMS // 3
print(f"📌 Checkpoint joint/scaffold points: {NUM_CHECKPOINT_JOINTS}")
if not HAS_TRUE_21_JOINTS:
    print("⚠️ Checkpoint joint branch is not a 21-joint skeleton. Fallback visualization will show checkpoint scaffold points only.")

model = ScaffoldedPointPredictor(num_joints=NUM_JOINTS, num_verts=NUM_VERTS, num_vectors=NUM_VECTORS).to(DEVICE)
model.load_state_dict(checkpoint)
model.eval()

# --- 2. PREPROCESSING PIPELINE ---
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def normalize_lighting(img_bgr):
    """Apply CLAHE on the L channel to even out lighting across photos."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def crop_hand_bbox(img_rgb, mask, pad_ratio=BBOX_PAD_RATIO):
    """Crop tightly around the largest non-black foreground region from rembg."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        print("  ⚠️  No foreground pixels found after rembg — using full masked image.")
        return img_rgb

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    h, w = img_rgb.shape[:2]
    pad_x = int((x2 - x1 + 1) * pad_ratio)
    pad_y = int((y2 - y1 + 1) * pad_ratio)

    x1 = max(0, x1 - pad_x)
    x2 = min(w, x2 + pad_x + 1)
    y1 = max(0, y1 - pad_y)
    y2 = min(h, y2 + pad_y + 1)

    cropped = img_rgb[y1:y2, x1:x2]
    if cropped.size == 0:
        print("  ⚠️  Empty crop after bbox — using full masked image.")
        return img_rgb
    return cropped

def remove_background(img_bgr):
    """Apply rembg and return black-background RGB image plus binary foreground mask."""
    _, buf = cv2.imencode('.png', img_bgr)
    output_bytes = rembg_remove(buf.tobytes())
    rgba = Image.open(io.BytesIO(output_bytes)).convert('RGBA')

    rgba_np = np.array(rgba)
    alpha = rgba_np[:, :, 3]
    mask = (alpha > 0).astype(np.uint8) * 255

    rgb = rgba_np[:, :, :3]
    black_bg = np.zeros_like(rgb)
    black_bg[mask > 0] = rgb[mask > 0]
    return black_bg, mask


# --- Resize and center helper ---
def resize_and_center(img_rgb, size=224, target_ratio=TARGET_RATIO):
    """Resize while preserving aspect ratio, then paste onto a black square canvas."""
    h, w = img_rgb.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)

    scale = target_ratio * size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img_rgb, (new_w, new_h))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas

def predict_on_image(img_path, i):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"❌ Failed to read image: {img_path}")
        return None, None, None

    original_img = img_bgr.copy()
    #cv2.imwrite(f"debug_raw_{i}.png", original_img)

    img_bgr = normalize_lighting(img_bgr)      # 1. even out lighting
    #cv2.imwrite(f"debug_light_{i}.png", img_bgr)

    img_rgb, fg_mask = remove_background(img_bgr)   # 2. rembg first
    #cv2.imwrite(f"debug_mask_{i}.png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    img_rgb = crop_hand_bbox(img_rgb, fg_mask)      # 3. bbox from non-black / foreground pixels
    #cv2.imwrite(f"debug_crop_{i}.png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    final_img = resize_and_center(img_rgb, IMG_SIZE)
    #cv2.imwrite(f"debug_final_{i}.png", cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

    pil_img = Image.fromarray(final_img)
    input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        p_joints, _, p_verts = model(input_tensor)

    joint_values = p_joints.cpu().numpy().reshape(-1)
    pred_j = None
    if joint_values.size % 3 == 0 and joint_values.size > 0:
        pred_j = joint_values.reshape(-1, 3)
    pred_v = p_verts.cpu().numpy().reshape(NUM_VERTS, 3)

    display = final_img
    return display, pred_j, pred_v

def set_axes_equal_3d(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    if radius <= 0:
        radius = 1.0

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)

# --- 3. MAIN LOOP ---
image_files = sorted(
    [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
)

if not image_files:
    print(f"❌ No images found in '{IMAGE_FOLDER}' folder.")
else:
    print(f"🖼️ Found {len(image_files)} images. Starting inference...")

for i, filename in enumerate(image_files):
    img_path = os.path.join(IMAGE_FOLDER, filename)
    print(f"🔍 Processing: {filename}")
    
    result = predict_on_image(img_path, i)
    if result[0] is None:
        continue
    img_display, p_j, p_v = result
    # img_display, p_j, p_v = predict_on_image(img_path, i)
    
    fig = plt.figure(figsize=(15, 7))
    fig.suptitle(f"Inference: {filename}", fontsize=14)
    
    # Left: Original Image
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img_display)
    ax1.set_title("Input (Resized)")
    ax1.axis('off')
    
    # Right: 3D Reconstruction
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # 1. Plot the Mesh (Red cloud)
    ax2.scatter(p_v[:, 0], p_v[:, 1], p_v[:, 2], s=2, c='red', alpha=0.4, label='Pred Mesh')
    
    # 2. Plot the checkpoint joint/scaffold outputs
    if p_j is not None:
        ax2.scatter(p_j[:, 0], p_j[:, 1], p_j[:, 2], s=40, c='black', marker='x', label='Pred Joints / Scaffold')

        if p_j.shape[0] == NUM_JOINTS:
            # 3. Draw the 21-joint skeleton only when the checkpoint really outputs 21 joints
            for finger in SKEL_CONNECTIONS:
                ax2.plot(p_j[finger, 0], p_j[finger, 1], p_j[finger, 2], color='blue', linewidth=2)
            ax2.set_title("3D Mesh + 21-Joint Skeleton")
        else:
            ax2.set_title(f"3D Mesh + {p_j.shape[0]} Checkpoint Scaffold Points")
    else:
        ax2.set_title("3D Mesh Prediction")
    set_axes_equal_3d(ax2, p_v)
    ax2.view_init(elev=-90, azim=-90) # Match FreiHAND camera view
    
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()