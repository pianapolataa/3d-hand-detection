import os
import io
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms
from model import ScaffoldedPointPredictor

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'checkpoints/best_point_model_newest_12560.pth'
IMAGE_FOLDER = 'eval_images_converted'
IMG_SIZE = 224
NUM_VERTS = 778

import mediapipe as mp
from rembg import remove as rembg_remove
_mp_hands = mp.solutions.hands
print("✅ Using MediaPipe + rembg for background removal.")

# Skeletal Connection Mapping (MANO Standard)
SKEL_CONNECTIONS = [
    [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]
]

# --- 1. LOAD MODEL ---
print(f"🔄 Loading Super-Fusion model from {MODEL_PATH}...")
model = ScaffoldedPointPredictor(num_joints=21, num_verts=NUM_VERTS).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
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

def crop_hand_bbox(img_bgr):
    """Use MediaPipe landmarks to crop tightly around the hand, excluding sleeve/arm."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    with _mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        print("  ⚠️  No hand detected by MediaPipe — using full image.")
        return img_bgr
    xs = [lm.x for lm in results.multi_hand_landmarks[0].landmark]
    ys = [lm.y for lm in results.multi_hand_landmarks[0].landmark]
    pad = 0.08  # fractional padding around landmark bounding box
    x1 = max(0, int((min(xs) - pad) * w))
    x2 = min(w, int((max(xs) + pad) * w))
    y1 = max(0, int((min(ys) - pad) * h))
    y2 = min(h, int((max(ys) + pad) * h))
    cropped = img_bgr[y1:y2, x1:x2]
    if cropped.size == 0:
        print("  ⚠️  Empty crop — using full image.")
        return img_bgr
    return cropped

def remove_background(img_bgr):
    """Apply rembg on the already hand-cropped image, composite onto black."""
    _, buf = cv2.imencode('.png', img_bgr)
    output_bytes = rembg_remove(buf.tobytes())
    rgba = Image.open(io.BytesIO(output_bytes)).convert('RGBA')
    black_bg = Image.new('RGBA', rgba.size, (0, 0, 0, 255))
    black_bg.paste(rgba, mask=rgba.split()[3])
    return np.array(black_bg.convert('RGB'))


# --- Resize and center helper ---
def resize_and_center(img_rgb, size=224, target_ratio=0.88):
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
    cv2.imwrite(f"debug_raw_{i}.png", original_img)

    img_bgr = normalize_lighting(img_bgr)      # 1. even out lighting
    cv2.imwrite(f"debug_light_{i}.png", img_bgr)

    img_bgr = crop_hand_bbox(img_bgr)          # 2. crop to hand only (excludes sleeve)
    cv2.imwrite(f"debug_crop_{i}.png", img_bgr)

    img_rgb = remove_background(img_bgr)       # 3. rembg on tight crop → black background
    cv2.imwrite(f"debug_mask_{i}.png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    final_img = resize_and_center(img_rgb, IMG_SIZE)
    cv2.imwrite(f"debug_final_{i}.png", cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

    pil_img = Image.fromarray(final_img)
    input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        p_joints, _, p_verts = model(input_tensor)

    pred_j = p_joints.cpu().numpy().reshape(21, 3)
    pred_v = p_verts.cpu().numpy().reshape(NUM_VERTS, 3)

    display = final_img
    return display, pred_j, pred_v

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
    
    # 2. Plot the Joints (Black crosses)
    ax2.scatter(p_j[:, 0], p_j[:, 1], p_j[:, 2], s=40, c='black', marker='x', label='Pred Joints')
    
    # 3. Draw the Skeleton (Blue lines)
    for finger in SKEL_CONNECTIONS:
        ax2.plot(p_j[finger, 0], p_j[finger, 1], p_j[finger, 2], color='blue', linewidth=2)

    # 4. Formatting
    ax2.set_title("3D Scaffold Prediction")
    ax2.view_init(elev=-90, azim=-90) # Match FreiHAND camera view
    
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()