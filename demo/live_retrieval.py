"""
live_retrieval.py — Real-time pose retrieval from webcam.

Pipeline per frame:
  webcam → MediaPipe Hands → crop + convex-hull mask → 224×224 (FreiHAND format)
        → model backbone embedding → cosine search index → top-k results

Display (single OpenCV window):
  [  Webcam + MediaPipe overlay  ] [ Preprocessed input ] [ Top-k: image | cloud ]

Controls:
  Q / ESC  — quit
  S        — save current frame to demo_capture.png
  1/2/3    — show top-1 / top-2 / top-3 results (default: 3)

Usage (from repo root or demo/):
    python3 demo/live_retrieval.py
    python3 demo/live_retrieval.py --index data/retrieval_index_600_verts.npz
    python3 demo/live_retrieval.py --k 3 --update-every 5
"""

import argparse
import os
import sys
import time

import subprocess
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image
from torchvision import transforms

# ── path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from model import ScaffoldedPointPredictor

# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_INDEX      = os.path.join(REPO_ROOT, "data", "retrieval_index_600_verts.npz")
DEFAULT_MODEL      = os.path.join(REPO_ROOT, "scripts", "checkpoints", "model_600_verts_15_vectors.pth")
LANDMARKER_PATH    = os.path.join(REPO_ROOT, "demo", "hand_landmarker.task")
LANDMARKER_URL     = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
NUM_VERTS      = 600
NUM_VECTORS    = 15
IMG_SIZE       = 224   # FreiHAND input resolution
THUMB_SIZE     = 140   # display size for retrieved images
CLOUD_SIZE     = 180   # display size for cloud thumbnails
PAD_FRACTION   = 0.25  # bounding-box padding as fraction of longest side

SKEL_CONNECTIONS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]

PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# colours (BGR)
C_BLUE   = (237, 149, 100)
C_ORANGE = (0, 165, 255)
C_RED    = (80, 80, 220)
C_WHITE  = (255, 255, 255)
C_GRAY   = (80, 80, 80)
C_GREEN  = (80, 200, 80)
C_YELLOW = (0, 220, 220)


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index",        default=DEFAULT_INDEX)
    p.add_argument("--model",        default=DEFAULT_MODEL)
    p.add_argument("--k",            type=int, default=3, help="Number of retrievals to show (max 3)")
    p.add_argument("--update-every", type=int, default=3, help="Run retrieval every N frames")
    p.add_argument("--camera",       type=int, default=0, help="Camera device index")
    p.add_argument("--num-verts",    type=int, default=NUM_VERTS)
    p.add_argument("--num-vectors",  type=int, default=NUM_VECTORS)
    return p.parse_args()


# ── model + hook ──────────────────────────────────────────────────────────────

def load_model(model_path, num_verts, num_vectors, device):
    model = ScaffoldedPointPredictor(
        num_joints=21, num_verts=num_verts, num_vectors=num_vectors,
        pretrained_backbone=False,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    captured = {}
    def _hook(module, inp, out):
        captured["feat"] = out
    model.backbone_layers.register_forward_hook(_hook)

    return model, captured


@torch.no_grad()
def embed_pil(pil_img, model, captured, device):
    """Return L2-normalised 512-dim embedding for a PIL image."""
    tensor = PREPROCESS(pil_img).unsqueeze(0).to(device)
    model(tensor)
    feat = captured["feat"].squeeze().cpu().numpy()
    feat /= max(np.linalg.norm(feat), 1e-8)
    return feat


# ── MediaPipe preprocessing ───────────────────────────────────────────────────

def landmarks_to_bbox(landmarks, frame_h, frame_w):
    """Return (x1, y1, x2, y2) pixel bounding box from MediaPipe landmarks."""
    xs = [lm.x * frame_w for lm in landmarks]
    ys = [lm.y * frame_h for lm in landmarks]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    return x1, y1, x2, y2


def preprocess_frame(frame_rgb, landmarks, frame_h, frame_w):
    """
    Crop + mask the hand region to match FreiHAND's format:
      - square crop with padding around the landmark bounding box
      - convex-hull mask from landmark positions (black background outside hand)
      - resize to 224×224

    Returns the preprocessed uint8 numpy image (224, 224, 3) or None on failure.
    """
    x1, y1, x2, y2 = landmarks_to_bbox(landmarks, frame_h, frame_w)
    side = max(x2 - x1, y2 - y1)
    pad  = int(side * PAD_FRACTION)

    # Square crop, clamped to frame bounds
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half = side // 2 + pad
    cx1, cy1 = max(0, cx - half), max(0, cy - half)
    cx2, cy2 = min(frame_w, cx + half), min(frame_h, cy + half)

    if cx2 - cx1 < 10 or cy2 - cy1 < 10:
        return None

    crop = frame_rgb[cy1:cy2, cx1:cx2].copy()
    ch, cw = crop.shape[:2]

    # Project landmarks into crop coordinates
    lm_pts = np.array([
        [(lm.x * frame_w - cx1) * IMG_SIZE / cw,
         (lm.y * frame_h - cy1) * IMG_SIZE / ch]
        for lm in landmarks
    ], dtype=np.float32)

    # Resize crop to 224×224
    crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

    # Convex-hull mask from projected landmarks
    hull = cv2.convexHull(lm_pts.astype(np.int32))
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # Morphological dilation to widen the mask slightly
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel)

    # Apply mask — black background outside hand
    masked = cv2.bitwise_and(crop_resized, crop_resized, mask=mask)
    return masked  # uint8, RGB, 224×224


# ── Point cloud 2D renderer ───────────────────────────────────────────────────

def render_cloud_2d(verts, joints, size=CLOUD_SIZE):
    """
    Fast 2D projection of a point cloud (x,y plane — matches eval.py's
    elev=-90, azim=-90 view).  Returns a uint8 BGR image of shape (size, size, 3).
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)

    pts2d = np.vstack([verts[:, :2], joints[:, :2]])
    mn, mx = pts2d.min(axis=0), pts2d.max(axis=0)
    span = (mx - mn).max()
    if span < 1e-6:
        return img

    scale = (size * 0.82) / span
    center = (mn + mx) / 2.0

    def to_px(p):
        px = ((p[:, :2] - center) * scale + size / 2).astype(int)
        # flip y so hand reads upright in image coords
        px[:, 1] = size - 1 - px[:, 1]
        return px

    v_px = to_px(verts)
    j_px = to_px(joints)

    # verts
    for p in v_px:
        if 0 <= p[0] < size and 0 <= p[1] < size:
            cv2.circle(img, (p[0], p[1]), 2, C_BLUE, -1)

    # skeleton lines
    for finger in SKEL_CONNECTIONS:
        for ki in range(len(finger) - 1):
            a = tuple(j_px[finger[ki]])
            b = tuple(j_px[finger[ki + 1]])
            cv2.line(img, a, b, C_RED, 1, cv2.LINE_AA)

    # joints
    for p in j_px:
        if 0 <= p[0] < size and 0 <= p[1] < size:
            cv2.circle(img, (p[0], p[1]), 3, C_ORANGE, -1)

    return img


def prerender_all_clouds(pred_verts, pred_joints, num_verts):
    """Pre-render cloud thumbnails for every sample in the index."""
    N = len(pred_verts)
    clouds = []
    print(f"Pre-rendering {N} cloud thumbnails...", end=" ", flush=True)
    t0 = time.time()
    for i in range(N):
        v = pred_verts[i].reshape(-1, 3)
        j = pred_joints[i].reshape(21, 3)
        clouds.append(render_cloud_2d(v, j))
    print(f"done ({time.time() - t0:.1f}s)")
    return clouds


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query_emb, embeddings, k):
    sims = embeddings @ query_emb
    top_k = np.argsort(sims)[::-1][:k]
    return top_k, sims[top_k]


# ── Display helpers ───────────────────────────────────────────────────────────

def _fit(img, w, h):
    """Resize an image to fit within (w, h), maintaining aspect ratio, with black letterbox."""
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def _label(img, text, pos=(6, 18), color=C_WHITE, scale=0.45, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,  thickness,   cv2.LINE_AA)


def build_display(
    webcam_bgr, preprocessed_rgb, db_images, cloud_thumbs,
    top_k_indices, top_k_sims, k,
    no_hand=False, fps=0.0,
):
    """
    Assemble the full display frame:
      [  webcam 480×480  ][  input 200×200  ][  retrieval panel  ]
    """
    PANEL_H = 480
    CAM_W   = 480
    PRE_W   = 200
    RET_W   = CLOUD_SIZE + THUMB_SIZE + 10  # cloud + image + gap

    total_w = CAM_W + PRE_W + RET_W + 4  # 2px separators

    canvas = np.zeros((PANEL_H, total_w, 3), dtype=np.uint8)

    # ── webcam panel ──────────────────────────────────────────────────────────
    cam_panel = _fit(webcam_bgr, CAM_W, PANEL_H)
    _label(cam_panel, f"Live  {fps:.0f} fps", pos=(6, 22), scale=0.5)
    if no_hand:
        _label(cam_panel, "No hand detected", pos=(6, PANEL_H - 10), color=C_YELLOW, scale=0.5)
    canvas[:, :CAM_W] = cam_panel

    # separator
    canvas[:, CAM_W:CAM_W+2] = 40

    # ── preprocessed input panel ──────────────────────────────────────────────
    pre_x = CAM_W + 2
    pre_panel = np.zeros((PANEL_H, PRE_W, 3), dtype=np.uint8)
    if preprocessed_rgb is not None:
        pre_img = cv2.cvtColor(preprocessed_rgb, cv2.COLOR_RGB2BGR)
        pre_thumb = _fit(pre_img, PRE_W, PRE_W)
        pre_panel[10:10 + PRE_W] = pre_thumb
        _label(pre_panel, "Model input", pos=(6, PRE_W + 24), scale=0.42, color=C_GRAY)
    else:
        _label(pre_panel, "Model input", pos=(6, PANEL_H // 2), scale=0.45, color=C_GRAY)
    canvas[:, pre_x:pre_x + PRE_W] = pre_panel

    # separator
    sep_x = pre_x + PRE_W
    canvas[:, sep_x:sep_x + 2] = 40

    # ── retrieval panel ───────────────────────────────────────────────────────
    ret_x = sep_x + 2
    row_h  = PANEL_H // 3

    for rank in range(k):
        ry = rank * row_h

        if top_k_indices is not None and rank < len(top_k_indices):
            db_idx = top_k_indices[rank]
            sim    = top_k_sims[rank]

            # dataset image thumbnail
            db_img = cv2.cvtColor(db_images[db_idx], cv2.COLOR_RGB2BGR)
            thumb  = _fit(db_img, THUMB_SIZE, row_h - 20)
            th, tw = thumb.shape[:2]
            ty = ry + (row_h - 20 - th) // 2
            canvas[ty:ty + th, ret_x:ret_x + tw] = thumb
            _label(canvas, f"#{rank+1}  sim={sim:.3f}", pos=(ret_x + 2, ry + row_h - 6),
                   scale=0.42, color=C_GREEN)

            # cloud thumbnail
            cloud = cloud_thumbs[db_idx]
            cth, ctw = cloud.shape[:2]
            cy_raw = ry + (row_h - cth) // 2
            cy = max(0, cy_raw)
            cx = ret_x + THUMB_SIZE + 10
            cloud_offset = cy - cy_raw
            cy_end = min(PANEL_H, cy + cth - cloud_offset)
            canvas[cy:cy_end, cx:cx + ctw] = cloud[cloud_offset:cloud_offset + (cy_end - cy)]

        else:
            _label(canvas, f"Rank {rank+1}: —", pos=(ret_x + 4, ry + row_h // 2),
                   scale=0.45, color=C_GRAY)

        if rank < k - 1:
            canvas[ry + row_h - 1, ret_x:] = 30  # thin row divider

    # header
    _label(canvas, "Retrieved poses", pos=(ret_x + 4, 16), scale=0.5, color=C_WHITE)

    return canvas


# ── MediaPipe hand drawing (tasks API has no built-in draw helper) ────────────

# Standard 21-point hand connections
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),           # thumb
    (0,5),(5,6),(6,7),(7,8),           # index
    (5,9),(9,10),(10,11),(11,12),      # middle
    (9,13),(13,14),(14,15),(15,16),    # ring
    (0,17),(13,17),(17,18),(18,19),(19,20),  # pinky + palm
]

def draw_hand_landmarks(frame_bgr, landmarks, frame_h, frame_w):
    """Draw 21 landmarks and connections onto a BGR frame in-place."""
    pts = [
        (int(lm.x * frame_w), int(lm.y * frame_h))
        for lm in landmarks
    ]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], (0, 220, 0), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame_bgr, p, 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, p, 4, (0, 150, 0),    1, cv2.LINE_AA)


def ensure_landmarker_model(path, url):
    """Download the hand_landmarker.task file if it isn't already cached.
    Uses curl so the download works regardless of Python SSL cert configuration."""
    if os.path.exists(path):
        return
    print(f"Downloading MediaPipe hand landmarker model → {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    result = subprocess.run(["curl", "-fsSL", "-o", path, url])
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to download model from {url}\n"
            f"Download manually and place at: {path}"
        )
    print("Download complete.")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    k = max(1, min(args.k, 3))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load index
    if not os.path.exists(args.index):
        print(f"Index not found: {args.index}\nRun  python3 scripts/build_index.py  first.")
        sys.exit(1)

    idx_data    = np.load(args.index)
    db_images   = idx_data["images"]        # (N, 224, 224, 3) uint8
    embeddings  = idx_data["embeddings"]    # (N, 512) L2-normalised
    pred_joints = idx_data["pred_joints"]   # (N, 63)
    pred_verts  = idx_data["pred_verts"]    # (N, V*3)
    print(f"Index loaded: {len(db_images)} samples")

    # Pre-render all cloud thumbnails
    cloud_thumbs = prerender_all_clouds(pred_verts, pred_joints, args.num_verts)

    # Load model
    model, captured = load_model(args.model, args.num_verts, args.num_vectors, device)
    print("Model loaded.")

    # MediaPipe hand landmarker (tasks API)
    ensure_landmarker_model(LANDMARKER_PATH, LANDMARKER_URL)
    lm_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(lm_options)

    # Camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nRunning — press Q or ESC to quit, S to save frame")
    print(f"Retrieval updates every {args.update_every} frames\n")

    # State
    frame_count    = 0
    fps            = 0.0
    t_fps          = time.time()
    preprocessed   = None
    top_k_indices  = None
    top_k_sims     = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # FPS every 30 frames
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - t_fps)
            t_fps = time.time()

        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection = landmarker.detect(mp_image)
        no_hand = not detection.hand_landmarks

        # Draw MediaPipe overlay on BGR frame copy
        frame_display = frame.copy()
        if not no_hand:
            draw_hand_landmarks(frame_display, detection.hand_landmarks[0], h, w)

        # Retrieval every N frames when hand is present
        if not no_hand and frame_count % args.update_every == 0:
            lms = detection.hand_landmarks[0]
            preprocessed = preprocess_frame(frame_rgb, lms, h, w)
            if preprocessed is not None:
                pil = Image.fromarray(preprocessed)
                emb = embed_pil(pil, model, captured, device)
                top_k_indices, top_k_sims = retrieve(emb, embeddings, k)

        if no_hand:
            preprocessed  = None
            top_k_indices = None
            top_k_sims    = None

        # Build display
        display = build_display(
            webcam_bgr=frame_display,
            preprocessed_rgb=preprocessed,
            db_images=db_images,
            cloud_thumbs=cloud_thumbs,
            top_k_indices=top_k_indices,
            top_k_sims=top_k_sims,
            k=k,
            no_hand=no_hand,
            fps=fps,
        )

        cv2.imshow("3D Hand Pose Retrieval", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # Q or ESC
            break
        elif key == ord('s'):
            fname = f"demo_capture_{int(time.time())}.png"
            cv2.imwrite(fname, display)
            print(f"Saved {fname}")
        elif key == ord('1'):
            k = 1
        elif key == ord('2'):
            k = 2
        elif key == ord('3'):
            k = 3

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Done.")


if __name__ == "__main__":
    main()
