"""
Real-time hand pose retrieval from webcam using MediaPipe landmarks.

Pipeline per frame:
  webcam → MediaPipe Hand Landmarker → wrist-centered joint embedding
        → cosine search → top-k retrieved poses

The embedding is purely geometric: 21 landmarks, wrist-centered, L2-normalised.
No custom model is needed at query time — MediaPipe generalises well to
uncalibrated cameras, whereas our trained backbone does not.

Display:
  [  Webcam + skeleton overlay  ] [  Top-k: dataset image | point cloud  ]

Controls:
  Q / ESC  — quit
  S        — save frame to disk
  1/2/3    — show top-1/2/3 results (default: 3)

"""

import argparse
import os
import subprocess
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INDEX  = os.path.join(REPO_ROOT, "data", "retrieval_index_600_verts.npz")
LANDMARKER_PATH = os.path.join(REPO_ROOT, "demo", "hand_landmarker.task")
LANDMARKER_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

THUMB_SIZE  = 140
CLOUD_SIZE  = 180
PANEL_H     = 480
CAM_W       = 480

SKEL_CONNECTIONS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]

_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (0,17),(13,17),(17,18),(18,19),(19,20),
]

# BGR colours
C_BLUE   = (237, 149, 100)
C_ORANGE = (0, 165, 255)
C_RED    = (80, 80, 220)
C_WHITE  = (255, 255, 255)
C_GRAY   = (80, 80, 80)
C_GREEN  = (80, 200, 80)
C_YELLOW = (0, 220, 220)


# ── Embedding ─────────────────────────────────────────────────────────────────

def landmarks_to_embedding(landmarks):
    """
    Convert 21 MediaPipe NormalizedLandmarks to a 63-dim L2-normalised vector.

    Matches the index embedding: wrist-centered (subtract lm[0]), flattened,
    then L2-normalised for cosine similarity.
    Returns None if the hand is degenerate.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]           # center on wrist
    vec  = pts.flatten()    # (63,)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return None
    return vec / norm


# ── Retrieval ─────────────────────────────────────────────────────────────────

def cosine_search(query_emb, embeddings, k):
    sims  = embeddings @ query_emb
    top_k = np.argsort(sims)[::-1][:k]
    return top_k, sims[top_k]


# ── Point cloud renderer ──────────────────────────────────────────────────────

def render_cloud_2d(verts, joints, size=CLOUD_SIZE):
    """2D projection (x-y plane) of a point cloud. Returns BGR uint8 image."""
    img = np.zeros((size, size, 3), dtype=np.uint8)

    pts2d = np.vstack([verts[:, :2], joints[:, :2]])
    mn, mx = pts2d.min(axis=0), pts2d.max(axis=0)
    span   = (mx - mn).max()
    if span < 1e-6:
        return img

    scale  = (size * 0.82) / span
    center = (mn + mx) / 2.0

    def to_px(p):
        px = ((p[:, :2] - center) * scale + size / 2).astype(int)
        px[:, 1] = size - 1 - px[:, 1]   # flip y → upright
        return px

    v_px = to_px(verts)
    j_px = to_px(joints)

    for p in v_px:
        if 0 <= p[0] < size and 0 <= p[1] < size:
            cv2.circle(img, (p[0], p[1]), 2, C_BLUE, -1)

    for finger in SKEL_CONNECTIONS:
        for ki in range(len(finger) - 1):
            cv2.line(img, tuple(j_px[finger[ki]]), tuple(j_px[finger[ki+1]]),
                     C_RED, 1, cv2.LINE_AA)

    for p in j_px:
        if 0 <= p[0] < size and 0 <= p[1] < size:
            cv2.circle(img, (p[0], p[1]), 3, C_ORANGE, -1)

    return img


def prerender_clouds(gt_verts, gt_joints):
    N = len(gt_verts)
    print(f"Pre-rendering {N} cloud thumbnails...", end=" ", flush=True)
    t0 = time.time()
    clouds = [
        render_cloud_2d(gt_verts[i].reshape(-1, 3), gt_joints[i].reshape(21, 3))
        for i in range(N)
    ]
    print(f"done ({time.time()-t0:.1f}s)")
    return clouds


# ── Display ───────────────────────────────────────────────────────────────────

def _fit(img, w, h):
    ih, iw = img.shape[:2]
    scale  = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    canvas[oy:oy+nh, ox:ox+nw] = cv2.resize(img, (nw, nh))
    return canvas


def _label(img, text, pos=(6, 18), color=C_WHITE, scale=0.45, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,   thickness,   cv2.LINE_AA)


def build_display(webcam_bgr, db_images, cloud_thumbs,
                  top_k_indices, top_k_sims, k, no_hand=False, fps=0.0):
    """
    Layout:  [ Webcam 480×480 ] [ Retrieval panel: image + cloud per rank ]
    """
    RET_W   = THUMB_SIZE + CLOUD_SIZE + 10
    total_w = CAM_W + 2 + RET_W
    canvas  = np.zeros((PANEL_H, total_w, 3), dtype=np.uint8)

    # webcam
    cam = _fit(webcam_bgr, CAM_W, PANEL_H)
    _label(cam, f"Live  {fps:.0f} fps", pos=(6, 22), scale=0.5)
    if no_hand:
        _label(cam, "No hand detected", pos=(6, PANEL_H - 10), color=C_YELLOW, scale=0.5)
    canvas[:, :CAM_W] = cam
    canvas[:, CAM_W:CAM_W+2] = 40  # separator

    # retrieval panel
    ret_x = CAM_W + 2
    row_h = PANEL_H // 3

    for rank in range(k):
        ry = rank * row_h

        if top_k_indices is not None and rank < len(top_k_indices):
            db_idx = top_k_indices[rank]
            sim    = top_k_sims[rank]

            # dataset image thumbnail
            thumb     = _fit(cv2.cvtColor(db_images[db_idx], cv2.COLOR_RGB2BGR),
                             THUMB_SIZE, row_h - 20)
            th, tw    = thumb.shape[:2]
            ty        = ry + (row_h - 20 - th) // 2
            canvas[ty:ty+th, ret_x:ret_x+tw] = thumb
            _label(canvas, f"#{rank+1}  sim={sim:.3f}",
                   pos=(ret_x+2, ry+row_h-6), scale=0.42, color=C_GREEN)

            # cloud thumbnail
            cloud         = cloud_thumbs[db_idx]
            cth, ctw      = cloud.shape[:2]
            cy_raw        = ry + (row_h - cth) // 2
            cy            = max(0, cy_raw)
            cx            = ret_x + THUMB_SIZE + 10
            cloud_offset  = cy - cy_raw
            cy_end        = min(PANEL_H, cy + cth - cloud_offset)
            canvas[cy:cy_end, cx:cx+ctw] = cloud[cloud_offset:cloud_offset+(cy_end-cy)]

        else:
            _label(canvas, f"Rank {rank+1}: —",
                   pos=(ret_x+4, ry+row_h//2), scale=0.45, color=C_GRAY)

        if rank < k - 1:
            canvas[ry+row_h-1, ret_x:] = 30

    _label(canvas, "Retrieved poses", pos=(ret_x+4, 16), scale=0.5, color=C_WHITE)
    return canvas


# ── MediaPipe helpers ─────────────────────────────────────────────────────────

def draw_hand_landmarks(frame_bgr, landmarks, frame_h, frame_w):
    pts = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], (0, 220, 0), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame_bgr, p, 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, p, 4, (0, 150, 0),    1, cv2.LINE_AA)


def ensure_landmarker(path, url):
    if os.path.exists(path):
        return
    print(f"Downloading MediaPipe hand landmarker → {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = subprocess.run(["curl", "-fsSL", "-o", path, url])
    if r.returncode != 0:
        raise RuntimeError(f"Download failed. Place the file manually at: {path}")
    print("Download complete.")


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index",        default=DEFAULT_INDEX)
    p.add_argument("--k",            type=int, default=3, help="Top-k results (max 3)")
    p.add_argument("--update-every", type=int, default=3, help="Run retrieval every N frames")
    p.add_argument("--camera",       type=int, default=0)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    k    = max(1, min(args.k, 3))

    if not os.path.exists(args.index):
        print(f"Index not found: {args.index}")
        print("Run  python3 knn/build_index.py  first.")
        raise SystemExit(1)

    d          = np.load(args.index)
    db_images  = d["images"]      # (N, 224, 224, 3) uint8
    embeddings = d["embeddings"]  # (N, 63)
    gt_joints  = d["gt_joints"]   # (N, 63)
    gt_verts   = d["gt_verts"]    # (N, V*3)
    print(f"Index loaded: {len(db_images)} samples  emb_dim={embeddings.shape[1]}")

    cloud_thumbs = prerender_clouds(gt_verts, gt_joints)

    ensure_landmarker(LANDMARKER_PATH, LANDMARKER_URL)
    lm_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(lm_opts)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        raise SystemExit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nRunning — Q/ESC to quit, S to save frame, 1/2/3 to change k\n")

    frame_count   = 0
    fps           = 0.0
    t_fps         = time.time()
    top_k_indices = None
    top_k_sims    = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            fps   = 30 / (time.time() - t_fps)
            t_fps = time.time()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w      = frame.shape[:2]

        mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection = landmarker.detect(mp_img)
        no_hand   = not detection.hand_landmarks

        frame_display = frame.copy()
        if not no_hand:
            draw_hand_landmarks(frame_display, detection.hand_landmarks[0], h, w)

        if not no_hand and frame_count % args.update_every == 0:
            emb = landmarks_to_embedding(detection.hand_landmarks[0])
            if emb is not None:
                top_k_indices, top_k_sims = cosine_search(emb, embeddings, k)

        if no_hand:
            top_k_indices = None
            top_k_sims    = None

        display = build_display(
            webcam_bgr=frame_display,
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
        if key in (ord('q'), 27):
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
