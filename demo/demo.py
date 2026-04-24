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


# ── Paths / MediaPipe model ──────────────────────────────────────────────────

REPO_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANDMARKER_PATH  = os.path.join(REPO_ROOT, "demo", "hand_landmarker.task")
REFERENCES_PATH  = os.path.join(REPO_ROOT, "demo", "reference_poses.npz")
LANDMARKER_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


# ── MediaPipe landmark indices (same convention as FreiHAND/MANO) ────────────

WRIST         = 0
INDEX_MCP     = 5
MIDDLE_MCP    = 9
PINKY_MCP     = 17
FINGERTIPS    = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky

HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (0,17),(13,17),(17,18),(18,19),(19,20),
]


# ── Predefined poses (drastically different on purpose) ──────────────────────

POSES = ["FIST", "OPEN_PALM", "PEACE", "THUMBS_UP", "POINT"]
POSE_HINTS = {
    "FIST":      "all fingers curled into a fist",
    "OPEN_PALM": "all five fingers fully extended",
    "PEACE":     "index + middle up, others curled",
    "THUMBS_UP": "thumb extended up, other fingers curled",
    "POINT":     "index finger extended, others curled",
}

MATCH_THRESHOLD   = 0.92   # cosine sim below this is shown as "no match"
COLLECT_SECONDS   = 6.0    # per-pose capture window
COUNTDOWN_SECONDS = 3.0


# ── Math: hand-frame change of basis ─────────────────────────────────────────

def _safe_normalize(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def hand_frame_axes(joints):
    """
    Given 21x3 landmarks, return orthonormal basis (v1, v2, v3) of the local
    hand frame: v1 = palm normal, v2 = wrist->middle_mcp projected onto the
    palm plane, v3 = v1 x v2.
    """
    wrist      = joints[WRIST]
    index_mcp  = joints[INDEX_MCP]  - wrist
    pinky_mcp  = joints[PINKY_MCP]  - wrist
    middle_mcp = joints[MIDDLE_MCP] - wrist

    v1 = _safe_normalize(np.cross(index_mcp, pinky_mcp))
    v2 = _safe_normalize(middle_mcp - np.dot(middle_mcp, v1) * v1)
    v3 = _safe_normalize(np.cross(v1, v2))
    return v1.astype(np.float32), v2.astype(np.float32), v3.astype(np.float32)


def to_hand_frame(joints):
    """Wrist-center joints, then project onto (v1, v2, v3). Returns (21, 3)."""
    centered = joints - joints[WRIST]
    v1, v2, v3 = hand_frame_axes(joints)
    R = np.stack([v1, v2, v3], axis=0)     # 3x3: rows are basis vectors
    return (centered @ R.T).astype(np.float32)


def fingertip_embedding(joints):
    """
    Rotation-invariant fingertip embedding: 15 dims of hand-frame tip positions
    plus 5 dims of tip-to-wrist distances (an explicit curled-vs-extended
    signal, so thumb vs index can't be washed out by the three fingers that
    match across similar poses). Returns a 20-dim L2-normalised vector, or
    None if the hand is degenerate.
    """
    hf    = to_hand_frame(joints)
    tips  = hf[FINGERTIPS]                       # (5, 3) — wrist-centered
    dists = np.linalg.norm(tips, axis=1)         # (5,)   — tip-to-wrist
    feat  = np.concatenate([tips.reshape(-1), dists])
    norm  = np.linalg.norm(feat)
    if norm < 1e-6:
        return None
    return (feat / norm).astype(np.float32)


def match_pose(query_emb, ref_names, ref_embs):
    """Return (best_name, best_sim) via cosine similarity."""
    sims = ref_embs @ query_emb
    best = int(np.argmax(sims))
    return ref_names[best], float(sims[best])


# ── MediaPipe ────────────────────────────────────────────────────────────────

def ensure_landmarker(path, url):
    if os.path.exists(path):
        return
    print(f"Downloading MediaPipe hand landmarker -> {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = subprocess.run(["curl", "-fsSL", "-o", path, url])
    if r.returncode != 0:
        raise RuntimeError(f"Download failed. Place the file manually at: {path}")


def make_landmarker():
    ensure_landmarker(LANDMARKER_PATH, LANDMARKER_URL)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def detect_joints(landmarker, frame_bgr):
    """Return (21, 3) float32 MediaPipe landmarks for the first hand, or None."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    det       = landmarker.detect(mp_img)
    if not det.hand_landmarks:
        return None
    return np.array([[lm.x, lm.y, lm.z] for lm in det.hand_landmarks[0]],
                    dtype=np.float32)


# ── Drawing ──────────────────────────────────────────────────────────────────

def draw_skeleton(frame, joints):
    h, w = frame.shape[:2]
    pts = [(int(j[0] * w), int(j[1] * h)) for j in joints]
    for a, b in HAND_EDGES:
        cv2.line(frame, pts[a], pts[b], (0, 220, 0), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 255, 255), -1, cv2.LINE_AA)


def put_text(img, text, org, scale=0.8, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


# ── Calibration ──────────────────────────────────────────────────────────────

def collect_references(landmarker, cap):
    """
    Walk the user through each pose: countdown, then a ~3s capture window
    where they slowly rotate/tilt the hand. Returns dict {pose_name: 15-dim}.
    """
    refs = {}
    print("\n── Calibration ──")
    print("For each pose, hold the shape and slowly rotate/tilt your hand")
    print(f"during the {COLLECT_SECONDS:.0f}s capture window.\n")

    for idx, pose in enumerate(POSES, start=1):
        print(f"  [{idx}/{len(POSES)}] {pose}  — {POSE_HINTS[pose]}")

        # Countdown phase
        t_end = time.time() + COUNTDOWN_SECONDS
        while time.time() < t_end:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            remain = int(np.ceil(t_end - time.time()))
            put_text(frame, f"{pose}: get ready ({remain})", (30, 60),
                     scale=1.0, color=(0, 220, 220))
            put_text(frame, POSE_HINTS[pose], (30, 110), scale=0.7)
            cv2.imshow("Hand Pose Demo", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                raise SystemExit("Calibration aborted.")

        # Capture phase
        embeds = []
        t_end = time.time() + COLLECT_SECONDS
        while time.time() < t_end:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            joints = detect_joints(landmarker, frame)
            if joints is not None:
                draw_skeleton(frame, joints)
                e = fingertip_embedding(joints)
                if e is not None:
                    embeds.append(e)

            remain = t_end - time.time()
            frac   = 1.0 - remain / COLLECT_SECONDS
            bar_w  = int(frac * (frame.shape[1] - 60))
            cv2.rectangle(frame, (30, frame.shape[0] - 40),
                          (30 + bar_w, frame.shape[0] - 20),
                          (80, 200, 80), -1)
            put_text(frame, f"Capturing {pose}  rotate slowly", (30, 60),
                     scale=1.0, color=(80, 200, 80))
            put_text(frame, f"samples: {len(embeds)}", (30, 110), scale=0.7)
            cv2.imshow("Hand Pose Demo", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                raise SystemExit("Calibration aborted.")

        if len(embeds) < 5:
            print(f"     only {len(embeds)} valid frames — retrying {pose}")
            return collect_references(landmarker, cap)

        avg = np.mean(np.stack(embeds, axis=0), axis=0)
        avg = avg / max(np.linalg.norm(avg), 1e-8)
        refs[pose] = avg.astype(np.float32)
        print(f"     captured {len(embeds)} frames, embedding norm {np.linalg.norm(avg):.3f}")

    return refs


def save_references(refs, path):
    names = np.array(list(refs.keys()))
    embs  = np.stack([refs[n] for n in names], axis=0).astype(np.float32)
    np.savez_compressed(path, names=names, embeddings=embs)
    print(f"Saved references -> {path}")


def load_references(path):
    d = np.load(path, allow_pickle=False)
    return list(d["names"]), d["embeddings"].astype(np.float32)


# ── Live demo ────────────────────────────────────────────────────────────────

def run_live(landmarker, cap, ref_names, ref_embs):
    print("\n── Live demo ──  Q/ESC to quit, R to recalibrate\n")
    recalibrate = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        joints = detect_joints(landmarker, frame)
        if joints is None:
            put_text(frame, "No hand detected", (30, 50),
                     scale=0.9, color=(0, 220, 220))
        else:
            draw_skeleton(frame, joints)
            emb = fingertip_embedding(joints)
            if emb is not None:
                name, sim = match_pose(emb, ref_names, ref_embs)
                if sim >= MATCH_THRESHOLD:
                    label, color = name, (80, 220, 80)
                else:
                    label, color = "(no confident match)", (80, 200, 220)
                put_text(frame, label, (30, 60), scale=1.4, color=color, thickness=3)
                put_text(frame, f"sim = {sim:.3f}", (30, 110),
                         scale=0.8, color=(255, 255, 255))

        put_text(frame, "Q quit   R recalibrate",
                 (30, frame.shape[0] - 20), scale=0.6, color=(180, 180, 180))
        cv2.imshow("Hand Pose Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            recalibrate = True
            break

    return recalibrate


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--collect", action="store_true",
                   help="Force reference pose calibration before the demo")
    p.add_argument("--camera", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    landmarker = make_landmarker()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    expected_dim = len(FINGERTIPS) * 3 + len(FINGERTIPS)   # 15 positions + 5 distances

    try:
        need_collect = args.collect or not os.path.exists(REFERENCES_PATH)
        while True:
            if not need_collect:
                ref_names, ref_embs = load_references(REFERENCES_PATH)
                if ref_embs.shape[1] != expected_dim:
                    print(f"Reference embedding dim ({ref_embs.shape[1]}) "
                          f"does not match current ({expected_dim}); recalibrating.")
                    need_collect = True

            if need_collect:
                refs = collect_references(landmarker, cap)
                save_references(refs, REFERENCES_PATH)
                ref_names, ref_embs = load_references(REFERENCES_PATH)
                need_collect = False

            print("Loaded references:", ", ".join(ref_names))

            recalibrate = run_live(landmarker, cap, ref_names, ref_embs)
            if not recalibrate:
                break
            need_collect = True
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("Done.")


if __name__ == "__main__":
    main()
