# 3d-hand-detection

This project explores 3D hand detection from RGB images, including:

- dense 3D hand point cloud reconstruction from a single image
- camera-angle clustering
- hand-pose clustering
- knn hand pose retrieval from a live webcam using MediaPipe

## Installation

From the repo root:

```bash
python -m pip install -e .
```

This installs the dependencies used across the repo.

## Dataset Setup

Download FreiHAND Dataset v2 from:

`https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html`

After unzipping, place `FreiHAND_pub_v2` at the repo root:

```text
3d-hand-detection/
  FreiHAND_pub_v2/
```

Then preprocess the data:

```bash
python data/process_data.py
```

That produces the processed `.npz` files used by the training/eval/clustering scripts.

## Main Streamlit Demo

Launch the website with:

```bash
streamlit run app.py
```

The app currently includes:

- model prediction demo
- camera-angle clustering demo
- hand-pose clustering demo
- live knn hand-pose retrieval webcam demo

## Webcam Hand Pose Detection Demo

For the standalone OpenCV demo of our knn hand pose detection:

```bash
python demo/demo.py
```

We provide a predefined list of hand poses for the following poses, and the script automatically retrieves the closest of these poses to the camera input:

- `FIST`
- `OPEN_PALM`
- `PEACE`
- `THUMBS_UP`
- `POINT`

You can force recalibration with:

```bash
python demo/demo.py --collect
```

Controls:

- `Q` quit
- `R` recalibrate

## Model Evaluation

To run the original local eval script of our RGB -> point cloud detection model:

```bash
python scripts/eval.py
```
Which was trained using the script
```
python scripts/train.py
```

## Clustering Demos

### Camera-angle clustering

```bash
python clustering/angle_clustering_eval.py \
  --clusters-path clustering/train_camera_clusters_k8.npz \
  --npz-path data/train_data_600_verts.npz \
  --samples-per-cluster 5
```

Expected behavior:
hands in each cluster should generally face the same direction, even when the exact hand pose differs.

### Hand-pose clustering

```bash
python clustering/pose_clustering_eval.py \
  --clusters-path clustering/train_pose_clusters_k10.npz \
  --npz-path data/train_data_600_verts.npz \
  --samples-per-cluster 6
```

Expected behavior:
hands in each cluster should generally share a similar pose, even when the camera orientation differs.

## kNN Retrieval Scripts

Build the retrieval index:

```bash
python knn/build_index.py
```

Offline retrieval:

```bash
python knn/retrieve.py
```

Live retrieval:

```bash
python knn/live_demo.py
```

## Notes

- `scripts/checkpoints/model_600_verts_15_vectors.pth` is the RGB -> point cloud model we trained and used in plotting and clustering demos.
- `demo/hand_landmarker.task` is required for the MediaPipe webcam demos.
- `demo/reference_poses.npz` is used by the live hand-pose matcher after calibration.
- `data/retrieval_index_600_verts.npz` is used by the kNN retrieval demo.
