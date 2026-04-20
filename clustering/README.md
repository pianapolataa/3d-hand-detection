# Camera-Angle Clustering

This pipeline groups hand images by viewpoint using the model-predicted 3D hand pose.

## What The Pipeline Does

1. Loads a processed split of masked RGB hand images.
2. Runs the trained 3D hand reconstruction network in inference mode.
3. Extracts predicted joints for each image.
4. Builds a 6D coordinate-frame feature per sample:
   - first 3 values: normalized palm normal computed from wrist-to-index-MCP and wrist-to-pinky-MCP
   - next 3 values: normalized wrist-to-middle-MCP direction
5. Clusters these 6D vectors using cosine (spherical) k-means.
6. Ranks samples within each cluster by cosine similarity to the cluster center.
7. Writes machine-readable outputs plus visual diagnostics.

## How The Core Logic Works

### Inference and feature extraction

- Images are normalized with ImageNet statistics before model inference.
- The model predicts joints, auxiliary vectors, and mesh vertices; clustering uses the predicted joints.
- Coordinate-frame vectors are L2-normalized, then concatenated into one 6D feature vector per image.

### Clustering behavior

- Features are normalized row-wise before clustering.
- Cluster centers are initialized with a distance-aware probabilistic seeding strategy.
- Empty clusters are re-seeded from random samples.
- Multiple initializations are tried; the run with the highest cosine objective is kept.

### Ranking and diagnostics

- For each cluster, samples are sorted by similarity to the cluster center.
- Preview figures show representative examples and associated 3D axes.
- A dedicated axes visualization shows only joints plus the two frame directions:
  - red: palm normal
  - blue: wrist-to-middle-MCP direction

## Outputs You Get

Each run creates:

- coordinate-frame dataset with sample index mapping back to original data ids
- cluster assignments and centers
- cluster size summary (CSV)
- per-cluster ranked membership table with similarity scores (CSV)
- preview montage of top representatives per cluster
- 3D axes-only visualization for viewpoint sanity checking

## Ways To Run

Run from repository root.

### 1) Single run (local)

```bash
python clustering/run_camera_angle_clustering.py
```

### 2) Single run with custom settings

```bash
python clustering/run_camera_angle_clustering.py \
  --num-clusters 4 \
  --batch-size 64 \
  --num-verts 600 \
  --max-iters 100 \
  --n-init 5 \
  --samples-per-cluster 6
```

### 3) Choose split and explicit paths

```bash
python clustering/run_camera_angle_clustering.py \
  --split train \
  --npz-path data/train_data_600_verts.npz \
  --checkpoint-path scripts/checkpoints/model_600_verts_15_vectors.pth \
  --output-dir clustering/outputs/manual_run
```

## Practical Notes

- The number of vertices and vectors must match both dataset and checkpoint.
- If not specified, model dimensions are inferred directly from checkpoint weights.
- Device selection defaults to CUDA when available, otherwise CPU.
