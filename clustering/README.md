# Camera-Angle Clustering

This folder contains a self-contained pipeline for clustering hand images by camera angle using the predicted 3D hand coordinate frame.

The pipeline:

1. Loads masked RGB images from `data/train_data_600_verts.npz`.
2. Runs the trained reconstruction model checkpoint.
3. Extracts the predicted 21 joints.
4. Builds a 6D hand-frame feature per image:
   - `v1 = normalized cross(wrist -> index_mcp, wrist -> pinky_mcp)`
   - `v2 = normalized (wrist -> middle_mcp)`
5. Clusters those 6D features with a scratch cosine k-means implementation.
6. Saves cluster assignments, preview image grids, and 3D coordinate-axis visualizations.

One small repo-specific note: the preprocessing script on disk is `data/process_data.py`.

## Run

From the repo root:

```bash
python clustering/run_camera_angle_clustering.py
```

Useful flags:

```bash
python clustering/run_camera_angle_clustering.py \
  --num-clusters 4 \
  --batch-size 64 \
  --num-verts 600 \
  --samples-per-cluster 6
```

Outputs are written to `clustering/outputs/`.

The processed FreiHAND dataset stays under `data/`, for example:

```bash
data/train_data_600_verts.npz
data/eval_data_600_verts.npz
```

The preprocessing step also preserves the original FreiHAND sample ids in:

```bash
sample_indices_train
sample_indices_val
sample_indices_test
```

So if `sample_indices_train[2] == 157`, then processed dataset row `2` came from FreiHAND image `00000157.jpg`.

## SLURM

Submit the full clustering job with:

```bash
sbatch clustering/run_camera_angle_clustering.slurm
```

The default SLURM sweep runs:

- `NUM_VERTS in {600, 70}`
- `k in {4, 5, 6, 7, 8}`

Common overrides:

```bash
sbatch --export=ALL,ENV_NAME=3dhand,MODEL_VERTS_LIST="600 70",K_VALUES="4 5 6 7 8",BATCH_SIZE=64 clustering/run_camera_angle_clustering.slurm
```

The SLURM launcher writes a timestamped sweep directory inside `clustering/outputs/`, and then one subdirectory per combo such as:

```bash
clustering/outputs/<RUN_ID>/verts600_k4/
clustering/outputs/<RUN_ID>/verts600_k5/
clustering/outputs/<RUN_ID>/verts70_k4/
...
```

Each combo directory includes:

- `train_coord_frames.npz`
- `train_clusters_k<k>.npz`
- `train_cluster_summary_k<k>.csv`
- `train_cluster_previews_k<k>.png`
- `train_cluster_axes_k<k>.png`

The `train_cluster_axes_k<k>.png` artifact plots the two coordinate-frame vectors used for clustering:

- red = palm normal
- blue = wrist -> middle MCP
