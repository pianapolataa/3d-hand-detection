# 3d-hand-detection
This project addresses the challenge of reconstructing an accurate 3D hand representation from an RGB image, transitioning from skeleton-based 21-point tracking to dense surface point cloud modeling.


## Download Dataset
Scroll to the bottom of this website and download FreiHAND Dataset v2 (3.7GB)

```https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html```

Unzip the file and move the folder FreiHAND_pub_v2 to the root directory, so your repo looks like 3d-hand-detection/FreiHAND_pub_v2

Then, go to the data directory ```cd data``` and run process_data.py (which will take a bit). 

## Running the basic RGB -> hand mesh model
Simply run 
```
cd scripts/
python eval.py
```
This script plots predictions for 20 random images in the evaluation dataset, unseen during training. You can drag the 3d plot around to view the points plotted in 3d!

## Running our clustering demos
We trained a model that clusters by camera angle / hand orientation, and one that clusters by hand pose.

For visualization of our clustering by hand orientation/camera angle model, simply run
```
python clustering/angle_clustering_eval.py \
  --clusters-path clustering/train_camera_clusters_k8.npz \
  --npz-path data/train_data_600_verts.npz \
  --samples-per-cluster 5
```
You should see that the hands in each cluster are facing the same direction, even though the hand positions are different.

For visualization of our clustering by hand pose model, run
```
python clustering/pose_clustering_eval.py \
  --clusters-path clustering/train_pose_clusters_k10.npz \
  --npz-path data/train_data_600_verts.npz \
  --samples-per-cluster 6
```
You should see that the hands in each cluster are moved to the same pose (ex pinch in cluster 4, even though the orientations are different).
