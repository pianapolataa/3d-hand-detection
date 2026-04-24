# 3d-hand-detection
This project addresses the challenge of reconstructing an accurate 3D hand representation from an RGB image, transitioning from skeleton-based 21-point tracking to dense surface point cloud modeling.


## Download Dataset
Scroll to the bottom of this website and download FreiHAND Dataset v2 (3.7GB)

```https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html```

Unzip the file and move the folder FreiHAND_pub_v2 to the root directory, so your repo looks like 3d-hand-detection/FreiHAND_pub_v2

Then, go to the data directory ```cd data``` and run process_data.py (which will take a bit). Then run verify_processed.py and make sure the point clouds plotted look correct.

Note: the xyz points are normalized (scaled) and the RGB images are masked. During pose matching we need to mask our input image as well.

## Running the basic RGB -> hand mesh model
Simply run 
```
cd scripts/
python eval.py
```
This script plots predictions for 20 random images in the training dataset. You can drag the 3d plot around to view the points plotted in 3d!

## Running our clustering demos
We trained a model that clusters by camera angle / hand orientation, and one that clusters by hand pose.

For visualization of the accuracy of our hand orientation clustering model, simply run
```
python clustering/angle_clustering_eval.py \
  --clusters-path clustering/train_camera_clusters_k8.npz \
  --npz-path data/train_data_600_verts.npz \
  --samples-per-cluster 5
```

For visualization of the accuracy of our hand pose clustering model, run
