# 3d-hand-detection
This project addresses the challenge of reconstructing an accurate 3D hand representation from an RGB image, transitioning from skeleton-based 21-point tracking to dense surface point cloud modeling.


## Download Dataset
Scroll to the bottom of this website and download FreiHAND Dataset v2 (3.7GB)

```https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html```

Unzip the file and move the folder FreiHAND_pub_v2 to the root directory, so your repo looks like 3d-hand-detection/FreiHAND_pub_v2

Then, go to the data directory ```cd data``` and run process_data.py (which will take a bit). Then run verify_processed.py and make sure the point clouds plotted look correct.

Note: the xyz points are normalized (scaled) and the RGB images are masked. During pose matching we need to mask our input image as well.
