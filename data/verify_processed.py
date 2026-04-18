import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


NPZ_PATH = 'eval_data_70_verts.npz'


def verify_saved_data():
   print(f"📂 Loading {NPZ_PATH}...")
   data = np.load(NPZ_PATH)
  
   # Check what keys are actually in there
   print(f"Keys found: {list(data.keys())}")
  
   images = data['x_train']
   joints_flat = data['y_train_joints']
   verts_flat = data['y_train_verts']
  
   # Dynamically determine the number of vertices
   # Total values / 3 (for x, y, z)
   num_verts = verts_flat.shape[1] // 3
   print(f"📊 Detected {num_verts} vertices per sample.")


   indices = np.random.choice(len(images), 3, replace=False)


   for idx in indices:
       img = images[idx]
      
       # Reshape using the detected count
       v = verts_flat[idx].reshape(num_verts, 3)
       j = joints_flat[idx].reshape(21, 3)


       fig = plt.figure(figsize=(15, 6))
      
       # Subplot 1: RGB Input (Black Background)
       ax1 = fig.add_subplot(1, 2, 1)
       ax1.imshow(img)
       ax1.set_title(f"Sample {idx}: Model Input")
       ax1.axis('off')


       # Subplot 2: 3D Point Cloud + Skeleton
       ax2 = fig.add_subplot(1, 2, 2, projection='3d')
      
       # 1. Plot the mesh points (Blue cloud)
       ax2.scatter(v[:, 0], v[:, 1], v[:, 2], s=5, c='blue', alpha=0.5, label='Mesh Points')
      
       # 2. Plot the joints (Red dots)
       ax2.scatter(j[:, 0], j[:, 1], j[:, 2], s=40, c='red', marker='o', label='Joints')
      
       # 3. Highlight the Wrist (Origin)
       ax2.scatter(0, 0, 0, s=100, c='black', marker='X', label='Wrist (Anchor)')


       # Connection mapping to draw the skeleton lines
       SKEL_CONNECTIONS = [
           [0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12], [0,13,14,15,16], [0,17,18,19,20]
       ]
       for finger in SKEL_CONNECTIONS:
           ax2.plot(j[finger, 0], j[finger, 1], j[finger, 2], color='red', linewidth=1.5, alpha=0.7)


       ax2.set_title(f"3D Structure ({num_verts} pts)")
       ax2.view_init(elev=-90, azim=-90)
       ax2.legend()
      
      
       plt.tight_layout()
       plt.show()


if __name__ == "__main__":
   verify_saved_data()
