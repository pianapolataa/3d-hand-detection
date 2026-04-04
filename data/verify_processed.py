import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

NPZ_PATH = './train_data_778_2560_random_background.npz'

def verify_saved_data():
    print(f"📂 Loading {NPZ_PATH}...")
    data = np.load(NPZ_PATH)
    
    images = data['x_train']
    verts_flat = data['y_train']
    
    indices = np.random.choice(len(images), 3, replace=False)

    for idx in indices:
        img = images[idx]
        
        # CHANGE 2: Reshape the flat vector (2334,) back to (778, 3) for plotting
        v = verts_flat[idx].reshape(778, 3)

        fig = plt.figure(figsize=(12, 5))
        
        # Subplot 1: RGB
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img)
        ax1.set_title(f"Sample {idx}: Masked Input")
        ax1.axis('off')

        # Subplot 2: 3D Point Cloud
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        # Plotting x, y, z columns
        ax2.scatter(v[:, 0], v[:, 1], v[:, 2], s=2, c='blue', alpha=0.6)
        
        # Plot the origin (the Wrist root we centered on)
        ax2.scatter(0, 0, 0, s=100, c='red', marker='X', label='Wrist')

        ax2.set_title("Normalized 3D Points (Reshaped)")
        # Fix view angle for better hand visualization
        ax2.view_init(elev=-90, azim=-90) 
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    verify_saved_data()