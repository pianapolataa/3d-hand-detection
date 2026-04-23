import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cluster centers from a saved CosineKMeans model.")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the saved model (.pkl)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("cluster_centers_plot.png"),
        help="Path to save the resulting plot image.",
    )
    return parser.parse_args()


def plot_cluster_centers(centers: np.ndarray, output_path: Path) -> None:
    """
    Plots the cluster centers (which are 6D: [palm_normal_x,y,z, middle_axis_x,y,z]).
    """
    num_clusters = centers.shape[0]
    fig = plt.figure(figsize=(4.0 * num_clusters, 4.0))

    for i in range(num_clusters):
        ax = fig.add_subplot(1, num_clusters, i + 1, projection="3d")
        
        # Center is [palm_normal (3), middle_axis (3)]
        palm_normal = centers[i, 0:3]
        middle_axis = centers[i, 3:6]
        
        # We also compute an approximate 'thumb' axis via cross product just for visualization
        # Note: in compute_hand_frame_components, palm_normal = cross(index, pinky)
        # We can just visualize the two main vectors from the wrist (0,0,0)
        wrist = np.array([0.0, 0.0, 0.0])
        
        # Plot origin
        ax.scatter([0], [0], [0], s=40, c="orange", marker="o", label="Wrist")
        
        # Plot the canonical vectors
        ax.quiver(
            0, 0, 0,
            palm_normal[0], palm_normal[1], palm_normal[2],
            color="crimson", linewidth=2.0, arrow_length_ratio=0.18, label="Palm Normal"
        )
        ax.quiver(
            0, 0, 0,
            middle_axis[0], middle_axis[1], middle_axis[2],
            color="royalblue", linewidth=2.0, arrow_length_ratio=0.18, label="Middle Axis"
        )
        
        # Set limits to [-1, 1] since they are unit vectors
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        ax.view_init(elev=20, azim=-60)
        ax.set_title(f"Cluster {i}", fontsize=12)
        
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"CosineKMeans Cluster Centers (K={num_clusters})", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cluster centers plot to {output_path}")


def main() -> None:
    args = parse_args()
    
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found at {args.model_path}")
        
    print(f"Loading model from {args.model_path}...")
    with open(args.model_path, "rb") as f:
        clusterer = pickle.load(f)
        
    centers = clusterer.centers_
    if centers is None:
        raise ValueError("The loaded model has not been fitted (centers_ is None).")
        
    print(f"Loaded CosineKMeans model with {centers.shape[0]} clusters.")
    plot_cluster_centers(centers, args.output_path)


if __name__ == "__main__":
    main()
