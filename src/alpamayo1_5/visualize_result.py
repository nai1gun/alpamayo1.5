# SPDX-License-Identifier: Apache-2.0
"""Visualize Alpamayo 1.5 inference results from saved data."""

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from alpamayo1_5.viz_utils import make_camera_grid


def main() -> None:
    """Load saved inference results and create visualization."""
    results_dir = Path(__file__).parent.parent.parent / "results"
    results_path = results_dir / "inference_results.npz"

    if not results_path.exists():
        print(f"Error: No results found at {results_path}")
        print("Run test_inference.py first to generate results.")
        return

    print(f"Loading results from: {results_path}")
    data = np.load(results_path, allow_pickle=True)

    clip_id = str(data["clip_id"])
    cot = str(data["cot"])
    pred_xyz = data["pred_xyz"]
    gt_future_xyz = data["gt_future_xyz"]
    image_frames = torch.from_numpy(data["image_frames"])
    camera_indices = torch.from_numpy(data["camera_indices"])
    min_ade = float(data["min_ade"])

    print(f"Clip: {clip_id}")
    print(f"Chain-of-Causation: {cot}")
    print(f"minADE: {min_ade:.3f}m")

    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Top: Camera grid
    camera_grid = make_camera_grid(image_frames, camera_indices)
    axes[0].imshow(camera_grid)
    axes[0].set_title(f"Camera Views at t0\nClip: {clip_id[:20]}...", fontsize=10)
    axes[0].axis("off")

    # Bottom: Trajectory plot
    ax = axes[1]
    gt_xy = gt_future_xyz[0, 0, :, :2].T

    for i in range(pred_xyz.shape[2]):
        pred_xy = pred_xyz[0, 0, i, :, :2].T
        ax.plot(pred_xy[0], pred_xy[1], "b-", alpha=0.7, linewidth=2, label="Predicted" if i == 0 else None)

    ax.plot(gt_xy[0], gt_xy[1], "r-", linewidth=2, label="Ground Truth")
    ax.plot(0, 0, "ko", markersize=10, label="Ego Vehicle")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Trajectory Prediction\nReasoning: \"{cot}\"", fontsize=10)
    ax.legend(loc="best")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    ax.text(0.02, 0.98, f"minADE: {min_ade:.3f}m", transform=ax.transAxes,
            fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat"))

    plt.tight_layout()
    output_path = results_dir / "alpamayo_result.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
