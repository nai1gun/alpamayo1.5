# SPDX-License-Identifier: Apache-2.0
"""Visualize Alpamayo 1.5 inference results with camera images and trajectory plot."""

import os
import numpy as np
import torch

# Limit CPU usage to avoid system lag during inference
NUM_THREADS = 4
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(2)
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)

from transformers import BitsAndBytesConfig
import matplotlib.pyplot as plt

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.viz_utils import make_camera_grid


def main() -> None:
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    print(f"Loading dataset for clip_id: {clip_id}...")
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    print("Dataset loaded.")

    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading model with 4-bit quantization...")
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "7GiB", "cpu": "40GiB"},  # Leave RAM for system
        offload_folder="./offload",
        attn_implementation="eager",
        quantization_config=bnb_config,
    )
    processor = helper.get_processor(model.tokenizer)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }

    model_inputs = helper.to_device(model_inputs, model.device)

    print("Running inference...")
    torch.cuda.manual_seed_all(42)
    with torch.autocast(str(model.device).split(":")[0], dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )

    cot = extra["cot"][0][0]
    print(f"\nChain-of-Causation: {cot}")

    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Top: Camera grid
    camera_grid = make_camera_grid(data["image_frames"], data["camera_indices"])
    axes[0].imshow(camera_grid)
    axes[0].set_title(f"Camera Views at t0\nClip: {clip_id[:20]}...", fontsize=10)
    axes[0].axis("off")

    # Bottom: Trajectory plot
    ax = axes[1]
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()

    for i in range(pred_xyz.shape[2]):
        pred_xy = pred_xyz.cpu()[0, 0, i, :, :2].T.numpy()
        ax.plot(pred_xy[0], pred_xy[1], "b-", alpha=0.7, linewidth=2, label="Predicted" if i == 0 else None)

    ax.plot(gt_xy[0], gt_xy[1], "r-", linewidth=2, label="Ground Truth")
    ax.plot(0, 0, "ko", markersize=10, label="Ego Vehicle")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Trajectory Prediction\nReasoning: \"{cot}\"", fontsize=10)
    ax.legend(loc="best")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    # Compute minADE
    pred_xy_all = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy_all - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    ax.text(0.02, 0.98, f"minADE: {min_ade:.3f}m", transform=ax.transAxes,
            fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat"))

    plt.tight_layout()
    output_path = "alpamayo_result.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
