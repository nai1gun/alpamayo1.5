# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end example script for the inference pipeline.

Loads a dataset, runs inference, computes the minADE, and saves results for visualization.
"""

import os
from pathlib import Path
import numpy as np
import torch

# Limit CPU usage to avoid system lag during inference
# Adjust these values based on your CPU (e.g., half of your cores)
NUM_THREADS = 4  # Limit PyTorch CPU threads
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(2)
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)

from transformers import BitsAndBytesConfig

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5


def main() -> None:
    """Run inference on an example clip and report minADE."""
    clip_id = "bd65ae5a-7c50-4d33-a953-bd382c108d04"
    t0_us = 12_000_000  # "Yield to truck crossing intersection" scenario
    print(f"Loading dataset for clip_id: {clip_id}...")
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    print("Dataset loaded.")
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

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

    print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    print("minADE:", min_ade, "meters")
    if min_ade >= 1.0:
        print(f"WARNING: minADE ({min_ade:.2f}m) is above 1.0m. Model sampling can be stochastic.")

    # Save results for visualization
    results_dir = Path(__file__).parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "inference_results.npz"

    np.savez(
        results_path,
        clip_id=clip_id,
        t0_us=t0_us,
        cot=extra["cot"][0][0],
        pred_xyz=pred_xyz.cpu().numpy(),
        gt_future_xyz=data["ego_future_xyz"].cpu().numpy(),
        image_frames=data["image_frames"].numpy(),
        camera_indices=data["camera_indices"].numpy(),
        min_ade=min_ade,
    )
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
