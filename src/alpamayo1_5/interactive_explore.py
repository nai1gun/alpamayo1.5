# SPDX-License-Identifier: Apache-2.0
"""Interactive exploration script for Alpamayo 1.5.

Workflow:
1. Preview scene (camera frames) - no model needed
2. Load model (once)
3. Run trajectory inference with optional navigation
4. Ask VQA questions interactively
5. Visualize results

Usage:
    python src/alpamayo1_5/interactive_explore.py
"""

import os
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving to files
import matplotlib.pyplot as plt

# Limit CPU usage
NUM_THREADS = 4
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(2)
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)

from transformers import BitsAndBytesConfig

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.viz_utils import make_camera_grid


class AlpamayoExplorer:
    """Interactive explorer for Alpamayo 1.5 scenes."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.data = None
        self.clip_id = None
        self.t0_us = None
        self.last_trajectory_result = None

    def load_scene(self, clip_id: str, t0_us: int | None = None):
        """Load a scene from the PhysicalAI-AV dataset.

        Args:
            clip_id: The clip ID to load
            t0_us: Optional timestamp in microseconds (default: start of clip)
        """
        print(f"Loading scene: {clip_id}")
        if t0_us:
            print(f"  Timestamp: {t0_us / 1_000_000:.2f}s")

        self.data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
        self.clip_id = clip_id
        self.t0_us = t0_us
        print("Scene loaded successfully.")
        return self

    def show_scene(self, figsize=(14, 8), output_path: str | None = None):
        """Display the camera views for the current scene.

        Args:
            figsize: Figure size tuple
            output_path: Path to save the image. If None, saves to results/scene_preview.png
        """
        if self.data is None:
            print("Error: No scene loaded. Call load_scene() first.")
            return

        camera_grid = make_camera_grid(
            self.data["image_frames"],
            camera_indices=self.data["camera_indices"]
        )

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(camera_grid)
        ax.set_title(f"Scene: {self.clip_id}\nTimestamp: {(self.t0_us or 0) / 1_000_000:.2f}s", fontsize=12)
        ax.axis("off")
        plt.tight_layout()

        if output_path is None:
            results_dir = Path(__file__).parent.parent.parent / "results"
            results_dir.mkdir(exist_ok=True)
            output_path = results_dir / "scene_preview.png"

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Scene preview saved to: {output_path}")

    def show_video(self, num_frames: int = 4, frame_interval_s: float = 1.0):
        """Create a video GIF showing the scene over time.

        This loads frames independently for visualization only - does NOT affect
        the trajectory inference data (which uses time_step=0.1s).

        Args:
            num_frames: Number of frames to show (default: 4)
            frame_interval_s: Seconds between frames for visualization (default: 1.0)
        """
        if self.clip_id is None or self.t0_us is None:
            print("Error: No scene loaded. Call load_scene() first.")
            return

        from PIL import Image, ImageDraw
        import physical_ai_av

        print(f"Creating video preview ({num_frames} frames, {frame_interval_s}s apart)...")

        avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
        camera_features = [
            avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        ]

        # Camera layout
        layout = {
            0: (1, 0),  # cross_left
            1: (1, 1),  # front_wide
            2: (1, 2),  # cross_right
            6: (0, 1),  # front_tele
        }
        camera_name_to_index = {
            "camera_cross_left_120fov": 0,
            "camera_front_wide_120fov": 1,
            "camera_cross_right_120fov": 2,
            "camera_front_tele_30fov": 6,
        }

        # Calculate timestamps for video frames
        interval_us = int(frame_interval_s * 1_000_000)
        timestamps = np.array([
            self.t0_us - (num_frames - 1 - i) * interval_us
            for i in range(num_frames)
        ], dtype=np.int64)

        # Load frames from each camera
        all_frames = []
        cam_indices = []
        for cam_feature in camera_features:
            camera = avdi.get_clip_feature(self.clip_id, cam_feature, maybe_stream=True)
            frames, _ = camera.decode_images_from_timestamps(timestamps)
            all_frames.append(frames)  # (num_frames, H, W, 3)
            cam_name = cam_feature.split("/")[-1].lower()
            cam_indices.append(camera_name_to_index.get(cam_name, 0))

        H, W = all_frames[0].shape[1], all_frames[0].shape[2]
        scale = 0.2
        new_h, new_w = int(H * scale), int(W * scale)
        ncols, nrows = 3, 2

        gif_frames = []
        for t in range(num_frames):
            grid = np.zeros((nrows * new_h, ncols * new_w, 3), dtype=np.uint8)
            for cam_idx, cam_id in enumerate(cam_indices):
                if cam_id in layout:
                    r, c = layout[cam_id]
                    img = Image.fromarray(all_frames[cam_idx][t]).resize((new_w, new_h), Image.LANCZOS)
                    grid[r*new_h:(r+1)*new_h, c*new_w:(c+1)*new_w] = np.array(img)

            pil_grid = Image.fromarray(grid)
            draw = ImageDraw.Draw(pil_grid)
            time_offset = -(num_frames - 1 - t) * frame_interval_s
            draw.text((10, 10), f"t = {time_offset:.1f}s", fill=(255, 255, 0))
            gif_frames.append(pil_grid)

        # Save GIF and individual frames
        results_dir = Path(__file__).parent.parent.parent / "results"
        results_dir.mkdir(exist_ok=True)

        gif_path = results_dir / "scene_video.gif"
        gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                          duration=int(frame_interval_s * 800), loop=0)
        print(f"Video GIF saved to: {gif_path}")

        for i, frame in enumerate(gif_frames):
            frame.save(results_dir / f"frame_{i}.png")
        print(f"Individual frames saved to results/frame_0.png - frame_{num_frames-1}.png")

    def load_model(self):
        """Load the Alpamayo 1.5 model with 4-bit quantization for low-VRAM GPUs."""
        if self.model is not None:
            print("Model already loaded.")
            return self

        print("Loading Alpamayo 1.5 model with 4-bit quantization...")
        print("  This will take a few minutes on first run (downloading ~22GB)...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = Alpamayo1_5.from_pretrained(
            "nvidia/Alpamayo-1.5-10B",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "7GiB", "cpu": "40GiB"},
            offload_folder="./offload",
            attn_implementation="eager",
            quantization_config=bnb_config,
        )
        self.processor = helper.get_processor(self.model.tokenizer)
        print("Model loaded successfully.")
        return self

    def predict_trajectory(self, nav_text: str | None = None, num_samples: int = 1, show: bool = True):
        """Run trajectory prediction on the current scene.

        Args:
            nav_text: Optional navigation instruction (e.g., "Turn right in 30m")
            num_samples: Number of trajectory samples to generate
            show: Whether to display the visualization

        Returns:
            dict with keys: pred_xyz, cot (chain-of-causation), min_ade
        """
        if self.model is None:
            print("Error: Model not loaded. Call load_model() first.")
            return None
        if self.data is None:
            print("Error: No scene loaded. Call load_scene() first.")
            return None

        print(f"Running trajectory prediction...")
        if nav_text:
            print(f"  Navigation instruction: {nav_text}")

        # Create message with optional navigation
        messages = helper.create_message(
            frames=self.data["image_frames"].flatten(0, 1),
            camera_indices=self.data["camera_indices"],
            nav_text=nav_text,
        )

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": self.data["ego_history_xyz"],
            "ego_history_rot": self.data["ego_history_rot"],
        }
        model_inputs = helper.to_device(model_inputs, self.model.device)

        torch.cuda.manual_seed_all(42)
        with torch.autocast(str(self.model.device).split(":")[0], dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=num_samples,
                max_generation_length=256,
                return_extra=True,
            )

        cot = extra["cot"][0][0] if extra["cot"][0] else ""
        print(f"\nChain-of-Causation:\n  {cot}")

        # Calculate minADE if ground truth available
        gt_xy = self.data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
        pred_xy_all = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
        diff = np.linalg.norm(pred_xy_all - gt_xy[None, ...], axis=1).mean(-1)
        min_ade = diff.min()
        print(f"minADE: {min_ade:.3f}m")

        self.last_trajectory_result = {
            "pred_xyz": pred_xyz,
            "cot": cot,
            "min_ade": min_ade,
            "nav_text": nav_text,
        }

        if show:
            self._show_trajectory_result()

        return self.last_trajectory_result

    def _show_trajectory_result(self, output_path: str | None = None):
        """Visualize the last trajectory prediction result."""
        if self.last_trajectory_result is None:
            return

        pred_xyz = self.last_trajectory_result["pred_xyz"]
        cot = self.last_trajectory_result["cot"]
        min_ade = self.last_trajectory_result["min_ade"]
        nav_text = self.last_trajectory_result["nav_text"]

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Top: Camera grid
        camera_grid = make_camera_grid(
            self.data["image_frames"],
            camera_indices=self.data["camera_indices"]
        )
        axes[0].imshow(camera_grid)
        title = f"Scene: {self.clip_id[:30]}..."
        if nav_text:
            title += f"\nNav: {nav_text}"
        axes[0].set_title(title, fontsize=10)
        axes[0].axis("off")

        # Bottom: Trajectory plot
        ax = axes[1]
        gt_xy = self.data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()

        for i in range(pred_xyz.shape[2]):
            pred_xy = pred_xyz.cpu()[0, 0, i, :, :2].T.numpy()
            alpha = 0.7 if pred_xyz.shape[2] == 1 else 0.3
            ax.plot(pred_xy[0], pred_xy[1], "b-", alpha=alpha, linewidth=2,
                    label="Predicted" if i == 0 else None)

        ax.plot(gt_xy[0], gt_xy[1], "r-", linewidth=2, label="Ground Truth")
        ax.plot(0, 0, "ko", markersize=10, label="Ego Vehicle")

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"Reasoning: \"{cot[:100]}{'...' if len(cot) > 100 else ''}\"", fontsize=10)
        ax.legend(loc="best")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, f"minADE: {min_ade:.3f}m", transform=ax.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat"))

        plt.tight_layout()

        if output_path is None:
            results_dir = Path(__file__).parent.parent.parent / "results"
            results_dir.mkdir(exist_ok=True)
            output_path = results_dir / "trajectory_result.png"

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Trajectory visualization saved to: {output_path}")

    def ask(self, question: str):
        """Ask a visual question about the current scene.

        Args:
            question: The question to ask (e.g., "Describe the scene.",
                     "What obstacles are ahead?", "Is it safe to turn left?")

        Returns:
            The model's answer as a string
        """
        if self.model is None:
            print("Error: Model not loaded. Call load_model() first.")
            return None
        if self.data is None:
            print("Error: No scene loaded. Call load_scene() first.")
            return None

        print(f"Question: {question}")

        messages = helper.create_vqa_message(
            self.data["image_frames"].flatten(0, 1),
            question=question,
            camera_indices=self.data["camera_indices"],
        )

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {"tokenized_data": inputs}
        model_inputs = helper.to_device(model_inputs, self.model.device)

        torch.cuda.manual_seed_all(42)
        with torch.autocast(str(self.model.device).split(":")[0], dtype=torch.bfloat16):
            extra = self.model.generate_text(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_samples=1,
                max_generation_length=256,
            )

        answer = extra["answer"][0]
        print(f"\nAnswer: {answer}\n")
        return answer


def interactive_session():
    """Run an interactive exploration session."""
    explorer = AlpamayoExplorer()

    # Example scenes to try
    example_scenes = [
        ("bd65ae5a-7c50-4d33-a953-bd382c108d04", 12_000_000, "Yield to truck crossing intersection"),
        ("ea7bbd31-b7a5-4972-8dbd-7089e6b53de4", 4_000_000, "Intersection in Sweden - turn decision"),
        ("c9c045a3-ebe9-4569-9ce3-a44068cf2e3b", 2_000_000, "Right turn scenario"),
    ]

    print("=" * 60)
    print("Alpamayo 1.5 Interactive Explorer")
    print("=" * 60)
    print("\nExample scenes:")
    for i, (clip_id, t0, desc) in enumerate(example_scenes):
        print(f"  {i + 1}. {desc}")
        print(f"     clip_id='{clip_id}', t0_us={t0}")
    print()

    print("Quick start commands:")
    print("  explorer.load_scene(clip_id, t0_us)  # Load a scene")
    print("  explorer.show_scene()                 # Show camera views")
    print("  explorer.load_model()                 # Load model (once)")
    print("  explorer.predict_trajectory()         # Run trajectory prediction")
    print("  explorer.predict_trajectory(nav_text='Turn right in 30m')")
    print("  explorer.ask('Describe the scene.')   # Ask a question")
    print("  explorer.ask('What obstacles are ahead?')")
    print("=" * 60)

    return explorer


if __name__ == "__main__":
    # Start interactive session
    explorer = interactive_session()

    # Demo: Load first example scene and show it
    print("\nLoading example scene...")
    explorer.load_scene("bd65ae5a-7c50-4d33-a953-bd382c108d04", t0_us=12_000_000)
    explorer.show_scene()

    print("\nScene preview complete! Open the image file to see the driving scenario.")
    print("To continue, use the 'explorer' object:")
    print("  explorer.load_model()                 # Load model (takes a few minutes)")
    print("  explorer.predict_trajectory()         # Run trajectory prediction")
    print("  explorer.ask('Describe the scene.')   # Ask a VQA question")
