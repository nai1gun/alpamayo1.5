# SPDX-License-Identifier: Apache-2.0
"""Benchmark Alpamayo inference with resource sampling.

This script runs the same example scene as ``test_inference.py`` while recording
phase timings and process/GPU resource usage. It is intentionally self-contained
so different model checkpoints can be compared with the same harness.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import BitsAndBytesConfig

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.viz_utils import make_camera_grid

try:
    import psutil
except ImportError:  # pragma: no cover - fallback for minimal environments.
    psutil = None


DEFAULT_TORCH_NUM_THREADS = 4
DEFAULT_TORCH_NUM_INTEROP_THREADS = 2


@dataclass
class PhaseResult:
    name: str
    wall_s: float
    process_cpu_s: float
    process_cpu_percent_of_one_core: float
    torch_cuda_peak_allocated_mib: float | None
    torch_cuda_peak_reserved_mib: float | None


def log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def safe_float(value: str) -> float | None:
    value = value.strip()
    if not value or value.lower() in {"[not supported]", "n/a", "nan"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return value or "run"


def first_text(value: Any) -> str:
    while isinstance(value, (list, tuple)):
        if not value:
            return ""
        value = value[0]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        return first_text(value.tolist())
    return str(value)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_cpu_affinity(value: str) -> list[int]:
    cpus: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            cpus.extend(range(int(start_text), int(end_text) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def apply_cpu_polite_defaults(args: argparse.Namespace) -> None:
    """Apply a conservative CPU-side resource profile unless overridden."""
    if args.polite:
        args.cpu_polite = True

    if args.torch_num_threads is None:
        args.torch_num_threads = 2 if args.cpu_polite else DEFAULT_TORCH_NUM_THREADS
    if args.torch_num_interop_threads is None:
        args.torch_num_interop_threads = (
            1 if args.cpu_polite else DEFAULT_TORCH_NUM_INTEROP_THREADS
        )
    if args.process_priority is None:
        args.process_priority = "below-normal" if args.cpu_polite else "normal"
    if args.cpu_polite and args.cpu_max_memory == "40GiB":
        args.cpu_max_memory = "16GiB"
    if args.cpu_polite and args.sample_interval_s == 1.0:
        args.sample_interval_s = 10.0
    if args.cpu_polite and args.watchdog_max_rss_gib is None:
        args.watchdog_max_rss_gib = 24.0
    if args.watchdog_action is None:
        args.watchdog_action = "exit" if args.cpu_polite else "record"


def apply_runtime_limits(args: argparse.Namespace) -> dict[str, Any]:
    """Best-effort process controls for keeping the desktop responsive."""
    result: dict[str, Any] = {"requested": {}, "applied": {}, "warnings": []}

    thread_env = {
        "OMP_NUM_THREADS": str(args.torch_num_threads),
        "MKL_NUM_THREADS": str(args.torch_num_threads),
        "OPENBLAS_NUM_THREADS": str(args.torch_num_threads),
        "NUMEXPR_NUM_THREADS": str(args.torch_num_threads),
    }
    os.environ.update(thread_env)
    result["requested"].update(thread_env)
    result["requested"]["torch_num_threads"] = args.torch_num_threads
    result["requested"]["torch_num_interop_threads"] = args.torch_num_interop_threads

    torch.set_num_threads(args.torch_num_threads)
    result["applied"]["torch_num_threads"] = torch.get_num_threads()
    try:
        torch.set_num_interop_threads(args.torch_num_interop_threads)
        result["applied"]["torch_num_interop_threads"] = torch.get_num_interop_threads()
    except RuntimeError as exc:
        result["warnings"].append(f"Could not set torch interop threads: {exc}")
        result["applied"]["torch_num_interop_threads"] = torch.get_num_interop_threads()

    if psutil is None:
        result["warnings"].append("psutil is unavailable; priority and affinity not applied.")
        return result

    process = psutil.Process(os.getpid())
    result["requested"]["process_priority"] = args.process_priority
    try:
        if os.name == "nt":
            priority_map = {
                "idle": psutil.IDLE_PRIORITY_CLASS,
                "below-normal": psutil.BELOW_NORMAL_PRIORITY_CLASS,
                "normal": psutil.NORMAL_PRIORITY_CLASS,
            }
            process.nice(priority_map[args.process_priority])
        else:
            priority_map = {"idle": 19, "below-normal": 10, "normal": 0}
            process.nice(priority_map[args.process_priority])
        result["applied"]["process_priority"] = args.process_priority
    except (OSError, psutil.Error) as exc:
        result["warnings"].append(f"Could not set process priority: {exc}")

    if args.cpu_affinity:
        requested_cpus = parse_cpu_affinity(args.cpu_affinity)
        result["requested"]["cpu_affinity"] = requested_cpus
        try:
            allowed_cpus = set(process.cpu_affinity())
            cpus = [cpu for cpu in requested_cpus if cpu in allowed_cpus]
            if not cpus:
                raise ValueError(f"None of {requested_cpus} are available to this process.")
            process.cpu_affinity(cpus)
            result["applied"]["cpu_affinity"] = process.cpu_affinity()
        except (AttributeError, OSError, ValueError, psutil.Error) as exc:
            result["warnings"].append(f"Could not set CPU affinity: {exc}")

    return result


def run_resource_self_test(seconds: float) -> dict[str, Any]:
    """Run a small CPU-only workload to verify priority, affinity, and sampling."""
    deadline = time.perf_counter() + seconds
    iterations = 0
    x = torch.randn((256, 256), dtype=torch.float32)
    while time.perf_counter() < deadline:
        x = x @ x
        x = x / (x.abs().max() + 1e-6)
        iterations += 1
        if iterations % 4 == 0:
            time.sleep(0.02)
    return {
        "seconds_requested": seconds,
        "iterations": iterations,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


class ResourceMonitor:
    """Sample process, system, and GPU metrics in a background thread."""

    def __init__(
        self,
        interval_s: float,
        watchdog_max_rss_gib: float | None = None,
        watchdog_max_gpu_memory_gib: float | None = None,
        watchdog_max_system_ram_percent: float | None = None,
        watchdog_action: str = "record",
        watchdog_marker_path: Path | None = None,
    ) -> None:
        self.interval_s = interval_s
        self.watchdog_max_rss_gib = watchdog_max_rss_gib
        self.watchdog_max_gpu_memory_gib = watchdog_max_gpu_memory_gib
        self.watchdog_max_system_ram_percent = watchdog_max_system_ram_percent
        self.watchdog_action = watchdog_action
        self.watchdog_marker_path = watchdog_marker_path
        self.watchdog_violation: dict[str, Any] | None = None
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start = time.perf_counter()
        self._process = psutil.Process(os.getpid()) if psutil is not None else None
        self._start_io = self._io_counters()
        if self._process is not None:
            self._process.cpu_percent(None)
            psutil.cpu_percent(None)

    def start(self) -> None:
        self._start = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s * 2))
        self.sample()

    def _io_counters(self) -> Any | None:
        if self._process is None:
            return None
        try:
            return self._process.io_counters()
        except (AttributeError, OSError, psutil.Error):
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval_s)

    def sample(self) -> None:
        sample: dict[str, Any] = {"t_s": round(time.perf_counter() - self._start, 3)}

        if self._process is not None:
            try:
                mem = self._process.memory_info()
                sample["process_rss_mib"] = mem.rss / 1024**2
                sample["process_vms_mib"] = mem.vms / 1024**2
                sample["process_cpu_percent"] = self._process.cpu_percent(None)
                sample["process_threads"] = self._process.num_threads()
                vm = psutil.virtual_memory()
                sample["system_ram_used_percent"] = vm.percent
                sample["system_ram_available_mib"] = vm.available / 1024**2
                sample["system_cpu_percent"] = psutil.cpu_percent(None)
            except psutil.Error:
                pass

        if torch.cuda.is_available():
            try:
                sample["torch_cuda_allocated_mib"] = torch.cuda.memory_allocated() / 1024**2
                sample["torch_cuda_reserved_mib"] = torch.cuda.memory_reserved() / 1024**2
                sample["torch_cuda_max_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
                sample["torch_cuda_max_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
            except RuntimeError:
                pass

        sample.update(self._gpu_sample())
        self.samples.append(sample)
        self._check_watchdog(sample)

    def _check_watchdog(self, sample: dict[str, Any]) -> None:
        if self.watchdog_violation is not None:
            return

        violations: list[str] = []
        rss_mib = sample.get("process_rss_mib")
        gpu_mib = sample.get("gpu_memory_used_mib")
        system_ram_percent = sample.get("system_ram_used_percent")

        if (
            self.watchdog_max_rss_gib is not None
            and isinstance(rss_mib, (int, float))
            and rss_mib > self.watchdog_max_rss_gib * 1024
        ):
            violations.append(
                f"process RSS {rss_mib / 1024:.2f} GiB > {self.watchdog_max_rss_gib:.2f} GiB"
            )
        if (
            self.watchdog_max_gpu_memory_gib is not None
            and isinstance(gpu_mib, (int, float))
            and gpu_mib > self.watchdog_max_gpu_memory_gib * 1024
        ):
            violations.append(
                f"GPU memory {gpu_mib / 1024:.2f} GiB > "
                f"{self.watchdog_max_gpu_memory_gib:.2f} GiB"
            )
        if (
            self.watchdog_max_system_ram_percent is not None
            and isinstance(system_ram_percent, (int, float))
            and system_ram_percent > self.watchdog_max_system_ram_percent
        ):
            violations.append(
                f"system RAM {system_ram_percent:.1f}% > "
                f"{self.watchdog_max_system_ram_percent:.1f}%"
            )

        if not violations:
            return

        self.watchdog_violation = {
            "t_s": sample.get("t_s"),
            "violations": violations,
            "sample": sample,
            "action": self.watchdog_action,
        }
        message = "Watchdog triggered: " + "; ".join(violations)
        log(message)

        if self.watchdog_marker_path is not None:
            try:
                with self.watchdog_marker_path.open("w", encoding="utf-8") as f:
                    json.dump(self.watchdog_violation, f, indent=2)
            except OSError:
                pass

        if self.watchdog_action == "exit":
            log("Exiting immediately because watchdog-action=exit.")
            os._exit(88)

    def _gpu_sample(self) -> dict[str, Any]:
        query = (
            "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
        )
        cmd = [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        except (OSError, subprocess.SubprocessError):
            return {}
        if proc.returncode != 0 or not proc.stdout.strip():
            return {}

        first_line = proc.stdout.strip().splitlines()[0]
        parts = [part.strip() for part in first_line.split(",")]
        if len(parts) < 6:
            return {}
        return {
            "gpu_name": parts[0],
            "gpu_util_percent": safe_float(parts[1]),
            "gpu_memory_used_mib": safe_float(parts[2]),
            "gpu_memory_total_mib": safe_float(parts[3]),
            "gpu_temperature_c": safe_float(parts[4]),
            "gpu_power_w": safe_float(parts[5]),
        }

    def write_csv(self, path: Path) -> None:
        if not self.samples:
            return
        fieldnames = sorted({key for sample in self.samples for key in sample})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.samples)

    def summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {"samples": len(self.samples)}
        numeric_keys = [
            "process_rss_mib",
            "process_vms_mib",
            "process_cpu_percent",
            "system_ram_used_percent",
            "system_ram_available_mib",
            "system_cpu_percent",
            "torch_cuda_allocated_mib",
            "torch_cuda_reserved_mib",
            "torch_cuda_max_allocated_mib",
            "torch_cuda_max_reserved_mib",
            "gpu_util_percent",
            "gpu_memory_used_mib",
            "gpu_memory_total_mib",
            "gpu_temperature_c",
            "gpu_power_w",
        ]
        for key in numeric_keys:
            values = [
                sample[key]
                for sample in self.samples
                if isinstance(sample.get(key), (int, float)) and sample.get(key) is not None
            ]
            if not values:
                continue
            summary[f"{key}_avg"] = float(np.mean(values))
            summary[f"{key}_max"] = float(np.max(values))
            summary[f"{key}_min"] = float(np.min(values))

        end_io = self._io_counters()
        if self._start_io is not None and end_io is not None:
            summary["process_read_mib"] = (end_io.read_bytes - self._start_io.read_bytes) / 1024**2
            summary["process_write_mib"] = (
                end_io.write_bytes - self._start_io.write_bytes
            ) / 1024**2
        return summary


class BenchmarkTimer:
    def __init__(self) -> None:
        self.phases: list[PhaseResult] = []

    @contextmanager
    def phase(self, name: str):
        log(f"Starting phase: {name}")
        cuda_sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        try:
            yield
        finally:
            cuda_sync()
            wall_s = time.perf_counter() - start_wall
            cpu_s = time.process_time() - start_cpu
            peak_alloc = None
            peak_reserved = None
            if torch.cuda.is_available():
                peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
                peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
            self.phases.append(
                PhaseResult(
                    name=name,
                    wall_s=wall_s,
                    process_cpu_s=cpu_s,
                    process_cpu_percent_of_one_core=(cpu_s / wall_s * 100.0)
                    if wall_s > 0
                    else 0.0,
                    torch_cuda_peak_allocated_mib=peak_alloc,
                    torch_cuda_peak_reserved_mib=peak_reserved,
                )
            )
            log(f"Finished phase: {name} in {wall_s:.2f}s")

    def as_dicts(self) -> list[dict[str, Any]]:
        return [phase.__dict__ for phase in self.phases]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--clip-id", default="bd65ae5a-7c50-4d33-a953-bd382c108d04")
    parser.add_argument("--t0-us", type=int, default=12_000_000)
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--gpu-max-memory", default="7GiB")
    parser.add_argument("--cpu-max-memory", default="40GiB")
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--sample-interval-s", type=float, default=1.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument(
        "--cpu-polite",
        action="store_true",
        help=(
            "Apply CPU-side politeness: fewer Torch threads, below-normal priority, "
            "16GiB CPU offload target, slower resource sampling, and a RAM watchdog. "
            "This does not reduce GPU VRAM/utilization."
        ),
    )
    parser.add_argument(
        "--polite",
        action="store_true",
        help="Deprecated alias for --cpu-polite.",
    )
    parser.add_argument("--torch-num-threads", type=int, default=None)
    parser.add_argument("--torch-num-interop-threads", type=int, default=None)
    parser.add_argument(
        "--process-priority",
        choices=["idle", "below-normal", "normal"],
        default=None,
    )
    parser.add_argument(
        "--cpu-affinity",
        default=None,
        help="Comma/range list such as '0,1' or '0-3'. Optional; threads are capped separately.",
    )
    parser.add_argument("--watchdog-max-rss-gib", type=float, default=None)
    parser.add_argument("--watchdog-max-gpu-memory-gib", type=float, default=None)
    parser.add_argument("--watchdog-max-system-ram-percent", type=float, default=None)
    parser.add_argument("--watchdog-action", choices=["record", "exit"], default=None)
    parser.add_argument(
        "--resource-self-test-seconds",
        type=float,
        default=0.0,
        help="Run a small CPU-only workload to verify limits, then exit without loading Alpamayo.",
    )
    return parser


def save_visualization(
    output_path: Path,
    data: dict[str, Any],
    pred_xyz: torch.Tensor,
    cot: str,
    min_ade: float,
    clip_id: str,
    model_id: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    camera_grid = make_camera_grid(data["image_frames"], camera_indices=data["camera_indices"])
    axes[0].imshow(camera_grid)
    axes[0].set_title(f"Camera Views at t0\nClip: {clip_id[:30]}...", fontsize=10)
    axes[0].axis("off")

    ax = axes[1]
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_np = pred_xyz.detach().cpu().numpy()

    for i in range(pred_np.shape[2]):
        pred_xy = pred_np[0, 0, i, :, :2].T
        ax.plot(
            pred_xy[0],
            pred_xy[1],
            "b-",
            alpha=0.7 if pred_np.shape[2] == 1 else 0.3,
            linewidth=2,
            label="Predicted" if i == 0 else None,
        )

    ax.plot(gt_xy[0], gt_xy[1], "r-", linewidth=2, label="Ground Truth")
    ax.plot(0, 0, "ko", markersize=10, label="Ego Vehicle")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{model_id}\nReasoning: \"{cot[:140]}{'...' if len(cot) > 140 else ''}\"", fontsize=9)
    ax.legend(loc="best")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02,
        0.98,
        f"minADE: {min_ade:.3f}m",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()
    apply_cpu_polite_defaults(args)
    run_name = sanitize_name(args.run_name or args.model_id.replace("/", "__"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_output = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).resolve().parents[2] / "results" / "benchmarks"
    )
    run_dir = root_output / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    offload_folder = Path(args.offload_folder) if args.offload_folder else run_dir / "offload"
    offload_folder.mkdir(parents=True, exist_ok=True)
    runtime_limits = apply_runtime_limits(args)

    log(f"Run directory: {run_dir}")
    log(f"Model: {args.model_id}")
    log(f"Runtime limits: {json.dumps(runtime_limits, sort_keys=True)}")
    monitor = ResourceMonitor(
        interval_s=args.sample_interval_s,
        watchdog_max_rss_gib=args.watchdog_max_rss_gib,
        watchdog_max_gpu_memory_gib=args.watchdog_max_gpu_memory_gib,
        watchdog_max_system_ram_percent=args.watchdog_max_system_ram_percent,
        watchdog_action=args.watchdog_action,
        watchdog_marker_path=run_dir / "watchdog_violation.json",
    )
    timer = BenchmarkTimer()
    monitor.start()
    start_wall = time.perf_counter()
    pred_xyz: torch.Tensor | None = None
    min_ade: float | None = None
    cot = ""
    data: dict[str, Any] | None = None
    self_test_result: dict[str, Any] | None = None

    try:
        if args.resource_self_test_seconds > 0:
            with timer.phase("resource_self_test"):
                self_test_result = run_resource_self_test(args.resource_self_test_seconds)
            return

        with timer.phase("load_dataset"):
            data = load_physical_aiavdataset(args.clip_id, t0_us=args.t0_us)

        with timer.phase("prepare_prompt"):
            messages = helper.create_message(
                frames=data["image_frames"].flatten(0, 1),
                camera_indices=data["camera_indices"],
            )

        with timer.phase("load_model"):
            model_kwargs: dict[str, Any] = {
                "dtype": torch.bfloat16,
                "device_map": "auto",
                "max_memory": {0: args.gpu_max_memory, "cpu": args.cpu_max_memory},
                "offload_folder": str(offload_folder),
                "attn_implementation": args.attn_implementation,
                "local_files_only": args.local_files_only,
                "trust_remote_code": args.trust_remote_code,
            }
            if args.revision:
                model_kwargs["revision"] = args.revision
            if not args.no_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            model = Alpamayo1_5.from_pretrained(args.model_id, **model_kwargs)
            processor = helper.get_processor(model.tokenizer)

        with timer.phase("tokenize_inputs"):
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

        with timer.phase("inference"):
            torch.cuda.manual_seed_all(args.seed)
            device_type = str(model.device).split(":")[0]
            with torch.autocast(device_type, dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    num_traj_samples=args.num_traj_samples,
                    max_generation_length=args.max_generation_length,
                    return_extra=True,
                )
            cot = first_text(extra.get("cot", ""))

        with timer.phase("compute_metrics"):
            gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
            pred_xy = pred_xyz.detach().cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
            diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
            min_ade = float(diff.min())

        with timer.phase("save_outputs"):
            results_path = run_dir / "inference_results.npz"
            np.savez(
                results_path,
                model_id=args.model_id,
                clip_id=args.clip_id,
                t0_us=args.t0_us,
                cot=cot,
                pred_xyz=pred_xyz.detach().cpu().numpy(),
                gt_future_xyz=data["ego_future_xyz"].cpu().numpy(),
                image_frames=data["image_frames"].cpu().numpy(),
                camera_indices=data["camera_indices"].cpu().numpy(),
                min_ade=min_ade,
            )
            visual_path = run_dir / "visualization.png"
            save_visualization(visual_path, data, pred_xyz, cot, min_ade, args.clip_id, args.model_id)

    finally:
        monitor.stop()
        total_wall_s = time.perf_counter() - start_wall
        samples_path = run_dir / "samples.csv"
        monitor.write_csv(samples_path)
        metrics = {
            "created_at": datetime.now().isoformat(),
            "model_id": args.model_id,
            "run_name": run_name,
            "clip_id": args.clip_id,
            "t0_us": args.t0_us,
            "args": vars(args),
            "total_wall_s": total_wall_s,
            "runtime_limits": runtime_limits,
            "phases": timer.as_dicts(),
            "resource_summary": monitor.summary(),
            "watchdog_violation": monitor.watchdog_violation,
            "self_test_result": self_test_result,
            "cot": cot,
            "min_ade": min_ade,
            "output_dir": str(run_dir),
        }
        metrics_path = run_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        log(f"Samples saved to: {samples_path}")
        log(f"Metrics saved to: {metrics_path}")

    log("Benchmark complete")
    if self_test_result is not None:
        log(f"Self-test result: {self_test_result}")
    log(f"Chain-of-Causation: {cot}")
    if min_ade is not None:
        log(f"minADE: {min_ade:.3f}m")
    log(f"Visualization: {run_dir / 'visualization.png'}")


if __name__ == "__main__":
    main()
