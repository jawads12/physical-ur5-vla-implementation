"""Run a LeRobot policy on saved observations only; never imports or publishes ROS commands."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .action_semantics import (
    ABSOLUTE_ACTION_SEMANTICS, DELTA_ACTION_SEMANTICS, delta_to_absolute,
)


def qualification_indices(dataset, samples_per_episode: int) -> list[int]:
    """Uniformly cover every episode, including both boundaries."""
    if samples_per_episode < 1:
        raise ValueError("samples_per_episode must be positive")
    result = []
    for episode in range(dataset.meta.total_episodes):
        item = dataset.meta.episodes[episode]
        start = int(item["dataset_from_index"])
        stop = int(item["dataset_to_index"])
        result.extend(np.linspace(start, stop - 1, min(samples_per_episode, stop - start), dtype=int).tolist())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="JSON report; default: OUTPUT_CSV.report.json")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument(
        "--samples-per-episode", type=int,
        help="uniform samples from every episode; overrides --max-frames",
    )
    parser.add_argument(
        "--max-target-delta-rad", type=float, default=0.012566,
        help="fail if any predicted arm target differs from the sampled state by more than this",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--inference-seed", type=int,
        help="reset policy sampling to this seed before every observation",
    )
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    args = parser.parse_args()
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root, video_backend=args.video_backend)
    conversion_path = args.dataset_root / "meta" / "ur5_vla_conversion.json"
    if not conversion_path.is_file():
        raise SystemExit(f"missing action-semantics metadata: {conversion_path}")
    action_semantics = json.loads(conversion_path.read_text()).get("action_semantics")
    if action_semantics not in {ABSOLUTE_ACTION_SEMANTICS, DELTA_ACTION_SEMANTICS}:
        raise SystemExit(f"unsupported action semantics: {action_semantics!r}")
    config = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    config.device = args.device
    policy = get_policy_class(config.type).from_pretrained(args.checkpoint, config=config, local_files_only=True).to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=args.checkpoint,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    rows, predictions, states, demonstrations, latencies = [], [], [], [], []
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        indices = (
            qualification_indices(dataset, args.samples_per_episode)
            if args.samples_per_episode is not None
            else np.linspace(0, len(dataset) - 1, min(args.max_frames, len(dataset)), dtype=int).tolist()
        )
        for index in indices:
            sample = dataset[int(index)]
            batch = {}
            for key, value in sample.items():
                if isinstance(value, torch.Tensor): batch[key] = value.unsqueeze(0).to(args.device)
            batch["task"] = [dataset.meta.tasks.index[int(sample["task_index"])]]
            processed = preprocessor(batch)
            if args.inference_seed is not None:
                torch.manual_seed(args.inference_seed)
                if args.device.startswith("cuda"):
                    torch.cuda.manual_seed_all(args.inference_seed)
            if args.device.startswith("cuda"): torch.cuda.synchronize()
            started = time.perf_counter()
            predicted = policy.predict_action_chunk(processed)
            predicted = postprocessor(predicted)
            if args.device.startswith("cuda"): torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)
            values = predicted.detach().cpu().numpy()
            if values.ndim == 2: values = values[None, ...]
            if values.ndim != 3 or values.shape[-1] != 7 or not np.isfinite(values).all():
                raise ValueError(f"invalid policy output shape/values: {values.shape}")
            raw_first = values[0, 0]
            demonstrated = np.asarray(sample["action"])
            state = np.asarray(sample["observation.state"])
            if action_semantics == DELTA_ACTION_SEMANTICS:
                # Reject before integration. This is deliberately not saturation/clipping.
                first = delta_to_absolute(state, raw_first, args.max_target_delta_rad)
                demonstrated_absolute = delta_to_absolute(state, demonstrated, args.max_target_delta_rad)
            else:
                first = raw_first
                demonstrated_absolute = demonstrated
            predictions.append(first)
            states.append(state)
            demonstrations.append(demonstrated_absolute)
            rows.append([int(index), *demonstrated_absolute.tolist(), *first.tolist()])
    predictions = np.asarray(predictions)
    states = np.asarray(states)
    demonstrations = np.asarray(demonstrations)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["frame", *[f"demo_{i}" for i in range(7)], *[f"pred_{i}" for i in range(7)]]
    with args.output_csv.open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(header); writer.writerows(rows)
    jumps = np.max(np.abs(np.diff(predictions, axis=0)), axis=0) if len(predictions) > 1 else np.zeros(7)
    print("predicted ranges:", list(zip(predictions.min(0), predictions.max(0))))
    print("maximum sampled consecutive jumps:", jumps.tolist())
    print("gripper predictions:", predictions[:, 6].tolist())
    print("offline comparison:", args.output_csv)
    report_path = args.report or Path(f"{args.output_csv}.report.json")
    report = {
        "checkpoint": str(args.checkpoint), "samples": len(predictions),
        "action_semantics": action_semantics,
        "episode_coverage": dataset.meta.total_episodes if args.samples_per_episode is not None else None,
        "inference_seed": args.inference_seed,
        "output_shape": [len(predictions), 7],
        "predicted_min": predictions.min(0).tolist(), "predicted_max": predictions.max(0).tolist(),
        "maximum_sampled_consecutive_jump": jumps.tolist(),
        "gripper_predictions": predictions[:, 6].tolist(),
        "inference_latency_s": {"mean": float(np.mean(latencies)), "median": float(np.median(latencies)), "max": float(np.max(latencies))},
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0,
        "finite": bool(np.isfinite(predictions).all()), "comparison_csv": str(args.output_csv),
    }
    target_deltas = np.max(np.abs(predictions[:, :6] - states[:, :6]), axis=1)
    action_errors = np.max(np.abs(predictions[:, :6] - demonstrations[:, :6]), axis=1)
    threshold_failures = int(np.count_nonzero(target_deltas > args.max_target_delta_rad))
    report["predicted_target_delta_rad"] = {
        "limit": args.max_target_delta_rad,
        "mean": float(np.mean(target_deltas)),
        "median": float(np.median(target_deltas)),
        "max": float(np.max(target_deltas)),
        "failures": threshold_failures,
    }
    report["predicted_vs_demonstrated_action_error_rad"] = {
        "mean": float(np.mean(action_errors)),
        "median": float(np.median(action_errors)),
        "max": float(np.max(action_errors)),
    }
    report["pass"] = bool(report["finite"] and threshold_failures == 0)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("offline validation report:", report_path)
    if not report["pass"]:
        raise SystemExit(
            f"OFFLINE GATE FAIL: {threshold_failures}/{len(target_deltas)} predicted targets exceed "
            f"{args.max_target_delta_rad:.6f} rad"
        )


if __name__ == "__main__":
    main()
