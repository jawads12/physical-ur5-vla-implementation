#!/usr/bin/env python3
"""Display an estimated progress dashboard for the active delta-policy run."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path


PID = 973789
TOTAL_STEPS = 20_000
MEASURED_STEPS_PER_SECOND = 1.61
CHECKPOINT_ROOT = Path(__file__).resolve().parents[4] / "dump/downloaded_models/Checkpoints/train_clean20_delta_20k/checkpoints"


def command(*args: str) -> str:
    return subprocess.run(args, check=False, capture_output=True, text=True).stdout.strip()


while Path(f"/proc/{PID}").exists():
    elapsed_text = command("ps", "-p", str(PID), "-o", "etimes=")
    elapsed = int(elapsed_text or 0)
    estimated_step = min(TOTAL_STEPS, int(elapsed * MEASURED_STEPS_PER_SECOND))
    percent = 100.0 * estimated_step / TOTAL_STEPS
    remaining = max(0, int((TOTAL_STEPS - estimated_step) / MEASURED_STEPS_PER_SECOND))
    width = 50
    filled = min(width, int(width * percent / 100.0))
    bar = "#" * filled + "-" * (width - filled)
    checkpoints = sorted(path.name for path in CHECKPOINT_ROOT.glob("*") if path.is_dir())
    gpu = command(
        "nvidia-smi", "--id=0",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    )
    print("\033[2J\033[H", end="")
    print("UR5 DELTA POLICY TRAINING — LIVE ESTIMATE\n")
    print(f"[{bar}] {percent:6.2f}%")
    print(f"Estimated step : {estimated_step:,} / {TOTAL_STEPS:,}")
    print(f"Elapsed        : {elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d}")
    print(f"Estimated ETA  : {remaining // 3600:02d}:{remaining % 3600 // 60:02d}:{remaining % 60:02d}")
    print(f"Measured rate  : {MEASURED_STEPS_PER_SECOND:.2f} steps/s")
    print(f"GPU 0          : {gpu or 'unavailable'}  (util %, MiB used/total, C)")
    print(f"Checkpoints    : {', '.join(checkpoints) if checkpoints else 'none yet (first at 5K)'}")
    print("\nPercentage is estimated from measured throughput; checkpoints are exact.")
    time.sleep(5)

print("\nTraining process is no longer running.")
input("Press Enter to close...")
