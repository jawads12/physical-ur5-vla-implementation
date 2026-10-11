"""Generate the 53-episode, 80k-step SmolVLA Colab notebook with W&B."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks/train_ur5_cube_30ep_smolvla_40000_colab.ipynb"
OUTPUT = ROOT / "notebooks/train_ur5_cube_clean_53ep_smolvla_80000_wandb_colab.ipynb"


def cell_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def make_cell(cell_type: str, text: str) -> dict:
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    }
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


notebook = json.loads(BASE.read_text())

replacements = (
    ("ur5_cube_30ep_smolvla_40000_steps", "ur5_cube_clean_53ep_smolvla_80000_steps"),
    ("ur5_cube_30ep", "ur5_cube_clean_53ep"),
    ("28 episodes / 40,000 steps", "53 episodes / 80,000 steps"),
    ("28-episode", "53-episode"),
    ("28 episodes", "53 episodes"),
    ("40,000", "80,000"),
    ("040000", "080000"),
    ("TRAIN_STEPS = 40000", "TRAIN_STEPS = 80000"),
    ("dataset.meta.total_episodes == 28 and dataset.meta.total_frames == 47204",
     "dataset.meta.total_episodes == 53 and dataset.meta.total_frames == 67621"),
    ("dataset_episodes\": 28, \"dataset_frames\": 47204",
     "dataset_episodes\": 53, \"dataset_frames\": 67621"),
    ("plt.subplots(28, 2, figsize=(12, 84))", "plt.subplots(53, 2, figsize=(12, 159))"),
    ("range(28)", "range(53)"),
    ("--wandb.enable=false", "--wandb.enable=true",),
)

for cell in notebook["cells"]:
    text = cell_source(cell)
    for old, new in replacements:
        text = text.replace(old, new)
    cell["source"] = text.splitlines(keepends=True)

# Add an explicit project name to the LeRobot command.
for cell in notebook["cells"]:
    text = cell_source(cell)
    if '"--wandb.enable=true"' in text:
        text = text.replace(
            '"--wandb.enable=true",',
            '"--wandb.enable=true", "--wandb.project=smolvla_53ep_80k", "--wandb.mode=online",',
        )
        cell["source"] = text.splitlines(keepends=True)

wandb_markdown = make_cell(
    "markdown",
    """
## E — Authenticate Weights & Biases

In Colab, open **Secrets** (key icon), create `WANDB_API_KEY`, paste your W&B API key, and enable notebook access. The key is read securely and is never written to Drive or the notebook. Metrics are logged to the W&B project `smolvla_53ep_80k`.
""",
)
wandb_code = make_cell(
    "code",
    """
import os
from google.colab import userdata
import wandb

try:
    wandb_key = userdata.get("WANDB_API_KEY")
except Exception as exc:
    raise RuntimeError(
        "Add WANDB_API_KEY in Colab Secrets and enable notebook access before continuing."
    ) from exc

assert wandb_key, "WANDB_API_KEY is empty"
os.environ["WANDB_API_KEY"] = wandb_key
wandb.login(key=wandb_key, relogin=True)
del wandb_key
print("W&B authentication PASS; project: smolvla_53ep_80k")
""",
)

insert_at = next(
    i for i, cell in enumerate(notebook["cells"])
    if "## E — Configure the 80,000-step training run" in cell_source(cell)
)
notebook["cells"][insert_at:insert_at] = [wandb_markdown, wandb_code]

# W&B becomes section E, so shift the later headings for readability.
for cell in notebook["cells"][insert_at + 2:]:
    text = cell_source(cell)
    text = text.replace("## E — Configure", "## F — Configure")
    text = text.replace("## F — Check", "## G — Check")
    text = text.replace("## G — Train", "## H — Train")
    cell["source"] = text.splitlines(keepends=True)

notebook["metadata"].setdefault("colab", {})["name"] = OUTPUT.name
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n")
print(OUTPUT)
