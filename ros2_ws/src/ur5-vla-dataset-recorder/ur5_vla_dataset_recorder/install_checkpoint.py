"""Verify and safely install a complete LeRobot checkpoint archive."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from .package_dataset import safe_extract, sha256_file


def install_checkpoint(archive: Path, checksum: Path, output: Path, overwrite: bool = False,
                       verify_load: bool = True) -> Path:
    archive, checksum, output = archive.resolve(), checksum.resolve(), output.resolve()
    if not archive.is_file() or not checksum.is_file():
        raise FileNotFoundError("checkpoint archive and checksum file are required")
    expected = checksum.read_text().split()[0].lower()
    actual = sha256_file(archive)
    if actual != expected:
        raise ValueError(f"checkpoint SHA-256 mismatch: {actual} != {expected}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing overwrite; review path and pass --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ur5_checkpoint_verify_") as temporary:
        extracted_root = Path(temporary)
        safe_extract(archive, extracted_root)
        model = extracted_root / "pretrained_model"
        required = [model / "model.safetensors", model / "config.json"]
        if not all(path.is_file() for path in required):
            raise ValueError(f"incomplete checkpoint; required files missing: {required}")
        processor_files = list(model.glob("*processor*.json")) + list(model.glob("*tokenizer*"))
        if not processor_files:
            raise ValueError("incomplete checkpoint: processor/tokenizer assets are absent")
        if verify_load:
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors
            config = PreTrainedConfig.from_pretrained(model, local_files_only=True)
            get_policy_class(config.type).from_pretrained(model, config=config, local_files_only=True)
            make_pre_post_processors(config, pretrained_path=model)
        temporary_output = output.with_name(f".{output.name}.installing")
        if temporary_output.exists():
            shutil.rmtree(temporary_output)
        shutil.copytree(model, temporary_output)
        if output.exists():
            shutil.rmtree(output)
        temporary_output.rename(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checksum", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-model-load", action="store_true")
    args = parser.parse_args()
    installed = install_checkpoint(args.archive, args.checksum, args.output, args.overwrite, not args.skip_model_load)
    print(f"Installed verified complete checkpoint: {installed}")


if __name__ == "__main__":
    main()
