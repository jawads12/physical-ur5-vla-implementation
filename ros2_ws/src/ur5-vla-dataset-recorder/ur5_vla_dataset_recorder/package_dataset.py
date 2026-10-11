"""Package a local LeRobot dataset as an uncompressed, verified tar archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checked_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive path: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"links are not allowed in dataset archives: {member.name}")
    return members


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:") as archive:
        members = checked_members(archive)
        archive.extractall(destination, members=members, filter="data")


def _metadata_summary(dataset_root: Path, repo_id: str) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root, video_backend="pyav")
    info = dataset.meta.info
    return {
        "dataset_name": dataset_root.name,
        "repo_id": repo_id,
        "lerobot_version": "0.4.3",
        "codebase_version": info.get("codebase_version"),
        "episode_count": dataset.meta.total_episodes,
        "frame_count": dataset.meta.total_frames,
        "fps": dataset.meta.fps,
        "tasks": dataset.meta.tasks.index.tolist(),
        "features": dataset.meta.features,
    }


def package_dataset(dataset_root: Path, archive_path: Path, repo_id: str,
                    overwrite: bool = False) -> dict:
    dataset_root = dataset_root.resolve()
    archive_path = archive_path.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    if archive_path.suffix != ".tar":
        raise ValueError("archive path must end in .tar (no recompression)")
    outputs = {
        "archive": archive_path,
        "checksum": Path(f"{archive_path}.sha256"),
        "metadata_checksums": Path(f"{archive_path}.metadata.sha256"),
        "manifest": Path(f"{archive_path}.manifest.json"),
        "validation_report": Path(f"{archive_path}.validation.md"),
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing overwrite; use --overwrite after reviewing: {existing}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    summary = _metadata_summary(dataset_root, repo_id)
    dataset_bytes = sum(path.stat().st_size for path in dataset_root.rglob("*") if path.is_file())
    temporary_archive = archive_path.with_name(f".{archive_path.name}.tmp")
    if temporary_archive.exists():
        temporary_archive.unlink()
    try:
        with tarfile.open(temporary_archive, "w", format=tarfile.PAX_FORMAT) as archive:
            archive.add(dataset_root, arcname=dataset_root.name, recursive=True)
        with tarfile.open(temporary_archive, "r:") as archive:
            members = checked_members(archive)
            if not members or not all(PurePosixPath(member.name).parts[0] == dataset_root.name for member in members):
                raise ValueError("archive does not preserve a single dataset root directory")
        os.replace(temporary_archive, archive_path)
    finally:
        if temporary_archive.exists():
            temporary_archive.unlink()

    archive_sha256 = sha256_file(archive_path)
    important = [
        dataset_root / "meta/info.json",
        dataset_root / "meta/stats.json",
        dataset_root / "meta/tasks.parquet",
        dataset_root / "meta/episodes/chunk-000/file-000.parquet",
        dataset_root / "meta/ur5_vla_conversion.json",
        dataset_root / "validation_report.md",
    ]
    metadata_hashes = {str(path.relative_to(dataset_root)): sha256_file(path) for path in important if path.is_file()}
    manifest = {
        **summary,
        "dataset_size_bytes": dataset_bytes,
        "archive_size_bytes": archive_path.stat().st_size,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": archive_sha256,
        "archive_filename": archive_path.name,
        "metadata_sha256": metadata_hashes,
        "source_dataset_root": str(dataset_root),
        "compression": "none",
    }
    outputs["checksum"].write_text(f"{archive_sha256}  {archive_path.name}\n")
    outputs["metadata_checksums"].write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(metadata_hashes.items()))
    )
    outputs["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")

    from .validate_lerobot import validate_dataset
    with tempfile.TemporaryDirectory(prefix="ur5_lerobot_verify_") as temporary:
        extraction_root = Path(temporary)
        safe_extract(archive_path, extraction_root)
        extracted = extraction_root / dataset_root.name
        errors = validate_dataset(
            extracted, repo_id, outputs["validation_report"],
            expected_episodes=summary["episode_count"],
            expected_frames=summary["frame_count"],
            task=summary["tasks"][0] if len(summary["tasks"]) == 1 else None,
            video_backend="pyav",
        )
        if errors:
            raise ValueError(f"extracted dataset validation failed: {errors}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = package_dataset(args.dataset_root, args.archive, args.repo_id, args.overwrite)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
