"""Split a verified archive for services with conservative per-file limits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .package_dataset import sha256_file


def split_archive(archive: Path, part_size: int = 400 * 1024 * 1024,
                  overwrite: bool = False) -> list[Path]:
    archive = archive.resolve()
    if not archive.is_file() or part_size <= 0:
        raise ValueError("a readable archive and positive part size are required")
    part_count = (archive.stat().st_size + part_size - 1) // part_size
    parts = [Path(f"{archive}.part-{index:03d}") for index in range(part_count)]
    checksum_path = Path(f"{archive}.parts.sha256")
    manifest_path = Path(f"{archive}.parts.json")
    outputs = [*parts, checksum_path, manifest_path]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing overwrite; review and pass --overwrite: {existing}")
    temporary_parts = [path.with_name(f".{path.name}.tmp") for path in parts]
    for path in temporary_parts:
        if path.exists(): path.unlink()
    try:
        with archive.open("rb") as source:
            for temporary in temporary_parts:
                remaining = part_size
                with temporary.open("wb") as output:
                    while remaining:
                        chunk = source.read(min(8 * 1024 * 1024, remaining))
                        if not chunk: break
                        output.write(chunk); remaining -= len(chunk)
        for temporary, final in zip(temporary_parts, parts, strict=True):
            temporary.replace(final)
    finally:
        for path in temporary_parts:
            if path.exists(): path.unlink()
    hashes = {part.name: sha256_file(part) for part in parts}
    checksum_path.write_text("".join(f"{digest}  {name}\n" for name, digest in hashes.items()))
    manifest_path.write_text(json.dumps({
        "archive_filename": archive.name, "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive), "part_size_bytes": part_size,
        "parts": [{"filename": part.name, "size_bytes": part.stat().st_size, "sha256": hashes[part.name]} for part in parts],
        "reassembly": f"concatenate parts in lexical filename order to recreate {archive.name}",
    }, indent=2) + "\n")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--part-size-mib", type=int, default=400)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    parts = split_archive(args.archive, args.part_size_mib * 1024 * 1024, args.overwrite)
    for part in parts: print(part, part.stat().st_size, sha256_file(part))


if __name__ == "__main__":
    main()
