import io
import tarfile
from pathlib import Path

import pytest

from ur5_vla_dataset_recorder.package_dataset import checked_members, safe_extract, sha256_file
from ur5_vla_dataset_recorder.install_checkpoint import install_checkpoint
from ur5_vla_dataset_recorder.split_archive import split_archive


def test_sha256_file(tmp_path):
    path = tmp_path / "value"; path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_safe_extract_preserves_hierarchy(tmp_path):
    source = tmp_path / "source"; source.mkdir(); (source / "meta").mkdir(); (source / "meta/info.json").write_text("{}")
    archive_path = tmp_path / "dataset.tar"
    with tarfile.open(archive_path, "w") as archive: archive.add(source, arcname="dataset")
    output = tmp_path / "output"; safe_extract(archive_path, output)
    assert (output / "dataset/meta/info.json").read_text() == "{}"


def test_archive_traversal_is_rejected(tmp_path):
    archive_path = tmp_path / "bad.tar"
    with tarfile.open(archive_path, "w") as archive:
        info = tarfile.TarInfo("../escape"); data = b"bad"; info.size = len(data); archive.addfile(info, io.BytesIO(data))
    with tarfile.open(archive_path, "r:") as archive:
        with pytest.raises(ValueError, match="unsafe archive path"): checked_members(archive)
    with pytest.raises(ValueError): safe_extract(archive_path, tmp_path / "output")


def test_archive_links_are_rejected(tmp_path):
    archive_path = tmp_path / "bad-link.tar"
    with tarfile.open(archive_path, "w") as archive:
        info = tarfile.TarInfo("dataset/link"); info.type = tarfile.SYMTYPE; info.linkname = "/etc/passwd"; archive.addfile(info)
    with tarfile.open(archive_path, "r:") as archive:
        with pytest.raises(ValueError, match="links are not allowed"): checked_members(archive)


def test_complete_checkpoint_install_is_non_destructive(tmp_path):
    model = tmp_path / "source/pretrained_model"; model.mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "config.json").write_text("{}")
    (model / "preprocessor_config.json").write_text("{}")
    archive = tmp_path / "checkpoint.tar"
    with tarfile.open(archive, "w") as output: output.add(model, arcname="pretrained_model")
    checksum = tmp_path / "checkpoint.tar.sha256"
    checksum.write_text(f"{sha256_file(archive)}  {archive.name}\n")
    installed = install_checkpoint(archive, checksum, tmp_path / "installed", verify_load=False)
    assert (installed / "model.safetensors").read_bytes() == b"weights"
    with pytest.raises(FileExistsError):
        install_checkpoint(archive, checksum, installed, verify_load=False)


def test_checkpoint_checksum_mismatch_is_rejected(tmp_path):
    archive = tmp_path / "checkpoint.tar"; archive.write_bytes(b"not a tar")
    checksum = tmp_path / "checkpoint.tar.sha256"; checksum.write_text(f"{'0'*64}  {archive.name}\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        install_checkpoint(archive, checksum, tmp_path / "installed", verify_load=False)


def test_split_archive_reassembles_exact_bytes(tmp_path):
    archive = tmp_path / "dataset.tar"; archive.write_bytes(bytes(range(251)) * 10)
    parts = split_archive(archive, part_size=1000)
    assert len(parts) == 3 and all(part.stat().st_size <= 1000 for part in parts)
    assert b"".join(part.read_bytes() for part in parts) == archive.read_bytes()
    assert Path(f"{archive}.parts.sha256").is_file() and Path(f"{archive}.parts.json").is_file()
    with pytest.raises(FileExistsError): split_archive(archive, part_size=1000)
