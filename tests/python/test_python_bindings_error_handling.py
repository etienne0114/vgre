"""Real-filesystem tests for GGUF Python binding errors and cleanup."""

import os
import struct
import sys
import threading
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "bindings" / "python"))

NATIVE_NAME = {
    "darwin": "libvgre.dylib",
    "win32": "vgre.dll",
}.get(sys.platform, "libvgre.so")
NATIVE_PATH = REPO_ROOT / "build" / NATIVE_NAME
if NATIVE_PATH.is_file():
    os.environ["VGRE_LIB_PATH"] = str(NATIVE_PATH)
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        part for part in (str(NATIVE_PATH.parent), os.environ.get("LD_LIBRARY_PATH", "")) if part
    )

from vgre import GGUFMetadataReader
from vgre._native import NATIVE_AVAILABLE


pytestmark = pytest.mark.skipif(not NATIVE_AVAILABLE, reason="VGRE native library is required")


def _write_empty_gguf(path: Path) -> None:
    header = b"GGUF" + struct.pack("<IQQ", 3, 0, 0)
    path.write_bytes(header + b"\0" * ((32 - len(header) % 32) % 32))


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        GGUFMetadataReader(tmp_path / "missing.gguf")


def test_invalid_magic_and_truncated_file_raise_value_error(tmp_path):
    invalid_path = tmp_path / "invalid.gguf"
    invalid_path.write_bytes(b"not-a-gguf")
    with pytest.raises(ValueError, match="not a GGUF"):
        GGUFMetadataReader(invalid_path)

    truncated_path = tmp_path / "truncated.gguf"
    truncated_path.write_bytes(b"GGUF\x03\x00\x00\x00")
    with pytest.raises(ValueError, match="Invalid or truncated"):
        GGUFMetadataReader(truncated_path)


def test_directory_path_raises_directory_error(tmp_path):
    with pytest.raises(IsADirectoryError, match="is a directory"):
        GGUFMetadataReader(tmp_path)


def test_empty_valid_gguf_has_optional_metadata_and_closes(tmp_path):
    gguf_path = tmp_path / "empty.gguf"
    _write_empty_gguf(gguf_path)

    with GGUFMetadataReader(gguf_path) as reader:
        assert reader.get_block_count() is None
        assert reader.get_embedding_length() is None
        assert reader.get_head_count() is None
        assert reader.get_head_count_kv() is None
        assert reader.get_feed_forward_length() is None
        assert reader.get_rope_freq_base() is None
        assert reader.get_norm_eps() is None
    assert reader._h is None


def test_context_manager_closes_when_processing_raises(tmp_path):
    gguf_path = tmp_path / "empty.gguf"
    _write_empty_gguf(gguf_path)

    with pytest.raises(RuntimeError, match="processing failed"):
        with GGUFMetadataReader(gguf_path) as reader:
            raise RuntimeError("processing failed")
    assert reader._h is None


def test_concurrent_missing_file_errors_are_python_file_errors(tmp_path):
    missing_path = tmp_path / "missing.gguf"
    errors = []

    def open_missing_file():
        try:
            GGUFMetadataReader(missing_path)
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=open_missing_file) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(errors) == len(threads)
    assert all(isinstance(error, FileNotFoundError) for error in errors)


def test_repeated_missing_file_errors_do_not_open_native_handles(tmp_path):
    for index in range(50):
        with pytest.raises(FileNotFoundError):
            GGUFMetadataReader(tmp_path / f"missing-{index}.gguf")
