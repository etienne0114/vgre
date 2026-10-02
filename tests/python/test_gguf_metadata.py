"""Python-facing contract tests for native GGUF metadata access."""

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "bindings" / "python"))

from vgre import GGUFMetadataReader
from vgre._native import NATIVE_AVAILABLE, _lib


def test_gguf_metadata_c_api_is_exported():
    if not NATIVE_AVAILABLE:
        pytest.skip("native VGRE library is not available")
    assert _lib is not None
    assert hasattr(_lib, "vgre_gguf_metadata_open")


def test_missing_gguf_path_raises_file_not_found(tmp_path):
    missing_path = tmp_path / "absent.gguf"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        GGUFMetadataReader(missing_path)
