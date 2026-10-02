"""End-to-end regression tests for GGUF auto-configuration and generation."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bindings" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_gguf_auto_config import _write_gguf
from vgre import GGUFMetadataReader, LanguageModel
from vgre._native import NATIVE_AVAILABLE
from vgre.auto_config import GGUFFileError, ModelAutoConfigurator


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="requires libvgre")
def test_metadata_to_generation_pipeline_is_repeatable(tmp_path):
    model_path = tmp_path / "tiny-llama.gguf"
    _write_gguf(model_path)

    with GGUFMetadataReader(str(model_path)) as reader:
        assert reader.get_block_count() == 1
        assert reader.get_embedding_length() == 64
        assert reader.get_head_count() == 1
        assert reader.get_vocabulary_size() == 256

    first_config = ModelAutoConfigurator.auto_configure(str(model_path))
    second_config = ModelAutoConfigurator.auto_configure(str(model_path))
    assert first_config == second_config

    model = LanguageModel.from_gguf(str(model_path))
    try:
        first_output = model.generate([1, 2], n_new=2, temperature=0.0)
        second_output = model.generate([1, 2], n_new=2, temperature=0.0)
        assert first_output == second_output
        assert len(first_output) == 4
        assert all(0 <= token < model.vocab for token in first_output)
    finally:
        model.close()


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="requires libvgre")
def test_json_sidecar_overrides_metadata_and_invalid_json_is_reported(tmp_path):
    model_path = tmp_path / "sidecar-model.gguf"
    _write_gguf(model_path)
    sidecar_path = model_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps({"rope_theta": 12345.0}), encoding="utf-8")

    config = ModelAutoConfigurator.auto_configure(str(model_path))
    assert config["rope_theta"] == 12345.0

    model = LanguageModel.from_gguf(str(model_path))
    try:
        assert model.rope_theta == 12345.0
    finally:
        model.close()

    sidecar_path.write_text("{ invalid json", encoding="utf-8")
    with pytest.raises(GGUFFileError, match="Failed to parse JSON sidecar"):
        ModelAutoConfigurator.auto_configure(str(model_path))


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="requires libvgre")
def test_concurrent_metadata_reads_are_consistent(tmp_path):
    model_path = tmp_path / "concurrent-model.gguf"
    _write_gguf(model_path)

    with ThreadPoolExecutor(max_workers=4) as executor:
        configs = list(executor.map(
            lambda _: ModelAutoConfigurator.auto_configure(str(model_path)), range(8)
        ))

    assert all(config == configs[0] for config in configs[1:])
