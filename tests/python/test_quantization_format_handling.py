"""End-to-end checks for GGML tensor types supported by the GGUF loader."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bindings" / "python"))

from test_q4k_model_loading import _write_quantized_gguf
from vgre import LanguageModel
from vgre._native import NATIVE_AVAILABLE


SUPPORTED_TYPES = [
    pytest.param(0, id="f32"),
    pytest.param(1, id="f16"),
    pytest.param(2, id="q4_0"),
    pytest.param(3, id="q4_1"),
    pytest.param(6, id="q5_0"),
    pytest.param(8, id="q8_0"),
    pytest.param(12, id="q4_k"),
    pytest.param(14, id="q6_k"),
]


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="requires libvgre")
@pytest.mark.parametrize("ggml_type", SUPPORTED_TYPES)
def test_supported_gguf_weight_type_loads_and_generates(tmp_path, ggml_type):
    model_path = tmp_path / f"tiny-type-{ggml_type}.gguf"
    _write_quantized_gguf(model_path, weight_type=ggml_type)

    model = LanguageModel.from_gguf(str(model_path))
    try:
        assert model.vocab == 256
        assert model.rope_theta == 500000.0
        assert model.norm_eps == pytest.approx(1e-6, rel=1e-6)
        generated = model.generate([1, 2], n_new=2, temperature=0.0)
        assert len(generated) == 4
        assert all(0 <= token < model.vocab for token in generated)
    finally:
        model.close()
