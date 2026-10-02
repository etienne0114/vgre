"""End-to-end Q4_K GGUF loading regression."""

import struct
import sys
import io
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bindings" / "python"))

from vgre import GGUFMetadataReader, LanguageModel
from vgre._native import NATIVE_AVAILABLE
from vgre.cli import main as cli_main


def _string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _write_quantized_gguf(path, weight_type=12):
    metadata = {
        "general.architecture": "llama",
        "llama.block_count": "1",
        "llama.embedding_length": "64",
        "llama.attention.head_count": "1",
        "llama.attention.head_count_kv": "1",
        "llama.feed_forward_length": "128",
        "llama.vocab_size": "256",
        "llama.context_length": "32",
        "llama.rope.freq_base": "500000",
        "llama.attention.layer_norm_rms_epsilon": "0.000001",
    }
    tensors = [
        ("token_embd.weight", (64, 256), weight_type, False),
        ("output_norm.weight", (64,), 0, True),
        ("blk.0.attn_norm.weight", (64,), 0, True),
        ("blk.0.attn_q.weight", (64, 64), weight_type, False),
        ("blk.0.attn_k.weight", (64, 64), weight_type, False),
        ("blk.0.attn_v.weight", (64, 64), weight_type, False),
        ("blk.0.attn_output.weight", (64, 64), weight_type, False),
        ("blk.0.ffn_norm.weight", (64,), 0, True),
        ("blk.0.ffn_gate.weight", (64, 128), weight_type, False),
        ("blk.0.ffn_up.weight", (64, 128), weight_type, False),
        ("blk.0.ffn_down.weight", (128, 64), weight_type, False),
    ]

    header = bytearray(b"GGUF")
    header += struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        header += _string(key)
        header += struct.pack("<I", 8)
        header += _string(value)

    offsets = []
    offset = 0
    for name, shape, tensor_type, _ in tensors:
        offset = (offset + 31) & ~31
        offsets.append(offset)
        elements = 1
        for dimension in shape:
            elements *= dimension
        if tensor_type == 0:
            size = elements * 4
        elif tensor_type == 1:
            size = elements * 2
        else:
            block_sizes = {2: (32, 18), 3: (32, 20), 6: (32, 22),
                           8: (32, 34), 12: (256, 144), 14: (256, 210)}
            block_size, byte_size = block_sizes[tensor_type]
            if elements % block_size:
                raise ValueError(f"tensor size {elements} is invalid for GGML type {tensor_type}")
            size = elements // block_size * byte_size
        offset += size

    for (name, shape, tensor_type, _), tensor_offset in zip(tensors, offsets):
        header += _string(name)
        header += struct.pack("<I", len(shape))
        header += struct.pack(f"<{len(shape)}Q", *shape)
        header += struct.pack("<IQ", tensor_type, tensor_offset)
    header.extend(b"\0" * ((-len(header)) % 32))

    payload = bytearray()
    for (_, shape, tensor_type, ones), tensor_offset in zip(tensors, offsets):
        payload.extend(b"\0" * (tensor_offset - len(payload)))
        elements = 1
        for dimension in shape:
            elements *= dimension
        if tensor_type == 0:
            values = [1.0 if ones else 0.0] * elements
            payload.extend(struct.pack(f"<{elements}f", *values))
        elif tensor_type == 1:
            payload.extend(b"\0" * (elements * 2))
        else:
            block_size, byte_size = block_sizes[tensor_type]
            payload.extend(b"\0" * (elements // block_size * byte_size))

    path.write_bytes(header + payload)


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="requires libvgre")
def test_q4k_gguf_auto_config_loads_and_generates(tmp_path):
    model_path = tmp_path / "tiny-q4k.gguf"
    _write_quantized_gguf(model_path)

    with GGUFMetadataReader(str(model_path)) as reader:
        assert reader.get_block_count() == 1
        assert reader.get_embedding_length() == 64
        assert reader.get_feed_forward_length() == 128
        assert reader.get_rope_freq_base() == 500000.0

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


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="requires libvgre")
def test_chat_loads_gguf_and_runs_without_sidecars(tmp_path, monkeypatch, capsys):
    model_path = tmp_path / "tiny-chat.gguf"
    _write_quantized_gguf(model_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO("hello\n"))

    result = cli_main([
        "chat", str(model_path), "--max-tokens", "2", "--temperature", "0",
    ])

    captured = capsys.readouterr()
    assert result == 0, captured.err
    assert "loaded GGUF model" in captured.err
    assert "Assistant:" in captured.out
