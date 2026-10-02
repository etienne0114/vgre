import struct
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bindings" / "python"))

from vgre import LanguageModel
from vgre._native import NATIVE_AVAILABLE
from vgre.auto_config import ErrorRecovery, GGUFMetadataError
from vgre.cli import _generate, _validate_hf_file, build_parser


def _string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _metadata_entry(key, value_type, value):
    entry = _string(key) + struct.pack("<I", value_type)
    if value_type == 4:
        return entry + struct.pack("<I", value)
    if value_type == 6:
        return entry + struct.pack("<f", value)
    if value_type == 8:
        return entry + _string(value)
    raise ValueError(value_type)


def _write_gguf(path, architecture="llama"):
    metadata = [
        _metadata_entry("general.architecture", 8, architecture),
        _metadata_entry(f"{architecture}.block_count", 4, 1),
        _metadata_entry(f"{architecture}.embedding_length", 4, 64),
        _metadata_entry(f"{architecture}.attention.head_count", 4, 1),
        _metadata_entry(f"{architecture}.attention.head_count_kv", 4, 1),
        _metadata_entry(f"{architecture}.feed_forward_length", 4, 128),
        _metadata_entry(f"{architecture}.context_length", 4, 8192),
        _metadata_entry(f"{architecture}.rope.freq_base", 6, 500000.0),
        _metadata_entry(f"{architecture}.attention.layer_norm_rms_epsilon", 6, 1e-6),
        _metadata_entry("tokenizer.chat_template", 8, "<|im_start|>user"),
    ]
    tokens = struct.pack("<Q", len("tokenizer.ggml.tokens")) + b"tokenizer.ggml.tokens"
    tokens += struct.pack("<I IQ", 9, 8, 256)
    tokens += b"".join(_string(f"t{index}") for index in range(256))
    metadata.append(tokens)

    dimensions = 64
    feed_forward = 128
    vocab = 256
    tensors = [
        ("token_embd.weight", (dimensions, vocab), False),
        ("output_norm.weight", (dimensions,), True),
        ("blk.0.attn_norm.weight", (dimensions,), True),
        ("blk.0.attn_q.weight", (dimensions, dimensions), False),
        ("blk.0.attn_k.weight", (dimensions, dimensions), False),
        ("blk.0.attn_v.weight", (dimensions, dimensions), False),
        ("blk.0.attn_output.weight", (dimensions, dimensions), False),
        ("blk.0.ffn_norm.weight", (dimensions,), True),
        ("blk.0.ffn_gate.weight", (dimensions, feed_forward), False),
        ("blk.0.ffn_up.weight", (dimensions, feed_forward), False),
        ("blk.0.ffn_down.weight", (feed_forward, dimensions), False),
    ]
    if architecture == "qwen2":
        tensors.extend((f"blk.0.attn_{projection}.bias", (dimensions,), True)
                       for projection in ("q", "k", "v"))
    data = bytearray()
    offsets = []
    for _, dims, ones in tensors:
        aligned = (len(data) + 31) & ~31
        data.extend(b"\0" * (aligned - len(data)))
        offsets.append(aligned)
        count = 1
        for dim in dims:
            count *= dim
        if ones:
            data.extend(struct.pack(f"<{count}f", *([1.0] * count)))
        else:
            data.extend(b"\0" * (count * 4))

    header = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(metadata)))
    for entry in metadata:
        header.extend(entry)
    for (name, dims, _), offset in zip(tensors, offsets):
        header.extend(_string(name))
        header.extend(struct.pack("<I", len(dims)))
        header.extend(struct.pack(f"<{len(dims)}Q", *dims))
        header.extend(struct.pack("<IQ", 0, offset))
    header.extend(b"\0" * (((len(header) + 31) & ~31) - len(header)))
    path.write_bytes(header + data)


def test_real_typed_gguf_auto_configuration_and_generation():
    if not NATIVE_AVAILABLE:
        raise RuntimeError("libvgre is required for this integration test")
    with tempfile.TemporaryDirectory() as temp_dir:
        for architecture in ("llama", "qwen2"):
            model_path = Path(temp_dir) / f"tiny-{architecture}.gguf"
            _write_gguf(model_path, architecture)
            model = LanguageModel.from_gguf(str(model_path))
            try:
                assert model.vocab == 256
                assert model.rope_theta == 500000.0
                assert abs(model.norm_eps - 1e-6) < 1e-12
                generated = model.generate([1, 2], n_new=2, temperature=0.0)
                assert len(generated) == 4
                assert all(0 <= token < model.vocab for token in generated)

                if architecture == "llama":
                    cli_args = SimpleNamespace(
                        bf16=True, speculative=False, temperature=0.0, max_tokens=2,
                        top_k=40, top_p=0.95, repetition_penalty=1.1, seed=0,
                    )
                    bf16_generated = _generate(model, [1, 2], cli_args)
                    assert len(bf16_generated) == 4

                    cli_args.bf16 = False
                    cli_args.speculative = True
                    speculative_generated = _generate(model, [1, 2], cli_args)
                    assert speculative_generated == generated

                    cli_args.temperature = 0.8
                    try:
                        _generate(model, [1, 2], cli_args)
                    except ValueError as error:
                        assert "temperature 0" in str(error)
                    else:
                        raise AssertionError(
                            "speculative decoding must reject sampling temperature"
                        )
            finally:
                model.close()


def test_pull_cache_validation():
    with tempfile.TemporaryDirectory() as temp_dir:
        directory = Path(temp_dir)
        tokenizer_path = directory / "tokenizer.json"
        tokenizer_path.write_text("{}", encoding="utf-8")
        _validate_hf_file("tokenizer.json", tokenizer_path)

        model_path = directory / "model.gguf"
        model_path.write_bytes(b"not-gguf")
        try:
            _validate_hf_file("model.gguf", model_path)
        except OSError as error:
            assert "not a GGUF" in str(error)
        else:
            raise AssertionError("invalid cached GGUF must be rejected")


def test_missing_dimensions_are_not_guessed():
    try:
        ErrorRecovery.handle_missing_metadata("incomplete.gguf", ["vocab", "d_model"])
    except GGUFMetadataError as error:
        assert error.missing_params == ["vocab", "d_model"]
    else:
        raise AssertionError("missing GGUF dimensions must not be guessed")


def test_packed_ternary_inference_is_exposed_to_python_and_cli():
    args = SimpleNamespace(
        bf16=False, ternary=True, speculative=False, temperature=0.0,
        max_tokens=2, top_k=0, top_p=1.0, repetition_penalty=1.0, seed=7,
    )
    assert build_parser().parse_args(["generate", "--ternary"]).ternary
    assert build_parser().parse_args(["chat", "--ternary"]).ternary

    model = LanguageModel(vocab=64, n_layer=1, d_model=64, n_head=2,
                          d_ff=128, max_seq=16, seed=7)
    try:
        generated = _generate(model, [1, 2], args)
        assert len(generated) == 4
        assert all(0 <= token < model.vocab for token in generated)
    finally:
        model.close()


if __name__ == "__main__":
    if not NATIVE_AVAILABLE:
        raise SystemExit(77)
    test_real_typed_gguf_auto_configuration_and_generation()
    test_pull_cache_validation()
    test_missing_dimensions_are_not_guessed()
