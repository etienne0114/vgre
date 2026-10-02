"""
VGRE-LM — Python bindings for VGRE's in-tree language model + BPE tokenizer.

Trains and runs a Llama-style decoder entirely on CPU through VGRE's own SIMD
GEMM + autograd + AdamW stack — no external BLAS, no ML framework, no gated
checkpoint download. Thin ctypes wrappers over the C ABI in
include/vgre/xla/model_c_api.h.

Example
-------
    import vgre
    tok = vgre.Tokenizer()
    tok.train(open("corpus.txt").read(), num_merges=512)
    ids = tok.encode("Shall I compare")

    lm = vgre.LanguageModel(vocab=tok.vocab_size, n_layer=4, d_model=256, n_head=8)
    # ... training loop calling lm.train_step(window, targets, lr) ...
    out = lm.generate(ids, n_new=40, temperature=0.8)
    print(tok.decode(out))
"""

import ctypes
import os
from typing import List, Optional

from ._native import _lib, NATIVE_AVAILABLE

_bound = False


def _bind() -> None:
    """Declare argtypes/restypes once (idempotent)."""
    global _bound
    if _bound or _lib is None:
        return
    c, P = ctypes, ctypes.POINTER

    _lib.vgre_lm_create.argtypes = [c.c_int] * 6 + [c.c_float, c.c_int, c.c_uint]
    _lib.vgre_lm_create.restype = c.c_void_p
    _lib.vgre_lm_create_gqa.argtypes = [c.c_int] * 7 + [c.c_float, c.c_int, c.c_uint, c.c_int]
    _lib.vgre_lm_create_gqa.restype = c.c_void_p
    _lib.vgre_lm_set_rope_norm.argtypes = [c.c_void_p, c.c_float, c.c_float]
    _lib.vgre_lm_set_rope_norm.restype = c.c_int
    _lib.vgre_lm_create.restype = c.c_void_p
    _lib.vgre_lm_free.argtypes = [c.c_void_p]
    _lib.vgre_lm_num_params.argtypes = [c.c_void_p]
    _lib.vgre_lm_num_params.restype = c.c_longlong
    _lib.vgre_lm_set_bf16_inference.argtypes = [c.c_void_p, c.c_int]
    _lib.vgre_lm_set_ternary_inference.argtypes = [c.c_void_p, c.c_int]
    _lib.vgre_lm_set_int8_inference.argtypes = [c.c_void_p, c.c_int]
    _lib.vgre_lm_set_int8_kv_cache.argtypes = [c.c_void_p, c.c_int]
    _lib.vgre_lm_set_int4_kv_cache.argtypes = [c.c_void_p, c.c_int]
    _lib.vgre_lm_set_batched_prefill.argtypes = [c.c_void_p, c.c_int]
    _lib.vgre_lm_drop_fp32_weights.argtypes = [c.c_void_p]
    _lib.vgre_lm_train_step.argtypes = [c.c_void_p, P(c.c_int), P(c.c_int), c.c_int, c.c_float]
    _lib.vgre_lm_train_step.restype = c.c_float
    _lib.vgre_lm_loss.argtypes = [c.c_void_p, P(c.c_int), P(c.c_int), c.c_int]
    _lib.vgre_lm_loss.restype = c.c_float
    _lib.vgre_lm_accumulate.argtypes = [c.c_void_p, P(c.c_int), P(c.c_int), c.c_int, c.c_float]
    _lib.vgre_lm_accumulate.restype = c.c_float
    _lib.vgre_lm_optim_step.argtypes = [c.c_void_p, c.c_float, c.c_float]
    _lib.vgre_lm_generate.argtypes = [c.c_void_p, P(c.c_int), c.c_int, c.c_int,
                                      c.c_float, c.c_int, c.c_float, c.c_float,
                                      c.c_uint, P(c.c_int), c.c_int]
    _lib.vgre_lm_generate.restype = c.c_int
    _lib.vgre_lm_generate_speculative.argtypes = [c.c_void_p, P(c.c_int), c.c_int, c.c_int,
                                                  c.c_int, c.c_uint, P(c.c_int), c.c_int]
    _lib.vgre_lm_generate_speculative.restype = c.c_int
    _lib.vgre_lm_save.argtypes = [c.c_void_p, c.c_char_p]
    _lib.vgre_lm_save.restype = c.c_int
    _lib.vgre_lm_load.argtypes = [c.c_void_p, c.c_char_p]
    _lib.vgre_lm_load.restype = c.c_int
    _lib.vgre_lm_load_llama.argtypes = [c.c_void_p, c.c_char_p]
    _lib.vgre_lm_load_llama.restype = c.c_int
    _lib.vgre_lm_load_gguf.argtypes = [c.c_void_p, c.c_char_p]
    _lib.vgre_lm_load_gguf.restype = c.c_int

    # GGUF Metadata C API bindings
    _lib.vgre_gguf_metadata_open.argtypes = [c.c_char_p]
    _lib.vgre_gguf_metadata_open.restype = c.c_void_p
    _lib.vgre_gguf_metadata_free.argtypes = [c.c_void_p]
    _lib.vgre_gguf_metadata_get_block_count.argtypes = [c.c_void_p, P(c.c_int32)]
    _lib.vgre_gguf_metadata_get_block_count.restype = c.c_int
    _lib.vgre_gguf_metadata_get_embedding_length.argtypes = [c.c_void_p, P(c.c_int32)]
    _lib.vgre_gguf_metadata_get_embedding_length.restype = c.c_int
    _lib.vgre_gguf_metadata_get_head_count.argtypes = [c.c_void_p, P(c.c_int32)]
    _lib.vgre_gguf_metadata_get_head_count.restype = c.c_int
    _lib.vgre_gguf_metadata_get_head_count_kv.argtypes = [c.c_void_p, P(c.c_int32)]
    _lib.vgre_gguf_metadata_get_head_count_kv.restype = c.c_int
    _lib.vgre_gguf_metadata_get_feed_forward_length.argtypes = [c.c_void_p, P(c.c_int32)]
    _lib.vgre_gguf_metadata_get_feed_forward_length.restype = c.c_int
    _lib.vgre_gguf_metadata_get_vocabulary_size.argtypes = [c.c_void_p, P(c.c_int32)]
    _lib.vgre_gguf_metadata_get_vocabulary_size.restype = c.c_int
    _lib.vgre_gguf_metadata_get_context_length.argtypes = [c.c_void_p, P(c.c_int32)]
    _lib.vgre_gguf_metadata_get_context_length.restype = c.c_int
    _lib.vgre_gguf_metadata_get_rope_freq_base.argtypes = [c.c_void_p, P(c.c_float)]
    _lib.vgre_gguf_metadata_get_rope_freq_base.restype = c.c_int
    _lib.vgre_gguf_metadata_get_norm_eps.argtypes = [c.c_void_p, P(c.c_float)]
    _lib.vgre_gguf_metadata_get_norm_eps.restype = c.c_int
    _lib.vgre_gguf_metadata_has_attention_bias.argtypes = [c.c_void_p]
    _lib.vgre_gguf_metadata_has_attention_bias.restype = c.c_int
    _lib.vgre_gguf_metadata_has_output_weight.argtypes = [c.c_void_p]
    _lib.vgre_gguf_metadata_has_output_weight.restype = c.c_int
    _lib.vgre_gguf_metadata_get_architecture.argtypes = [c.c_void_p]
    _lib.vgre_gguf_metadata_get_architecture.restype = c.c_char_p
    _lib.vgre_gguf_metadata_get_chat_template.argtypes = [c.c_void_p]
    _lib.vgre_gguf_metadata_get_chat_template.restype = c.c_char_p
    _lib.vgre_gguf_metadata_is_valid.argtypes = [c.c_void_p]
    _lib.vgre_gguf_metadata_is_valid.restype = c.c_int
    _lib.vgre_gguf_metadata_get_last_error.argtypes = [c.c_void_p]
    _lib.vgre_gguf_metadata_get_last_error.restype = c.c_char_p

    _lib.vgre_cosine_lr.argtypes = [c.c_longlong, c.c_longlong, c.c_longlong, c.c_float, c.c_float]
    _lib.vgre_cosine_lr.restype = c.c_float

    _lib.vgre_bpe_create.restype = c.c_void_p
    _lib.vgre_bpe_free.argtypes = [c.c_void_p]
    _lib.vgre_bpe_train.argtypes = [c.c_void_p, c.c_char_p, c.c_int]
    _lib.vgre_bpe_vocab_size.argtypes = [c.c_void_p]
    _lib.vgre_bpe_vocab_size.restype = c.c_int
    _lib.vgre_bpe_encode.argtypes = [c.c_void_p, c.c_char_p, P(c.c_int), c.c_int]
    _lib.vgre_bpe_encode.restype = c.c_int
    _lib.vgre_bpe_decode.argtypes = [c.c_void_p, P(c.c_int), c.c_int, c.c_char_p, c.c_int]
    _lib.vgre_bpe_decode.restype = c.c_int
    _lib.vgre_bpe_load_hf.argtypes = [c.c_void_p, c.c_char_p]
    _lib.vgre_bpe_load_hf.restype = c.c_int
    _lib.vgre_bpe_load_gpt2.argtypes = [c.c_void_p, c.c_char_p, c.c_char_p]
    _lib.vgre_bpe_load_gpt2.restype = c.c_int
    _lib.vgre_bpe_special_id.argtypes = [c.c_void_p, c.c_char_p]
    _lib.vgre_bpe_special_id.restype = c.c_int
    # Premium surface: reset, structured errors, checked APIs, introspection.
    _lib.vgre_bpe_reset.argtypes = [c.c_void_p]
    _lib.vgre_bpe_last_error.argtypes = [c.c_void_p]
    _lib.vgre_bpe_last_error.restype = c.c_int
    _lib.vgre_bpe_last_error_message.argtypes = [c.c_void_p]
    _lib.vgre_bpe_last_error_message.restype = c.c_char_p
    _lib.vgre_bpe_error_string.argtypes = [c.c_int]
    _lib.vgre_bpe_error_string.restype = c.c_char_p
    _lib.vgre_bpe_validate.argtypes = [c.c_void_p]
    _lib.vgre_bpe_validate.restype = c.c_int
    _lib.vgre_bpe_is_valid_utf8.argtypes = [c.c_char_p, c.c_int, P(c.c_int), P(c.c_int)]
    _lib.vgre_bpe_is_valid_utf8.restype = c.c_int
    _lib.vgre_bpe_encode_checked.argtypes = [c.c_void_p, c.c_char_p, P(c.c_int), c.c_int]
    _lib.vgre_bpe_encode_checked.restype = c.c_int
    _lib.vgre_bpe_decode_checked.argtypes = [c.c_void_p, P(c.c_int), c.c_int, c.c_char_p, c.c_int]
    _lib.vgre_bpe_decode_checked.restype = c.c_int
    _lib.vgre_bpe_added_token_count.argtypes = [c.c_void_p]
    _lib.vgre_bpe_added_token_count.restype = c.c_int
    _lib.vgre_bpe_added_token.argtypes = [c.c_void_p, c.c_int, c.c_char_p, c.c_int,
                                          P(c.c_int), P(c.c_int), P(c.c_int), P(c.c_int),
                                          P(c.c_int), P(c.c_int)]
    _lib.vgre_bpe_added_token.restype = c.c_int
    _bound = True


def _require():
    if not NATIVE_AVAILABLE or _lib is None:
        raise RuntimeError("libvgre not found — build it and set LD_LIBRARY_PATH/VGRE_LIB_PATH")
    _bind()


def cosine_lr(step: int, warmup: int, total: int, base_lr: float, min_lr: float = 0.0) -> float:
    """Cosine learning-rate schedule with linear warmup (matches the C++ trainer)."""
    _require()
    return float(_lib.vgre_cosine_lr(int(step), int(warmup), int(total), float(base_lr), float(min_lr)))


class GGUFMetadataReader:
    """Python interface for GGUF metadata extraction.

    This class provides a Pythonic wrapper around the C API for extracting
    model configuration parameters from GGUF files. It handles automatic
    memory management and converts C errors to Python exceptions.

    Example:
        reader = GGUFMetadataReader("model.gguf")
        params = reader.get_model_parameters()
        print(f"Model has {params.get('n_layer', 'unknown')} layers")
    """

    def __init__(self, path: str):
        """Open a GGUF file for metadata reading.

        Args:
            path: Path to the GGUF file

        Raises:
            RuntimeError: If the file cannot be opened or is invalid
        """
        _require()
        path = os.fspath(path)
        if not os.path.isfile(path):
            if os.path.isdir(path):
                raise IsADirectoryError(f"GGUF model path is a directory: {path}")
            raise FileNotFoundError(f"GGUF model file does not exist: {path}")
        with open(path, 'rb') as gguf_file:
            if gguf_file.read(4) != b'GGUF':
                raise ValueError(f"File is not a GGUF model: {path}")

        self._h = _lib.vgre_gguf_metadata_open(path.encode('utf-8'))
        if not self._h:
            raise ValueError(f"Invalid or truncated GGUF file: {path}")

        if not _lib.vgre_gguf_metadata_is_valid(self._h):
            error_msg = self._get_last_error()
            self.close()
            raise ValueError(f"Invalid GGUF file: {error_msg}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def get_model_parameters(self) -> dict:
        """Extract all available model parameters from GGUF metadata.

        Returns:
            Dictionary containing extracted parameters with keys:
            - n_layer: Number of transformer layers
            - d_model: Embedding/hidden dimension size
            - n_head: Number of attention heads
            - n_kv_head: Number of key-value heads (for GQA)
            - d_ff: Feed-forward dimension
            - rope_theta: RoPE frequency base
            - norm_eps: Layer normalization epsilon
        """
        params = {}

        for param_name, getter_func in self._PARAMETER_GETTERS.items():
            value = getter_func()
            if value is not None:
                params[param_name] = value
            else:
                error = self._get_last_error()
                if error.startswith("Invalid numeric value"):
                    raise ValueError(error)

        params["attn_bias"] = self.has_attention_bias
        params["tie_embeddings"] = not self.has_output_weight
        if self.architecture:
            params["architecture"] = self.architecture
        if self.chat_template:
            params["chat_template"] = self.chat_template

        return params

    def get_block_count(self) -> Optional[int]:
        """Extract n_layer parameter from GGUF metadata.

        Returns:
            Number of transformer layers, or None if not found
        """
        value = ctypes.c_int32()
        result = _lib.vgre_gguf_metadata_get_block_count(self._h, ctypes.byref(value))
        return value.value if result == 1 else None

    def get_embedding_length(self) -> Optional[int]:
        """Extract d_model parameter from GGUF metadata.

        Returns:
            Embedding dimension size, or None if not found
        """
        value = ctypes.c_int32()
        result = _lib.vgre_gguf_metadata_get_embedding_length(self._h, ctypes.byref(value))
        return value.value if result == 1 else None

    def get_head_count(self) -> Optional[int]:
        """Extract n_head parameter from GGUF metadata.

        Returns:
            Number of attention heads, or None if not found
        """
        value = ctypes.c_int32()
        result = _lib.vgre_gguf_metadata_get_head_count(self._h, ctypes.byref(value))
        return value.value if result == 1 else None

    def get_head_count_kv(self) -> Optional[int]:
        """Extract n_kv_head parameter from GGUF metadata.

        Returns:
            Number of key-value heads, or None if not found
        """
        value = ctypes.c_int32()
        result = _lib.vgre_gguf_metadata_get_head_count_kv(self._h, ctypes.byref(value))
        return value.value if result == 1 else None

    def get_feed_forward_length(self) -> Optional[int]:
        """Extract d_ff parameter from GGUF metadata.

        Returns:
            Feed-forward dimension, or None if not found
        """
        value = ctypes.c_int32()
        result = _lib.vgre_gguf_metadata_get_feed_forward_length(self._h, ctypes.byref(value))
        return value.value if result == 1 else None

    def get_vocabulary_size(self) -> Optional[int]:
        value = ctypes.c_int32()
        result = _lib.vgre_gguf_metadata_get_vocabulary_size(self._h, ctypes.byref(value))
        return value.value if result == 1 else None

    def get_context_length(self) -> Optional[int]:
        value = ctypes.c_int32()
        result = _lib.vgre_gguf_metadata_get_context_length(self._h, ctypes.byref(value))
        return value.value if result == 1 else None

    def get_rope_freq_base(self) -> Optional[float]:
        """Extract rope_theta parameter from GGUF metadata.

        Returns:
            RoPE frequency base, or None if not found
        """
        value = ctypes.c_float()
        result = _lib.vgre_gguf_metadata_get_rope_freq_base(self._h, ctypes.byref(value))
        return value.value if result == 1 else None

    def get_norm_eps(self) -> Optional[float]:
        """Extract norm_eps parameter from GGUF metadata.

        Returns:
            Layer normalization epsilon, or None if not found
        """
        value = ctypes.c_float()
        result = _lib.vgre_gguf_metadata_get_norm_eps(self._h, ctypes.byref(value))
        return value.value if result == 1 else None

    @property
    def has_attention_bias(self) -> bool:
        return _lib.vgre_gguf_metadata_has_attention_bias(self._h) == 1

    @property
    def has_output_weight(self) -> bool:
        return _lib.vgre_gguf_metadata_has_output_weight(self._h) == 1

    @property
    def architecture(self) -> Optional[str]:
        value = _lib.vgre_gguf_metadata_get_architecture(self._h)
        return value.decode("utf-8") if value else None

    @property
    def chat_template(self) -> Optional[str]:
        value = _lib.vgre_gguf_metadata_get_chat_template(self._h)
        return value.decode("utf-8") if value else None

    def _get_last_error(self) -> str:
        """Get the last error message from the C API.

        Returns:
            Error message string, or empty string if no error
        """
        error_ptr = _lib.vgre_gguf_metadata_get_last_error(self._h)
        return error_ptr.decode('utf-8') if error_ptr else ""

    def close(self):
        """Release resources associated with this metadata reader."""
        if getattr(self, '_h', None):
            _lib.vgre_gguf_metadata_free(self._h)
            self._h = None

    def __del__(self):
        self.close()

    # Parameter getter mapping for convenience
    @property
    def _PARAMETER_GETTERS(self):
        return {
            'n_layer': self.get_block_count,
            'd_model': self.get_embedding_length,
            'n_head': self.get_head_count,
            'n_kv_head': self.get_head_count_kv,
            'd_ff': self.get_feed_forward_length,
            'vocab': self.get_vocabulary_size,
            'context_length': self.get_context_length,
            'rope_theta': self.get_rope_freq_base,
            'norm_eps': self.get_norm_eps,
        }


class Tokenizer:
    """A from-scratch byte-level BPE tokenizer (trainable on any text)."""

    def __init__(self) -> None:
        _require()
        self._h = _lib.vgre_bpe_create()
        if not self._h:
            raise RuntimeError("vgre_bpe_create failed")

    def train(self, corpus: str, num_merges: int = 512) -> "Tokenizer":
        _lib.vgre_bpe_train(self._h, corpus.encode("utf-8"), int(num_merges))
        return self

    def load_hf(self, tokenizer_json_path: str) -> "Tokenizer":
        """Load a Hugging Face `tokenizer.json` (byte-level BPE family: GPT-2/
        Whisper, Llama-3, Qwen, Phi, DeepSeek, …) incl. special tokens; encode/
        decode then emit/consume that model's exact ids."""
        if not _lib.vgre_bpe_load_hf(self._h, tokenizer_json_path.encode("utf-8")):
            raise RuntimeError(f"could not load tokenizer.json from {tokenizer_json_path}")
        return self

    def load_gpt2(self, vocab_json_path: str, merges_txt_path: str) -> "Tokenizer":
        """Load the two-file GPT-2 tokenizer form (vocab.json + merges.txt)."""
        if not _lib.vgre_bpe_load_gpt2(self._h, vocab_json_path.encode("utf-8"),
                                       merges_txt_path.encode("utf-8")):
            raise RuntimeError("could not load GPT-2 vocab.json/merges.txt")
        return self

    def special_id(self, content: str) -> int:
        """Id of an added/special token (e.g. '<|im_end|>'); -1 if absent."""
        return int(_lib.vgre_bpe_special_id(self._h, content.encode("utf-8")))

    @property
    def vocab_size(self) -> int:
        return int(_lib.vgre_bpe_vocab_size(self._h))

    def encode(self, text: str, max_tokens: int = 1 << 20) -> List[int]:
        buf = (ctypes.c_int * max_tokens)()
        n = _lib.vgre_bpe_encode(self._h, text.encode("utf-8"), buf, max_tokens)
        if n < 0:
            raise RuntimeError("encode failed")
        return list(buf[:n])

    def decode(self, ids: List[int], max_bytes: int = 1 << 20) -> str:
        arr = (ctypes.c_int * len(ids))(*ids)
        out = ctypes.create_string_buffer(max_bytes)
        n = _lib.vgre_bpe_decode(self._h, arr, len(ids), out, max_bytes)
        if n < 0:
            raise RuntimeError("decode failed")
        return out.value.decode("utf-8", errors="replace")

    # ── Premium surface ─────────────────────────────────────────────────────
    def reset(self) -> "Tokenizer":
        """Return to the pristine 256-byte base state (no merges/model/specials),
        so this instance can load a different model with zero carryover."""
        _lib.vgre_bpe_reset(self._h)
        return self

    @property
    def last_error(self) -> int:
        """Integer code of the most recent operation (0 == ok)."""
        return int(_lib.vgre_bpe_last_error(self._h))

    @property
    def last_error_message(self) -> str:
        msg = _lib.vgre_bpe_last_error_message(self._h)
        return msg.decode("utf-8", errors="replace") if msg else ""

    @staticmethod
    def error_string(code: int) -> str:
        """Stable, human-readable name for an error code."""
        _require()
        s = _lib.vgre_bpe_error_string(int(code))
        return s.decode("utf-8", errors="replace") if s else "unknown error"

    def validate(self) -> bool:
        """Check the loaded state is self-consistent (see last_error on failure)."""
        return bool(_lib.vgre_bpe_validate(self._h))

    @staticmethod
    def is_valid_utf8(data) -> bool:
        """Strict RFC 3629 UTF-8 validation (rejects overlong forms, surrogate
        codepoints and values above U+10FFFF). Accepts str or bytes."""
        _require()
        raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        return bool(_lib.vgre_bpe_is_valid_utf8(raw, len(raw), None, None))

    def encode_checked(self, text: str, max_tokens: int = 1 << 20) -> List[int]:
        """Like encode(), but validates UTF-8 and guarantees no byte is lost to
        the model vocabulary; raises ValueError(last_error_message) on failure."""
        buf = (ctypes.c_int * max_tokens)()
        n = _lib.vgre_bpe_encode_checked(self._h, text.encode("utf-8"), buf, max_tokens)
        if n < 0:
            raise ValueError(self.last_error_message or "encode_checked failed")
        return list(buf[:n])

    def decode_checked(self, ids: List[int], max_bytes: int = 1 << 20) -> str:
        """Like decode(), but every id must be resolvable; raises ValueError on
        an invalid id instead of silently skipping it."""
        arr = (ctypes.c_int * len(ids))(*ids)
        out = ctypes.create_string_buffer(max_bytes)
        n = _lib.vgre_bpe_decode_checked(self._h, arr, len(ids), out, max_bytes)
        if n < 0:
            raise ValueError(self.last_error_message or "decode_checked failed")
        return out.value.decode("utf-8", errors="replace")

    def added_tokens(self) -> List[dict]:
        """Rich metadata for every HF added/special token (after load_hf)."""
        count = int(_lib.vgre_bpe_added_token_count(self._h))
        result: List[dict] = []
        for i in range(count):
            content = ctypes.create_string_buffer(256)
            tid = ctypes.c_int(0)
            special = ctypes.c_int(0)
            lstrip = ctypes.c_int(0)
            rstrip = ctypes.c_int(0)
            single_word = ctypes.c_int(0)
            normalized = ctypes.c_int(0)
            if _lib.vgre_bpe_added_token(self._h, i, content, 256,
                                         ctypes.byref(tid), ctypes.byref(special),
                                         ctypes.byref(lstrip), ctypes.byref(rstrip),
                                         ctypes.byref(single_word), ctypes.byref(normalized)):
                result.append({
                    "content": content.value.decode("utf-8", errors="replace"),
                    "id": tid.value,
                    "special": bool(special.value),
                    "lstrip": bool(lstrip.value),
                    "rstrip": bool(rstrip.value),
                    "single_word": bool(single_word.value),
                    "normalized": bool(normalized.value),
                })
        return result

    def close(self) -> None:
        if getattr(self, "_h", None):
            _lib.vgre_bpe_free(self._h)
            self._h = None

    def __del__(self):
        self.close()


class LanguageModel:
    """An in-tree Llama-style decoder LM (RMSNorm, RoPE, causal attention, SwiGLU)."""

    def __init__(self, vocab: int, n_layer: int = 4, d_model: int = 256,
                 n_head: int = 8, d_ff: int = 0, max_seq: int = 256,
                 dropout: float = 0.0, tie_embeddings: bool = False,
                 seed: int = 1234, n_kv_head: int = 0, attn_bias: bool = False,
                 # New parameters for enhanced model support
                 rope_theta: float = 10000.0, norm_eps: float = 1e-5) -> None:
        _require()
        self.vocab = int(vocab)
        self._gguf_chat_template = None
        self._rope_theta = rope_theta
        self._norm_eps = norm_eps

        # n_kv_head < n_head → grouped-query attention (smaller K/V + KV cache);
        # 0 or == n_head → plain multi-head attention. attn_bias → Qwen2-style
        # Q/K/V projection biases.
        self._h = _lib.vgre_lm_create_gqa(int(vocab), int(n_layer), int(d_model),
                                          int(n_head), int(n_kv_head), int(d_ff), int(max_seq),
                                          float(dropout), 1 if tie_embeddings else 0,
                                          int(seed) & 0xFFFFFFFF, 1 if attn_bias else 0)
        if not self._h:
            raise RuntimeError("vgre_lm_create failed (check d_model % n_head == 0 and head_dim even)")
        if not _lib.vgre_lm_set_rope_norm(self._h, float(rope_theta), float(norm_eps)):
            _lib.vgre_lm_free(self._h)
            self._h = None
            raise ValueError("rope_theta and norm_eps must be finite and positive")

    @property
    def num_parameters(self) -> int:
        return int(_lib.vgre_lm_num_params(self._h))

    @property
    def rope_theta(self) -> float:
        """Rotational Position Embedding frequency base."""
        return self._rope_theta

    @property
    def norm_eps(self) -> float:
        """Layer normalization epsilon parameter."""
        return self._norm_eps

    def set_bf16_inference(self, on: bool = True) -> None:
        """Run generation on bf16-cached weights (half the matmul-weight
        footprint/bandwidth, fp32 accumulation). Training stays fp32."""
        _lib.vgre_lm_set_bf16_inference(self._h, 1 if on else 0)

    def set_int8_inference(self, on: bool = True) -> None:
        """Run generation on weight-only int8 weights (~4× smaller, per-channel
        scale, fp32 accumulation). Mutually exclusive with bf16; training stays
        fp32."""
        _lib.vgre_lm_set_int8_inference(self._h, 1 if on else 0)

    def set_ternary_inference(self, on: bool = True) -> None:
        """Use packed 2-bit ternary weights for faster inference at reduced quality."""
        _lib.vgre_lm_set_ternary_inference(self._h, 1 if on else 0)

    def set_int8_kv_cache(self, on: bool = True) -> None:
        """Store the generation KV cache as int8 with a per-(position, head)
        absmax scale instead of fp32 (~3.8x less KV memory at Dh=64). At long
        context the KV cache, not the weights, dominates footprint. Affects
        generation only; weights and activations are untouched."""
        _lib.vgre_lm_set_int8_kv_cache(self._h, 1 if on else 0)

    def set_int4_kv_cache(self, on: bool = True) -> None:
        """Store the generation KV cache as 4-bit packed (2 codes/byte) with a
        per-(position, head) absmax scale — ~7x less KV memory than fp32, a
        further ~2x over int8. Coarser than int8: a memory/quality tradeoff
        (takes precedence over int8 when both are enabled)."""
        _lib.vgre_lm_set_int4_kv_cache(self._h, 1 if on else 0)

    def set_batched_prefill(self, on: bool = True) -> None:
        """Batched prompt prefill (one GEMM per projection over the whole prompt
        instead of a per-token GEMV) — on by default; bit-identical to sequential
        prefill. Turn off for strict per-token determinism/debugging."""
        _lib.vgre_lm_set_batched_prefill(self._h, 1 if on else 0)

    def drop_fp32_weights(self) -> None:
        """Free the fp32 master weights after enabling bf16/int8 inference, so the
        resident footprint truly drops to ½×/¼×. SERVE-ONLY afterwards (no more
        training)."""
        _lib.vgre_lm_drop_fp32_weights(self._h)

    def train_step(self, ids: List[int], targets: List[int], lr: float = 3e-3) -> float:
        if len(ids) != len(targets):
            raise ValueError("ids and targets must be the same length")
        T = len(ids)
        a = (ctypes.c_int * T)(*ids)
        b = (ctypes.c_int * T)(*targets)
        loss = _lib.vgre_lm_train_step(self._h, a, b, T, float(lr))
        if loss < 0:
            raise RuntimeError("train_step failed")
        return float(loss)

    def train_batch(self, batch, lr: float, clip: float = 1.0) -> float:
        """Mini-batch AdamW step: accumulate gradients over a batch of
        (ids, targets) sequences (each loss averaged by 1/len(batch)), then one
        clipped optimizer update. Returns the mean batch loss. More stable than
        the single-sequence train_step."""
        if not batch:
            return 0.0
        inv = 1.0 / len(batch)
        total = 0.0
        for ids, tgt in batch:
            if len(ids) != len(tgt):
                raise ValueError("ids and targets must be the same length")
            T = len(ids)
            a = (ctypes.c_int * T)(*ids)
            b = (ctypes.c_int * T)(*tgt)
            l = _lib.vgre_lm_accumulate(self._h, a, b, T, ctypes.c_float(inv))
            if l < 0:
                raise RuntimeError("accumulate failed")
            total += l
        _lib.vgre_lm_optim_step(self._h, ctypes.c_float(lr), ctypes.c_float(clip))
        return total * inv

    def loss(self, ids: List[int], targets: List[int]) -> float:
        """Forward-only cross-entropy loss (no gradient step) — for validation."""
        if len(ids) != len(targets):
            raise ValueError("ids and targets must be the same length")
        T = len(ids)
        a = (ctypes.c_int * T)(*ids)
        b = (ctypes.c_int * T)(*targets)
        v = _lib.vgre_lm_loss(self._h, a, b, T)
        if v < 0:
            raise RuntimeError("loss failed")
        return float(v)

    def generate(self, prompt: List[int], n_new: int = 32, temperature: float = 0.0,
                 top_k: int = 0, top_p: float = 1.0, repetition_penalty: float = 1.0,
                 seed: int = 0) -> List[int]:
        """KV-cached autoregressive generation.

        temperature<=0 → greedy; top_k>0 keeps the top-k logits; top_p<1 applies
        nucleus (cumulative-probability) sampling; repetition_penalty>1 discourages
        repeating already-emitted tokens.
        """
        cap = len(prompt) + n_new + 8
        p = (ctypes.c_int * len(prompt))(*prompt)
        out = (ctypes.c_int * cap)()
        n = _lib.vgre_lm_generate(self._h, p, len(prompt), int(n_new),
                                  float(temperature), int(top_k), float(top_p),
                                  float(repetition_penalty), int(seed) & 0xFFFFFFFF, out, cap)
        if n < 0:
            raise RuntimeError("generate failed")
        return list(out[:n])

    def generate_speculative(self, prompt: List[int], n_new: int = 32,
                             draft_k: int = 8, seed: int = 0) -> List[int]:
        """Lossless greedy speculative decoding: a prompt-lookup drafter proposes
        up to ``draft_k`` tokens and one batched forward verifies them, so a run
        of correct guesses costs a single forward pass. The output is identical to
        ``generate(...)`` with greedy sampling — only faster on repetitive output.
        """
        cap = len(prompt) + n_new + 8
        p = (ctypes.c_int * len(prompt))(*prompt)
        out = (ctypes.c_int * cap)()
        n = _lib.vgre_lm_generate_speculative(self._h, p, len(prompt), int(n_new),
                                              int(draft_k), int(seed) & 0xFFFFFFFF, out, cap)
        if n < 0:
            raise RuntimeError("generate_speculative failed")
        return list(out[:n])

    def save(self, path: str) -> None:
        if not _lib.vgre_lm_save(self._h, str(path).encode("utf-8")):
            raise RuntimeError("save failed")

    def load(self, path: str) -> None:
        if not _lib.vgre_lm_load(self._h, str(path).encode("utf-8")):
            raise RuntimeError("load failed (config must match the checkpoint)")

    def load_llama(self, path: str) -> None:
        """Load a Hugging Face Llama-family safetensors checkpoint. Create this
        model with the checkpoint's dims first (vocab, n_layer, d_model, n_head,
        d_ff, tie_embeddings — from its config.json). Handles the HF weight
        transpose, RoPE convention, and grouped-query attention."""
        if not _lib.vgre_lm_load_llama(self._h, str(path).encode("utf-8")):
            raise RuntimeError("load_llama failed (config mismatch or missing tensor)")

    def load_gguf(self, path: str) -> None:
        """Load a llama.cpp GGUF checkpoint (quantized tensors are dequantized to
        f32). Create this model with the checkpoint's dims first. Handles the GGUF
        transpose and grouped-query attention (GGUF already bakes in the RoPE
        permute)."""
        if not _lib.vgre_lm_load_gguf(self._h, str(path).encode("utf-8")):
            raise RuntimeError("load_gguf failed (config mismatch or missing tensor)")

    @classmethod
    def from_gguf(cls, gguf_path: str,
                  json_sidecar_path: Optional[str] = None) -> 'LanguageModel':
        """Factory method for auto-configuration from GGUF files.

        Creates a LanguageModel instance automatically configured from GGUF metadata,
        eliminating the need for manual parameter specification or JSON sidecars.

        Args:
            gguf_path: Path to the GGUF file
            json_sidecar_path: Optional path to JSON sidecar (overrides GGUF metadata)

        Returns:
            Configured LanguageModel with weights loaded from GGUF

        Example:
            # Simple auto-configuration
            model = LanguageModel.from_gguf("llama-3-8b.gguf")

            # With JSON sidecar override
            model = LanguageModel.from_gguf("model.gguf", "config.json")
        """
        from .auto_config import ModelAutoConfigurator
        return ModelAutoConfigurator.from_gguf(gguf_path, json_sidecar_path)

    def close(self) -> None:
        if getattr(self, "_h", None):
            _lib.vgre_lm_free(self._h)
            self._h = None

    def __del__(self):
        self.close()
