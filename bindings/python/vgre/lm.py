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
from typing import List, Optional

from ._native import _lib, NATIVE_AVAILABLE

_bound = False


def _bind() -> None:
    """Declare argtypes/restypes once (idempotent)."""
    global _bound
    if _bound or _lib is None:
        return
    c, P = ctypes, ctypes.POINTER

    _lib.vgre_lm_create.argtypes = [c.c_int] * 6 + [c.c_uint]
    _lib.vgre_lm_create.restype = c.c_void_p
    _lib.vgre_lm_free.argtypes = [c.c_void_p]
    _lib.vgre_lm_num_params.argtypes = [c.c_void_p]
    _lib.vgre_lm_num_params.restype = c.c_longlong
    _lib.vgre_lm_set_bf16_inference.argtypes = [c.c_void_p, c.c_int]
    _lib.vgre_lm_set_int8_inference.argtypes = [c.c_void_p, c.c_int]
    _lib.vgre_lm_train_step.argtypes = [c.c_void_p, P(c.c_int), P(c.c_int), c.c_int, c.c_float]
    _lib.vgre_lm_train_step.restype = c.c_float
    _lib.vgre_lm_loss.argtypes = [c.c_void_p, P(c.c_int), P(c.c_int), c.c_int]
    _lib.vgre_lm_loss.restype = c.c_float
    _lib.vgre_lm_generate.argtypes = [c.c_void_p, P(c.c_int), c.c_int, c.c_int,
                                      c.c_float, c.c_int, c.c_float, c.c_float,
                                      c.c_uint, P(c.c_int), c.c_int]
    _lib.vgre_lm_generate.restype = c.c_int
    _lib.vgre_lm_save.argtypes = [c.c_void_p, c.c_char_p]
    _lib.vgre_lm_save.restype = c.c_int
    _lib.vgre_lm_load.argtypes = [c.c_void_p, c.c_char_p]
    _lib.vgre_lm_load.restype = c.c_int

    _lib.vgre_bpe_create.restype = c.c_void_p
    _lib.vgre_bpe_free.argtypes = [c.c_void_p]
    _lib.vgre_bpe_train.argtypes = [c.c_void_p, c.c_char_p, c.c_int]
    _lib.vgre_bpe_vocab_size.argtypes = [c.c_void_p]
    _lib.vgre_bpe_vocab_size.restype = c.c_int
    _lib.vgre_bpe_encode.argtypes = [c.c_void_p, c.c_char_p, P(c.c_int), c.c_int]
    _lib.vgre_bpe_encode.restype = c.c_int
    _lib.vgre_bpe_decode.argtypes = [c.c_void_p, P(c.c_int), c.c_int, c.c_char_p, c.c_int]
    _lib.vgre_bpe_decode.restype = c.c_int
    _bound = True


def _require():
    if not NATIVE_AVAILABLE or _lib is None:
        raise RuntimeError("libvgre not found — build it and set LD_LIBRARY_PATH/VGRE_LIB_PATH")
    _bind()


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

    def close(self) -> None:
        if getattr(self, "_h", None):
            _lib.vgre_bpe_free(self._h)
            self._h = None

    def __del__(self):
        self.close()


class LanguageModel:
    """An in-tree Llama-style decoder LM (RMSNorm, RoPE, causal attention, SwiGLU)."""

    def __init__(self, vocab: int, n_layer: int = 4, d_model: int = 256,
                 n_head: int = 8, d_ff: int = 0, max_seq: int = 256, seed: int = 1234) -> None:
        _require()
        self._h = _lib.vgre_lm_create(int(vocab), int(n_layer), int(d_model),
                                      int(n_head), int(d_ff), int(max_seq), int(seed) & 0xFFFFFFFF)
        if not self._h:
            raise RuntimeError("vgre_lm_create failed (check d_model % n_head == 0 and head_dim even)")

    @property
    def num_parameters(self) -> int:
        return int(_lib.vgre_lm_num_params(self._h))

    def set_bf16_inference(self, on: bool = True) -> None:
        """Run generation on bf16-cached weights (half the matmul-weight
        footprint/bandwidth, fp32 accumulation). Training stays fp32."""
        _lib.vgre_lm_set_bf16_inference(self._h, 1 if on else 0)

    def set_int8_inference(self, on: bool = True) -> None:
        """Run generation on weight-only int8 weights (~4× smaller, per-channel
        scale, fp32 accumulation). Mutually exclusive with bf16; training stays
        fp32."""
        _lib.vgre_lm_set_int8_inference(self._h, 1 if on else 0)

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

    def save(self, path: str) -> None:
        if not _lib.vgre_lm_save(self._h, str(path).encode("utf-8")):
            raise RuntimeError("save failed")

    def load(self, path: str) -> None:
        if not _lib.vgre_lm_load(self._h, str(path).encode("utf-8")):
            raise RuntimeError("load failed (config must match the checkpoint)")

    def close(self) -> None:
        if getattr(self, "_h", None):
            _lib.vgre_lm_free(self._h)
            self._h = None

    def __del__(self):
        self.close()
