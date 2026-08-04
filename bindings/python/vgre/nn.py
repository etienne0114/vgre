"""
vgre.nn — a tiny PyTorch-like autograd framework on VGRE's CPU engine.

Build and train arbitrary models (MLPs, CNNs, …) from pure Python — reverse-mode
autodiff, conv/pool/norm/activation ops, and AdamW — with no GPU and no external
ML library. Tensors wrap VGRE-Autograd nodes via the C ABI in
include/vgre/xla/autograd_c_api.h.

    import numpy as np, vgre.nn as nn
    W = nn.tensor(np.random.randn(4, 3) * 0.1, requires_grad=True)
    b = nn.tensor(np.zeros(3), requires_grad=True)
    opt = nn.AdamW([W, b], lr=1e-2)
    for step in range(200):
        opt.zero_grad()
        logits = nn.matmul(x, W) + b           # x: nn.Tensor [N,4]
        loss = nn.softmax_cross_entropy(logits, targets)
        loss.backward()
        opt.step()
"""
import ctypes
import json
import os
import struct
from typing import List, Sequence, Tuple

import numpy as np

from ._native import _lib, NATIVE_AVAILABLE

_bound = False

# Builder callback type for gradient checkpointing:
#   vgre_ag (*)(vgre_ag* inputs, int n, void* user)
_BUILDER = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
                            ctypes.c_int, ctypes.c_void_p)
# Keep checkpoint callbacks alive from forward through backward (they are invoked
# during backward to recompute). Tensor.backward() clears this afterwards.
_CKPT_CBS: list = []


def _bind() -> None:
    global _bound
    if _bound or _lib is None:
        return
    c, P, V = ctypes, ctypes.POINTER, ctypes.c_void_p
    I64 = c.c_int64
    _lib.vgre_ag_last_error.argtypes = []; _lib.vgre_ag_last_error.restype = c.c_char_p
    _lib.vgre_ag_clear_error.argtypes = []
    _lib.vgre_ag_tensor.argtypes = [P(I64), c.c_int, P(c.c_float), c.c_int]; _lib.vgre_ag_tensor.restype = V
    _lib.vgre_ag_free.argtypes = [V]
    _lib.vgre_ag_size.argtypes = [V]; _lib.vgre_ag_size.restype = I64
    _lib.vgre_ag_ndim.argtypes = [V]; _lib.vgre_ag_ndim.restype = c.c_int
    _lib.vgre_ag_shape.argtypes = [V, P(I64)]
    _lib.vgre_ag_get_data.argtypes = [V, P(c.c_float)]
    _lib.vgre_ag_set_data.argtypes = [V, P(c.c_float)]
    _lib.vgre_ag_get_grad.argtypes = [V, P(c.c_float)]
    for name in ("matmul", "add", "mul", "linear_tied", "bmm"):
        f = getattr(_lib, "vgre_ag_" + name); f.argtypes = [V, V]; f.restype = V
    _lib.vgre_ag_scale.argtypes = [V, c.c_float]; _lib.vgre_ag_scale.restype = V
    _lib.vgre_ag_dropout.argtypes = [V, c.c_float]; _lib.vgre_ag_dropout.restype = V
    _lib.vgre_ag_ternary_quantize.argtypes = [V]; _lib.vgre_ag_ternary_quantize.restype = V
    for name in ("relu", "gelu", "silu", "sigmoid", "tanh", "mean", "softmax", "transpose",
                 "all_reduce", "exp", "softplus"):
        f = getattr(_lib, "vgre_ag_" + name); f.argtypes = [V]; f.restype = V
    _lib.vgre_ag_concat.argtypes = [V, V, c.c_int]; _lib.vgre_ag_concat.restype = V
    _lib.vgre_ag_reshape.argtypes = [V, P(I64), c.c_int]; _lib.vgre_ag_reshape.restype = V
    _lib.vgre_ag_softmax_cross_entropy.argtypes = [V, P(c.c_int), c.c_int]; _lib.vgre_ag_softmax_cross_entropy.restype = V
    _lib.vgre_ag_layer_norm.argtypes = [V, V, V, c.c_float]; _lib.vgre_ag_layer_norm.restype = V
    _lib.vgre_ag_rms_norm.argtypes = [V, V, c.c_float]; _lib.vgre_ag_rms_norm.restype = V
    _lib.vgre_ag_embedding.argtypes = [V, P(c.c_int), c.c_int]; _lib.vgre_ag_embedding.restype = V
    _lib.vgre_ag_index_select.argtypes = [V, P(c.c_int), c.c_int]; _lib.vgre_ag_index_select.restype = V
    _lib.vgre_ag_index_add.argtypes = [c.c_longlong, V, P(c.c_int), c.c_int]; _lib.vgre_ag_index_add.restype = V
    _lib.vgre_ag_selective_scan.argtypes = [V, V]; _lib.vgre_ag_selective_scan.restype = V
    _lib.vgre_ag_rope.argtypes = [V, c.c_int, c.c_float]; _lib.vgre_ag_rope.restype = V
    _lib.vgre_ag_attention.argtypes = [V, V, V, c.c_int, c.c_int]; _lib.vgre_ag_attention.restype = V
    _lib.vgre_ag_flash_attention.argtypes = [V, V, V, c.c_int, c.c_int]; _lib.vgre_ag_flash_attention.restype = V
    _lib.vgre_ag_conv2d.argtypes = [V, V, V, c.c_int, c.c_int]; _lib.vgre_ag_conv2d.restype = V
    _lib.vgre_ag_max_pool2d.argtypes = [V, c.c_int, c.c_int]; _lib.vgre_ag_max_pool2d.restype = V
    _lib.vgre_ag_avg_pool2d.argtypes = [V, c.c_int, c.c_int]; _lib.vgre_ag_avg_pool2d.restype = V
    _lib.vgre_ag_batch_norm2d.argtypes = [V, V, V, P(c.c_float), P(c.c_float),
                                          c.c_int, c.c_int, c.c_float, c.c_float]
    _lib.vgre_ag_batch_norm2d.restype = V
    _lib.vgre_ag_backward.argtypes = [V]
    _lib.vgre_ag_checkpoint.argtypes = [_BUILDER, V, P(V), c.c_int]; _lib.vgre_ag_checkpoint.restype = V
    _lib.vgre_ag_all_reduce_grads.argtypes = [P(V), c.c_int]
    _lib.vgre_cluster_world_size.argtypes = []; _lib.vgre_cluster_world_size.restype = c.c_int
    _lib.vgre_ag_adamw.argtypes = [P(V), c.c_int, c.c_float, c.c_float]; _lib.vgre_ag_adamw.restype = V
    _lib.vgre_ag_sgd.argtypes = [P(V), c.c_int, c.c_float, c.c_float, c.c_float]; _lib.vgre_ag_sgd.restype = V
    _lib.vgre_ag_opt_set_lr.argtypes = [V, c.c_float]
    _lib.vgre_ag_opt_step.argtypes = [V, c.c_float]
    _lib.vgre_ag_opt_zero_grad.argtypes = [V]
    _lib.vgre_ag_opt_free.argtypes = [V]
    _bound = True


def _require():
    if not NATIVE_AVAILABLE or _lib is None:
        raise RuntimeError("libvgre not found — build it and set LD_LIBRARY_PATH/VGRE_LIB_PATH")
    _bind()


def _f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


class Tensor:
    """An autograd tensor. Wrap data with nn.tensor(); ops return new Tensors."""

    __slots__ = ("_h", "_shape", "_owns")

    def __init__(self, handle, shape, owns: bool = True):
        self._h = handle
        self._shape = tuple(int(s) for s in shape)
        self._owns = owns

    @classmethod
    def _from_handle(cls, handle, owns: bool = True) -> "Tensor":
        """Wrap an existing handle, querying its shape from the engine."""
        nd = _lib.vgre_ag_ndim(handle)
        buf = (ctypes.c_int64 * max(nd, 1))()
        _lib.vgre_ag_shape(handle, buf)
        return cls(handle, tuple(buf[i] for i in range(nd)), owns=owns)

    def _release(self):
        """Relinquish ownership of the handle (caller/engine now owns it)."""
        self._h = None

    @property
    def shape(self):
        return self._shape

    @property
    def size(self) -> int:
        return int(np.prod(self._shape)) if self._shape else 1

    def numpy(self) -> np.ndarray:
        out = np.empty(self.size, dtype=np.float32)
        _lib.vgre_ag_get_data(self._h, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return out.reshape(self._shape)

    def grad(self) -> np.ndarray:
        out = np.empty(self.size, dtype=np.float32)
        _lib.vgre_ag_get_grad(self._h, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return out.reshape(self._shape)

    def set_(self, data) -> None:
        d = _f32(data).reshape(-1)
        _lib.vgre_ag_set_data(self._h, d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    def item(self) -> float:
        return float(self.numpy().reshape(-1)[0])

    def backward(self) -> None:
        # backward() returns void; the C ABI clears the error channel on entry,
        # so a non-empty message afterwards means the backward pass actually
        # failed (e.g. a shape mismatch) instead of silently leaving stale grads.
        _lib.vgre_ag_backward(self._h)
        # checkpoint callbacks are invoked during backward; release them after.
        _CKPT_CBS.clear()
        err = _last_error()
        if err:
            raise RuntimeError(f"backward failed: {err}")

    # operator overloads
    def __matmul__(self, other): return matmul(self, other)
    def __add__(self, other):    return add(self, other)
    def __mul__(self, other):
        return scale(self, float(other)) if isinstance(other, (int, float)) else mul(self, other)
    __rmul__ = __mul__

    def __del__(self):
        h = getattr(self, "_h", None)
        if h and getattr(self, "_owns", True) and _lib is not None:
            _lib.vgre_ag_free(h)
            self._h = None


def _last_error() -> str:
    """Message from the last failed native op on this thread ('' if none)."""
    if _lib is None:
        return ""
    msg = _lib.vgre_ag_last_error()
    return msg.decode() if msg else ""


def tensor(data, requires_grad: bool = False) -> Tensor:
    _require()
    a = _f32(data)
    shp = (ctypes.c_int64 * a.ndim)(*a.shape)
    h = _lib.vgre_ag_tensor(shp, a.ndim, a.reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            1 if requires_grad else 0)
    if not h:
        raise RuntimeError(f"tensor creation failed: {_last_error() or 'unknown error'}")
    return Tensor(h, a.shape)


def _new(h, shape) -> Tensor:
    if not h:
        raise RuntimeError(f"op failed: {_last_error() or 'unknown error'}")
    return Tensor(h, shape)


# ── functional ops ───────────────────────────────────────────────────────────
def matmul(a: Tensor, b: Tensor) -> Tensor:
    return _new(_lib.vgre_ag_matmul(a._h, b._h), (a.shape[0], b.shape[1]))

def linear_tied(x: Tensor, w: Tensor) -> Tensor:
    """x[M,D] · Wᵀ where W is [V,D] (weight tying)."""
    return _new(_lib.vgre_ag_linear_tied(x._h, w._h), (x.shape[0], w.shape[0]))

def bmm(a: Tensor, b: Tensor) -> Tensor:
    """Batched matmul a[B,M,K] · b[B,K,N] -> [B,M,N]."""
    return _new(_lib.vgre_ag_bmm(a._h, b._h), (a.shape[0], a.shape[1], b.shape[2]))


def world_size() -> int:
    """Number of cluster nodes participating in collectives (1 if standalone)."""
    _require()
    return int(_lib.vgre_cluster_world_size())


def all_reduce_gradients(params: List["Tensor"]) -> None:
    """Distributed data-parallel gradient sync: average each parameter's gradient
    across all cluster nodes (sum-all-reduce over TCP/RDMA, then ÷ world_size).
    Call after backward(), before optimizer.step(). Single-node → no-op. This is
    what lets a model train across a CPU cluster: each node runs its own data
    shard, then this averages the gradients so every node stays in sync."""
    _require()
    arr = (ctypes.c_void_p * len(params))(*[t._h for t in params])
    _lib.vgre_ag_all_reduce_grads(arr, len(params))


def checkpoint(fn, inputs: List["Tensor"]) -> "Tensor":
    """Gradient checkpointing: run fn(inputs) but drop its intermediate
    activations, recomputing them during backward (saves memory, costs one extra
    forward). fn takes a list of Tensors and returns one Tensor. Output and
    gradients are identical to fn(inputs)."""
    _require()

    def _cb(handles, n, user):
        ins = [Tensor._from_handle(handles[i], owns=False) for i in range(n)]
        out = fn(ins)
        h = out._h
        out._release()          # hand the output handle to the engine
        return h

    cb = _BUILDER(_cb)
    _CKPT_CBS.append(cb)        # keep alive until backward() clears it
    arr = (ctypes.c_void_p * len(inputs))(*[t._h for t in inputs])
    h = _lib.vgre_ag_checkpoint(cb, None, arr, len(inputs))
    if not h:
        raise RuntimeError("checkpoint failed")
    return Tensor._from_handle(h)

def dropout(x: Tensor, p: float) -> Tensor:
    """Inverted dropout (stochastic; training only)."""
    return _new(_lib.vgre_ag_dropout(x._h, float(p)), x.shape)

def ternary_quantize(w: Tensor) -> Tensor:
    """BitNet b1.58 straight-through ternary quantization of a 2-D weight [K,N].
    Forward returns the dequantized ternary weight (per-column absmean); backward
    is the identity (STE), so gradients train the fp master weight. Composing it
    as matmul(x, ternary_quantize(W)) is quantization-aware training whose forward
    matches the inference-time multiplication-free ternary GEMM."""
    return _new(_lib.vgre_ag_ternary_quantize(w._h), w.shape)

def add(a: Tensor, b: Tensor) -> Tensor:
    return _new(_lib.vgre_ag_add(a._h, b._h), a.shape)

def mul(a: Tensor, b: Tensor) -> Tensor:
    return _new(_lib.vgre_ag_mul(a._h, b._h), a.shape)

def scale(a: Tensor, s: float) -> Tensor:
    return _new(_lib.vgre_ag_scale(a._h, ctypes.c_float(s)), a.shape)

def _unary(name, x):
    return _new(getattr(_lib, "vgre_ag_" + name)(x._h), x.shape)

def relu(x): return _unary("relu", x)
def gelu(x): return _unary("gelu", x)
def silu(x): return _unary("silu", x)
def sigmoid(x): return _unary("sigmoid", x)
def tanh(x): return _unary("tanh", x)
def exp(x): return _unary("exp", x)
def softplus(x): return _unary("softplus", x)
def mean(x): return _new(_lib.vgre_ag_mean(x._h), (1,))
def softmax(x): return _unary("softmax", x)
def all_reduce(x):
    """Differentiable sum-all-reduce across cluster ranks (identity backward).
    Tensor parallelism: a row-parallel layer is all_reduce(matmul(x_shard, W_shard))
    where each rank holds only W_shard, so a layer too big for one node fits."""
    return _unary("all_reduce", x)
def transpose(x: Tensor) -> Tensor:
    """Swap the last two dims: [M,N]->[N,M] or [B,M,N]->[B,N,M]."""
    s = x.shape
    out_shape = (s[1], s[0]) if len(s) == 2 else (s[0], s[2], s[1])
    return _new(_lib.vgre_ag_transpose(x._h), out_shape)
def concat(a: Tensor, b: Tensor, axis: int = 1) -> Tensor:
    if axis == 0:
        shape = (a.shape[0] + b.shape[0], a.shape[1])
    else:
        shape = (a.shape[0], a.shape[1] + b.shape[1])
    return _new(_lib.vgre_ag_concat(a._h, b._h, axis), shape)

def reshape(x: Tensor, shape: Sequence[int]) -> Tensor:
    shp = (ctypes.c_int64 * len(shape))(*shape)
    return _new(_lib.vgre_ag_reshape(x._h, shp, len(shape)), tuple(shape))

def softmax_cross_entropy(logits: Tensor, targets: Sequence[int]) -> Tensor:
    t = (ctypes.c_int * len(targets))(*[int(v) for v in targets])
    return _new(_lib.vgre_ag_softmax_cross_entropy(logits._h, t, len(targets)), (1,))

def layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    return _new(_lib.vgre_ag_layer_norm(x._h, weight._h, bias._h, ctypes.c_float(eps)), x.shape)

def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    return _new(_lib.vgre_ag_rms_norm(x._h, weight._h, ctypes.c_float(eps)), x.shape)

def embedding(weight: Tensor, ids: Sequence[int]) -> Tensor:
    t = (ctypes.c_int * len(ids))(*[int(v) for v in ids])
    return _new(_lib.vgre_ag_embedding(weight._h, t, len(ids)), (len(ids), weight.shape[1]))

def index_select(x: Tensor, idx: Sequence[int]) -> Tensor:
    """Gather rows of a 2-D tensor: out[i] = x[idx[i]]  (shape [len(idx), D])."""
    t = (ctypes.c_int * len(idx))(*[int(v) for v in idx])
    return _new(_lib.vgre_ag_index_select(x._h, t, len(idx)), (len(idx), x.shape[1]))

def index_add(rows: int, src: Tensor, idx: Sequence[int]) -> Tensor:
    """Scatter-add: out is [rows, D] with out[idx[i]] += src[i] (dual of gather)."""
    t = (ctypes.c_int * len(idx))(*[int(v) for v in idx])
    return _new(_lib.vgre_ag_index_add(int(rows), src._h, t, len(idx)), (rows, src.shape[1]))

def selective_scan(a: Tensor, b: Tensor) -> Tensor:
    """State-space-model (Mamba) recurrence h_t = a[t]*h_{t-1} + b[t] over a
    length-T sequence; a, b are [T, D]. Returns the states h [T, D]."""
    return _new(_lib.vgre_ag_selective_scan(a._h, b._h), a.shape)

def rope(x: Tensor, num_heads: int, base: float = 10000.0) -> Tensor:
    return _new(_lib.vgre_ag_rope(x._h, num_heads, ctypes.c_float(base)), x.shape)

def attention(q: Tensor, k: Tensor, v: Tensor, num_heads: int, causal: bool = True) -> Tensor:
    return _new(_lib.vgre_ag_attention(q._h, k._h, v._h, num_heads, 1 if causal else 0), q.shape)

def flash_attention(q: Tensor, k: Tensor, v: Tensor, num_heads: int, causal: bool = True) -> Tensor:
    return _new(_lib.vgre_ag_flash_attention(q._h, k._h, v._h, num_heads, 1 if causal else 0), q.shape)

def conv2d(x: Tensor, weight: Tensor, bias: Tensor = None, stride: int = 1, pad: int = 0) -> Tensor:
    N, _, H, W = x.shape
    Co, _, Kh, Kw = weight.shape
    Ho = (H + 2 * pad - Kh) // stride + 1
    Wo = (W + 2 * pad - Kw) // stride + 1
    bh = bias._h if bias is not None else None
    return _new(_lib.vgre_ag_conv2d(x._h, weight._h, bh, stride, pad), (N, Co, Ho, Wo))

def max_pool2d(x: Tensor, kernel: int, stride: int) -> Tensor:
    N, C, H, W = x.shape
    return _new(_lib.vgre_ag_max_pool2d(x._h, kernel, stride),
                (N, C, (H - kernel) // stride + 1, (W - kernel) // stride + 1))

def avg_pool2d(x: Tensor, kernel: int, stride: int) -> Tensor:
    N, C, H, W = x.shape
    return _new(_lib.vgre_ag_avg_pool2d(x._h, kernel, stride),
                (N, C, (H - kernel) // stride + 1, (W - kernel) // stride + 1))

def batch_norm2d(x: Tensor, gamma: Tensor, beta: Tensor, running_mean: np.ndarray,
                 running_var: np.ndarray, training: bool, momentum: float = 0.1,
                 eps: float = 1e-5) -> Tensor:
    C = x.shape[1]
    rm = np.ascontiguousarray(running_mean, dtype=np.float32)
    rv = np.ascontiguousarray(running_var, dtype=np.float32)
    h = _lib.vgre_ag_batch_norm2d(x._h, gamma._h, beta._h,
                                  rm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                  rv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                  C, 1 if training else 0, float(momentum), float(eps))
    running_mean[:] = rm   # buffers were updated in place
    running_var[:] = rv
    return _new(h, x.shape)


# ── Layer modules (PyTorch-like ergonomics over the functional ops) ──────────
class Module:
    """Base class: collects child Parameters/Modules for the optimizer."""

    training = True

    def _children(self):
        for v in vars(self).values():
            if isinstance(v, Module):
                yield v
            elif isinstance(v, (list, tuple)):
                for it in v:
                    if isinstance(it, Module):
                        yield it

    def train(self, mode: bool = True):
        self.training = mode
        for ch in self._children():
            ch.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def named_parameters(self, prefix: str = "") -> List[Tuple[str, "Tensor"]]:
        out: List[Tuple[str, Tensor]] = []
        for name, v in vars(self).items():
            if isinstance(v, Tensor):
                out.append((prefix + name, v))
            elif isinstance(v, Module):
                out += v.named_parameters(prefix + name + ".")
            elif isinstance(v, (list, tuple)):
                for i, it in enumerate(v):
                    if isinstance(it, Module):
                        out += it.named_parameters(f"{prefix}{name}.{i}.")
        return out

    def parameters(self) -> List["Tensor"]:
        return [t for _, t in self.named_parameters()]

    def __call__(self, x):
        return self.forward(x)


def _kaiming(fan_in: int, *shape) -> "Tensor":
    std = (2.0 / fan_in) ** 0.5
    return tensor(rng.standard_normal(shape) * std, requires_grad=True)


# module-level RNG so layer init is reproducible
rng = np.random.default_rng(1234)


def seed(s: int) -> None:
    """Reseed the layer-initialization RNG."""
    global rng
    rng = np.random.default_rng(s)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.W = _kaiming(in_features, in_features, out_features)
        self.b = tensor(np.zeros(out_features), requires_grad=True) if bias else None

    def forward(self, x):
        s = x.shape
        if len(s) == 3:                       # [B,T,D]: flatten → matmul → reshape
            flat = reshape(x, (s[0] * s[1], s[2]))
            y = matmul(flat, self.W)
            if self.b is not None:
                y = add(y, self.b)            # [B*T,O]+[O] bias broadcast (2-D)
            return reshape(y, (s[0], s[1], self.W.shape[1]))
        y = matmul(x, self.W)
        return add(y, self.b) if self.b is not None else y


class BitLinear(Module):
    """BitNet b1.58 linear layer: a full-precision master weight that is ternary-
    quantized {-1,0,+1} (per-column absmean) in the forward pass via a
    straight-through estimator, so the layer trains in fp but its forward matches
    the inference-time multiplication-free ternary GEMM. Drop-in for Linear —
    the path to running large models on CPUs with no multiplies in the matmul."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.W = _kaiming(in_features, in_features, out_features)
        self.b = tensor(np.zeros(out_features), requires_grad=True) if bias else None

    def forward(self, x):
        Wq = ternary_quantize(self.W)         # STE ternary weight (fp master trains)
        s = x.shape
        if len(s) == 3:                       # [B,T,D]: flatten → matmul → reshape
            flat = reshape(x, (s[0] * s[1], s[2]))
            y = matmul(flat, Wq)
            if self.b is not None:
                y = add(y, self.b)
            return reshape(y, (s[0], s[1], self.W.shape[1]))
        y = matmul(x, Wq)
        return add(y, self.b) if self.b is not None else y


class LoRALinear(Module):
    """Low-Rank Adaptation (LoRA, arXiv:2106.09685) over a frozen base weight:

        y = x @ W_frozen [+ b_frozen] + (alpha/r) · (x @ A) @ B

    Only A [in, r] and B [r, out] train — r·(in+out) parameters instead of
    in·out — which is what makes fine-tuning practical on a CPU/laptop. B is
    zero-initialized so training starts exactly at the base model. All compute
    (the three matmuls and the backward) runs in the C++ autograd engine; this
    class only composes existing differentiable ops.

        layer = nn.LoRALinear(768, 768, r=8, alpha=16)          # fresh base
        layer = nn.LoRALinear.from_linear(pretrained, r=8)      # adapt existing
        opt   = nn.AdamW(layer.adapter_parameters(), lr=1e-3)   # train A,B only
        layer.merge()      # fold the adapter into W for zero-overhead inference
    """

    def __init__(self, in_features: int, out_features: int, r: int = 8,
                 alpha: float = 16.0, bias: bool = True, base_weight=None,
                 base_bias=None):
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")
        # Frozen base (requires_grad=False → excluded from autograd updates).
        w = _f32(base_weight) if base_weight is not None else \
            _f32(rng.standard_normal((in_features, out_features)) *
                 (2.0 / in_features) ** 0.5)
        if w.shape != (in_features, out_features):
            raise ValueError(f"base_weight must be [{in_features},{out_features}]")
        self.W = tensor(w, requires_grad=False)
        if bias:
            b = _f32(base_bias) if base_bias is not None else np.zeros(out_features)
            self.b = tensor(b, requires_grad=False)
        else:
            self.b = None
        # Trainable adapter: A ~ N(0, 1/r) (standard LoRA init), B = 0 so the
        # initial delta is exactly zero.
        self.A = tensor(rng.standard_normal((in_features, r)) / float(r),
                        requires_grad=True)
        self.B = tensor(np.zeros((r, out_features)), requires_grad=True)
        self.r = r
        self.alpha = float(alpha)

    @classmethod
    def from_linear(cls, linear: "Linear", r: int = 8, alpha: float = 16.0) -> "LoRALinear":
        """Wrap an existing Linear's weights as the frozen base."""
        w = linear.W.numpy()
        b = linear.b.numpy() if linear.b is not None else None
        return cls(w.shape[0], w.shape[1], r=r, alpha=alpha,
                   bias=b is not None, base_weight=w, base_bias=b)

    @property
    def scaling(self) -> float:
        return self.alpha / self.r

    def adapter_parameters(self) -> List["Tensor"]:
        """The trainable tensors (A, B) — pass these to the optimizer."""
        return [self.A, self.B]

    def forward(self, x):
        s = x.shape
        if len(s) == 3:                       # [B,T,D]: flatten → compute → reshape
            x2 = reshape(x, (s[0] * s[1], s[2]))
            y2 = self._forward2d(x2)
            return reshape(y2, (s[0], s[1], self.W.shape[1]))
        return self._forward2d(x)

    def _forward2d(self, x):
        y = matmul(x, self.W)
        delta = scale(matmul(matmul(x, self.A), self.B), self.scaling)
        y = add(y, delta)
        return add(y, self.b) if self.b is not None else y

    def merge(self) -> None:
        """Fold the adapter into the base weight (W += (alpha/r)·A@B) and reset
        B to zero, so forward() is unchanged but the delta path contributes
        nothing — zero-overhead inference after merging."""
        self.W.set_(self.W.numpy() + self.scaling * (self.A.numpy() @ self.B.numpy()))
        self.B.set_(np.zeros(self.B.shape, dtype=np.float32))

    def save_adapter(self, path: str) -> None:
        """Persist only the adapter (A, B, r, alpha) — kilobytes, not the model."""
        np.savez(path, A=self.A.numpy(), B=self.B.numpy(),
                 r=np.int64(self.r), alpha=np.float64(self.alpha))

    def load_adapter(self, path: str) -> None:
        d = np.load(path)
        if int(d["r"]) != self.r:
            raise ValueError(f"adapter rank {int(d['r'])} != layer rank {self.r}")
        self.A.set_(d["A"])
        self.B.set_(d["B"])
        self.alpha = float(d["alpha"])


class QLoRALinear(Module):
    """QLoRA (arXiv:2305.14314): a frozen, low-bit-quantized base weight + a
    trainable LoRA adapter.

        y = x @ dequant(base) [+ b] + (alpha/r) · (x @ A) @ B

    The large base is stored quantized — ternary {-1,0,+1} with a per-column
    absmean scale, ~2 bits/weight — as plain arrays (NOT parameters), so it is
    never updated and forms no gradient; it is de-quantized transiently inside the
    forward matmul. Only the small A [in,r], B [r,out] adapters train. That is what
    lets a big layer be fine-tuned on a laptop: the base's memory is a fraction of
    fp32. B is zero-initialized so training starts exactly at the (quantized) base.
    Use adapter_parameters() (== parameters(), which holds only A and B)."""

    def __init__(self, in_features: int, out_features: int, r: int = 8,
                 alpha: float = 16.0, bias: bool = True, base_weight=None,
                 base_bias=None):
        if r <= 0:
            raise ValueError("QLoRA rank r must be positive")
        w = _f32(base_weight) if base_weight is not None else \
            _f32(rng.standard_normal((in_features, out_features)) *
                 (2.0 / in_features) ** 0.5)
        if w.shape != (in_features, out_features):
            raise ValueError(f"base_weight must be [{in_features},{out_features}]")
        # Freeze the base as ternary codes + per-column absmean scale (numpy, so
        # never registered as trainable parameters).
        scale = np.abs(w).mean(axis=0).astype(np.float32)          # [out]
        inv = np.where(scale == 0.0, 1.0, scale)
        self._codes = np.clip(np.rint(w / inv), -1, 1).astype(np.int8)   # [in,out]
        self._scale = scale
        self._bias = _f32(base_bias) if (bias and base_bias is not None) else \
            (np.zeros(out_features, np.float32) if bias else None)
        # Trainable adapter: A ~ N(0,1/r), B = 0 → initial delta is exactly zero.
        self.A = tensor(rng.standard_normal((in_features, r)) / float(r),
                        requires_grad=True)
        self.B = tensor(np.zeros((r, out_features)), requires_grad=True)
        self.r = r
        self.alpha = float(alpha)

    @classmethod
    def from_linear(cls, linear: "Linear", r: int = 8, alpha: float = 16.0) -> "QLoRALinear":
        """Quantize an existing Linear's weights as the frozen ternary base."""
        w = linear.W.numpy()
        b = linear.b.numpy() if linear.b is not None else None
        return cls(w.shape[0], w.shape[1], r=r, alpha=alpha,
                   bias=b is not None, base_weight=w, base_bias=b)

    @property
    def scaling(self) -> float:
        return self.alpha / self.r

    def adapter_parameters(self) -> List["Tensor"]:
        return [self.A, self.B]

    def base_bits_per_weight(self) -> float:
        """Effective storage of the frozen base: ~2-bit ternary + the tiny scale."""
        n = self._codes.size
        return (n * 2 + self._scale.size * 32) / n

    def _base(self):                                   # dequant transiently (no grad)
        return tensor(self._codes.astype(np.float32) * self._scale[None, :])

    def forward(self, x):
        s = x.shape
        if len(s) == 3:
            y2 = self._forward2d(reshape(x, (s[0] * s[1], s[2])))
            return reshape(y2, (s[0], s[1], self._codes.shape[1]))
        return self._forward2d(x)

    def _forward2d(self, x):
        y = matmul(x, self._base())
        y = add(y, scale(matmul(matmul(x, self.A), self.B), self.scaling))
        return add(y, tensor(self._bias)) if self._bias is not None else y


class Conv2d(Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1,
                 pad: int = 0, bias: bool = True):
        self.weight = _kaiming(in_ch * kernel * kernel, out_ch, in_ch, kernel, kernel)
        self.bias = tensor(np.zeros(out_ch), requires_grad=True) if bias else None
        self.stride, self.pad = stride, pad

    def forward(self, x):
        return conv2d(x, self.weight, self.bias, self.stride, self.pad)


class ReLU(Module):
    def forward(self, x): return relu(x)


class MoELayer(Module):
    """Mixture-of-Experts with a learned top-k router (Shazeer et al.; Mixtral /
    DeepSeek-style). Per token, the router picks its top-k experts by logit; the
    output is the softmax-gate-weighted sum of those experts' FFNs. The routing
    decision is discrete (chosen on the host), but the gate weights of the
    selected experts stay differentiable, so the router learns end-to-end.

    Why it matters: only k of E experts contribute per token, so a huge model has
    small *active* compute — ideal for CPU clusters. Dispatch is compute-sparse:
    each expert's FFN runs only on the tokens routed to it (via index_select /
    index_add gather-scatter), so the expensive expert matmuls scale with the
    active tokens, not tokens×experts. Expert-parallel over the cluster
    collectives is the follow-up (docs/implementationPlan.md T2).

    Input/output are 2-D [tokens, d_model].

    Expert-parallel (expert_parallel=True): experts are sharded across cluster
    ranks — each rank holds a disjoint block of experts and computes only their
    contribution; all_reduce sums the per-rank partials into the full output
    (like row-parallel TP, reusing the differentiable collective). The router is
    replicated and every expert is seeded by its GLOBAL index, so a sharded run
    over N ranks produces exactly the same result as a single-process run."""
    def __init__(self, d_model: int, num_experts: int, d_ff: int = 0,
                 top_k: int = 2, expert_parallel: bool = False, seed: int = 1234):
        d_ff = d_ff or 4 * d_model
        self.d = d_model
        self.E = num_experts
        self.k = max(1, min(top_k, num_experts))
        self.expert_parallel = expert_parallel

        if expert_parallel:
            # Deterministic init keyed on the GLOBAL expert index so any rank
            # builds identical weights for a given expert; shard the experts
            # across ranks (contiguous blocks), holding only the local ones.
            world = world_size()
            rank = int(os.environ.get("VGRE_NN_RANK", "0"))
            per = num_experts // world
            lo = rank * per
            hi = num_experts if rank == world - 1 else (rank + 1) * per
            self.local_ids = list(range(lo, hi)) if world > 1 else list(range(num_experts))
            rg = np.random.default_rng(seed)
            self.Wg = tensor((rg.standard_normal((d_model, num_experts))
                              * (2.0 / d_model) ** 0.5).astype(np.float32),
                             requires_grad=True)
            self.experts = [self._make_expert(d_model, d_ff, seed, e)
                            for e in self.local_ids]
        else:
            self.local_ids = list(range(num_experts))
            self.Wg = _kaiming(d_model, d_model, num_experts)     # router logits [d,E]
            self.experts = [Sequential(Linear(d_model, d_ff), ReLU(), Linear(d_ff, d_model))
                            for _ in range(num_experts)]

    @staticmethod
    def _make_expert(d_model, d_ff, seed, global_e):
        eg = np.random.default_rng(seed * 100003 + global_e)
        ex = Sequential(Linear(d_model, d_ff), ReLU(), Linear(d_ff, d_model))
        ex.layers[0].W.set_((eg.standard_normal((d_model, d_ff)) * (2.0 / d_model) ** 0.5).astype(np.float32))
        ex.layers[2].W.set_((eg.standard_normal((d_ff, d_model)) * (2.0 / d_ff) ** 0.5).astype(np.float32))
        return ex

    def forward(self, x):
        T = x.shape[0]
        logits = matmul(x, self.Wg)                           # [T, E] (replicated)
        L = logits.numpy()
        # Per-row top-k routing: keep the k largest logits, send the rest to -inf
        # so softmax puts ~0 weight there (only top-k experts contribute).
        topk = np.argpartition(-L, self.k - 1, axis=1)[:, :self.k]
        mask = np.full(L.shape, -1e9, dtype=np.float32)
        np.put_along_axis(mask, topk, 0.0, axis=1)
        gate = softmax(add(logits, tensor(mask)))             # [T, E]
        routed = mask == 0.0                                  # [T,E] bool: token→expert
        self._aux_logits = logits                             # kept for aux_loss()
        self._aux_frac = routed.mean(axis=0).astype(np.float32)  # f_e: load per expert

        ones_1d = tensor(np.ones((1, self.d), dtype=np.float32))
        out = None
        for li, e in enumerate(self.local_ids):               # local (possibly sharded) experts
            idx = np.nonzero(routed[:, e])[0].tolist()        # tokens routed to expert e
            if not idx:
                continue
            sel = np.zeros((self.E, 1), dtype=np.float32); sel[e, 0] = 1.0
            x_e = index_select(x, idx)                        # [n_e, d] — its tokens only
            y_e = self.experts[li](x_e)                       # expert FFN on the sub-batch
            g_e = matmul(index_select(gate, idx), tensor(sel))  # [n_e,1] = gate[idx,e]
            weighted = mul(y_e, matmul(g_e, ones_1d))         # [n_e, d]
            term = index_add(T, weighted, idx)                # scatter back to [T, d]
            out = term if out is None else add(out, term)
        if out is None:                                       # no token hit a local expert
            out = mul(x, tensor(np.zeros((T, self.d), np.float32)))
        # Expert-parallel: sum each rank's partial into the full output. At
        # world_size 1 this is the identity; across ranks it is the real TCP sum.
        if self.expert_parallel:
            out = all_reduce(out)
        return out

    def aux_loss(self):
        """Switch-Transformer load-balancing loss (∝ Σ_e f_e·P_e): f_e is the
        fraction of tokens routed to expert e (from the last forward), P_e the
        mean router probability for e. Minimizing it spreads tokens across
        experts and prevents router collapse. Add coef·aux_loss() to your loss
        after a forward pass. Differentiable through the router probabilities."""
        if getattr(self, "_aux_logits", None) is None:
            raise RuntimeError("aux_loss() requires a preceding forward()")
        P = softmax(self._aux_logits)                         # [T,E] unmasked probs
        Fbc = tensor(np.broadcast_to(self._aux_frac, P.shape).copy())  # [T,E] f_e per col
        # E · mean_{t,e}(P·f) = Σ_e f_e·(mean_t P[t,e]) — the balance objective.
        return scale(mean(mul(P, Fbc)), float(self.E))


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        self.p = p
    def forward(self, x):
        return dropout(x, self.p) if (self.training and self.p > 0) else x


class BatchNorm2d(Module):
    def __init__(self, channels: int, momentum: float = 0.1, eps: float = 1e-5):
        self.gamma = tensor(np.ones(channels), requires_grad=True)
        self.beta = tensor(np.zeros(channels), requires_grad=True)
        # Persistent (non-grad) running statistics.
        self.running_mean = np.zeros(channels, dtype=np.float32)
        self.running_var = np.ones(channels, dtype=np.float32)
        self.momentum, self.eps = momentum, eps

    def forward(self, x):
        return batch_norm2d(x, self.gamma, self.beta, self.running_mean,
                            self.running_var, self.training, self.momentum, self.eps)


class Embedding(Module):
    def __init__(self, vocab: int, dim: int):
        self.weight = tensor(rng.standard_normal((vocab, dim)) * 0.02, requires_grad=True)
    def forward(self, ids):
        return embedding(self.weight, ids)


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = tensor(np.ones(dim), requires_grad=True)
        self.bias = tensor(np.zeros(dim), requires_grad=True)
        self.eps = eps
    def forward(self, x):
        return layer_norm(x, self.weight, self.bias, self.eps)


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = tensor(np.ones(dim), requires_grad=True)
        self.eps = eps
    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


class MultiHeadAttention(Module):
    def __init__(self, dim: int, num_heads: int, causal: bool = True, use_rope: bool = True):
        self.q, self.k, self.v, self.o = (Linear(dim, dim, bias=False) for _ in range(4))
        self.num_heads, self.causal, self.use_rope = num_heads, causal, use_rope
    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        if self.use_rope:
            q = rope(q, self.num_heads); k = rope(k, self.num_heads)
        a = attention(q, k, v, self.num_heads, self.causal)
        return self.o(a)


class TransformerBlock(Module):
    """Pre-norm decoder block: x + Attn(RMSNorm(x)), then x + FFN(RMSNorm(x)).

    The FFN is a dense MLP by default, or a sparse Mixture-of-Experts when
    moe_experts > 0 (a drop-in replacement — only k of moe_experts FFNs run per
    token). With expert_parallel=True the experts shard across cluster ranks."""
    def __init__(self, dim: int, num_heads: int, ff: int = 0, causal: bool = True,
                 moe_experts: int = 0, moe_top_k: int = 2,
                 expert_parallel: bool = False):
        ff = ff or 4 * dim
        self.n1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, causal)
        self.n2 = RMSNorm(dim)
        if moe_experts > 0:
            self.moe = MoELayer(dim, moe_experts, ff, moe_top_k,
                                expert_parallel=expert_parallel)
        else:
            self.moe = None
            self.fc1, self.fc2 = Linear(dim, ff), Linear(ff, dim)

    def _ffn(self, h):
        if self.moe is None:
            return self.fc2(gelu(self.fc1(h)))
        s = h.shape                                  # MoE takes 2-D [tokens, dim]
        if len(s) == 3:
            return reshape(self.moe(reshape(h, (s[0] * s[1], s[2]))), s)
        return self.moe(h)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self._ffn(self.n2(x))

    def aux_loss(self):
        """MoE load-balancing loss for this block (None if the FFN is dense)."""
        return self.moe.aux_loss() if self.moe is not None else None


class SSMBlock(Module):
    """Selective state-space (Mamba / S6) mixer with a real d_state-dimensional
    diagonal SSM. Per channel it keeps a state h ∈ ℝ^N and runs the selective
    recurrence with input-dependent Δ, B, C:

        Δ_t = softplus(dt·x_t)                 (per-channel step, data-dependent)
        Ā_t = exp(Δ_t ⊙ A),  A = −exp(A_log)   (stable diagonal decay)
        h_t[:,n] = Ā_t[:,n] ⊙ h_{t-1}[:,n] + Δ_t ⊙ B_t[n] ⊙ u_t
        y_t = Σ_n C_t[n] ⊙ h_t[:,n] + D ⊙ u_t

    Linear-time, constant memory per step — no attention, no growing KV cache.
    Each state channel is one `selective_scan`; the input-dependent B/C give the
    selectivity that makes Mamba competitive with attention. Operates on [T,dim]."""
    def __init__(self, dim: int, d_state: int = 8):
        self.dim = dim
        self.N = d_state
        self.in_proj = Linear(dim, dim)                 # u_t (SSM input channels)
        self.dt_proj = Linear(dim, dim)                 # Δ_t (per-channel step)
        self.B_proj = Linear(dim, d_state)              # B_t [T,N]
        self.C_proj = Linear(dim, d_state)              # C_t [T,N]
        A0 = np.log(np.tile(np.arange(1, d_state + 1, dtype=np.float32), (dim, 1)))
        self.A_log = tensor(A0.astype(np.float32), requires_grad=True)   # [dim,N]
        self.Dskip = tensor(np.ones(dim, np.float32), requires_grad=True)
        self.out_proj = Linear(dim, dim)

    def forward(self, x):
        s = x.shape
        if len(s) == 3:
            # Batched [B,T,dim]: the recurrence is per sequence and must NEVER
            # carry state across batch boundaries, so each sequence is scanned
            # independently and the results re-stacked (a flat [B*T,dim] scan
            # would leak the end of one sequence into the start of the next).
            B, T = s[0], s[1]
            flat = reshape(x, (B * T, s[2]))
            out = None
            for b in range(B):
                yb = self._forward2d(index_select(flat, list(range(b * T, (b + 1) * T))))
                out = yb if out is None else concat(out, yb, 0)
            return reshape(out, (B, T, self.dim))
        return self._forward2d(x)

    def _forward2d(self, x):
        T, dim, N = x.shape[0], self.dim, self.N
        u = self.in_proj(x)                              # [T,dim]
        delta = softplus(self.dt_proj(x))               # [T,dim]  Δ>0
        B = self.B_proj(x)                              # [T,N]
        C = self.C_proj(x)                              # [T,N]
        A = scale(exp(self.A_log), -1.0)                # [dim,N] = -exp(A_log)
        onesT = tensor(np.ones((T, 1), np.float32))
        ones_d = tensor(np.ones((1, dim), np.float32))
        y = None
        for n in range(N):
            sel = tensor(np.eye(N, dtype=np.float32)[:, n:n + 1].copy())  # [N,1]
            A_row = matmul(onesT, reshape(matmul(A, sel), (1, dim)))      # [T,dim] A[:,n]
            B_col = matmul(matmul(B, sel), ones_d)                        # [T,dim] B[:,n]
            C_col = matmul(matmul(C, sel), ones_d)                        # [T,dim] C[:,n]
            Abar = exp(mul(delta, A_row))                                 # [T,dim] Ā
            bbar = mul(mul(delta, B_col), u)                             # [T,dim] Δ·B·u
            h = selective_scan(Abar, bbar)                              # [T,dim]
            yn = mul(h, C_col)
            y = yn if y is None else add(y, yn)
        y = add(y, mul(u, matmul(onesT, reshape(self.Dskip, (1, dim)))))  # + D·u skip
        return self.out_proj(y)


class MambaBlock(Module):
    """Pre-norm residual state-space block: x + SSM(RMSNorm(x)). A drop-in
    sequence mixer alternative to a transformer block (attention-free)."""
    def __init__(self, dim: int, d_state: int = 8):
        self.norm = RMSNorm(dim)
        self.ssm = SSMBlock(dim, d_state=d_state)

    def forward(self, x):
        return x + self.ssm(self.norm(x))


def hybrid_blocks(dim: int, n_layers: int, num_heads: int, attn_every: int = 4,
                  d_state: int = 8, ff: int = 0, causal: bool = True,
                  moe_experts: int = 0, moe_top_k: int = 2) -> List[Module]:
    """Build a hybrid Mamba–Transformer stack (Jamba-style): every `attn_every`-th
    layer is a `TransformerBlock`, the rest are `MambaBlock`s.

    Why hybridize: the SSM layers are linear-time and keep **no KV cache**, so
    memory and cost at long context are dominated only by the few attention
    layers, while attention is retained where exact long-range recall matters.
    With attn_every=4 only 1 layer in 4 holds a KV cache — a ~4× cut in KV memory
    versus an all-attention stack of the same depth.

    Both block types take and return [T,dim] or [B,T,dim] with the same residual
    stream, so they interleave directly.

        blocks = nn.hybrid_blocks(256, n_layers=8, num_heads=8, attn_every=4)
        # -> Mamba, Mamba, Mamba, Transformer, Mamba, Mamba, Mamba, Transformer
    """
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if attn_every <= 0:
        raise ValueError("attn_every must be positive")
    blocks: List[Module] = []
    for i in range(n_layers):
        if (i + 1) % attn_every == 0:
            blocks.append(TransformerBlock(dim, num_heads, ff=ff, causal=causal,
                                           moe_experts=moe_experts, moe_top_k=moe_top_k))
        else:
            blocks.append(MambaBlock(dim, d_state=d_state))
    return blocks


class MTPHeads(Module):
    """Multi-Token-Prediction heads (DeepSeek-V3 style): K linear heads on a
    shared trunk representation, where head j predicts the token j+1 positions
    ahead (head 0 is the usual next-token head). Training with the summed
    cross-entropy over all K heads is a denser learning signal; at inference the
    heads emit K draft tokens from a SINGLE trunk forward, so the model is its
    own draft model for speculative decoding (see mtp_draft_fn)."""
    def __init__(self, dim: int, vocab: int, n_predict: int = 4, bias: bool = False):
        if n_predict <= 0:
            raise ValueError("n_predict must be positive")
        self.K = n_predict
        self.heads = [Linear(dim, vocab, bias=bias) for _ in range(n_predict)]

    def forward(self, h):
        """h [T,dim] → list of K logits [T,vocab] (one per look-ahead distance)."""
        return [hd(h) for hd in self.heads]


class MaxPool2d(Module):
    def __init__(self, kernel: int, stride: int = None):
        self.kernel, self.stride = kernel, stride or kernel
    def forward(self, x): return max_pool2d(x, self.kernel, self.stride)


class AvgPool2d(Module):
    def __init__(self, kernel: int, stride: int = None):
        self.kernel, self.stride = kernel, stride or kernel
    def forward(self, x): return avg_pool2d(x, self.kernel, self.stride)


class Flatten(Module):
    def forward(self, x):
        N = x.shape[0]
        return reshape(x, (N, x.size // N))


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


# ── Generation: greedy + lossless speculative decoding ──────────────────────
def greedy_generate(model, prompt: Sequence[int], n_new: int) -> List[int]:
    """Vanilla greedy decode: append argmax of the last-position logits, n_new
    times. `model(ids)` must return logits of shape [len(ids), vocab]. Returns
    the generated tokens (excluding the prompt)."""
    ctx = list(prompt)
    for _ in range(n_new):
        ctx.append(int(np.argmax(model(ctx).numpy()[-1])))
    return ctx[len(prompt):]


def ngram_draft_fn(k: int = 4, n: int = 2):
    """A zero-cost drafter: propose the tokens that followed the most recent
    earlier occurrence of the current n-token suffix (prompt-lookup decoding).
    Great for repetitive / structured text; proposes nothing when unseen."""
    def draft(ctx: List[int]) -> List[int]:
        if len(ctx) < n + 1:
            return []
        suffix = tuple(ctx[-n:])
        for i in range(len(ctx) - n - 1, -1, -1):
            if tuple(ctx[i:i + n]) == suffix:
                return list(ctx[i + n:i + n + k])
        return []
    return draft


def model_draft_fn(draft_model, k: int = 4):
    """A drafter backed by a (smaller/cheaper) draft MODEL: greedily roll out k
    tokens from draft_model. Use with speculative_generate — the output is still
    identical to greedy decoding of the TARGET regardless of the draft model's
    quality; a better draft just gets more tokens accepted per target forward."""
    def draft(ctx: List[int]) -> List[int]:
        toks: List[int] = []
        dc = list(ctx)
        for _ in range(k):
            t = int(np.argmax(draft_model(dc).numpy()[-1]))
            toks.append(t); dc.append(t)
        return toks
    return draft


def _dist_from_logits(row, temperature: float, top_p: float):
    z = np.asarray(row, dtype=np.float64) / max(temperature, 1e-6)
    z -= z.max()
    p = np.exp(z); p /= p.sum()
    if top_p < 1.0:                                   # nucleus filter, renormalized
        order = np.argsort(-p)
        csum = np.cumsum(p[order])
        keep = csum <= top_p
        keep[0] = True                                # always keep the top token
        mask = np.zeros_like(p); mask[order[keep]] = 1.0
        p = p * mask; p /= p.sum()
    return p


def speculative_sample(target, draft, prompt: Sequence[int], n_new: int,
                       k: int = 4, temperature: float = 1.0, top_p: float = 1.0,
                       seed: int = 0):
    """Sampler-exact speculative decoding (Leviathan et al. / Chen et al. 2023).
    The draft model proposes k tokens by sampling its own distribution p; the
    target verifies them in ONE forward, accepting each token d with probability
    min(1, q(d)/p(d)) and, on the first rejection, resampling from the residual
    norm(relu(q − p)); if all k are accepted it samples a bonus from q. The
    produced tokens are distributed EXACTLY as sampling directly from the target
    at the same temperature/top_p — no approximation — while using one target
    forward per accepted run. Returns (tokens, num_target_forwards)."""
    rng = np.random.default_rng(seed)
    ctx = list(prompt); start = len(ctx); forwards = 0

    def sample(p):
        return int(rng.choice(len(p), p=p))

    while len(ctx) - start < n_new:
        # 1. Draft k tokens autoregressively, recording each draft distribution.
        d_toks: List[int] = []
        d_probs = []
        dc = list(ctx)
        for _ in range(k):
            p = _dist_from_logits(draft(dc).numpy()[-1], temperature, top_p)
            t = sample(p)
            d_toks.append(t); d_probs.append(p); dc.append(t)

        # 2. Verify all k in ONE target forward over the extended sequence.
        tl = target(ctx + d_toks).numpy(); forwards += 1
        n = len(ctx)
        accepted: List[int] = []
        all_ok = True
        for j in range(k):
            q = _dist_from_logits(tl[n - 1 + j], temperature, top_p)
            p, d = d_probs[j], d_toks[j]
            ratio = (q[d] / p[d]) if p[d] > 0 else 1.0
            if rng.random() < min(1.0, ratio):
                accepted.append(d)                    # accept the draft token
            else:
                resid = np.maximum(q - p, 0.0)        # resample from the residual
                s = resid.sum()
                accepted.append(sample(resid / s) if s > 0 else sample(q))
                all_ok = False
                break
        if all_ok:                                    # bonus token from the target
            accepted.append(sample(_dist_from_logits(tl[n - 1 + k], temperature, top_p)))

        room = n_new - (len(ctx) - start)
        ctx.extend(accepted[:room])
    return ctx[start:], forwards


def mtp_loss(head_logits: List["Tensor"], targets: Sequence[int]) -> "Tensor":
    """Summed multi-token-prediction loss. `head_logits[j]` is head j's logits
    [T,vocab]; head j at position t is trained to predict targets[t+j] (j tokens
    further ahead), so its supervised range shrinks by j at the tail. `targets`
    is the length-T next-token sequence (head 0's targets)."""
    T = head_logits[0].shape[0]
    total = None
    for j, lg in enumerate(head_logits):
        n = T - j
        if n <= 0:
            break
        lj = index_select(lg, list(range(n)))            # positions 0..n-1
        ce = softmax_cross_entropy(lj, list(targets[j:j + n]))
        total = ce if total is None else add(total, ce)
    return total


def mtp_draft_fn(mtp_model, k: int = 0):
    """Self-speculative drafter: `mtp_model(ctx)` returns the list of per-head
    logits [T,vocab]; head j predicts j tokens ahead, so the argmax of each head
    at the LAST position gives K draft tokens from ONE forward — no separate draft
    model. Pair with speculative_generate whose target is head 0; the output stays
    identical to greedy while several tokens are verified per target forward."""
    def draft(ctx: List[int]) -> List[int]:
        heads = mtp_model(ctx)
        K = len(heads) if k <= 0 else min(k, len(heads))
        return [int(np.argmax(heads[j].numpy()[-1])) for j in range(K)]
    return draft


def speculative_generate(model, prompt: Sequence[int], n_new: int,
                         draft_fn, k: int = 4):
    """Lossless speculative decoding (greedy). Each step the drafter proposes up
    to k tokens; the target model verifies them in ONE forward over the extended
    sequence, accepting the longest prefix whose tokens equal the target's own
    greedy argmax, plus one correction/bonus token. The output is IDENTICAL to
    greedy_generate on the target — the speedup comes from accepting several
    tokens per target forward.

    Returns (generated_tokens, num_forwards). tokens/num_forwards > 1 is the
    per-forward token yield (the speculative speedup vs one token per forward)."""
    ctx = list(prompt)
    start = len(ctx)
    forwards = 0
    while len(ctx) - start < n_new:
        draft = draft_fn(ctx)[:k]
        n = len(ctx)
        logits = model(ctx + draft).numpy()      # [n + len(draft), vocab]
        forwards += 1
        # Greedily verify each drafted token against the target's own argmax.
        accepted = []
        matched_all = True
        for j in range(len(draft)):
            t_j = int(np.argmax(logits[n - 1 + j]))
            if t_j == draft[j]:
                accepted.append(draft[j])
            else:
                accepted.append(t_j)             # correction = target's greedy token
                matched_all = False
                break
        if matched_all:
            accepted.append(int(np.argmax(logits[n - 1 + len(draft)])))  # bonus
        # Never overshoot the requested count.
        room = n_new - (len(ctx) - start)
        ctx.extend(accepted[:room])
    return ctx[start:], forwards


# ── Checkpoint I/O (standard safetensors; loadable by VGRE's SafeTensors) ────
def save(model: "Module", path: str) -> None:
    """Save a module's named parameters as a standard safetensors file."""
    header, blob = {}, bytearray()
    for name, t in model.named_parameters():
        a = np.ascontiguousarray(t.numpy(), dtype=np.float32)
        start = len(blob)
        blob += a.tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [start, len(blob)]}
    hb = json.dumps(header).encode("utf-8")
    hb += b" " * ((-(8 + len(hb))) % 8)            # 8-byte align the data blob
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(blob)


def load(model: "Module", path: str) -> None:
    """Load parameters by name into a module with a matching architecture."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        blob = f.read()
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        b, e = info["data_offsets"]
        tensors[name] = np.frombuffer(blob[b:e], dtype=np.float32).reshape(info["shape"])
    for name, t in model.named_parameters():
        if name not in tensors:
            raise KeyError(f"checkpoint missing parameter '{name}'")
        t.set_(tensors[name])


class _Optimizer:
    """Shared optimizer handle wrapper (generic step/zero_grad/lr/free)."""

    def __init__(self, handle, params):
        self._params = list(params)
        self._o = handle

    def set_lr(self, lr: float):
        _lib.vgre_ag_opt_set_lr(self._o, float(lr))

    def step(self, clip: float = 1.0):
        _lib.vgre_ag_opt_step(self._o, float(clip))

    def zero_grad(self):
        _lib.vgre_ag_opt_zero_grad(self._o)

    def __del__(self):
        o = getattr(self, "_o", None)
        if o and _lib is not None:
            _lib.vgre_ag_opt_free(o)
            self._o = None


class AdamW(_Optimizer):
    """AdamW over a list of parameter Tensors (those created requires_grad=True)."""

    def __init__(self, params: List[Tensor], lr: float = 1e-3, weight_decay: float = 0.01):
        _require()
        arr = (ctypes.c_void_p * len(params))(*[p._h for p in params])
        super().__init__(_lib.vgre_ag_adamw(arr, len(params), float(lr), float(weight_decay)), params)


class SGD(_Optimizer):
    """SGD (optional momentum) over a list of parameter Tensors."""

    def __init__(self, params: List[Tensor], lr: float = 1e-2, momentum: float = 0.0,
                 weight_decay: float = 0.0):
        _require()
        arr = (ctypes.c_void_p * len(params))(*[p._h for p in params])
        super().__init__(_lib.vgre_ag_sgd(arr, len(params), float(lr), float(momentum),
                                          float(weight_decay)), params)
