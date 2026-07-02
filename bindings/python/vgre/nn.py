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
    for name in ("relu", "gelu", "silu", "sigmoid", "tanh", "mean", "softmax", "transpose", "all_reduce"):
        f = getattr(_lib, "vgre_ag_" + name); f.argtypes = [V]; f.restype = V
    _lib.vgre_ag_concat.argtypes = [V, V, c.c_int]; _lib.vgre_ag_concat.restype = V
    _lib.vgre_ag_reshape.argtypes = [V, P(I64), c.c_int]; _lib.vgre_ag_reshape.restype = V
    _lib.vgre_ag_softmax_cross_entropy.argtypes = [V, P(c.c_int), c.c_int]; _lib.vgre_ag_softmax_cross_entropy.restype = V
    _lib.vgre_ag_layer_norm.argtypes = [V, V, V, c.c_float]; _lib.vgre_ag_layer_norm.restype = V
    _lib.vgre_ag_rms_norm.argtypes = [V, V, c.c_float]; _lib.vgre_ag_rms_norm.restype = V
    _lib.vgre_ag_embedding.argtypes = [V, P(c.c_int), c.c_int]; _lib.vgre_ag_embedding.restype = V
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
    """Pre-norm decoder block: x + Attn(RMSNorm(x)), then x + MLP(RMSNorm(x))."""
    def __init__(self, dim: int, num_heads: int, ff: int = 0, causal: bool = True):
        ff = ff or 4 * dim
        self.n1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, causal)
        self.n2 = RMSNorm(dim)
        self.fc1, self.fc2 = Linear(dim, ff), Linear(ff, dim)
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        h = self.fc2(gelu(self.fc1(self.n2(x))))
        return x + h


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
