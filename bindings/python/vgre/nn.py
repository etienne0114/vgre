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
    _lib.vgre_ag_softmax_cross_entropy_soft.argtypes = [V, V]; _lib.vgre_ag_softmax_cross_entropy_soft.restype = V
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
def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    """max(x, negative_slope*x), composed from existing autograd ops (no new C
    kernel): leaky = (1-slope)*relu(x) + slope*x, which is x for x>0 and
    slope*x for x<0, and differentiable through the same primitives relu uses."""
    r = relu(x)
    return add(scale(r, 1.0 - negative_slope), scale(x, negative_slope))
def gelu(x): return _unary("gelu", x)
def silu(x): return _unary("silu", x)
def sigmoid(x): return _unary("sigmoid", x)
def tanh(x): return _unary("tanh", x)
def exp(x): return _unary("exp", x)
def softplus(x): return _unary("softplus", x)
def mish(x: Tensor) -> Tensor:
    """mish(x) = x*tanh(softplus(x)), composed from existing autograd ops (no
    new C kernel): a smooth, self-gated activation like SiLU/GELU, and
    differentiable through the same tanh/softplus/mul primitives they use."""
    return mul(x, tanh(softplus(x)))
def log_sigmoid(x: Tensor) -> Tensor:
    """log(sigmoid(x)) = -softplus(-x), composed from existing autograd ops (no
    new C kernel): the numerically-stable log-sigmoid used e.g. in
    binary-cross-entropy-with-logits, differentiable through the same
    softplus/scale primitives softplus already uses."""
    return scale(softplus(scale(x, -1.0)), -1.0)
def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """ELU(x) = x for x>0, alpha*(exp(x)-1) for x<=0, composed from existing
    autograd ops (no new C kernel): x - relu(x) == min(x, 0), so
    relu(x) + alpha*(exp(x - relu(x)) - 1) is x on the positive branch and
    alpha*(exp(x)-1) on the negative one, differentiable through the same
    relu/exp/scale/add primitives leaky_relu and mish already use."""
    r = relu(x)
    neg = add(x, scale(r, -1.0))
    ones = tensor(np.ones(x.shape, dtype=np.float32))
    return add(r, scale(add(exp(neg), scale(ones, -1.0)), alpha))
def celu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """CELU(x) = x for x>0, alpha*(exp(x/alpha)-1) for x<=0, composed from
    existing autograd ops (no new C kernel): generalizes ELU by scaling
    inside the exponent too (ELU is the alpha=1 case), which keeps CELU
    1-Lipschitz for any alpha>0. Uses the same x - relu(x) == min(x, 0)
    trick as elu/leaky_relu, differentiable through the same
    relu/exp/scale/add primitives."""
    r = relu(x)
    neg = add(x, scale(r, -1.0))
    ones = tensor(np.ones(x.shape, dtype=np.float32))
    return add(r, scale(add(exp(scale(neg, 1.0 / alpha)), scale(ones, -1.0)), alpha))
def hardsigmoid(x: Tensor) -> Tensor:
    """Hardsigmoid(x) = clip((x+3)/6, 0, 1), composed from existing autograd
    ops (no new C kernel): the piecewise-linear sigmoid approximation used in
    MobileNetV3. min(y,1) is built the same way ELU builds min(x,0): 1 -
    relu(1-y) is y when y<=1 and 1 when y>1; clip(y,0,1) then follows from
    relu(y) (clamps below at 0) fed through that same min-with-1 step."""
    three = tensor(np.full(x.shape, 3.0, dtype=np.float32))
    ones = tensor(np.ones(x.shape, dtype=np.float32))
    lo = relu(scale(add(x, three), 1.0 / 6.0))
    return add(ones, scale(relu(add(ones, scale(lo, -1.0))), -1.0))
def hardswish(x: Tensor) -> Tensor:
    """Hardswish(x) = x*hardsigmoid(x), composed from existing autograd ops
    (no new C kernel): a piecewise-linear, division-free approximation of
    SiLU/Swish used in MobileNetV3, differentiable through the same
    relu/add/scale/mul primitives hardsigmoid and mish already use."""
    return mul(x, hardsigmoid(x))
def relu6(x: Tensor) -> Tensor:
    """ReLU6(x) = min(relu(x), 6), composed from existing autograd ops (no
    new C kernel): the clipped ReLU used in MobileNetV1/V2 to bound
    activation range for low-precision inference. Uses the same min(y,c) ==
    c - relu(c-y) trick hardsigmoid already uses to clip its upper bound,
    applied to relu(x) instead of a raw affine map."""
    r = relu(x)
    six = tensor(np.full(x.shape, 6.0, dtype=np.float32))
    return add(six, scale(relu(add(six, scale(r, -1.0))), -1.0))
def softshrink(x: Tensor, lambd: float = 0.5) -> Tensor:
    """Softshrink(x) = x-lambd for x>lambd, x+lambd for x<-lambd, else 0,
    composed from existing autograd ops (no new C kernel): the
    soft-thresholding operator used in LASSO/wavelet denoising. relu(x-lambd)
    is x-lambd on the upper branch and 0 elsewhere; relu(-x-lambd) is
    -(x+lambd) on the lower branch and 0 elsewhere (the branches never
    overlap for lambd>=0), so pos - neg reproduces all three pieces exactly,
    differentiable through the same relu/scale/add primitives elu/celu use."""
    lam = tensor(np.full(x.shape, lambd, dtype=np.float32))
    pos = relu(add(x, scale(lam, -1.0)))
    neg = relu(add(scale(x, -1.0), scale(lam, -1.0)))
    return add(pos, scale(neg, -1.0))
def tanhshrink(x: Tensor) -> Tensor:
    """Tanhshrink(x) = x - tanh(x), composed from existing autograd ops (no
    new C kernel): near-identity for large |x| (tanh saturates to +-1) and
    cubic-small near 0 (tanh(x)~=x), differentiable through the same
    tanh/add/scale primitives log_sigmoid/mish already use."""
    return add(x, scale(tanh(x), -1.0))
def hardtanh(x: Tensor, min_val: float = -1.0, max_val: float = 1.0) -> Tensor:
    """Hardtanh(x) = clamp(x, min_val, max_val), composed from existing
    autograd ops (no new C kernel): a piecewise-linear saturating gate used
    as a cheap tanh substitute and (at min_val=0, max_val=6) generalized by
    relu6 above. Two applications of the same max(a,b) == b + relu(a-b)
    trick relu6 uses: first clip the top (min(x,max_val) ==
    max_val - relu(max_val-x)), then clip the bottom of that
    (max(t,min_val) == min_val + relu(t-min_val)), differentiable through
    the same relu/add/scale primitives."""
    mx = tensor(np.full(x.shape, max_val, dtype=np.float32))
    mn = tensor(np.full(x.shape, min_val, dtype=np.float32))
    upper = add(mx, scale(relu(add(mx, scale(x, -1.0))), -1.0))
    return add(mn, relu(add(upper, scale(mn, -1.0))))
_SELU_ALPHA = 1.6732632423543772
_SELU_SCALE = 1.0507009873554804
def selu(x: Tensor) -> Tensor:
    """SELU(x) = scale*(x for x>0, alpha*(exp(x)-1) for x<=0) with the fixed
    self-normalizing constants from Klambauer et al., composed from existing
    autograd ops (no new C kernel): SELU is just ELU(x, alpha) rescaled, so
    this reuses elu's relu/exp/scale/add composition and adds one more scale."""
    return scale(elu(x, _SELU_ALPHA), _SELU_SCALE)
def mean(x): return _new(_lib.vgre_ag_mean(x._h), (1,))
def softmax(x): return _unary("softmax", x)
def softmin(x: Tensor) -> Tensor:
    """Softmin(x) = softmax(-x) over the last dim, composed from existing
    autograd ops (no new C kernel): the smallest entries get the largest
    weight, differentiable through the same softmax/scale primitives
    log_sigmoid already uses to negate its input."""
    return softmax(scale(x, -1.0))
def gaussian(x: Tensor) -> Tensor:
    """Gaussian(x) = exp(-x^2), composed from existing autograd ops (no new C
    kernel): the RBF-network activation, bell-shaped and bounded in (0,1],
    peaking at x=0 and vanishing as |x|->infinity (unlike the monotone
    activations above). Built from the same mul/scale/exp primitives elu
    already uses, with x squared via mul(x,x) rather than a new square op."""
    return exp(scale(mul(x, x), -1.0))
def maximum(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise max(a,b) = a + relu(b-a), composed from existing autograd
    ops (no new C kernel): the same max(x,c) == c + relu(x-c) trick
    relu6/hardtanh/softshrink already use against a constant, generalized
    here to two same-shape tensors. Differentiable through add/relu/scale:
    the gradient routes to whichever input was larger at each position (ties,
    where a==b, route through a, matching relu's tie-break at 0)."""
    return add(a, relu(add(b, scale(a, -1.0))))
def minimum(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise min(a,b) = -max(-a,-b), composed from existing autograd
    ops (no new C kernel): reuses maximum() above on the negated inputs,
    differentiable through the same add/relu/scale primitives."""
    return scale(maximum(scale(a, -1.0), scale(b, -1.0)), -1.0)
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

def softmax_cross_entropy_soft(logits: Tensor, soft_targets) -> Tensor:
    """Mean soft-target cross-entropy over rows of logits[M,V].

    `soft_targets` may be a Tensor or any array-like [M,V]. Rows are normalized
    by the native op and treated as constants, which is the loss needed for
    teacher/student distillation.
    """
    tgt = soft_targets if isinstance(soft_targets, Tensor) else tensor(soft_targets)
    return _new(_lib.vgre_ag_softmax_cross_entropy_soft(logits._h, tgt._h), (1,))

def label_smoothing_cross_entropy(logits: Tensor, targets: Sequence[int], smoothing: float = 0.1) -> Tensor:
    """Cross-entropy with label smoothing (Szegedy et al. 2016): instead of
    putting all target mass on the true class, spreads `smoothing` of it
    uniformly over all V classes — target[true] = 1-smoothing+smoothing/V,
    target[other] = smoothing/V — which discourages overconfident logits and
    tends to improve calibration/generalization. Built by constructing that
    distribution in NumPy and handing it to the existing
    `softmax_cross_entropy_soft` op (already row-normalizes and has a grad),
    so no new C++ kernel is needed. smoothing=0 reduces to `targets` getting
    all the mass, i.e. ordinary cross-entropy."""
    m, v = logits.shape
    smooth = np.full((m, v), smoothing / v, dtype=np.float32)
    smooth[np.arange(m), np.asarray(targets, dtype=np.int64)] += 1.0 - smoothing
    return softmax_cross_entropy_soft(logits, smooth)

def huber_loss(pred: Tensor, target: Tensor, delta: float = 1.0, reduction: str = "mean") -> Tensor:
    """Huber/SmoothL1 loss: 0.5*e^2 for |e|<=delta, delta*(|e|-0.5*delta) for
    |e|>delta, composed from existing autograd ops (no new C kernel, no
    division needed): 0.5*clip(|e|,0,delta)^2 + delta*(|e|-clip(|e|,0,delta))
    is exactly the piecewise formula above, since clip(|e|,0,delta)==|e| in
    the quadratic region (making the second term 0) and ==delta in the linear
    region (making the second term delta*(|e|-delta)). |e| is built the same
    relu(x)+relu(-x) way abs is elsewhere in this module, clip reuses
    hardtanh's relu/add/scale composition, and mean/mul/add/scale are the
    same primitives softmax_cross_entropy_soft-adjacent losses use; less
    sensitive to outliers than a squared-error loss, smoother than L1 near 0."""
    e = add(pred, scale(target, -1.0))
    abs_e = add(relu(e), relu(scale(e, -1.0)))
    clipped = hardtanh(abs_e, 0.0, delta)
    quadratic = scale(mul(clipped, clipped), 0.5)
    linear = scale(add(abs_e, scale(clipped, -1.0)), delta)
    per_elem = add(quadratic, linear)
    if reduction == "none":
        return per_elem
    if reduction == "sum":
        return scale(mean(per_elem), float(per_elem.shape[0] if len(per_elem.shape) == 1
                                            else int(np.prod(per_elem.shape))))
    return mean(per_elem)

def mse_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Mean squared error: (pred-target)^2, composed from existing autograd
    ops (add/scale/mul/mean, no new C kernel): e = pred-target, per-elem =
    e*e. Matches huber_loss's reduction handling ("mean"/"sum"/"none")."""
    e = add(pred, scale(target, -1.0))
    per_elem = mul(e, e)
    if reduction == "none":
        return per_elem
    if reduction == "sum":
        return scale(mean(per_elem), float(per_elem.shape[0] if len(per_elem.shape) == 1
                                            else int(np.prod(per_elem.shape))))
    return mean(per_elem)

def l1_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Mean absolute error: |pred-target|, composed from existing autograd
    ops (add/scale/relu/mean, no new C kernel): e = pred-target, |e| =
    relu(e)+relu(-e) (same abs-via-relu composition huber_loss uses).
    Matches huber_loss/mse_loss's reduction handling ("mean"/"sum"/"none")."""
    e = add(pred, scale(target, -1.0))
    per_elem = add(relu(e), relu(scale(e, -1.0)))
    if reduction == "none":
        return per_elem
    if reduction == "sum":
        return scale(mean(per_elem), float(per_elem.shape[0] if len(per_elem.shape) == 1
                                            else int(np.prod(per_elem.shape))))
    return mean(per_elem)

def kl_div_loss(logits: Tensor, target_probs, reduction: str = "mean") -> Tensor:
    """KL(target || softmax(logits)) = sum_v target*(log target - log softmax(logits)),
    averaged over rows ("mean", PyTorch's batchmean) or summed ("sum") — no new
    C kernel. `softmax_cross_entropy_soft` already computes the mean cross-entropy
    H(target, logits) = -sum target*log softmax(logits) with a correct gradient
    (mean row-wise (softmax(logits)-target)); KL = H(target,logits) - H(target),
    and the target entropy H(target) is a fixed NumPy scalar added as a
    constant (requires_grad=False) tensor, so it contributes nothing to the
    backward pass — the gradient w.r.t. logits is exactly softmax_cross_entropy_soft's,
    which already equals the analytic KL gradient since d/dlogits H(target) = 0."""
    probs = (target_probs.numpy() if isinstance(target_probs, Tensor)
             else np.asarray(target_probs, dtype=np.float32))
    m = probs.shape[0]
    row_entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    ce = softmax_cross_entropy_soft(logits, probs)
    kl = add(ce, tensor(np.array([-float(row_entropy.mean())], dtype=np.float32)))
    if reduction == "sum":
        return scale(kl, float(m))
    if reduction != "mean":
        raise ValueError("reduction must be 'mean' or 'sum'")
    return kl

def binary_cross_entropy_with_logits(logits: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """BCEWithLogits(x,t) = -[t*log(sigmoid(x)) + (1-t)*log(sigmoid(-x))],
    composed from existing autograd ops (no new C kernel): log_sigmoid(x) and
    log_sigmoid(-x) are the numerically-stable log(sigmoid) and log(1-sigmoid)
    terms (log_sigmoid already routes through softplus for stability), and
    lerping between them by t via b + t*(a-b) is exactly the two-term sum
    above since (1-t)*b + t*a == b + t*(a-b). Matches mse_loss/l1_loss's
    reduction handling ("mean"/"sum"/"none"); target may be soft (any value
    in [0,1], not just 0/1)."""
    pos = log_sigmoid(logits)
    neg = log_sigmoid(scale(logits, -1.0))
    per_elem = scale(add(neg, mul(target, add(pos, scale(neg, -1.0)))), -1.0)
    if reduction == "none":
        return per_elem
    if reduction == "sum":
        return scale(mean(per_elem), float(per_elem.shape[0] if len(per_elem.shape) == 1
                                            else int(np.prod(per_elem.shape))))
    return mean(per_elem)

def hinge_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Hinge loss for margin-based binary classification (linear SVMs):
    max(0, 1 - target*pred), target in {-1,+1}, composed from existing
    autograd ops (mul/add/scale/relu/mean, no new C kernel): margin =
    1 - target*pred (built via the same `tensor(np.full(...))` constant
    trick relu6/hardtanh use to add a literal), then relu(margin) is exactly
    the hinge — 0 once a sample clears the margin on the correct side,
    linear in the (uncapped) violation otherwise. Matches mse_loss/l1_loss's
    reduction handling ("mean"/"sum"/"none")."""
    one = tensor(np.full(pred.shape, 1.0, dtype=np.float32))
    margin = add(one, scale(mul(target, pred), -1.0))
    per_elem = relu(margin)
    if reduction == "none":
        return per_elem
    if reduction == "sum":
        return scale(mean(per_elem), float(per_elem.shape[0] if len(per_elem.shape) == 1
                                            else int(np.prod(per_elem.shape))))
    return mean(per_elem)

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


def structured_nm_mask(weight, n: int = 2, m: int = 4) -> np.ndarray:
    """Return a fixed N:M mask for a 2-D linear weight [in,out].

    For each output column and each group of `m` input-channel weights, the `n`
    largest-magnitude entries are kept. This is the standard contraction-axis
    layout used by 2:4 structured sparsity, and it composes directly with
    Linear's [in,out] weight layout.
    """
    if n <= 0 or m <= 0 or n > m:
        raise ValueError("structured sparsity requires 0 < n <= m")
    w = weight.numpy() if isinstance(weight, Tensor) else _f32(weight)
    if w.ndim != 2:
        raise ValueError("structured_nm_mask expects a 2-D weight [in,out]")
    rows, cols = w.shape
    mask = np.zeros((rows, cols), dtype=np.float32)
    for c in range(cols):
        for start in range(0, rows, m):
            stop = min(start + m, rows)
            group = np.abs(w[start:stop, c])
            keep = min(n, stop - start)
            if keep <= 0:
                continue
            if keep == group.size:
                idx = np.arange(group.size)
            else:
                idx = np.argpartition(-group, keep - 1)[:keep]
            mask[start + idx, c] = 1.0
    return mask


def structured_nm_metadata(mask, n: int = 2, m: int = 4) -> np.ndarray:
    """Pack an N:M mask into bit metadata [ceil(in/m), out].

    Bit `i` in each group is 1 when the corresponding input-channel weight is
    present. The function validates that every full group has exactly `n` kept
    weights and every tail group has `min(n, tail)` kept weights.
    """
    if n <= 0 or m <= 0 or n > m:
        raise ValueError("structured sparsity requires 0 < n <= m")
    ma = _f32(mask)
    if ma.ndim != 2:
        raise ValueError("structured_nm_metadata expects a 2-D mask")
    rows, cols = ma.shape
    groups = (rows + m - 1) // m
    meta = np.zeros((groups, cols), dtype=np.uint32)
    for c in range(cols):
        for g, start in enumerate(range(0, rows, m)):
            stop = min(start + m, rows)
            bits = 0
            kept = 0
            for i, v in enumerate(ma[start:stop, c]):
                if v != 0.0:
                    bits |= (1 << i)
                    kept += 1
            expected = min(n, stop - start)
            if kept != expected:
                raise ValueError(f"group {g}, column {c} keeps {kept}, expected {expected}")
            meta[g, c] = bits
    return meta


class StructuredSparseLinear(Module):
    """N:M structured sparse linear layer with a fixed pruning mask.

    The trainable fp32 master weight is physically zeroed outside the mask and
    forward() uses W * mask, so pruned positions contribute no activation and
    receive zero gradient. This is quantization/pruning-aware training without
    adding a dependency or a new backend kernel; metadata() exposes the compact
    bit patterns a sparse micro-kernel can consume.
    """
    def __init__(self, in_features: int, out_features: int, n: int = 2, m: int = 4,
                 bias: bool = True, base_weight=None, base_bias=None):
        w = _f32(base_weight) if base_weight is not None else \
            _f32(rng.standard_normal((in_features, out_features)) *
                 (2.0 / in_features) ** 0.5)
        if w.shape != (in_features, out_features):
            raise ValueError(f"base_weight must be [{in_features},{out_features}]")
        self.n, self.m = int(n), int(m)
        self._mask = structured_nm_mask(w, self.n, self.m)
        self.W = tensor(w * self._mask, requires_grad=True)
        if bias:
            b = _f32(base_bias) if base_bias is not None else np.zeros(out_features)
            self.b = tensor(b, requires_grad=True)
        else:
            self.b = None

    @classmethod
    def from_linear(cls, linear: "Linear", n: int = 2, m: int = 4) -> "StructuredSparseLinear":
        w = linear.W.numpy()
        b = linear.b.numpy() if linear.b is not None else None
        return cls(w.shape[0], w.shape[1], n=n, m=m,
                   bias=b is not None, base_weight=w, base_bias=b)

    @property
    def mask(self) -> np.ndarray:
        return self._mask.copy()

    def density(self) -> float:
        return float(np.mean(self._mask != 0.0))

    def metadata(self) -> np.ndarray:
        return structured_nm_metadata(self._mask, self.n, self.m)

    def sparse_weight(self) -> np.ndarray:
        return self.W.numpy() * self._mask

    def prune_from_current(self) -> None:
        """Recompute the N:M mask from current weights and zero pruned entries."""
        w = self.W.numpy()
        self._mask = structured_nm_mask(w, self.n, self.m)
        self.W.set_(w * self._mask)

    def _masked_weight(self) -> Tensor:
        return mul(self.W, tensor(self._mask))

    def forward(self, x):
        Wm = self._masked_weight()
        s = x.shape
        if len(s) == 3:
            flat = reshape(x, (s[0] * s[1], s[2]))
            y = matmul(flat, Wm)
            if self.b is not None:
                y = add(y, self.b)
            return reshape(y, (s[0], s[1], self.W.shape[1]))
        y = matmul(x, Wm)
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


# OCP Microscaling MXFP4: 4-bit E2M1 elements (sign + 2-bit exp + 1-bit mantissa
# -> magnitudes {0,.5,1,1.5,2,3,4,6}) sharing one power-of-two E8M0 scale per
# block of 32 elements along the reduction (in_features) axis. Mirrors the
# quantize/dequantize contract of include/vgre/xla/mxfp4.h, in plain NumPy so
# the Python base path needs no new C-ABI surface. 4 + 8/32 = 4.25 bits/weight.
_MXFP4_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_MXFP4_BLOCK = 32


def _mxfp4_quantize(w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """w: [K,N] fp32. Returns (codes [K,N] int8 in [-7,7], scale [nblocks,N]
    fp32 power-of-two per 32-row block). codes encode sign*magnitude_index
    into _MXFP4_MAGNITUDES; 0 is its own sign."""
    K, N = w.shape
    nblocks = (K + _MXFP4_BLOCK - 1) // _MXFP4_BLOCK
    pad = nblocks * _MXFP4_BLOCK - K
    wp = np.pad(w, ((0, pad), (0, 0))) if pad else w
    wp = wp.reshape(nblocks, _MXFP4_BLOCK, N)
    amax = np.abs(wp).max(axis=1)                                    # [nblocks,N]
    safe_amax = np.where(amax == 0.0, 1.0, amax)
    scale = np.where(amax == 0.0, 1.0,
                      np.exp2(np.floor(np.log2(safe_amax)) - 2.0)).astype(np.float32)
    scaled = wp / scale[:, None, :]
    mag_idx = np.argmin(np.abs(np.abs(scaled)[..., None] - _MXFP4_MAGNITUDES), axis=-1)
    codes = np.where(scaled < 0, -mag_idx, mag_idx).astype(np.int8)
    return codes.reshape(nblocks * _MXFP4_BLOCK, N)[:K], scale


def _mxfp4_dequantize(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    K = codes.shape[0]
    mags = _MXFP4_MAGNITUDES[np.abs(codes)]
    vals = np.where(codes < 0, -mags, mags)
    scale_full = np.repeat(scale, _MXFP4_BLOCK, axis=0)[:K]
    return vals * scale_full


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
                 base_bias=None, base_format: str = "ternary"):
        if r <= 0:
            raise ValueError("QLoRA rank r must be positive")
        if base_format not in ("ternary", "int4", "mxfp4"):
            raise ValueError("base_format must be 'ternary', 'int4', or 'mxfp4'")
        w = _f32(base_weight) if base_weight is not None else \
            _f32(rng.standard_normal((in_features, out_features)) *
                 (2.0 / in_features) ** 0.5)
        if w.shape != (in_features, out_features):
            raise ValueError(f"base_weight must be [{in_features},{out_features}]")
        self.base_format = base_format
        # Freeze the base as low-bit codes + a per-column scale (numpy, so never
        # registered as trainable parameters). ternary: {-1,0,+1} at the absmean
        # scale (~2 bits). int4: symmetric [-7,7] at the absmax/7 scale (~4 bits,
        # more faithful for weights with a few large outliers).
        if base_format == "ternary":
            scale = np.abs(w).mean(axis=0).astype(np.float32)          # [out]
            inv = np.where(scale == 0.0, 1.0, scale)
            self._codes = np.clip(np.rint(w / inv), -1, 1).astype(np.int8)
            self._bits = 2
        elif base_format == "int4":
            scale = (np.abs(w).max(axis=0) / 7.0).astype(np.float32)   # [out]
            inv = np.where(scale == 0.0, 1.0, scale)
            self._codes = np.clip(np.rint(w / inv), -7, 7).astype(np.int8)
            self._bits = 4
        else:  # mxfp4: codes [in,out] int8, scale [nblocks,out] power-of-two
            self._codes, scale = _mxfp4_quantize(w)
            self._bits = 4
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
    def from_linear(cls, linear: "Linear", r: int = 8, alpha: float = 16.0,
                    base_format: str = "ternary") -> "QLoRALinear":
        """Quantize an existing Linear's weights as the frozen base."""
        w = linear.W.numpy()
        b = linear.b.numpy() if linear.b is not None else None
        return cls(w.shape[0], w.shape[1], r=r, alpha=alpha,
                   bias=b is not None, base_weight=w, base_bias=b, base_format=base_format)

    @property
    def scaling(self) -> float:
        return self.alpha / self.r

    def adapter_parameters(self) -> List["Tensor"]:
        return [self.A, self.B]

    def base_bits_per_weight(self) -> float:
        """Effective storage of the frozen base: the code width (~2-bit ternary /
        ~4-bit int4 / ~4-bit mxfp4) plus the scale overhead — one fp32 per column
        for ternary/int4, one 8-bit E8M0 exponent per 32-row block for mxfp4."""
        n = self._codes.size
        scale_bits = 8 if self.base_format == "mxfp4" else 32
        return (n * self._bits + self._scale.size * scale_bits) / n

    def _base(self):                                   # dequant transiently (no grad)
        if self.base_format == "mxfp4":
            return tensor(_mxfp4_dequantize(self._codes, self._scale))
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


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01):
        self.negative_slope = negative_slope

    def forward(self, x): return leaky_relu(x, self.negative_slope)


class GELU(Module):
    def forward(self, x): return gelu(x)


class SiLU(Module):
    def forward(self, x): return silu(x)


class Sigmoid(Module):
    def forward(self, x): return sigmoid(x)


class Tanh(Module):
    def forward(self, x): return tanh(x)


class Softplus(Module):
    def forward(self, x): return softplus(x)


class Mish(Module):
    def forward(self, x): return mish(x)


class LogSigmoid(Module):
    def forward(self, x): return log_sigmoid(x)


class ELU(Module):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def forward(self, x): return elu(x, self.alpha)


class CELU(Module):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def forward(self, x): return celu(x, self.alpha)


class Hardsigmoid(Module):
    def forward(self, x): return hardsigmoid(x)


class Hardswish(Module):
    def forward(self, x): return hardswish(x)


class ReLU6(Module):
    def forward(self, x): return relu6(x)


class Softshrink(Module):
    def __init__(self, lambd: float = 0.5):
        self.lambd = lambd

    def forward(self, x): return softshrink(x, self.lambd)


class Tanhshrink(Module):
    def forward(self, x): return tanhshrink(x)


class Hardtanh(Module):
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x): return hardtanh(x, self.min_val, self.max_val)


class SELU(Module):
    def forward(self, x): return selu(x)


class Softmin(Module):
    def forward(self, x): return softmin(x)


class Gaussian(Module):
    def forward(self, x): return gaussian(x)


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


class SwiGLUMLP(Module):
    """Gated FFN: down(silu(gate(x)) * up(x)) — the SwiGLU feed-forward used in
    place of a ReLU/GELU two-layer MLP in Llama/Mistral/Qwen-style transformers
    (Shazaeer, 2020). Composed entirely from existing ops (Linear, silu, mul),
    no new C kernel: silu(gate(x)) is the gate that modulates up(x) elementwise
    before the down projection back to `dim`."""
    def __init__(self, dim: int, hidden: int = 0):
        hidden = hidden or 4 * dim
        self.gate = Linear(dim, hidden, bias=False)
        self.up = Linear(dim, hidden, bias=False)
        self.down = Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(mul(silu(self.gate(x)), self.up(x)))


class GeGLUMLP(Module):
    """Gated FFN: down(gelu(gate(x)) * up(x)) — the GEGLU feed-forward used in
    place of a plain ReLU/GELU two-layer MLP in T5 v1.1/PaLM-style transformers
    (Shazeer, 2020). Composed entirely from existing ops (Linear, gelu, mul),
    no new C kernel: identical structure to SwiGLUMLP with the gate activation
    swapped from silu to gelu."""
    def __init__(self, dim: int, hidden: int = 0):
        hidden = hidden or 4 * dim
        self.gate = Linear(dim, hidden, bias=False)
        self.up = Linear(dim, hidden, bias=False)
        self.down = Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(mul(gelu(self.gate(x)), self.up(x)))


class GLU(Module):
    """Gated Linear Unit: (xW+b) * sigmoid(xV+c) (Dauphin et al. 2016) -- the
    atomic gating building block behind the `gate * up` pattern SwiGLUMLP and
    GeGLUMLP already wrap around a down-projection, exposed standalone for
    gated convnets and PixelCNN-style gated activations. The reference
    formulation splits one projection's output in half; this uses two
    separate Linear layers instead (mathematically equivalent -- a single
    [in, 2*out] projection split in half is the same as two independent
    [in, out] projections, and the C ABI here has no split/slice op), the
    same trick SwiGLUMLP/GeGLUMLP already use for their gate/up pair.
    Composed entirely from existing ops (Linear, sigmoid, mul), no new C
    kernel."""
    def __init__(self, in_features: int, out_features: int):
        self.value = Linear(in_features, out_features)
        self.gate = Linear(in_features, out_features)

    def forward(self, x):
        return mul(self.value(x), sigmoid(self.gate(x)))


class Maxout(Module):
    """Maxout(in_features, out_features, num_pieces=2): each output unit is
    the elementwise max over `num_pieces` independent affine projections of
    the input (Goodfellow et al. 2013) -- a piecewise-linear, non-saturating
    activation that is itself learned rather than fixed (with tied pieces it
    can represent ReLU, abs, and other convex piecewise-linear functions).
    Built from `num_pieces` Linear layers plus the maximum() primitive above;
    no new C kernel needed since maximum() is itself an add/relu/scale
    composition. `self.pieces` is a plain list of Modules, which Module's
    `_children`/`named_parameters` already walk (see SwiGLUMLP's gate/up/down
    for the single-child equivalent)."""
    def __init__(self, in_features: int, out_features: int, num_pieces: int = 2):
        assert num_pieces >= 2, "Maxout needs at least 2 pieces"
        self.pieces = [Linear(in_features, out_features) for _ in range(num_pieces)]

    def forward(self, x):
        out = self.pieces[0](x)
        for piece in self.pieces[1:]:
            out = maximum(out, piece(x))
        return out


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
    def __init__(self, dim: int, d_state: int = 8, conv_kernel: int = 4):
        self.dim = dim
        self.N = d_state
        self.conv_kernel = conv_kernel
        self.in_proj = Linear(dim, dim)                 # u_t (SSM input channels)
        self.dt_proj = Linear(dim, dim)                 # Δ_t (per-channel step)
        self.B_proj = Linear(dim, d_state)              # B_t [T,N]
        self.C_proj = Linear(dim, d_state)              # C_t [T,N]
        A0 = np.log(np.tile(np.arange(1, d_state + 1, dtype=np.float32), (dim, 1)))
        self.A_log = tensor(A0.astype(np.float32), requires_grad=True)   # [dim,N]
        self.Dskip = tensor(np.ones(dim, np.float32), requires_grad=True)
        self.out_proj = Linear(dim, dim)
        if conv_kernel > 0:
            # Depthwise causal short conv over the sequence (one filter per
            # channel), the Mamba component that mixes a few adjacent tokens
            # before the SSM. Weights [dim, kernel]; identity-ish init (last tap 1).
            cw = np.zeros((dim, conv_kernel), np.float32); cw[:, -1] = 1.0
            self.conv_w = tensor(cw, requires_grad=True)
            self.conv_b = tensor(np.zeros(dim, np.float32), requires_grad=True)

    def _causal_conv1d(self, u):
        # Depthwise causal conv: y[t,d] = Σ_i conv_w[d,i]·u[t-(k-1)+i, d] + b[d],
        # left-padded with zeros so it never reads the future. Built from concat +
        # index_select + broadcast-mul (all differentiable), then SiLU-gated.
        T, dim, k = u.shape[0], self.dim, self.conv_kernel
        up = concat(tensor(np.zeros((k - 1, dim), np.float32)), u, 0)     # [T+k-1, dim]
        onesT = tensor(np.ones((T, 1), np.float32))
        y = None
        for i in range(k):
            win = index_select(up, list(range(i, i + T)))                # [T,dim] u[t-(k-1)+i]
            wi = matmul(onesT, reshape(matmul(self.conv_w,
                        tensor(np.eye(k, dtype=np.float32)[:, i:i + 1].copy())), (1, dim)))
            term = mul(win, wi)
            y = term if y is None else add(y, term)
        y = add(y, matmul(onesT, reshape(self.conv_b, (1, dim))))        # + bias
        return silu(y)

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
        if self.conv_kernel > 0:
            u = self._causal_conv1d(u)                   # depthwise short conv + SiLU
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


def _flatten_logits(logits: "Tensor") -> "Tensor":
    if len(logits.shape) == 2:
        return logits
    if len(logits.shape) == 3:
        return reshape(logits, (logits.shape[0] * logits.shape[1], logits.shape[2]))
    raise ValueError("logits must be [M,V] or [B,M,V]")


def _softmax_rows_np(a, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    z = np.asarray(a, dtype=np.float64) / float(temperature)
    z -= z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    return p.astype(np.float32)


def distillation_loss(student_logits: "Tensor", teacher_logits_or_probs,
                      temperature: float = 2.0, hard_targets: Sequence[int] = None,
                      hard_weight: float = 0.0, teacher_is_probs: bool = False) -> "Tensor":
    """Teacher/student distillation loss.

    The soft part is T^2 * CE(softmax(student/T), softmax(teacher/T)); teacher
    probabilities are materialized as constants, so gradients update only the
    student. Optionally blend hard-label CE with `hard_weight` in [0,1].
    """
    if not 0.0 <= hard_weight <= 1.0:
        raise ValueError("hard_weight must be in [0,1]")
    student2 = _flatten_logits(student_logits)
    teacher = (teacher_logits_or_probs.numpy()
               if isinstance(teacher_logits_or_probs, Tensor)
               else np.asarray(teacher_logits_or_probs, dtype=np.float32))
    teacher = teacher.reshape((-1, teacher.shape[-1]))
    if teacher.shape != student2.shape:
        raise ValueError(f"teacher shape {teacher.shape} does not match student logits {student2.shape}")
    if teacher_is_probs:
        probs = np.asarray(teacher, dtype=np.float32)
        rows = probs.sum(axis=1, keepdims=True)
        if np.any(rows <= 0.0) or np.any(probs < 0.0):
            raise ValueError("teacher probabilities must be non-negative with positive row sums")
        probs = probs / rows
    else:
        probs = _softmax_rows_np(teacher, temperature)
    soft = scale(softmax_cross_entropy_soft(scale(student2, 1.0 / float(temperature)), probs),
                 float(temperature) * float(temperature))
    if hard_targets is None or hard_weight == 0.0:
        return soft
    hard = softmax_cross_entropy(student2, list(np.asarray(hard_targets).reshape(-1)))
    if hard_weight == 1.0:
        return hard
    return add(scale(soft, 1.0 - hard_weight), scale(hard, hard_weight))


def distill_step(student, teacher, inputs, optimizer: "_Optimizer",
                 temperature: float = 2.0, hard_targets: Sequence[int] = None,
                 hard_weight: float = 0.0, teacher_is_probs: bool = False,
                 clip: float = 1.0) -> float:
    """One optimizer step for framework-free distillation.

    `student` and `teacher` are callables that accept `inputs`, or `teacher` may
    be precomputed logits/probabilities. The teacher path is converted to
    constants before the loss is built.
    """
    optimizer.zero_grad()
    student_logits = student(inputs)
    teacher_logits = teacher(inputs) if callable(teacher) else teacher
    loss = distillation_loss(student_logits, teacher_logits, temperature,
                             hard_targets=hard_targets, hard_weight=hard_weight,
                             teacher_is_probs=teacher_is_probs)
    loss.backward()
    optimizer.step(clip=clip)
    return loss.item()


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
