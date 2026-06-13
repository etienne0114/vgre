"""StableHLO (MLIR) → VGRE HLO translation.

Walks the module JAX actually lowers a jitted function to (`jax.jit(f).lower(...)
.compiler_ir('stablehlo')`) and rebuilds it as a VGRE HLO module via the builder
C ABI, then compiles it to an executable. This is the front half of an XLA/PJRT
backend: consume what the framework emits and run it on the VGRE engine.

Only float32 dense tensors are handled (VGRE HLO's domain). Unsupported ops raise
— nothing is silently approximated.
"""
from __future__ import annotations

import re

import numpy as np
from jaxlib.mlir.ir import ShapedType

from .runtime import VgreHlo

_BINARY = {
    "stablehlo.add": "Add", "stablehlo.subtract": "Subtract",
    "stablehlo.multiply": "Multiply", "stablehlo.divide": "Divide",
    "stablehlo.maximum": "Maximum", "stablehlo.minimum": "Minimum",
    "stablehlo.power": "Power",
    "stablehlo.and": "And", "stablehlo.or": "Or", "stablehlo.xor": "Xor",
}
_UNARY = {
    "stablehlo.negate": "Negate", "stablehlo.exponential": "Exp",
    "stablehlo.log": "Log", "stablehlo.tanh": "Tanh",
    "stablehlo.abs": "Abs", "stablehlo.rsqrt": "Rsqrt",
    "stablehlo.sqrt": "Sqrt", "stablehlo.sign": "Sign",
    "stablehlo.floor": "Floor", "stablehlo.ceil": "Ceil",
    "stablehlo.logistic": "Logistic",
    "stablehlo.cosine": "Cos", "stablehlo.sine": "Sin",
    "stablehlo.not": "Not", "stablehlo.is_finite": "IsFinite",
    "stablehlo.round_nearest_even": "RoundNearestEven",
    "stablehlo.round_nearest_afz": "RoundNearestAfz",
    "chlo.erf": "Erf", "chlo.erfc": "Erfc",
}
_REDUCE_KIND = {
    "stablehlo.add": "sum", "stablehlo.maximum": "max",
    "stablehlo.minimum": "min", "stablehlo.multiply": "prod",
}


def _shape(value) -> list[int]:
    return list(ShapedType(value.type).shape)


def _i64_list(attr) -> list[int]:
    return [int(x) for x in attr]


def _const_floats(attr):
    """Extract a flat float list from a Dense*ElementsAttr."""
    try:
        return np.asarray(attr, dtype=np.float32).ravel()
    except Exception:
        pass
    # splat / fallback paths
    try:
        return np.array([float(x) for x in attr], dtype=np.float32)
    except Exception as e:  # pragma: no cover
        raise NotImplementedError(f"cannot read constant attr {attr}") from e


class Translator:
    def __init__(self, lib_path: str | None = None):
        self.rt = VgreHlo(lib_path)

    def compile_module(self, module) -> "Executable":
        rt = self.rt
        b = rt.new_builder()
        # index every func by its symbol name so func.call can be inlined
        self._funcs = {}
        for op in module.body.operations:
            if op.operation.name == "func.func":
                self._funcs[str(op.attributes["sym_name"]).strip('"@')] = op
        try:
            fn = self._main_func(module)
            block = fn.body.blocks[0]
            arg_ids = [rt.parameter(b, i, _shape(a)) for i, a in enumerate(block.arguments)]
            ids, _ = self._emit_block(b, block, arg_ids)
            if not ids:
                raise NotImplementedError("function has no return")
            # output leaf shapes (a function may return several arrays / a pytree)
            ret = [op for op in block.operations
                   if op.operation.name in ("func.return", "stablehlo.return")][-1]
            out_shapes = [_shape(v) for v in ret.operands]
            root = rt.tuple(b, ids) if len(ids) > 1 else ids[0]
            rt.set_root(b, root)
            exe = rt.compile(b)
            b = 0  # consumed
            return Executable(rt, exe, out_shapes)
        finally:
            if b:
                rt.free_builder(b)

    def _emit_block(self, b, block, arg_ids):
        """Emit a block with its arguments pre-bound to instruction ids.

        Returns (return_ids, return_scalars) for the block's terminator — a list
        because while-bodies / multi-result calls yield several values."""
        rt = self.rt
        val2id: dict = dict(zip(block.arguments, arg_ids))
        const_scalar: dict = {}  # value -> python float (for reduce inits / call args)
        for op in block.operations:
            name = op.operation.name
            if name in ("func.return", "stablehlo.return"):
                ids = [val2id[o] for o in op.operands]
                scs = [const_scalar.get(o) for o in op.operands]
                return ids, scs
            if name == "stablehlo.while":
                rids = self._emit_while(b, op, val2id, const_scalar)
            elif name == "stablehlo.sort":
                rids = self._emit_sort(b, op, val2id)
            elif name == "stablehlo.reduce" and len(op.results) > 1:
                rids = self._emit_reduce_variadic(b, op, val2id)
            elif name in ("chlo.top_k", "stablehlo.top_k"):
                rids = self._emit_top_k(b, op, val2id)
            elif name in ("func.call", "call"):
                rids, rscs = self._emit_call(b, op, val2id, const_scalar)
                for k, res in enumerate(op.results):
                    val2id[res] = rids[k]
                    if k < len(rscs) and rscs[k] is not None:
                        const_scalar[res] = rscs[k]
                continue
            else:
                rid, sc = self._emit(rt, b, op, name, val2id, const_scalar)
                rids = [rid] if rid is not None else []
                if rid is not None and sc is not None:
                    const_scalar[op.results[0]] = sc
            for k, res in enumerate(op.results):
                if k < len(rids):
                    val2id[res] = rids[k]
        return [], []

    def _emit_call(self, b, op, val2id, const_scalar):
        callee_name = str(op.attributes["callee"]).strip('"@')
        callee = self._funcs.get(callee_name)
        if callee is None:
            raise NotImplementedError(f"call to unknown function {callee_name}")
        arg_ids = [val2id[o] for o in op.operands]
        return self._emit_block(b, callee.body.blocks[0], arg_ids)

    def _emit_while(self, b, op, val2id, const_scalar):
        """Translate a stablehlo.while: build cond/body sub-modules, emit the loop,
        then expose each loop-carried result via get_tuple_element."""
        rt = self.rt
        init_ids = [val2id[o] for o in op.operands]
        cond_region, body_region = op.regions[0], op.regions[1]

        cb = rt.new_builder()
        cblk = cond_region.blocks[0]
        cargs = [rt.parameter(cb, i, _shape(a)) for i, a in enumerate(cblk.arguments)]
        cids, _ = self._emit_block(cb, cblk, cargs)
        rt.set_root(cb, cids[0])

        bb = rt.new_builder()
        bblk = body_region.blocks[0]
        bargs = [rt.parameter(bb, i, _shape(a)) for i, a in enumerate(bblk.arguments)]
        bids, _ = self._emit_block(bb, bblk, bargs)
        rt.set_root(bb, rt.tuple(bb, bids))  # body yields the carried tuple

        primary = _shape(op.results[0]) if len(op.results) else []
        w = rt.while_op(b, init_ids, cb, bb, primary)  # consumes cb, bb
        return [rt.get_tuple_element(b, w, k, _shape(res)) for k, res in enumerate(op.results)]

    def _emit_sort(self, b, op, val2id):
        """stablehlo.sort: build the comparator sub-module, emit Sort, expose
        each sorted operand via get_tuple_element."""
        rt = self.rt
        operands = [val2id[o] for o in op.operands]
        dim = int(op.attributes["dimension"])
        cblk = op.regions[0].blocks[0]
        cb = rt.new_builder()
        cargs = [rt.parameter(cb, i, _shape(a)) for i, a in enumerate(cblk.arguments)]
        cids, _ = self._emit_block(cb, cblk, cargs)
        rt.set_root(cb, cids[0])
        s = rt.sort(b, operands, dim, cb)  # consumes cb
        return [rt.get_tuple_element(b, s, k, _shape(res)) for k, res in enumerate(op.results)]

    def _emit_reduce_variadic(self, b, op, val2id):
        """Variadic stablehlo.reduce (e.g. argmax over value+index): build the
        reducer body sub-module and emit ReduceGeneral."""
        rt = self.rt
        operands = [val2id[o] for o in op.operands]   # n inputs then n inits
        dims = _i64_list(op.attributes["dimensions"])
        bblk = op.regions[0].blocks[0]
        bb = rt.new_builder()
        bargs = [rt.parameter(bb, i, _shape(a)) for i, a in enumerate(bblk.arguments)]
        bids, _ = self._emit_block(bb, bblk, bargs)
        rt.set_root(bb, rt.tuple(bb, bids))
        r = rt.reduce_general(b, operands, dims, bb, _shape(op.results[0]))
        return [rt.get_tuple_element(b, r, k, _shape(res)) for k, res in enumerate(op.results)]

    def _emit_top_k(self, b, op, val2id):
        """chlo.top_k -> iota + descending sort(value, index) + slice first k.
        Returns (top values, top indices)."""
        rt = self.rt
        operand = val2id[op.operands[0]]
        in_shape = _shape(op.operands[0])
        out_shape = _shape(op.results[0])
        last = len(in_shape) - 1
        k = out_shape[last]
        idx = rt.iota(b, in_shape, last)
        # descending comparator: a precedes b iff a_value > b_value
        cb = rt.new_builder()
        a0 = rt.parameter(cb, 0, [])
        b0 = rt.parameter(cb, 1, [])
        rt.parameter(cb, 2, [])
        rt.parameter(cb, 3, [])
        rt.set_root(cb, rt.compare(cb, a0, b0, "GT"))
        s = rt.sort(b, [operand, idx], last, cb)
        sv = rt.get_tuple_element(b, s, 0, in_shape)
        si = rt.get_tuple_element(b, s, 1, in_shape)
        starts = [0] * len(in_shape)
        limits = list(in_shape)
        limits[last] = k
        strides = [1] * len(in_shape)
        return [rt.slice(b, sv, starts, limits, strides, out_shape),
                rt.slice(b, si, starts, limits, strides, out_shape)]

    # ── op emission ──────────────────────────────────────────────────────────
    def _emit(self, rt, b, op, name, val2id, const_scalar):
        def ref(i):
            return val2id[op.operands[i]]

        if name in _BINARY:
            return rt.binary(b, _BINARY[name], ref(0), ref(1)), None
        if name in _UNARY:
            return rt.unary(b, _UNARY[name], ref(0)), None
        if name == "chlo.square":
            return rt.binary(b, "Multiply", ref(0), ref(0)), None
        if name == "stablehlo.constant":
            data = _const_floats(op.attributes["value"])
            shp = _shape(op.results[0])
            sc = float(data[0]) if data.size == 1 else None
            return rt.constant(b, shp, data), sc
        if name == "stablehlo.dot":
            # plain 2-D matmul [M,K]·[K,N] (TF emits this; jax emits dot_general)
            return rt.dot(b, ref(0), ref(1)), None
        if name == "stablehlo.dot_general":
            lb, rb, lc, rc = self._dot_dims(op)
            lshape, rshape = _shape(op.operands[0]), _shape(op.operands[1])
            # fast path: a plain 2-D matmul [M,K]·[K,N] (no batch) uses the
            # contiguous Dot kernel instead of the general per-element gather.
            if (not lb and not rb and lc == [1] and rc == [0]
                    and len(lshape) == 2 and len(rshape) == 2):
                return rt.dot(b, ref(0), ref(1)), None
            return rt.dot_general(b, ref(0), ref(1), lb, rb, lc, rc, _shape(op.results[0])), None
        if name == "stablehlo.concatenate":
            dim = int(op.attributes["dimension"])
            ids = [val2id[o] for o in op.operands]
            return rt.concatenate(b, ids, dim, _shape(op.results[0])), None
        if name == "stablehlo.slice":
            starts = _i64_list(op.attributes["start_indices"])
            limits = _i64_list(op.attributes["limit_indices"])
            strides = _i64_list(op.attributes["strides"])
            return rt.slice(b, ref(0), starts, limits, strides, _shape(op.results[0])), None
        if name == "stablehlo.pad":
            low = _i64_list(op.attributes["edge_padding_low"])
            high = _i64_list(op.attributes["edge_padding_high"])
            interior = _i64_list(op.attributes["interior_padding"])
            return rt.pad(b, ref(0), ref(1), low, high, interior, _shape(op.results[0])), None
        if name == "stablehlo.convolution":
            return self._emit_conv(rt, b, op, ref), None
        if name == "stablehlo.gather":
            return self._emit_gather(rt, b, op, ref), None
        if name == "stablehlo.scatter":
            return self._emit_scatter(b, op, val2id), None
        if name == "stablehlo.reduce_window":
            return self._emit_reduce_window(rt, b, op, ref), None
        if name == "stablehlo.dynamic_slice":
            sizes = _i64_list(op.attributes["slice_sizes"])
            starts = [val2id[o] for o in op.operands[1:]]
            return rt.dynamic_slice(b, ref(0), starts, sizes, _shape(op.results[0])), None
        if name == "stablehlo.dynamic_update_slice":
            starts = [val2id[o] for o in op.operands[2:]]
            return rt.dynamic_update_slice(b, ref(0), ref(1), starts, _shape(op.results[0])), None
        if name == "stablehlo.broadcast_in_dim":
            bdims = _i64_list(op.attributes["broadcast_dimensions"])
            return rt.broadcast(b, ref(0), _shape(op.results[0]), bdims), None
        if name == "stablehlo.reshape":
            return rt.reshape(b, ref(0), _shape(op.results[0])), None
        if name == "stablehlo.transpose":
            perm = _i64_list(op.attributes["permutation"])
            return rt.transpose(b, ref(0), perm), None
        if name == "stablehlo.reverse":
            dims = _i64_list(op.attributes["dimensions"])
            return rt.reverse(b, ref(0), dims), None
        if name == "stablehlo.iota":
            dim = int(op.attributes["iota_dimension"])
            return rt.iota(b, _shape(op.results[0]), dim), None
        if name == "stablehlo.compare":
            d = self._compare_dir(op)
            return rt.compare(b, ref(0), ref(1), d), None
        if name == "stablehlo.select":
            return rt.select(b, ref(0), ref(1), ref(2)), None
        if name == "stablehlo.reduce":
            return self._emit_reduce(rt, b, op, ref, const_scalar), None
        if name == "stablehlo.convert":
            # Single-dtype float32 engine: convert is dtype plumbing (f32<->bf16,
            # integer index params carried as float). Identity — and propagate a
            # known scalar so a converted constant can still seed a reduce init.
            return ref(0), const_scalar.get(op.operands[0])
        raise NotImplementedError(f"unsupported StableHLO op: {name}")

    def _emit_reduce(self, rt, b, op, ref, const_scalar):
        # operands: [input, init]; body block's combiner op decides the kind.
        body = op.regions[0].blocks[0]
        kind = None
        for inner in body.operations:
            if inner.operation.name in _REDUCE_KIND:
                kind = _REDUCE_KIND[inner.operation.name]
                break
        if kind is None:
            raise NotImplementedError(f"unsupported reduce combiner: {op}")
        init_val = const_scalar.get(op.operands[1])
        if init_val is None:
            raise NotImplementedError("reduce init must be a constant scalar")
        dims = _i64_list(op.attributes["dimensions"])
        return rt.reduce(b, ref(0), dims, kind, init_val)

    # ── attribute parsing helpers ────────────────────────────────────────────
    @staticmethod
    def _compare_dir(op) -> str:
        s = str(op.attributes["comparison_direction"])
        m = re.search(r"(GT|GE|LT|LE|EQ|NE)", s)
        if not m:
            raise NotImplementedError(f"compare direction: {s}")
        return m.group(1)

    @staticmethod
    def _ints(s, key):
        m = re.search(key + r"\s*=\s*\[([^\]]*)\]", s)
        if not m:
            return []
        return [int(x) for x in re.split(r"[,\s]+", m.group(1).strip()) if x]

    @classmethod
    def _dot_dims(cls, op):
        s = str(op.attributes["dot_dimension_numbers"])
        return (cls._ints(s, "lhs_batching_dimensions"),
                cls._ints(s, "rhs_batching_dimensions"),
                cls._ints(s, "lhs_contracting_dimensions"),
                cls._ints(s, "rhs_contracting_dimensions"))

    def _emit_conv(self, rt, b, op, ref):
        s = str(op.attributes["dimension_numbers"])
        # #stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>
        m = re.search(r"<\[([^\]]*)\]x\[([^\]]*)\]->\[([^\]]*)\]>", s)
        if not m:
            raise NotImplementedError(f"conv dimension_numbers: {s}")
        def spec(group):
            toks = [t.strip() for t in group.split(",")]
            return toks
        in_s, k_s, out_s = spec(m.group(1)), spec(m.group(2)), spec(m.group(3))
        dn = {
            "in_batch": in_s.index("b"), "in_feat": in_s.index("f"),
            "k_out": k_s.index("o"), "k_in": k_s.index("i"),
            "out_batch": out_s.index("b"), "out_feat": out_s.index("f"),
            "in_sp": [in_s.index(str(d)) for d in range(len(in_s) - 2)],
            "k_sp": [k_s.index(str(d)) for d in range(len(k_s) - 2)],
            "out_sp": [out_s.index(str(d)) for d in range(len(out_s) - 2)],
        }
        strides = _i64_list(op.attributes["window_strides"])
        rhs_dil = _i64_list(op.attributes["rhs_dilation"])
        pad = np.asarray(op.attributes["padding"]).reshape(-1, 2)
        pad_lo = [int(x) for x in pad[:, 0]]
        pad_hi = [int(x) for x in pad[:, 1]]
        groups = int(op.attributes["feature_group_count"])
        return rt.convolution(b, ref(0), ref(1), _shape(op.results[0]), dn,
                              strides, pad_lo, pad_hi, rhs_dil, groups)

    def _emit_reduce_window(self, rt, b, op, ref):
        # combiner op in the body decides the kind (max-pool / sum-pool / ...).
        body = op.regions[0].blocks[0]
        kind = None
        for inner in body.operations:
            if inner.operation.name in _REDUCE_KIND:
                kind = _REDUCE_KIND[inner.operation.name]
                break
        if kind is None:
            raise NotImplementedError(f"unsupported reduce_window combiner: {op}")
        win = _i64_list(op.attributes["window_dimensions"])
        strides = _i64_list(op.attributes["window_strides"])
        wdil = _i64_list(op.attributes["window_dilations"])
        bdil = _i64_list(op.attributes["base_dilations"])
        pad = np.asarray(op.attributes["padding"]).reshape(-1, 2)
        pad_lo = [int(v) for v in pad[:, 0]]
        pad_hi = [int(v) for v in pad[:, 1]]
        return rt.reduce_window(b, ref(0), ref(1), kind, _shape(op.results[0]),
                                win, strides, pad_lo, pad_hi, bdil, wdil)

    def _emit_scatter(self, b, op, val2id):
        rt = self.rt
        operand = val2id[op.operands[0]]
        indices = val2id[op.operands[1]]
        updates = val2id[op.operands[2]]
        s = str(op.attributes["scatter_dimension_numbers"])
        uwd = self._ints(s, "update_window_dims")
        iwd = self._ints(s, "inserted_window_dims")
        dto = self._ints(s, "scatter_dims_to_operand_dims")
        ivd_m = re.search(r"index_vector_dim\s*=\s*(\d+)", s)
        ivd = int(ivd_m.group(1)) if ivd_m else 0
        cblk = op.regions[0].blocks[0]
        cb = rt.new_builder()
        cargs = [rt.parameter(cb, i, _shape(a)) for i, a in enumerate(cblk.arguments)]
        cids, _ = self._emit_block(cb, cblk, cargs)
        rt.set_root(cb, cids[0])
        return rt.scatter(b, operand, indices, updates, cb, uwd, iwd, dto, ivd)

    def _emit_gather(self, rt, b, op, ref):
        s = str(op.attributes["dimension_numbers"])
        offset = self._ints(s, "offset_dims")
        collapsed = self._ints(s, "collapsed_slice_dims")
        start_map = self._ints(s, "start_index_map")
        ivd_m = re.search(r"index_vector_dim\s*=\s*(\d+)", s)
        ivd = int(ivd_m.group(1)) if ivd_m else 0
        slice_sizes = _i64_list(op.attributes["slice_sizes"])
        return rt.gather(b, ref(0), ref(1), _shape(op.results[0]),
                         offset, collapsed, start_map, slice_sizes, ivd)

    @staticmethod
    def _main_func(module):
        for op in module.body.operations:
            if op.operation.name == "func.func":
                # the public entry is named "main"
                if "main" in str(op.attributes["sym_name"]):
                    return op
        # fallback: first func
        for op in module.body.operations:
            if op.operation.name == "func.func":
                return op
        raise NotImplementedError("no func.func in module")


class Executable:
    def __init__(self, rt: VgreHlo, exe: int, out_shapes=None):
        self.rt = rt
        self.exe = exe
        self.out_shapes = out_shapes or []

    def __call__(self, *inputs) -> np.ndarray:
        arrs = [np.asarray(x, dtype=np.float32) for x in inputs]
        return self.rt.execute(self.exe, arrs)

    def call_multi(self, *inputs) -> list:
        """Execute and return one flat ndarray per output leaf (in order)."""
        arrs = [np.asarray(x, dtype=np.float32) for x in inputs]
        numels = [int(np.prod(s)) if s else 1 for s in self.out_shapes]
        return self.rt.execute_multi(self.exe, arrs, numels)

    @property
    def n_outputs(self) -> int:
        return len(self.out_shapes)

    def output_numel(self) -> int:
        return self.rt.output_numel(self.exe)

    def close(self):
        if self.exe:
            self.rt.free(self.exe)
            self.exe = 0
