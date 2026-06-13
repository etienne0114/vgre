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
}
_UNARY = {
    "stablehlo.negate": "Negate", "stablehlo.exponential": "Exp",
    "stablehlo.log": "Log", "stablehlo.tanh": "Tanh",
    "stablehlo.abs": "Abs", "stablehlo.rsqrt": "Rsqrt",
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
            root, _ = self._emit_block(b, block, arg_ids)
            if root is None:
                raise NotImplementedError("function has no return")
            rt.set_root(b, root)
            exe = rt.compile(b)
            b = 0  # consumed
            return Executable(rt, exe)
        finally:
            if b:
                rt.free_builder(b)

    def _emit_block(self, b, block, arg_ids):
        """Emit a block with its arguments pre-bound to instruction ids.

        Returns (return_id, return_scalar) for the block's func.return."""
        rt = self.rt
        val2id: dict = dict(zip(block.arguments, arg_ids))
        const_scalar: dict = {}  # value -> python float (for reduce inits / call args)
        for op in block.operations:
            name = op.operation.name
            if name == "func.return":
                rv = op.operands[0]
                return val2id[rv], const_scalar.get(rv)
            if name in ("func.call", "call"):
                rid, sc = self._emit_call(b, op, val2id, const_scalar)
            else:
                rid, sc = self._emit(rt, b, op, name, val2id, const_scalar)
            if rid is not None:
                val2id[op.results[0]] = rid
                if sc is not None:
                    const_scalar[op.results[0]] = sc
        return None, None

    def _emit_call(self, b, op, val2id, const_scalar):
        callee_name = str(op.attributes["callee"]).strip('"@')
        callee = self._funcs.get(callee_name)
        if callee is None:
            raise NotImplementedError(f"call to unknown function {callee_name}")
        arg_ids = [val2id[o] for o in op.operands]
        rid, sc = self._emit_block(b, callee.body.blocks[0], arg_ids)
        return rid, sc

    # ── op emission ──────────────────────────────────────────────────────────
    def _emit(self, rt, b, op, name, val2id, const_scalar):
        def ref(i):
            return val2id[op.operands[i]]

        if name in _BINARY:
            return rt.binary(b, _BINARY[name], ref(0), ref(1)), None
        if name in _UNARY:
            return rt.unary(b, _UNARY[name], ref(0)), None
        if name == "stablehlo.constant":
            data = _const_floats(op.attributes["value"])
            shp = _shape(op.results[0])
            sc = float(data[0]) if data.size == 1 else None
            return rt.constant(b, shp, data), sc
        if name == "stablehlo.dot_general":
            self._check_plain_matmul(op)
            return rt.dot(b, ref(0), ref(1)), None
        if name == "stablehlo.broadcast_in_dim":
            bdims = _i64_list(op.attributes["broadcast_dimensions"])
            return rt.broadcast(b, ref(0), _shape(op.results[0]), bdims), None
        if name == "stablehlo.reshape":
            return rt.reshape(b, ref(0), _shape(op.results[0])), None
        if name == "stablehlo.transpose":
            perm = _i64_list(op.attributes["permutation"])
            return rt.transpose(b, ref(0), perm), None
        if name == "stablehlo.compare":
            d = self._compare_dir(op)
            return rt.compare(b, ref(0), ref(1), d), None
        if name == "stablehlo.select":
            return rt.select(b, ref(0), ref(1), ref(2)), None
        if name == "stablehlo.reduce":
            return self._emit_reduce(rt, b, op, ref, const_scalar), None
        if name == "stablehlo.convert":
            # f32 -> f32 identity (dtype-narrowing is out of scope; verify same).
            if str(ShapedType(op.results[0].type).element_type) != \
               str(ShapedType(op.operands[0].type).element_type):
                raise NotImplementedError(f"convert across dtypes: {op}")
            return ref(0), None
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
    def _check_plain_matmul(op):
        s = str(op.attributes["dot_dimension_numbers"])
        # require lhs[...,K]·rhs[K,...] 2D matmul, no batch dims
        lhs_c = re.search(r"lhs_contracting_dimensions\s*=\s*\[([^\]]*)\]", s)
        rhs_c = re.search(r"rhs_contracting_dimensions\s*=\s*\[([^\]]*)\]", s)
        lhs_b = re.search(r"lhs_batching_dimensions\s*=\s*\[([^\]]*)\]", s)
        rhs_b = re.search(r"rhs_batching_dimensions\s*=\s*\[([^\]]*)\]", s)
        def vals(m):
            return [x for x in re.split(r"[,\s]+", m.group(1).strip()) if x] if m else []
        if vals(lhs_b) or vals(rhs_b):
            raise NotImplementedError(f"batched dot_general not supported: {s}")
        lc, rc = vals(lhs_c), vals(rhs_c)
        lrank = len(ShapedType(op.operands[0].type).shape)
        rrank = len(ShapedType(op.operands[1].type).shape)
        if lrank != 2 or rrank != 2 or lc != ["1"] or rc != ["0"]:
            raise NotImplementedError(f"only plain 2D matmul supported: {s}")

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
    def __init__(self, rt: VgreHlo, exe: int):
        self.rt = rt
        self.exe = exe

    def __call__(self, *inputs) -> np.ndarray:
        arrs = [np.asarray(x, dtype=np.float32) for x in inputs]
        return self.rt.execute(self.exe, arrs)

    def output_numel(self) -> int:
        return self.rt.output_numel(self.exe)

    def close(self):
        if self.exe:
            self.rt.free(self.exe)
            self.exe = 0
