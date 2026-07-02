"""vgre.vector — from-scratch HNSW approximate-nearest-neighbor index.

The retrieval half of a fully local RAG stack: embed with vgre.nn / vgre.lm,
index and search here — no FAISS, no external vector DB, pure in-tree C++
(src/xla/vector_index.cpp; real Malkov–Yashunin HNSW).

    import numpy as np
    from vgre.vector import VectorIndex

    ix = VectorIndex(dim=64, metric="cosine")
    ix.add(np.random.randn(1000, 64), ids=np.arange(1000))
    ids, dists = ix.search(query_vec, k=10)
    ix.save("docs.vgvx"); ix2 = VectorIndex.load("docs.vgvx")
"""
import ctypes
from typing import Optional, Tuple

import numpy as np

from ._native import _lib, NATIVE_AVAILABLE

_METRICS = {"cosine": 0, "l2": 1}
_bound = False


def _bind() -> None:
    global _bound
    if _bound or _lib is None:
        return
    c, P, V = ctypes, ctypes.POINTER, ctypes.c_void_p
    _lib.vgre_vindex_create.argtypes = [c.c_int, c.c_int, c.c_int, c.c_int]
    _lib.vgre_vindex_create.restype = V
    _lib.vgre_vindex_add.argtypes = [V, P(c.c_float), c.c_int64]
    _lib.vgre_vindex_add.restype = c.c_int
    _lib.vgre_vindex_search.argtypes = [V, P(c.c_float), c.c_int, c.c_int,
                                        P(c.c_int64), P(c.c_float)]
    _lib.vgre_vindex_search.restype = c.c_int
    _lib.vgre_vindex_size.argtypes = [V]
    _lib.vgre_vindex_size.restype = c.c_int64
    _lib.vgre_vindex_dim.argtypes = [V]
    _lib.vgre_vindex_dim.restype = c.c_int
    _lib.vgre_vindex_save.argtypes = [V, c.c_char_p]
    _lib.vgre_vindex_save.restype = c.c_int
    _lib.vgre_vindex_load.argtypes = [c.c_char_p]
    _lib.vgre_vindex_load.restype = V
    _lib.vgre_vindex_free.argtypes = [V]
    _bound = True


def _require():
    if not NATIVE_AVAILABLE or _lib is None:
        raise RuntimeError("libvgre not found — build it and set LD_LIBRARY_PATH/VGRE_LIB_PATH")
    _bind()


class VectorIndex:
    """HNSW ANN index over float32 vectors with caller-chosen int64 ids."""

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200,
                 metric: str = "cosine", _handle=None):
        _require()
        if _handle is not None:
            self._h = _handle
            self._dim = int(_lib.vgre_vindex_dim(self._h))
            return
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {sorted(_METRICS)}")
        self._h = _lib.vgre_vindex_create(int(dim), int(M), int(ef_construction),
                                          _METRICS[metric])
        if not self._h:
            raise RuntimeError("vgre_vindex_create failed")
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return int(_lib.vgre_vindex_size(self._h))

    def add(self, vectors, ids=None) -> None:
        """Insert one [dim] vector or a batch [n, dim]; ids default to the
        current size counting upward."""
        v = np.ascontiguousarray(vectors, dtype=np.float32)
        if v.ndim == 1:
            v = v[None, :]
        if v.shape[1] != self._dim:
            raise ValueError(f"expected dim {self._dim}, got {v.shape[1]}")
        if ids is None:
            start = len(self)
            ids = np.arange(start, start + v.shape[0], dtype=np.int64)
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.shape[0] != v.shape[0]:
            raise ValueError("ids length must match vector count")
        fp = ctypes.POINTER(ctypes.c_float)
        for i in range(v.shape[0]):
            rc = _lib.vgre_vindex_add(self._h, v[i].ctypes.data_as(fp), int(ids[i]))
            if rc != 0:
                raise RuntimeError(f"vgre_vindex_add failed (rc={rc})")

    def search(self, query, k: int = 10, ef: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """k-NN: returns (ids[int64], distances[float32]), nearest first.
        ef is the query beam width (0 → max(64, k); raise it for higher recall)."""
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._dim:
            raise ValueError(f"expected dim {self._dim}, got {q.shape[0]}")
        ids = np.empty(k, dtype=np.int64)
        dists = np.empty(k, dtype=np.float32)
        n = _lib.vgre_vindex_search(
            self._h, q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), int(k),
            int(ef), ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            dists.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if n < 0:
            raise RuntimeError(f"vgre_vindex_search failed (rc={n})")
        return ids[:n], dists[:n]

    def save(self, path: str) -> None:
        if _lib.vgre_vindex_save(self._h, path.encode()) != 0:
            raise RuntimeError(f"could not save index to {path}")

    @classmethod
    def load(cls, path: str) -> "VectorIndex":
        _require()
        h = _lib.vgre_vindex_load(path.encode())
        if not h:
            raise RuntimeError(f"could not load index from {path}")
        return cls(dim=0, _handle=h)

    def close(self) -> None:
        if getattr(self, "_h", None):
            _lib.vgre_vindex_free(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
