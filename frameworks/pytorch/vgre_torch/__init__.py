"""VGRE PyTorch backend. `import vgre_torch` registers the "vgre" device; then
move tensors with `.to("vgre")` and compute as usual (matmul runs on VGRE's
emulated cuBLAS; allocation/copies on its emulated CUDA runtime)."""
import types
import torch

from . import _C  # noqa: F401  (registers allocator / ops / device guard at import)

torch.utils.rename_privateuse1_backend("vgre")

_devmod = types.ModuleType("torch.vgre")
_devmod.current_device = lambda: 0
_devmod.device_count = lambda: 1
_devmod.is_available = lambda: True
_devmod.is_initialized = lambda: True
_devmod._is_in_bad_fork = lambda: False
_devmod.get_rng_state = lambda device=0: torch.zeros(1, dtype=torch.uint8)
_devmod.set_rng_state = lambda *a, **k: None
torch._register_device_module("vgre", _devmod)
try:
    torch.utils.generate_methods_for_privateuse1_backend(for_tensor=True, for_module=False, for_storage=False)
except Exception:
    pass

device = torch.device("vgre", 0)
