"""
VGRE — Virtual GPU Runtime Engine
Python Bindings Package
"""

from .device import VirtualDevice  # type: ignore
from .kernel import Kernel  # type: ignore
from .runtime import Runtime  # type: ignore
from .memory import DeviceArray, ManagedArray  # type: ignore
from .stream import Stream  # type: ignore
from .graph import Graph  # type: ignore
from .lm import LanguageModel, Tokenizer, cosine_lr  # type: ignore

try:
    from ._native import NATIVE_AVAILABLE  # type: ignore
except ImportError:
    NATIVE_AVAILABLE = False

__version__ = "0.1.0"
__all__ = [
    "VirtualDevice", "Kernel", "Runtime",
    "DeviceArray", "ManagedArray", "Stream", "Graph",
    "LanguageModel", "Tokenizer", "cosine_lr",
    "NATIVE_AVAILABLE",
]
