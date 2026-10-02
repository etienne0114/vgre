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
from .lm import LanguageModel, Tokenizer, GGUFMetadataReader, cosine_lr  # type: ignore
from .auto_config import ModelAutoConfigurator  # type: ignore
from . import nn  # type: ignore  # general autograd framework (vgre.nn)

try:
    from ._native import NATIVE_AVAILABLE  # type: ignore
except ImportError:
    NATIVE_AVAILABLE = False

__version__ = "0.1.4"
__all__ = [
    "VirtualDevice", "Kernel", "Runtime",
    "DeviceArray", "ManagedArray", "Stream", "Graph",
    "LanguageModel", "Tokenizer", "GGUFMetadataReader", "ModelAutoConfigurator", "cosine_lr", "nn",
    "NATIVE_AVAILABLE",
]
