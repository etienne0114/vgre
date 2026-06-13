"""Installable VGRE PyTorch backend. Build: `python setup.py build_ext --inplace`
(or `pip install -e .`). Requires the project's libvgre_cudart.so to be built;
point VGRE_BUILD_DIR at the CMake build directory (default: ../../build)."""
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BUILD = os.environ.get("VGRE_BUILD_DIR", os.path.join(ROOT, "build"))

setup(
    name="vgre_torch",
    version="0.1.0",
    description="VGRE PyTorch backend (PrivateUse1) — run CUDA-style PyTorch on the CPU emulator",
    packages=["vgre_torch"],
    ext_modules=[CppExtension(
        name="vgre_torch._C",
        sources=[os.path.join("csrc", "vgre_backend.cpp")],
        include_dirs=[os.path.join(ROOT, "include")],
        library_dirs=[BUILD],
        runtime_library_dirs=[BUILD],
        libraries=["vgre_cudart"],
        extra_compile_args=["-std=c++17", "-O2"],
    )],
    cmdclass={"build_ext": BuildExtension},
)
