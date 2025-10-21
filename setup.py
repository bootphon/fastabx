"""Build the DTW PyTorch C++ extension."""

import os
import sys

import torch
from setuptools import Extension, setup
from torch.torch_version import Version
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CppExtension, CUDAExtension


def get_openmp_flags() -> tuple[list[str], list[str]]:
    """Return the compiler and linker flags for OpenMP."""
    match sys.platform:
        case "linux":
            compile_flags, link_flags = ["-fopenmp"], ["-fopenmp"]
        case "win32":
            compile_flags, link_flags = ["-openmp"], []
        case _:  # On MacOS, we use the OpenMP version vendored by PyTorch
            return [], []
    return compile_flags, link_flags


def get_cuda_arch_list() -> str:
    """Supported CUDA architectures. Volta is not supported by CUDA 13.0."""
    if torch.version.cuda is None or Version(torch.version.cuda) < Version("13.0"):
        return "Volta;Turing;Ampere;Ada;Hopper"
    return "Turing;Ampere;Ada;Hopper"


def get_extension() -> Extension:
    """Either CUDA or CPU extension."""
    use_cuda = CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension
    openmp_flags = get_openmp_flags()
    extra_compile_args = {
        "cxx": ["-fdiagnostics-color=always", "-DPy_LIMITED_API=0x030C0000", "-O3"] + openmp_flags[0],
        "nvcc": ["-O3"],
    }
    sources = ["src/fastabx/csrc/dtw.cpp"]
    if use_cuda:
        os.environ["TORCH_CUDA_ARCH_LIST"] = get_cuda_arch_list()
        sources.append("src/fastabx/csrc/cuda/dtw.cu")
    return extension(
        "fastabx._C",
        sources,
        extra_compile_args=extra_compile_args,
        extra_link_args=openmp_flags[1],
        py_limited_api=True,
    )


if __name__ == "__main__":
    setup(
        ext_modules=[get_extension()],
        cmdclass={"build_ext": BuildExtension},
        options={"bdist_wheel": {"py_limited_api": "cp312"}},
    )
