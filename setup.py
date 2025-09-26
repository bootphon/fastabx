"""Build the DTW PyTorch C++ extension."""

import os
import sys

from setuptools import Extension, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CppExtension, CUDAExtension


def get_openmp_flags() -> tuple[list[str], list[str]]:
    """Return the compiler and linker flags for OpenMP."""
    match sys.platform:
        case "linux":
            return ["-fopenmp"], ["-fopenmp"]
        case "win32":
            return ["-openmp"], []
        case _:  # On MacOS, we use the OpenMP version vendored by PyTorch
            return [], []


def get_extension() -> Extension:
    """Either CUDA or CPU extension."""
    use_cuda = CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension
    openmp_flags = get_openmp_flags()
    extra_compile_args = {
        "cxx": ["-fdiagnostics-color=always", "-O3"] + openmp_flags[0],
        "nvcc": ["-O3"],
    }
    sources = ["src/fastabx/csrc/dtw.cpp"]
    if use_cuda:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "Volta;Turing;Ampere;Ada;Hopper"
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
