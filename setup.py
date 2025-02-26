"""Build the DTW PyTorch C++ extension."""

import os
import sys
from pathlib import Path

import numpy as np
from setuptools import Extension, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CppExtension, CUDAExtension


def get_cython_extension() -> Extension:
    """Cython DTW extension as a fallback."""
    return Extension(
        "fastabx._dtw_cython",
        sources=["src/fastabx/csrc/dtw.pyx"],
        extra_compile_args=["-O3", "-DNPY_NO_DEPRECATED_API=NPY_2_0_API_VERSION"],  # , "-DPy_LIMITED_API=0x030c0000"],
        include_dirs=[np.get_include()],
        # py_limited_api=True,
    )


def get_torch_extension() -> Extension:
    """Either CUDA or CPU extension."""
    use_cuda = CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension
    openmp = ["-fopenmp"] if sys.platform == "linux" else []
    extra_compile_args = {
        "cxx": ["-fdiagnostics-color=always", "-DPy_LIMITED_API=0x030c0000", "-O3", *openmp],
        "nvcc": ["-O3"],
    }
    sources = [Path("src/fastabx/csrc/dtw.cpp")]
    if use_cuda:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "Volta;Turing;Ampere;Ada;Hopper"
        sources.append(Path("src/fastabx/csrc/cuda/dtw.cu"))
    return extension(
        "fastabx._C",
        sources,
        extra_compile_args=extra_compile_args,
        extra_link_args=openmp,
        py_limited_api=True,
    )


setup(
    ext_modules=[get_cython_extension()],
    # cmdclass={"build_ext": BuildExtension},
    # options={"bdist_wheel": {"py_limited_api": "cp312"}},
)
