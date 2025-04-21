"""Build the DTW PyTorch C++ and Cython extensions."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from setuptools import Extension, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CppExtension, CUDAExtension


def get_openmp_flags() -> tuple[list[str], list[str]]:
    """Return the compiler and linker flags for OpenMP."""
    match sys.platform:
        case "linux":
            compile_flags, link_flags = ["-fopenmp"], ["-fopenmp"]
        case "win32":
            compile_flags, link_flags = ["-openmp"], []
        case "darwin":
            brew = ["brew", "--prefix", "libomp"]
            prefix = subprocess.run(brew, check=True, text=True, capture_output=True).stdout.strip()  # noqa: S603
            compile_flags = ["-Xclang", "-fopenmp"]
            link_flags = [f"-I{prefix}/include", f"-L{prefix}/lib", "-lomp", f"-Wl,-rpath,{prefix}/lib"]
        case _:
            return [], []
    return compile_flags, link_flags


def torch_extension() -> Extension:
    """Either CUDA or CPU extension."""
    use_cuda = CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension
    openmp_flags = get_openmp_flags()
    extra_compile_args = {
        "cxx": ["-fdiagnostics-color=always", "-DPy_LIMITED_API=0x030C0000", "-O3"] + openmp_flags[0],
        "nvcc": ["-O3"],
    }
    sources = [Path("src/fastabx/dtw/csrc/dtw.cpp")]
    if use_cuda:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "Volta;Turing;Ampere;Ada;Hopper"
        sources.append(Path("src/fastabx/dtw/csrc/cuda/dtw.cu"))
    return extension(
        "fastabx.dtw._C",
        sources,
        extra_compile_args=extra_compile_args,
        extra_link_args=openmp_flags[1],
        py_limited_api=True,
    )


def cython_extension() -> Extension:
    """Alternative cython backend for the DTW."""
    return Extension(
        "fastabx.dtw._dtw_numpy",
        sources=[Path("src/fastabx/dtw/_dtw_numpy.pyx")],
        extra_compile_args=["-O3", "-DNPY_NO_DEPRECATED_API=NPY_2_0_API_VERSION", "-DPy_LIMITED_API=0x030C0000"],
        include_dirs=[np.get_include()],
        py_limited_api=True,
    )


setup(
    ext_modules=[torch_extension(), cython_extension()],
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp312"}},
)
