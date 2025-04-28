"""Build the DTW PyTorch C++ extension."""

import os
import subprocess
import sys
from pathlib import Path

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


def get_extension() -> Extension:
    """Either CUDA or CPU extension."""
    use_cuda = CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension
    openmp_flags = get_openmp_flags()
    extra_compile_args = {
        "cxx": ["-fdiagnostics-color=always", "-O3"] + openmp_flags[0],
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
        extra_link_args=openmp_flags[1],
        py_limited_api=True,
    )


setup(
    ext_modules=[get_extension()],
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp312"}},
)
