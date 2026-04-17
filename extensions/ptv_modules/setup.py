import os

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

base_path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "functional", "src")
)

cpp_sources = [
    os.path.join("interpolate", "neighbor_interpolate_cpu.cpp"),
    os.path.join("interpolate", "trilinear_devox_cpu.cpp"),
    os.path.join("sampling", "sampling_cpu.cpp"),
    os.path.join("voxelization", "vox_cpu.cpp"),
    "bindings.cpp",
]

cu_sources = [
    os.path.join("interpolate", "neighbor_interpolate.cu"),
    os.path.join("interpolate", "trilinear_devox.cu"),
    os.path.join("sampling", "sampling.cu"),
    os.path.join("voxelization", "vox.cu"),
]

sources = [
    os.path.normpath(os.path.join(base_path, s))
    for s in cpp_sources + cu_sources
]

setup(
    name="ptv_modules",
    version="1.0.0",
    packages=find_packages(include=["ptv_modules", "ptv_modules.*"]),
    install_requires=["torch>=1.13"],
    ext_modules=[
        CUDAExtension(
            name="ptv_modules.functional._pvt_backend",
            sources=sources,
            include_dirs=[base_path],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
