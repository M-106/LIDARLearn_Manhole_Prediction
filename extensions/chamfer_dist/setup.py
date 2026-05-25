from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="chamfer_dist",
    version="2.0.0",
    packages=["chamfer_dist"],
    ext_modules=[
        CUDAExtension(
            "chamfer",
            sources=["chamfer_cuda.cpp", "chamfer.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
