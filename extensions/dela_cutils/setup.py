import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

src = "src"
sources = [
    os.path.join(root, f)
    for root, _, files in os.walk(src)
    for f in files
    if f.endswith((".cpp", ".cu"))
]

setup(
    name="dela_cutils",
    version="1.0",
    packages=["dela_cutils"],
    install_requires=["torch>=1.13", "numpy"],
    ext_modules=[
        CUDAExtension(
            name="dela_cutils._C",
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3", "-mavx2", "-funroll-loops"],
                "nvcc": ["-O2"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
