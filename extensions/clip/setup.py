from setuptools import setup, find_packages

setup(
    name="clip",
    version="1.0",
    description="OpenAI CLIP",
    author="OpenAI",
    py_modules=["clip"],
    packages=find_packages(exclude=["tests*"]),
    install_requires=["ftfy", "packaging", "regex"],
    include_package_data=True,
)
