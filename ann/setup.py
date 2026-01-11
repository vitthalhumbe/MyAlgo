from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import pybind11

ext_modules = [
    Extension(
        name="ann_cpp",
        sources=["ann.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "/usr/include/eigen3",
        ],
        language="c++",
    )
]

class BuildExt(build_ext):
    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = []

        if ct == "unix":
            opts.append("-O3")
            opts.append("-std=c++17")

        for ext in self.extensions:
            ext.extra_compile_args = opts

        build_ext.build_extensions(self)

setup(
    name="ann_cpp",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)