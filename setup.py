# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Vitthal Humbe

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "myalgo._linreg_cpp",
        ["src/myalgo/_cpp/linear_regression.cpp"],
        include_dirs=["/usr/include/eigen3"],
        extra_compile_args=["-O3", "-std=c++17"],
    ),
    Pybind11Extension(
        "myalgo._ann_cpp",
        ["src/myalgo/_cpp/ann.cpp"],
        include_dirs=["/usr/include/eigen3"],
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})