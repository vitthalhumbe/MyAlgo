from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "linreg_internal",
        ["../cpp/linear_regression.cpp"],
        include_dirs=[
            "/usr/include/eigen3", 
        ],
    ),
]

setup(
    name="linreg_internal",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
