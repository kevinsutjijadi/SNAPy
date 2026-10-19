from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import os
import sys
import numpy

here = os.path.dirname(os.path.abspath(__file__))
extra = [] if sys.platform == "win32" else ["-O2", "-std=c++11"]

ext_modules = [
    Extension(
        "graph",
        [os.path.join(here, "graph.pyx")],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra,
        language="c++",
    ),
    Extension(
        "geom",
        [os.path.join(here, "geom.pyx")],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra,
        language="c++",
    ),
]

setup(
    ext_modules=cythonize(
        ext_modules,
        language_level=3,
        compiler_directives={"language_level": 3},
    ),
)
# Windows (this folder):  python setup.py build_ext --inplace
# Linux .so via Docker:   docker build -t sgacy-linux --build-arg PY=3.11 .
#                         docker run --rm -v "${PWD}:/src" sgacy-linux
