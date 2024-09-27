from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

dir = "D:\\OneDrive\\0101 Python Module Dev\\SNAPy\\SNAPy\\SGACy\\"


ext_modules = [
    Extension(
        "graph",
        [dir+"graph.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "geom",
        [dir+"geom.pyx"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    ext_modules=cythonize(ext_modules),
)
# cd SNAPy\SGACy
# python setup.py build_ext --inplace