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
    )
]

# ext_modules = [
#     Extension(
#         "geom",
#         [dir+"geom.pyx"],
#     )
# ]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)
# cd SNAPy\SGACy
# python setup.py build_ext --inplace