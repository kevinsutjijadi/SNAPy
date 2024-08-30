from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

dir = "D:\\OneDrive\\0101 Python Module Dev\\SNAPy\\SNAPy\\SGACy\\"


ext_modules = [
    Extension(
        "graph",
        [dir+"graph.pyx"],
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
)
# cd SNAPy\SGACy
# python setup.py build_ext --inplace