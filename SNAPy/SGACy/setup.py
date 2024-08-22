from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

dir = "D:\\OneDrive\\0101 Python Module Dev\\SNAPy\\SNAPy\\SGACy\\"

# setup(
#     ext_modules=cythonize(dir+"ShapyCy.pyx"),
# )

ext_modules = [
    Extension(
        "graph",
        [dir+"graph.pyx"],
        # extra_compile_args=['/fopenmp'],
        # extra_link_args=['/fopenmp'],
    )
]

setup(
    ext_modules=cythonize(ext_modules),
)
# cd SNAPy\SGACy
# python setup.py build_ext --inplace