from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("src/ea/cython_f.pyx", annotate=True )
)