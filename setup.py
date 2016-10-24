#!/usr/bin/env python3

from setuptools import setup, Extension
from Cython.Build import cythonize
from numpy import get_include as get_numpy_include

exts = [Extension("nuc_cython", ["nuc_cython.pyx"],
                  include_dirs=[get_numpy_include()])]

setup(
    ext_modules = cythonize(exts)
)
