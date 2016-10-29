#!/usr/bin/env python3

from setuptools import setup, Extension
from Cython.Build import cythonize
from numpy import get_include as get_numpy_include
from os import path

exts = [Extension("nuc_dynamics.nuc_cython", [path.join("nuc_dynamics", "nuc_cython.pyx")],
                  include_dirs=[get_numpy_include()])]

setup(
    name="Nuc Dynamics",
    version="0.0.1",
    description="A tool to calculate a structure from Hi-C coordinates",
    packages=['nuc_dynamics'],
    ext_modules = cythonize(exts),
    entry_points={'console_scripts': ['nuc_dynamics=nuc_dynamics:main']},
    include_package_data=True,
)
