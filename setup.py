#!/usr/bin/env python3

from setuptools import setup

setup(
    name="Nuc Dynamics",
    version="0.0.1",
    description="A tool to calculate a structure from Hi-C coordinates",
    packages=['nuc_dynamics'],
    entry_points={'console_scripts': ['nuc_dynamics=nuc_dynamics:main']},
    include_package_data=True,
)
