from __future__ import absolute_import, division, print_function

import os
import sys

from setuptools import setup, Extension

import numpy as np

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    print("Could not import Cython. Install `cython` and rerun.")
    sys.exit(1)
    
ext = Extension(
    "cross_product.cross", ["cross_product/lib/cross.c", "cross_product/cross.pyx"],
    libraries=[],
    include_dirs=[np.get_include()]
)

setup(
    install_requires=['cython', 'numpy'],
    name="cross_product",
    version = "0.1",
    license = 'MIT',
    author="Daniel Filonik",
    author_email="d.filonik@hdr.qut.edu.au",
    cmdclass={"build_ext": build_ext},
    packages=['cross_product'],
    ext_modules=[ext]
)