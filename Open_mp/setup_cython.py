from setuptools import setup
from Cython.Build import cythonize
import numpy as np
import sys


if sys.platform == 'linux':
    # Linux
    extra_compile_args = ['-fopenmp']
    extra_link_args = ['-fopenmp']
else:
    # Windows (assuming MSVC with OpenMP support)
    extra_compile_args = ['/openmp']
    extra_link_args = []

setup(
    name='cython_open_mp.pyx',
    ext_modules=cythonize(
        "cython_open_mp.pyx",
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'cdivision': True
        },
    ),
    include_dirs=[np.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args
)