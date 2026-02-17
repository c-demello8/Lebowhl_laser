import sys
from distutils.core import setup
from LebwohlLasher2 import main
from Cython.Build import cythonize

setup(name="LebwohlLasher2",
      ext_modules = cythonize("LebwohlLasher2.pyx"))
    