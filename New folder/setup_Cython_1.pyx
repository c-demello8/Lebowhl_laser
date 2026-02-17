from distutils.core import setup
from Cython.Build import cythonize

setup(name="Cython_1",
      ext_modules=cythonize("Cython_1.pyx"))
# Just making the file into a cython file

    