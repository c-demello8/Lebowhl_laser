from distutils.core import setup
from Cython.Build import cythonize

setup(name="picalc_pyx2",
      ext_modules=cythonize("picalc_pyx2.pyx"))
# Just making the file into a cython file
