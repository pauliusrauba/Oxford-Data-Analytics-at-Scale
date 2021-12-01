from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='FINd_Cython',
    ext_modules = cythonize('FINd_Cython.pyx', include_path = [np.get_include()])
)