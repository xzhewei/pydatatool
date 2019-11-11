from Cython.Build import cythonize
import numpy as np

from setuptools import setup, find_packages, Extension



# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

ext_modules = [
    Extension(
        'pycocotools._mask',
        sources=['common/maskApi.c', 'pycocotools/_mask.pyx'],
        include_dirs = [np.get_include(), 'common'],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    )
]

setup(
      name        ='pydatatool',
      version     ='0.1',      
      packages    = find_packages(),

      author      = "Zhewei Xu",
      author_email= "xzhewei@gmail.com",
      description = "A data tool for CNN Training.",
      license     = "MIT",
      ext_modules = cythonize(ext_modules)
      )