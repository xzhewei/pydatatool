from distutils.core import setup

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

setup(name='pydatatool',
      packages=['pydatatool','pydatatool.caltech'],
      package_dir={'pydatatool':'pydatatool'},
      version='0.1',
      author='Zhewei Xu',
      author_email='xzhewei@gmail.com'
      )