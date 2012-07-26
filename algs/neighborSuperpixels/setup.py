
'''
Configure:
python setup.py build_ext --inplace
'''

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np      

ext_modules = [Extension("neighborSuperpixels", ["neighborSuperpixels.pyx"])]
# for e in ext_modules:
#     e.pyrex_directives = {"boundscheck": False}
#     e.pyrex_directives = {"wraparound": False}
setup(
  name = 'neighborSuperpixels',
  cmdclass = {'build_ext': build_ext},
  include_dirs = [np.get_include()],
  ext_modules = ext_modules
)