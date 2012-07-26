
#python dijkstras_setup.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np      

ext_modules = [Extension("dijkstrasGraph", ["dijkstras.pyx"])]
# for e in ext_modules:
#     e.pyrex_directives = {"boundscheck": False}
#     e.pyrex_directives = {"wraparound": False}
setup(
  name = 'dijkstrasGraph',
  cmdclass = {'build_ext': build_ext},
  include_dirs = [np.get_include()],
  ext_modules = ext_modules
)