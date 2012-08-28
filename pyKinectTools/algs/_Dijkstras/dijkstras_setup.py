
#python dijkstras_setup.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np      

ext_modules = [Extension("dijkstrasGraph", ["dijkstras.pyx"])]
setup(
  name = 'dijkstrasGraph',
  cmdclass = {'build_ext': build_ext},
  include_dirs = [np.get_include()],
  ext_modules = ext_modules
)

ext_modules = [Extension("dijkstrasGraphNew", ["dijkstras_New.pyx"])]
setup(
  name = 'dijkstrasGraphNew',
  cmdclass = {'build_ext': build_ext},
  include_dirs = [np.get_include()],
  ext_modules = ext_modules
)
