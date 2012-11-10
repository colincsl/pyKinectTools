
'''
Configure:
python setup.py build

pyKinectTools
Colin Lea
2012
'''

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np      

ext_modules = [Extension("pyKinectTools.algs.NeighborSuperpixels", ["pyKinectTools/algs/NeighborSuperpixels.pyx"]),
			Extension("pyKinectTools.algs.dijkstrasGraph", ["pyKinectTools/algs/dijkstras.pyx"])
				]
# Extension("pyKinectTools.algs.Dijkstras", ["pyKinectTools/algs/dijkstras_New.pyx"])

for e in ext_modules:
	e.pyrex_directives = {
						"boundscheck": False,
						"wraparound": False,
						"infer_types": True
						}
	e.extra_compile_args = ["-w"]


setup(
	author = 'Colin Lea',
	author_email = 'colincsl@gmail.com',
	license = "FreeBSD",
	version= "0.1",
	name = 'pyKinectTools',
	cmdclass = {'build_ext': build_ext},
	include_dirs = [np.get_include()],
	ext_modules = ext_modules,
	packages= [	"pyKinectTools",
				"pyKinectTools.algs",
				"pyKinectTools.utils",
				"pyKinectTools.data"
			]
)

