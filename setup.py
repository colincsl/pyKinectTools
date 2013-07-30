
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

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
# from Cython.Build import cythonize

ext_modules = [
				Extension("pyKinectTools_algs_Dijkstras", ["pyKinectTools/algs/dijkstras.pyx"],language='c++'),
				Extension("pyKinectTools_algs_local_occupancy_pattern", ["pyKinectTools/algs/LocalOccupancyPattern.pyx"],language='c++'),
				Extension("pyKinectTools_algs_dynamic_time_warping", ["pyKinectTools/algs/DynamicTimeWarping.pyx"],language='c++'),
				# cythonize("pyKinectTools/algs/DynamicTimeWarping.pyx"),
				]
# _Dijkstras
# Extension("pyKinectTools_algs_Pose_Tracking", ["pyKinectTools/algs/cPoseTracking.pyx"],language='c++'),
# Extension("pyKinectTools.NeighborSuperpixels", ["pyKinectTools/algs/NeighborSuperpixels.pyx"])
# Extension("pyKinectTools.algs.Dijkstras", ["pyKinectTools/algs/dijkstras_new.pyx"],)
# Extension("pyKinectTools.algs.NeighborSuperpixels", ["pyKinectTools/algs/NeighborSuperpixels.pyx"])
# Extension("pyKinectTools.algs.dijkstrasGraph", ["pyKinectTools/algs/dijkstras.pyx"])

for e in ext_modules:
	e.pyrex_directives = {
						"boundscheck": False,
						"wraparound": False,
						"infer_types": True
						}
	e.extra_compile_args = ["-w"]

print ext_modules
setup(
	author = 'Colin Lea',
	author_email = 'colincsl@gmail.com',
	description = '',
	license = "FreeBSD",
	version= "0.1",
	name = 'pyKinectTools',
	cmdclass = {'build_ext': build_ext},
	include_dirs = [np.get_include()],
	packages= [	"pyKinectTools",
				"pyKinectTools.algs",
				"pyKinectTools.utils",
				"pyKinectTools.configs",
				"pyKinectTools.dataset_readers"
				],
	package_data={'':['*.xml', '*.png', '*.yml', '*.txt']},
	ext_modules = ext_modules
)

