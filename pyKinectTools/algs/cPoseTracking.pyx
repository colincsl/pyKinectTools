"""
DON'T USE THIS FILE. LESS EFFICIENT THAN NUMPY VERSION

Efficient pose query
"""

import numpy as np
import cython
cimport cython
cimport numpy as cnp
cnp.import_array()

cdef extern from "math.h":
	int abs(int x)
	double sqrt(double x)
	double pow(double base, double exp)

from scipy.spatial import cKDTree
ctypedef cnp.float64_t FLOAT
ctypedef cnp.int16_t INT16

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cpdef inline cnp.ndarray[FLOAT, ndim=1] query_error(cnp.ndarray[FLOAT, ndim=2] markers, cnp.ndarray[FLOAT, ndim=3] trees, cnp.ndarray[INT16, ndim=1] search_joints):

	cdef int i,ii,j,m
	cdef float error_tmp, v
	cdef int marker_count = len(markers)
	cdef int pose_count = len(trees)
	cdef int joint_count = len(search_joints)

	# cdef cnp.ndarray[FLOAT, ndim=3, mode="c"] trees = trees_
	cdef cnp.ndarray[FLOAT, ndim=1, mode="c"] error = np.zeros(pose_count, np.float)
	cdef cnp.ndarray[FLOAT, ndim=1, mode="c"] marker = np.zeros_like(markers[0])
	cdef cnp.ndarray[FLOAT, ndim=1, mode="c"] diff = np.zeros(joint_count, np.float)
	# trees = trees[:,search_joints]

	for i in range(pose_count):
		# error[i] = trees[i].query(markers)[0].sum()
		error_tmp = 0
		for m in range(marker_count):
			marker = markers[m]
			for ii in range(joint_count):
				diff[ii] = 0.
				for j in range(3):
					# diff[ii] = diff[ii] + trees[i][search_joints][j] - marker[j]
					# diff[ii] = diff[ii] + trees[i][search_joints][j] -

					diff[ii] = diff[ii] + pow(trees[i][search_joints[ii]][j] - marker[j],2.)
				diff[ii] = sqrt(diff[ii])
			# min_diff = min(diff)
			error_tmp += cnp.min(diff)
			# error_tmp += cnp.min(cnp.sqrt(cnp.sum(cnp.pow((trees[i][search_joints] - marker), 2), -1)))
		error[i] = error_tmp

	# for i,tree in enumerate(trees):
		# error[i] = tree.query(markers)[0].sum()


	return np.array(error)
