
'''
Vanilla implementation of Dynamic Time Warping
Colin Lea (colincsl@gmail.com)
July 2013
'''
import time
import numpy as np
import cython
cimport cython
cimport numpy as np
from libcpp.vector cimport vector

np.import_array()

ctypedef np.float64_t FLOAT
ctypedef np.int16_t INT

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cdef float distance_fcn(np.ndarray[FLOAT, ndim=1, mode="c"] x, np.ndarray[FLOAT, ndim=1, mode="c"] y):
	'''
	Find squared euclidian distance
	'''
	cdef np.ndarray[FLOAT, ndim=1, mode="c"] diff = x-y
	return sum(diff*diff)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cpdef np.ndarray[FLOAT, ndim=2, mode="c"] DynamicTimeWarping_c(np.ndarray[FLOAT, ndim=2, mode="c"] x, np.ndarray[FLOAT, ndim=2, mode="c"] y):
	'''
	Implements vanilla DTW

	http://en.wikipedia.org/wiki/Dynamic_time_warping
	'''

	cdef int x_res, y_res, i, j
	x_res = len(x)
	y_res = len(y)

	cdef np.ndarray[FLOAT, ndim=2, mode="c"] cost_matrix = np.zeros([x_res, y_res], dtype=np.float)

	# Caclulate edges first (they only depend on one value)
	for i in xrange(1, x_res):
		cost_matrix[i,0] = cost_matrix[i-1,0] + distance_fcn(x[i], y[0])
	for j in xrange(1, y_res):
		cost_matrix[0,j] = cost_matrix[0,j-1] + distance_fcn(x[0], y[j])		

	# Calculate distance at each location
	for i in xrange(1, x_res):
		for j in xrange(1, y_res):
			cost = distance_fcn(x[i], y[j])
			cost_matrix[i,j] = cost + min(min(cost_matrix[i-1,j], cost_matrix[i,j-1]), cost_matrix[i-1,j-1])

	return cost_matrix


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cpdef np.ndarray[INT, ndim=2, mode="c"] dtw_path(np.ndarray[FLOAT, ndim=2, mode="c"] cost_matrix):
	# Calculate path from start to finish
	cdef int x_res = cost_matrix.shape[0]
	cdef int y_res = cost_matrix.shape[1]
	cdef vector[int] path_x
	cdef vector[int] path_y
	path_x.push_back(0)
	path_y.push_back(0)
	cdef vector[FLOAT] costs
	for i in xrange(3):
		costs.push_back(0.)

	cdef int idx_x,idx_y, count=0
	# Start at top left and move to bottom right of cost matrix
	while path_x[count] < x_res-1 or path_y[count] < y_res-1:
		if path_x[count] == x_res-1: # If at left edge
			path_x.push_back(path_x[count])
			path_y.push_back(path_y[count]+1)				
		elif path_y[count] == y_res-1: # If at bottom edge
			path_x.push_back(path_x[count]+1)
			path_y.push_back(path_y[count])				
		else: # Normal case
			idx_x = path_x[count]
			idx_y = path_y[count]+1
			costs[0] = cost_matrix[idx_x, idx_y]
			idx_x = path_x[count]+1
			idx_y = path_y[count]
			costs[1] = cost_matrix[idx_x, idx_y]
			idx_x = path_x[count]+1
			idx_y = path_y[count]+1
			costs[2] = cost_matrix[idx_x, idx_y]

			if costs[0] < costs[1] and costs[0] < costs[2]:
				path_x.push_back(path_x[count])
				path_y.push_back(path_y[count]+1)	
			elif costs[1] < costs[0] and costs[0] < costs[2]:
				path_x.push_back(path_x[count]+1)
				path_y.push_back(path_y[count])				
			else:
				path_x.push_back(path_x[count]+1)
				path_y.push_back(path_y[count]+1)
		count += 1

	# Output to list
	cdef int path_length = path_x.size()
	path = np.empty([path_length, 2], dtype=np.int16)
	for i in xrange(path_length):
		path[i,0] = path_x[i]
		path[i,1] = path_y[i]
	
	return path

		
