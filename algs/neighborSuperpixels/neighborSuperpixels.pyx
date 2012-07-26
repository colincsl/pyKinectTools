''' Build with: python setup.py build_ext --inplace '''

# import cython
# cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport abs, floor, sqrt

np.import_array()

# ctypedef np.uint16_t UINT16
# ctypedef np.int16_t INT16
ctypedef np.uint8_t UINT8

# cdef inline int int_max(int a, int b): return a if a >= b else b
# cdef inline int int_min(int a, int b): return a if a <= b else b
# cdef inline list ind2dim(int ind, int rezX, int rezY): 
# 	return [floor(ind/rezX), ind-rezX*floor(ind/rezX)]
# cdef inline int dim2ind(int i, int j, int rezX, int rezY): return (j*rezX+i)

'''Compute the weighted distance to a point on the graph'''
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.infer_types(True)
# @cython.profile(True)
cpdef inline list getNeighborEdges(np.ndarray[UINT8, ndim=2] regions):
	cdef int x,y, ind
	cdef int height = regions.shape[0]
	cdef int width = regions.shape[1]
	cdef list edges = []

	cdef UINT8* regions_c = <UINT8*>regions.data

	for y in range(1, height):
		for x in range(1, width):
			ind = y*width + x
			ind2 = (y-1)*width + x
			ind3 = y*width + x -1
			if regions_c[ind] - regions_c[ind2] != 0 and regions_c[ind] > 0 and regions_c[ind2] > 0:
				edges.append([regions_c[ind], regions_c[ind2]])
			if regions_c[ind] - regions_c[ind3] != 0 and regions_c[ind] > 0 and regions_c[ind2] > 0:
				edges.append([regions_c[ind], regions_c[ind3]])

	return edges


cpdef inline list getNeighborProfiles(np.ndarray[UINT8, ndim=2] im, list edges, list edgePositions):
	cdef int x,y, ind, dist
	cdef int height = im.shape[0]
	cdef int width = im.shape[1]

	cdef UINT8* im_c = <UINT8*>im.data

	cdef list edgeProfiles = []

	cdef int currentX, currentY
	for e in edges:
		currentY = edgePositions[e[0]][0]
		currentX = edgePositions[e[0]][1]
		endY = edgePositions[e[1]][0]
		endX = edgePositions[e[1]][1]

		ind = currentY*width + currentX
		edgeProfiles.append([im_c[ind]])
		
		while currentX != endX and currentY != endY:
			if abs(currentX - endX) > abs(currentY - endY):
				if currentX > endX:
					currentX -= 1
				else:
					currentX += 1
			else:
				if currentY > endY:
					currentY -= 1
				else:
					currentY += 1
			ind = currentY*width + currentX
			edgeProfiles[-1].append(im_c[ind])

	return edgeProfiles






