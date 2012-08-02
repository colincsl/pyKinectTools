''' Build with: python setup.py build_ext --inplace '''

import numpy as np
cimport numpy as np
from libc.math cimport abs

np.import_array()

ctypedef np.uint8_t UINT8

'''Compute all edges between segments'''
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

	# Look left and above current node. If it is a different label, add to edges.
	for y in range(1, height):
		for x in range(1, width):
			ind = y*width + x
			ind2 = (y-1)*width + x # Above
			ind3 = y*width + x -1  # Left
			# If not the same label and both labels are not the background (=0)
			if regions_c[ind] - regions_c[ind2] != 0 and regions_c[ind] > 0 and regions_c[ind2] > 0:
				edges.append([regions_c[ind], regions_c[ind2]])
			if regions_c[ind] - regions_c[ind3] != 0 and regions_c[ind] > 0 and regions_c[ind3] > 0:
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






