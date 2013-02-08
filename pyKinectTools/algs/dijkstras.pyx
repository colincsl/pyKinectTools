''' Build with: python dijkstras_setup.py build_ext --inplace '''

import numpy as np
import cython
cimport cython
cimport numpy as cnp
cnp.import_array()

# from python_ref cimport Py_INCREF, Py_DECREF

cdef extern from "math.h":
	int abs(int x)

ctypedef cnp.uint16_t UINT16
ctypedef cnp.int16_t INT16
ctypedef cnp.uint8_t UINT8


cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline void ind2dim(int ind, int current[2], int width, int height): 
	current[0] = int(ind/float(width))
	current[1] = ind - width*current[0]
	return
cdef inline int dim2ind(int i, int j, int width, int height): 
	return (i*width+j)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
# @cython.profile(True)
cpdef inline list graph_dijkstras(cnp.ndarray[UINT16, ndim=2, mode="c"] distsMat, cnp.ndarray[UINT8, ndim=2, mode="c"] visitMat, cnp.ndarray[UINT16, ndim=2, mode="c"] depthMat, cnp.ndarray[INT16, ndim=1, mode="c"] current_):
	'''
	Inputs:
		distsMat (uint16)
		visitMat
		depthMat
		current_ : Starting point
	Outputs:
	'''

	cdef int i, j, tmpX, tmpY
	cdef int minGradient, argGradient, gradient
	cdef int height = distsMat.shape[0]
	cdef int width = distsMat.shape[1]

	cdef int currentInd, tmpInd, currentCost
	cdef int current[2]
	current[0] = current_[0]
	current[1] = current_[1]

	cdef int TOUCHED=1, VISITED=254, OUTOFBOUNDS=255, MAXVALUE=32000
	cdef int neighborsVisited
	cdef int foundTouched

	cdef cnp.ndarray[dtype=UINT16, ndim=2, mode='c'] distsMat_c, depthMat_c
	cdef cnp.ndarray[dtype=UINT8, ndim=2, mode='c'] visitMat_c

	distsMat_c = distsMat #np.ascontiguousarray(distsMat
	visitMat_c = visitMat #np.ascontiguousarray(visitMat)
	depthMat_c = depthMat #np.ascontiguousarray(depthMat)

	currentInd = dim2ind(current[0], current[1], width, height)
	# print current[0], current[1]

	''' Set border to OUTOFBOUNDS '''
	visitMat_c[0, :] = OUTOFBOUNDS	
	visitMat_c[height-1:,] = OUTOFBOUNDS
	visitMat_c[:,0] = OUTOFBOUNDS	
	visitMat_c[:,width-1] = OUTOFBOUNDS

	''' --- Main --- '''
	''' Part 1: Get cost to each node from centroid '''
	while(1):

		if current[0] < 0 or current[0] > height or current[1] < 0 or current[1] > width:
			break

		currentCost = distsMat_c[current[0], current[1]]
		visitMat_c[current[0],current[1]] = VISITED

		# print current[0], current[1]
		''' Update neighbors '''
		minGradient = MAXVALUE
		argGradient = -1
		neighborsVisited = 0
		for i in range(-1,2):
			for j in range(-1,2):
				if not (i == 0 and j == 0):
					tmpInd = dim2ind(current[0]+i, current[1]+j, width, height)
					''' Only look at nodes that havent' been visited (that are in bounds) '''
					if visitMat_c[current[0]+i, current[1]+j] < VISITED:
						gradient = abs( depthMat_c[current[0]+i,current[1]+j] - depthMat_c[current[0],current[1]] )

						distsMat_c[current[0]+i, current[1]+j] = int_min(distsMat_c[current[0]+i,current[1]+j], currentCost + gradient + 1)
						visitMat_c[current[0]+i, current[1]+j] = int_max(visitMat_c[current[0]+i,current[1]+j], TOUCHED)

						if visitMat_c[current[0]+i, current[1]+j] < VISITED:
							if gradient < minGradient:
								minGradient = gradient
								argGradient = tmpInd

						''' Mark if previously visited or if out of bounds '''
						if visitMat_c[current[0]+i, current[1]+j] >= VISITED:
							neighborsVisited += 1

		''' If all neighbors have been visited, shift to new touched spot '''
		if neighborsVisited < 8 and argGradient >= 0:
			''' This means not all neighbors have been touched '''
			currentInd = argGradient
			ind2dim(currentInd, current, width, height)
		else:
			''' If all neighbors have been touched... '''
			tmpX = 0
			foundTouched = 0
			while (tmpX < width-1 and foundTouched==0):
				tmpY = 0
				while (tmpY  < height-1 and foundTouched==0):
					tmpInd = dim2ind(tmpY, tmpX, width, height)

					if visitMat_c[tmpY, tmpX] == TOUCHED:
						currentInd = tmpInd
						ind2dim(currentInd, current, width, height)
						foundTouched = 1
						break
					tmpY += 1
				tmpX += 1

			'''If no more open spots'''
			if foundTouched == 0:
				break

	# print 'Done with Part A'

	''' Part 2: Find best path from centroid to largest point '''

	''' Get max value '''
	cdef int maxVal=0
	cdef int maxCoord[2]	
	for y in range(height):
		for x in range(width):
			if visitMat_c[y,x] == VISITED and distsMat_c[y,x] > maxVal and distsMat_c[y,x] < MAXVALUE:
				maxVal = distsMat_c[y,x]
				maxCoord[0] = y
				maxCoord[1] = x

	current[0] = maxCoord[0]
	current[1] = maxCoord[1]
	cdef int currentVal = maxVal
	cdef int trailLength=1, done=0

	cdef int trail[1000][2]
	trail[0][0] = current[0];
	trail[0][1] = current[1]

	# print "max:", current[0], current[1], width, height, maxVal

	while currentVal!=0 and done != 1 and trailLength < 1000:
		minGradient = MAXVALUE
		argGradient = -1

		for i in range(-1,2):
			for j in range(-1,2):
				if not (i == 0 and j == 0):
					tmpInd = dim2ind(current[0]+i, current[1]+j, width, height)
					gradient = distsMat_c[current[0]+i, current[1]+j]-distsMat_c[current[0], current[1]]

					if gradient < minGradient and visitMat_c[current[0]+i, current[1]+j] == VISITED:
						minGradient = gradient
						argGradient = tmpInd

		if argGradient < 0:
			done = 1
			# print "Error"
			break

		ind2dim(argGradient, current, width, height)
		currentInd = argGradient
		currentVal = distsMat_c[current[0],current[1]]
		visitMat_c[current[0],current[1]] = OUTOFBOUNDS

		trail[trailLength][0] = current[0]
		trail[trailLength][1] = current[1]
		trailLength += 1

		for j in range(7,10):
			if trailLength-j > 0:
				tmpInd = trailLength-j
				if current[0] == trail[tmpInd][0] and current[1] == trail[tmpInd][1]:
					done = 1
					break

	cdef list trailOut = []
	for i in range(trailLength):
		trailOut.append([trail[i][0],trail[i][1]])


	return trailOut
