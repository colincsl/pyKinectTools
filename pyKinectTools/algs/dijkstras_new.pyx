''' Build with: python dijkstras_setup.py build_ext --inplace '''

import sys
import numpy as np
import cython
cimport cython
cimport numpy as np

from python_ref cimport Py_INCREF, Py_DECREF

cdef extern from "math.h":
	int abs(int x)

ctypedef np.uint16_t UINT16
ctypedef np.int16_t INT16
ctypedef np.uint8_t UINT8

np.import_array()

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline void ind2dim(int ind, int current[2], int rezX, int rezY): 
	current[0] = int(ind/rezX)
	current[1] = ind-rezX*((ind/rezX))
	return
	# return [(ind/rezX), ind-rezX*((ind/rezX))]
cdef inline int dim2ind(int i, int j, int rezX, int rezY): 
	return (i*rezX+j)
	# return (j*rezX+i)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
# @cython.profile(True)
# cpdef inline np.ndarray[INT16, ndim=2] graphDijkstras(np.ndarray[INT16, ndim=2] distsMat, np.ndarray[UINT8, ndim=2] visitMat, np.ndarray[INT16, ndim=2] depthMat, np.ndarray[INT16, ndim=1] current_):
cpdef inline list graphDijkstras(np.ndarray[UINT16, ndim=2, mode="c"] distsMat, np.ndarray[UINT8, ndim=2, mode="c"] visitMat, np.ndarray[UINT16, ndim=2, mode="c"] depthMat, np.ndarray[INT16, ndim=1, mode="c"] current_):
	'''
	Inputs:
		distsMat
		visitMat
		dists2
		current_
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
	# cdef int TOUCHED=1, MAXVALUE=32000
	# cdef UINT8 *VISITED=254, *OUTOFBOUNDS=255
	cdef int neighborsVisited
	cdef int foundTouched

	# if distsMat.flags.aligned == True:
	# 	print "yes"
	# else:
	# 	print "no"


	# Py_INCREF(distsMat)
	# Py_INCREF(visitMat)
	# Py_INCREF(depthMat)

	# cdef UINT16* distsMat_c = &distsMat[0,0]
	# cdef UINT8* visitMat_c = &visitMat[0,0]
	# cdef UINT16* depthMat_c = &depthMat[0,0]

	cdef UINT16 distsMat_c[307200]
	for y in xrange(height):
		for x in xrange(width):
			i = y*width + x
			distsMat_c[i] = distsMat[y,x]
	cdef UINT8 visitMat_c[307200]
	for y in xrange(height):
		for x in xrange(width):
			i = y*width + x
			visitMat_c[i] = visitMat[y,x]
	cdef UINT16 depthMat_c[307200]
	for y in xrange(height):
		for x in xrange(width):
			i = y*width + x
			depthMat_c[i] = depthMat[y,x]

	# cdef np.ndarray[np.double_t, ndim=1] rr
	# cdef np.ndarray[UINT16, ndim=1] distsMat_c = distsMat#np.zeros([height, width], dtype=np.uint16)
	# distsMat_c.data = <char*>&distsMat[0,0]
	# cdef np.ndarray[UINT8, ndim=2] visitMat_c = &visitMat[0,0]
	# cdef np.ndarray[UINT16, ndim=2] depthMat_c = &depthMat[0,0]


	currentInd = dim2ind(current[0], current[1], width, height)

	# Set border to OUTOFBOUNDS
	for i in range(0,height):
		tmpInd = dim2ind(i, 0, width, height)
		visitMat_c[tmpInd] = OUTOFBOUNDS
		tmpInd = dim2ind(i, width, width, height)
		visitMat_c[tmpInd] = OUTOFBOUNDS		
	for i in range(0,width):
		tmpInd = dim2ind(0, i, width, height)
		visitMat_c[tmpInd] = OUTOFBOUNDS
		tmpInd = dim2ind(height, i, width, height)
		visitMat_c[tmpInd] = OUTOFBOUNDS

	''' --- Main --- '''
	while(1):
		currentCost = distsMat_c[currentInd]
		ind2dim(currentInd, current, width, height)
		visitMat_c[currentInd] = VISITED

		''' Update neighbors '''
		minGradient = MAXVALUE
		argGradient = -1
		neighborsVisited = 0
		for i in range(-1,2):
			for j in range(-1,2):
				if not (i == 0 and j == 0):
					tmpInd = dim2ind(current[0]+i, current[1]+j, width, height)
					if visitMat_c[tmpInd] < VISITED:

						gradient = abs(depthMat_c[tmpInd]-depthMat_c[currentInd])
						# distsMat_c[tmpInd] = int_min(distsMat_c[tmpInd], currentCost + 1)
						distsMat_c[tmpInd] = int_min(distsMat_c[tmpInd], currentCost + gradient + 1)
						visitMat_c[tmpInd] = int_max(visitMat_c[tmpInd], TOUCHED)

						if visitMat_c[tmpInd] != VISITED:
							if gradient < minGradient:
								minGradient = gradient
								argGradient = tmpInd

						''' Mark if previously visited '''
						if visitMat_c[tmpInd] == VISITED or visitMat_c[tmpInd] == OUTOFBOUNDS:
							neighborsVisited += 1

		''' If all neighbors have been visited, shift to new touched spot '''
		if neighborsVisited < 8 and argGradient >= 0:
			currentInd = argGradient
		else:

			tmpX = 0
			foundTouched = 0
			while (tmpX < width-1 and foundTouched==0):
				tmpY = 0
				while (tmpY  < height-1 and foundTouched==0):
					tmpInd = dim2ind(tmpY, tmpX, width, height)

					if visitMat_c[tmpInd] == TOUCHED:
						currentInd = tmpInd
						foundTouched = 1
						break
					tmpY += 1
				tmpX += 1

			# If no more open spots
			if foundTouched == 0:
				break


	''' Find best path '''
	cdef int maxVal=0
	cdef int maxInd=-1
	# Get max value
	for y in range(height-1):
		for x in range(width-1):
			tmpInd = dim2ind(y, x, width, height)
			if distsMat_c[tmpInd] > maxVal and visitMat_c[tmpInd] == VISITED:
				maxVal = distsMat_c[tmpInd]
				maxInd = tmpInd
	ind2dim(maxInd, current, width, height)

	currentInd = maxInd
	cdef int currentVal = maxVal
	cdef int trailLength=1, done=0

	cdef int trail[1000][2]
	trail[0][0] = current[0];
	trail[0][1] = current[1]

	# print "start:", current[0], current[1]

	while currentVal!=0 and done != 1 and trailLength < 1000:
		minGradient = MAXVALUE
		argGradient = -1

		for i in range(-1,2):
			for j in range(-1,2):
				if not (i == 0 and j == 0):
					tmpInd = dim2ind(current[0]+i, current[1]+j, width, height)
					gradient = distsMat_c[tmpInd]-distsMat_c[currentInd]

					if gradient < minGradient and visitMat_c[tmpInd] == VISITED:
						minGradient = gradient
						argGradient = tmpInd

		if argGradient < 0:
			done = 1
			# print "Error"
			break

		ind2dim(argGradient, current, width, height)
		currentInd = argGradient
		currentVal = distsMat_c[argGradient]
		visitMat_c[currentInd] = OUTOFBOUNDS

		# print 'v:',currentVal, 'p:', current[0], current[1]
		trail[trailLength][0] = current[0]
		trail[trailLength][1] = current[1]
		trailLength += 1

		for j in range(7,10):
			if trailLength-j > 0:
				tmpInd = trailLength-j
				if current[0] == trail[tmpInd][0] and current[1] == trail[tmpInd][1]:
					done = 1
					break

	# print currentVal
	cdef list trailOut = []
	# trailOut = np.empty([trailLength, 2], dtype=int)
	for i in range(trailLength):
		# trailOut[i] = [trail[i][0],trail[i][1]]
		trailOut.append([trail[i][0],trail[i][1]])

		# py_incref
		# py_decref
	# del distsMat
	# del visitMat
	# del depthMat

	# distsMat.data = <char*>distsMat_c
	# visitMat.data = <char*>visitMat_c
	# depthMat.data = <char*>depthMat_c

	# distsMat = np.PyArray_SimpleNewFromData(2, [height, width], np.NPY_UINT16, <void*> distsMat_c)
	# visitMat = np.PyArray_SimpleNewFromData(2, [height, width], np.NPY_UINT8, <void*> visitMat_c)
	# depthMat = np.PyArray_SimpleNewFromData(2, [height, width], np.NPY_UINT16, <void*> depthMat_c)

	# cdef int x

	# print "refs:", sys.getrefcount(distsMat)
	# x = Py_DECREF(distsMat)
	# print x
	# Py_DECREF(visitMat)
	# Py_DECREF(depthMat)

	# for i in xrange(height*width):
	for y in xrange(height):
		for x in xrange(width):
			i = y*width + x
			distsMat[y,x] = distsMat_c[i]
	for y in xrange(height):
		for x in xrange(width):
			i = y*width + x
		visitMat[y,x] = visitMat_c[i]
	for y in xrange(height):
		for x in xrange(width):
			i = y*width + x
		depthMat[y,x] = depthMat_c[i]



	return trailOut




