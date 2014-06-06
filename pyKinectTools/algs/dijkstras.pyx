''' Build with: python dijkstras_setup.py build_ext --inplace '''

import numpy as np
import cython
cimport cython
cimport numpy as cnp
cnp.import_array()

from libcpp.vector cimport vector
from libc.math cimport floor

cdef extern from "math.h":
	int abs(int x)
	double sqrt(double x)
	double pow(double base, double exp)

ctypedef cnp.uint16_t UINT16
ctypedef cnp.int16_t INT16
ctypedef cnp.uint8_t UINT8
ctypedef cnp.float64_t FLOAT


cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline void ind2dim(int ind, int current[2], int width, int height):
	current[0] = int(ind/float(width))
	current[1] = ind - width*current[0]
	return
cdef inline int dim2ind(int i, int j, int width, int height):
	return (i*width+j)

# @cython.profile(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cpdef inline list graph_dijkstras(cnp.ndarray[UINT16, ndim=2, mode="c"] distsMat,
							cnp.ndarray[UINT8, ndim=2, mode="c"] visitMat,
							cnp.ndarray[UINT16, ndim=2, mode="c"] depthMat,
							cnp.ndarray[INT16, ndim=1, mode="c"] current_,
							int xy_scale):
	'''
	Inputs:
		distsMat (uint16)
		visitMat
		depthMat
		current_ : Starting point
		xy_scale : [used to compare depth and x/y values]
	Outputs:
	'''

	cdef int i, j, tmpX, tmpY
	cdef int minGradient, argGradient, gradient
	cdef int height = distsMat.shape[0]
	cdef int width = distsMat.shape[1]

	cdef int currentInd, tmpInd, current_cost
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

		current_cost = distsMat_c[current[0], current[1]]
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
						gradient = abs( depthMat_c[current[0]+i,current[1]+j] - depthMat_c[current[0],current[1]] ) + xy_scale # Add for distance to pixel

						distsMat_c[current[0]+i, current[1]+j] = int_min(distsMat_c[current[0]+i,current[1]+j], current_cost + gradient)
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



# @cython.profile(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cpdef inline distance_map(cnp.ndarray[UINT16, ndim=2, mode="c"] distsMat,
							cnp.ndarray[UINT8, ndim=2, mode="c"] visitMat,
							cnp.ndarray[UINT16, ndim=2, mode="c"] depthMat,
							cnp.ndarray[INT16, ndim=1, mode="c"] current_,
							int xy_scale):
	'''
	Inputs:
		distsMat (uint16)
		visitMat
		depthMat
		current_ : Starting point
		xy_scale : [used to compare depth and x/y values]
	Outputs:
	'''

	cdef int i, j, tmpX, tmpY
	cdef int minGradient, argGradient, gradient
	cdef int height = distsMat.shape[0]
	cdef int width = distsMat.shape[1]

	cdef int currentInd, tmpInd, current_cost
	cdef int current[2]
	current[0] = current_[0]
	current[1] = current_[1]

	cdef int TOUCHED=1, VISITED=254, OUTOFBOUNDS=255, MAXVALUE=32000
	cdef int neighborsVisited
	cdef int foundTouched

	currentInd = dim2ind(current[0], current[1], width, height)

	''' Set border to OUTOFBOUNDS '''
	visitMat[0, :] = OUTOFBOUNDS
	visitMat[height-1:,] = OUTOFBOUNDS
	visitMat[:,0] = OUTOFBOUNDS
	visitMat[:,width-1] = OUTOFBOUNDS

	# print "A"
	''' --- Main --- '''
	''' Part 1: Get cost to each node from centroid '''
	while(1):

		if current[0] < 0 or current[0] > height or current[1] < 0 or current[1] > width:
			break

		current_cost = distsMat[current[0], current[1]]
		visitMat[current[0],current[1]] = VISITED

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
					if visitMat[current[0]+i, current[1]+j] < VISITED:
						gradient = abs( depthMat[current[0]+i,current[1]+j] - depthMat[current[0],current[1]] ) + xy_scale

						distsMat[current[0]+i, current[1]+j] = int_min(distsMat[current[0]+i,current[1]+j], current_cost + gradient)
						visitMat[current[0]+i, current[1]+j] = int_max(visitMat[current[0]+i,current[1]+j], TOUCHED)

						if visitMat[current[0]+i, current[1]+j] < VISITED:
							if gradient < minGradient:
								minGradient = gradient
								argGradient = tmpInd

						''' Mark if previously visited or if out of bounds '''
						if visitMat[current[0]+i, current[1]+j] >= VISITED:
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

					if visitMat[tmpY, tmpX] == TOUCHED:
						currentInd = tmpInd
						ind2dim(currentInd, current, width, height)
						foundTouched = 1
						break
					tmpY += 1
				tmpX += 1

			'''If no more open spots'''
			if foundTouched == 0:
				break

	return



# # -------------------------------------------
''' This priority queue code is weakly adapted from pyDistances:
https://github.com/jakevdp/pyDistances/blob/master/ball_tree.pyx

DOESN'T WORK RIGHT NOW. EVERYTIME YOU INSERT IT REBUILDS A VECTOR... SO DO SOMETHING DIFFERENT..
'''

# cdef class PriorityQueue:
# 	cdef vector[INT16] queue_u
# 	cdef vector[INT16] queue_v
# 	cdef vector[INT16] queue_values

# 	cdef void init(self, INT16 size):
# 		self.queue_u.reserve(size)
# 		self.queue_v.reserve(size)
# 		self.queue_value.reserve(size)

# 	cdef INT16[] next(self):
# 		cdef INT16[] tmp = [self.queue_u.at(self.queue_u.size()-1), \
# 							self.queue_v.at(self.queue_v.size()-1)]
# 		self.queue_u.erase()
# 		self.queue_v.erase()
# 		self.vector.erase()
# 		return tmp

# 	cdef void insert(self, INT16 u, INT16 v, INT16 value):
# 		cdef int i_low = 0
# 		cdef int i_high = self.queue_values.size()-1
# 		cdef int i_mid

# 		# Search for the place to put the value in the queue
# 		if value >= self.values[i_high]:
# 			self.queue_u.push_back(u)
# 			self.queue_v.push_back(v)
# 			self.queue_values.push_back(value)
# 		elif value <= self.queue_values[i_low]:
# 			self.queue_u.insert(self.queue_u.begin(),u)
# 			self.queue_v.insert(self.queue_v.begin(),v)
# 			self.queue_values.insert(self.queue_values.begin(),value)
# 		else:
# 			while True:
# 				if (i_high - i_low) < 2:
# 					i_mid = i_low + 1
# 					self.queue_u.insert(self.queue_u.begin()+i_mid,u)
# 					self.queue_v.insert(self.queue_v.begin()+i_mid,v)
# 					self.queue_values.insert(self.queue_values.begin()+i_mid,value)
# 					break
# 				else:
# 					i_mid = (i_low + i_high) / 2

# 				if i_mid == i_low:
# 					i_mid = i_mid + 1
# 					self.queue_u.insert(self.queue_u.begin()+i_mid,u)
# 					self.queue_v.insert(self.queue_v.begin()+i_mid,v)
# 					self.queue_values.insert(self.queue_values.begin(),value)
# 					# self.queue_values.insert(queue_values.begin()+i_mid,value)
# 					break

# 				if value >= self.queue_values[i_mid]:
# 					i_low = i_mid
# 				else:
# 					i_high = i_mid



''' ------------------------------------------------- '''


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cpdef inline list geodesic_extrema_MPI(cnp.ndarray[INT16, ndim=3, mode="c"] depth_mat, \
								cnp.ndarray[INT16, ndim=1, mode="c"] start_pos, \
								int iterations):

	cdef int height = depth_mat.shape[0]
	cdef int width = depth_mat.shape[1]
	cdef cnp.ndarray[dtype=UINT16, ndim=2, mode='c'] cost_map = np.zeros([height, width], dtype=np.uint16)

	return geodesic_map_MPI(cost_map, depth_mat, start_pos, iterations, 0)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cpdef inline list geodesic_map_MPI(cnp.ndarray[UINT16, ndim=2, mode="c"] cost_map,\
								cnp.ndarray[INT16, ndim=3, mode="c"] depth_mat, \
								cnp.ndarray[INT16, ndim=1, mode="c"] start_pos, \
								int iterations,
								int visualize):
	'''
	Inputs:
		distsMat (uint16)
		visitMat
		depthMat
		current_ : Starting point
		xy_scale : [used to compare depth and x/y values]
	Outputs:

	Based on MPI algorithm. More efficient than Stanford method.

	(self-note: Length of queue gets to ~600)
	'''
	cdef int i, j, tmpX, tmpY
	cdef int minGradient, argGradient, gradient
	cdef int height = depth_mat.shape[0]
	cdef int width = depth_mat.shape[1]
	cdef int best_direction[0]

	cdef int currentInd, tmpInd, current_cost

	cdef int TOUCHED=1, VISITED=254, OUTOFBOUNDS=255, MAXVALUE=32000
	cdef int neighborsVisited
	cdef int foundTouched
	cdef double delta_cost

	# cdef cnp.ndarray[dtype=UINT16, ndim=2, mode='c'] cost_map = np.zeros([height, width], dtype=np.uint16)
	cdef cnp.ndarray[dtype=INT16, ndim=1, mode='c'] current = start_pos
	cost_map[depth_mat[:,:,2]!=0] += MAXVALUE
	cost_map[start_pos[0], start_pos[1]] = 0

	cdef int current_xyz[3]

	# cdef PriorityQueue queue = PriorityQueue(500)
	cdef vector[INT16] queue_u
	cdef vector[INT16] queue_v
	extrema = [start_pos]


	''' Set border to OUTOFBOUNDS '''
	depth_mat[0,:,2] = 0
	depth_mat[height-1:,2] = 0
	depth_mat[:,0,2] = 0
	depth_mat[:,width-1,2] = 0

	''' Part 1: Get cost to each node from centroid '''

	for iters in range(iterations):
		queue_u.push_back(current[0])
		queue_v.push_back(current[1])
		while queue_u.size() > 0:
			current[0] = queue_u.at(queue_u.size()-1)
			current[1] = queue_v.at(queue_v.size()-1)
			queue_u.pop_back()
			queue_v.pop_back()

			# Prevent from going over edge
			# if current[0] <= 0 or current[0] >= height-1 or current[1] <= 0 or current[1] >= width-1:
				# break

			current_cost = cost_map[current[0], current[1]]
			current_xyz[0] = depth_mat[current[0], current[1],0]
			current_xyz[1] = depth_mat[current[0], current[1],1]
			current_xyz[2] = depth_mat[current[0], current[1],2]

			# Find the neighbor with the minimum cost
			for i in range(-1,2):
				for j in range(-1,2):
					if (not (i == 0 and j == 0)):
						if current[0]+i >= 0 and current[0]+i < height and current[1]+j >= 0 and current[1]+j < width:
							if depth_mat[current[0]+i, current[1]+j, 2] != 0:
								delta_cost = sqrt(pow((depth_mat[current[0]+i, current[1]+j,0] - current_xyz[0]),2)+\
												  pow((depth_mat[current[0]+i, current[1]+j,1] - current_xyz[1]),2)+\
												  pow((depth_mat[current[0]+i, current[1]+j,2] - current_xyz[2]),2))
								# If this new value is lower than previous, mark new cost
								if current_cost+delta_cost < cost_map[current[0]+i, current[1]+j]:
									cost_map[current[0]+i, current[1]+j] = int(current_cost+delta_cost)
									queue_u.insert(queue_u.begin(), int(current[0]+i))
									queue_v.insert(queue_v.begin(), int(current[1]+j))

		cost_map[cost_map==MAXVALUE] = 0
		extrema_ind = np.argmax(cost_map)
		extrema.append(np.unravel_index(extrema_ind, [height, width]))
		current[0] = int(extrema[iters+1][0])
		current[1] = int(extrema[iters+1][1])
		cost_map[current[0], current[1]] = 0

	if visualize == 1:
		extrema += [np.copy(cost_map)]
	return extrema


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cpdef inline cnp.ndarray[UINT16, ndim=2, mode="c"] geodesic_trail(cnp.ndarray[UINT16, ndim=2, mode="c"] cost_map,\
								cnp.ndarray[INT16, ndim=1, mode="c"] start_pos):

	cdef int height = cost_map.shape[0]
	cdef int width = cost_map.shape[1]
	cdef list trail = [[start_pos[0],start_pos[1]]]
	cdef cnp.ndarray[dtype=INT16, ndim=1, mode='c'] current_uv = start_pos
	cdef int current_energy = cost_map[current_uv[0],current_uv[1]]
	cdef int min_step[2]
	cdef int min_step_cost = 0
	cdef int max_step_cost = 0
	cdef int diff_tmp

	while current_energy != 0 and min_step_cost < 32000:
		min_step_cost = 32000
		max_step_cost = 0
		min_step[0] = 0
		min_step[1] = 0
		for i in range(-1,2):
			for j in range(-1,2):
				# print current_uv
				if not (i == 0 and j == 0) \
					and current_uv[0]+i > 0 and current_uv[0]+i < height-1\
					and current_uv[1]+j > 0 and current_uv[1]+j < width-1:
					diff_tmp = cost_map[current_uv[0],current_uv[1]] - cost_map[current_uv[0]+i,current_uv[1]+j]
					if diff_tmp > 0 and cost_map[current_uv[0]+i,current_uv[1]+j] < 32000\
						and diff_tmp < 50:
						if cost_map[current_uv[0]+i,current_uv[1]+j] < min_step_cost:
							min_step_cost = cost_map[current_uv[0]+i,current_uv[1]+j]
							min_step[0] = i
							min_step[1] = j
						# else:
							# min_step_cost = cost_map[current_uv[0]+i,current_uv[1]+j]
							# min_step[0] = i
							# min_step[1] = j
		if min_step[0] == 0 and min_step[1] == 0:
			break
		current_uv[0] += min_step[0]
		current_uv[1] += min_step[1]
		current_energy = min_step_cost
		trail += [[current_uv[0], current_uv[1]]]


	return np.array(trail, dtype=np.int16)




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cpdef inline cnp.ndarray[FLOAT, ndim=3, mode="c"] relative_distance_features(\
									cnp.ndarray[FLOAT, ndim=3, mode="c"] depth_mat, \
									cnp.ndarray[FLOAT, ndim=2, mode="c"] markers, \
									cnp.ndarray[FLOAT, ndim=3, mode="c"] feature_map):
	'''
	Inputs:
		depthMat
	Outputs:
	'''
	cdef int r, c, m
	cdef int rows = depth_mat.shape[0]
	cdef int cols = depth_mat.shape[1]
	cdef int marker_count = markers.shape[0]

	for r in xrange(rows):
		for c in xrange(cols):
			for m in xrange(marker_count):
				if depth_mat[r,c,2] != 0:
					# print depth_mat[r,c],markers[m]
					feature_map[r,c,m] = sqrt(pow(depth_mat[r,c,0]-markers[m,0],2)+\
												pow(depth_mat[r,c,1]-markers[m,1],2)+
												pow(depth_mat[r,c,2]-markers[m,2],2)
											)


	return feature_map


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
cpdef inline cnp.ndarray[UINT8, ndim=3, mode="c"] local_histograms(\
									cnp.ndarray[UINT8, ndim=2, mode="c"] image, \
									int n_bins, int patch_size, int max_bound):
	'''
	Inputs:
	Outputs:
	'''
	assert patch_size%2 == 1, "patch_size must be odd"

	cdef int r, c, rr, cc
	cdef int rows = image.shape[0]
	cdef int cols = image.shape[1]
	cdef int patch_offset = (patch_size-1)/2
	cdef int bin_size = max_bound/n_bins + 1
	cdef cnp.ndarray[UINT8, ndim=3, mode="c"] output = np.zeros([rows, cols, n_bins], np.uint8)

	## Slower method
	# for r in xrange(patch_offset, rows-patch_offset):
	# 	for c in xrange(patch_offset, cols-patch_offset):
	# 		for rr in xrange(-patch_offset, patch_offset+1):
	# 			for cc in xrange(-patch_offset, patch_offset+1):
	# 				bin = image[r+rr,c+cc]/bin_size
	# 				output[r,c,bin] += 1

	for r in xrange(patch_offset, rows-patch_offset):
		for c in xrange(patch_offset, cols-patch_offset):
			bin = image[r,c]/bin_size
			output[r-patch_offset:r+patch_offset+1, c-patch_offset:c+patch_offset+1, bin] += 1

	return output


