''' Build with: python dijkstras_setup.py build_ext --inplace '''

import numpy as np
import cython
cimport cython
cimport numpy as np
from libc.math cimport abs, floor
# from cython.view import contig, indirect, strided, follow

ctypedef np.uint16_t UINT16
ctypedef np.int16_t INT16
ctypedef np.uint8_t UINT8

# from cython.parallel import prange
np.import_array()

# cython: profile=True

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline list ind2dim(int ind, int rezX, int rezY): 
	return [floor(ind/rezX), ind-rezX*floor(ind/rezX)]
cdef inline int dim2ind(int i, int j, int rezX, int rezY): return (j*rezX+i)

'''Compute the weighted distance to a point on the graph'''
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.infer_types(True)
@cython.profile(True)
cpdef inline list graphDijkstras(np.ndarray[INT16, ndim=2] dists2Tot, np.ndarray[UINT8, ndim=2] visitMat, np.ndarray[INT16, ndim=3] dists2, list current_):

	cdef unsigned int currentCost, direc,prevDirec, found, checkAll
	cdef int tmp, tmpMin=0, tmpNeigh=0
	cdef int currentVal
	cdef unsigned int d, length, i, x, y, j
	cdef unsigned int shape_[2]
	cdef int neigh[2][4]
	shape_[0] = np.shape(dists2)[0]# was 0
	shape_[1] = np.shape(dists2)[1]# was 1	
	# shape_[1] = np.shape(dists2)[0]
	# shape_[0] = np.shape(dists2)[1]
	# print shape_[0], shape_[1]

	cdef INT16 dists2_c[720][480][4]#[shape_[0]][shape_[1]]
	cdef INT16 dists2Tot_c[720][480]#[shape_[0]][shape_[1]]
	cdef UINT8 visitMat_c[720][480]#[shape_[0]][shape_[1]]
	''' Put images into c arrays '''
	for x in range(shape_[0]):
		for y in range(shape_[1]):
			for i in range(4):
				dists2_c[x][y][i] = dists2[x,y,i]
	for x in range(shape_[0]):
		for y in range(shape_[1]):
				dists2Tot_c[x][y] = dists2Tot[x,y]	
				visitMat_c[x][y] = visitMat[x,y]

	cdef int current[2]# = [int(current[0]), int(current[1])]
	current[0] = np.int(current_[0])
	current[1] = np.int(current_[1])

	dists2Tot_c[current[0]][current[1]] = 0

	length = shape_[0]*shape_[1]

	''' main '''
	while(1):
	# for i in range(length):
	# for i in prange(length, nogil=True):
		currentCost = dists2Tot_c[current[0]][current[1]]
		neigh[0][0] = current[0]-1; 	neigh[1][0] = current[1]
		neigh[0][1] = current[0]+1;		neigh[1][1] = current[1]
		neigh[0][2] = current[0];		neigh[1][2] = current[1]+1
		neigh[0][3] = current[0];		neigh[1][3] = current[1]-1
				
		visitMat_c[<unsigned int>current[0]][<unsigned int>current[1]] = 254

		''' Check if in bounds'''
		if (current[0] < 1) or (current[0] > shape_[0]-1) or (current[1] < 1) or (current[1] > shape_[1]-1):
		# if (current[0] < 1) or (current[0] > shape_[1]-1) or (current[1] < 1) or (current[1] > shape_[0]-1):
			found = 0
			for x in range(shape_[0]):
				for y in range(shape_[1]):	
					if visitMat_c[x][y] == 1:						
						current[0] = x
						current[1] = y
						found = 1
						break
				if found:
					break
			if found:
				continue
			else:
				break



		''' Compute new distances and add nodes to visit list'''
		tmpNeigh = 0
		for j in range(4):
			if visitMat_c[neigh[0][j]][neigh[1][j]] == 255:
				tmpNeigh += 1
				if tmpNeigh == 3:
					visitMat_c[<unsigned int>current[0]][<unsigned int>current[1]] = 255


			# if dists2_c[current[0]][current[1]][j] > 10000:
			visitMat_c[neigh[0][j]][neigh[1][j]] = int_max(1, visitMat_c[neigh[0][j]][neigh[1][j]])
			dists2Tot_c[neigh[0][j]][neigh[1][j]] = int_min(dists2Tot_c[neigh[0][j]][neigh[1][j]], (dists2_c[current[0]][current[1]][j] + currentCost +1))
			# if visitMat_c[neigh[0][j]][neigh[1][j]] == 255:
			# 	visitMat_c[<unsigned int>current[0]][<unsigned int>current[1]] = 255
			# else:
				# visitMat_c[neigh[0][j]][neigh[1][j]] = int_max(1, visitMat_c[neigh[0][j]][neigh[1][j]])
				# dists2Tot_c[neigh[0][j]][neigh[1][j]] = 0#int_min(dists2Tot_c[neigh[0][j]][neigh[1][j]], (dists2_c[current[0]][current[1]][j] + currentCost))

		''' Find minimum direction (gradient)'''
		tmpMin = 30000
		for j in range(4):
			if visitMat_c[neigh[0][j]][neigh[1][j]] < 250:
				tmp = dists2Tot_c[neigh[0][j]][neigh[1][j]]+visitMat_c[neigh[0][j]][neigh[1][j]]
				if tmp < tmpMin:
					tmpMin = tmp
					direc = j
		if direc == 0:
			current[0] -= 1
		elif direc == 1:
			current[0] += 1
		elif direc == 2:
			current[1] += 1
		elif direc == 3:
			current[1] -= 1

		'''	Check if neighbors have all been visited'''
		checkAll = 0
		for j in range(4):
			if visitMat_c[neigh[0][j]][neigh[1][j]] > 250:
				checkAll += 1
		if checkAll == 4:
			found = 0
			for x in range(shape_[0]):
				for y in range(shape_[1]):	
					if visitMat_c[x][y] == 1:						
						current[0] = x
						current[1] = y
						found = 1
						break
				if found:
					break
			if found:
				continue
			else:
				break

	'''Create trail going from max to min'''
	cdef int maxVal = 0
	cdef int maxInd[2]
	maxInd[0] = 0; maxInd[1] = 0
	for x in range(shape_[0]):
		for y in range(shape_[1]):
			if visitMat_c[x][y] == 254 and dists2Tot_c[x][y] > maxVal and dists2Tot_c[x][y] < 30000:
				maxVal = dists2Tot_c[x][y]
				maxInd[0] = x
				maxInd[1] = y

	# print "maxes: ", maxInd[0], maxInd[1], maxVal
	cdef int trailLen=1
	cdef int v
	cdef int trail[2000][2]	
	trail[0][0] = maxInd[0]; trail[0][1] = maxInd[1]
	current[0] = maxInd[0]; current[1] = maxInd[1]
	currentVal = maxVal
	prevDirec = 0
	while (currentVal != 0 and trailLen < 2000):
		neigh[0][0] = current[0]-1; 	neigh[1][0] = current[1]
		neigh[0][1] = current[0]+1;		neigh[1][1] = current[1]
		neigh[0][2] = current[0];		neigh[1][2] = current[1]+1
		neigh[0][3] = current[0];		neigh[1][3] = current[1]-1

		# Get downward direction
		# tmpMin = 9000#dists2Tot_c[neigh[0][0]][neigh[1][0]]
		tmpMin = 31000 # was +999
		direc = 3 # rand direction
		for j in range(4):
			# if j != prevDirec and \
			if 1 and \
					neigh[0][j] > 0 and neigh[1][j] >= 0 and \
					neigh[0][j] < shape_[0] and neigh[1][j] < shape_[1] and \
					visitMat_c[neigh[0][j]][neigh[1][j]] == 254:

				v = dists2Tot_c[neigh[0][j]][neigh[1][j]]
				tmp = dists2Tot_c[current[0]][current[1]] - v
				if tmp > 0 and tmp <= tmpMin:
					tmpMin = tmp
					direc = j
		# print tmp, j
		if direc == 0:
			current[0] -= 1
			prevDirec = 1
		elif direc == 1:
			current[0] += 1
			prevDirec = 0
		elif direc == 2:
			current[1] += 1
			prevDirec = 3
		elif direc == 3:
			current[1] -= 1
			prevDirec = 2
		currentVal = dists2Tot_c[current[0]][current[1]]
		trail[trailLen][0] = current[0]
		trail[trailLen][1] = current[1]
		trailLen += 1

		# print direc, tmpMin
	cdef list trailOut = []
	for x in range(trailLen):
		trailOut.append([trail[x][0], trail[x][1]])


	# Push data back to numpy array
	for x in range(shape_[0]):
		for y in range(shape_[1]):
				dists2Tot[x,y] = dists2Tot_c[x][y]
				visitMat[x,y] = visitMat_c[x][y]

	return trailOut


''' --------------AStar----------------------- '''

# cdef list UnstructuredDijkstras(int start, int end, list edges, list edgePositions):
# 	cdef int nodeCount = len(edgePositions)
# 	cdef list trail = []
# 	cdef list openSet []
# 	cdef list closedSet = []
# 	cdef int distances[nodeCount]
# 	for i in range(nodeCount):
# 		distance[i] = 9999
	
# 	for e in edges[start]:
# 		openSet.append(e)

		
# 	while openSet != []:
# 		''' find minimum score '''
# 		lowIndRaw = 0; lowInd = 0;	lowCost = 999



# 	while start[0] != end[0] and start[1] != end[1]:
# 		for e in edges[start]:
# 			openSet.append(e)

	




# cdef inline list ReconstructPath(int endPt[2], UINT8 visitMat_c[720][480], unsigned int shape_[2]):
# 	cdef list p = []
# 	cdef int end[2]
# 	print 2.1
# 	if visitMat_c[endPt[0]][endPt[1]] != 0:
# 		print 2.2
# 		print visitMat_c[endPt[0]][endPt[1]]
# 		tmp = ind2dim(visitMat_c[endPt[0]][endPt[1]], shape_[0], shape_[1])
# 		end[0] = tmp[0]
# 		end[1] = tmp[1]
# 		print 2.3
# 		p = ReconstructPath(end, visitMat_c, shape_)
# 		print 2.4
# 		p.append([endPt[0],endPt[1]])
# 		return 
# 	else:
# 		print 2.5
# 		return [end[0],end[1]]


# cpdef inline int AStar(list startPt, list endPt, np.ndarray[UINT8, ndim=2] visitMat, np.ndarray[INT16, ndim=3] dists2):
# 	cdef unsigned int i, j, x, y, lowInd, lowCost, added,indNeigh
# 	cdef unsigned int shape_[2]	
# 	shape_[0] = np.shape(dists2)[0]
# 	shape_[1] = np.shape(dists2)[1]
# 	cdef int neigh[2][4]

# 	cdef list closedSet = []
# 	cdef unsigned int closedLength = 0
# 	cdef list openSet = [dim2ind(startPt[0],startPt[1], shape_[0], shape_[1])]

# 	cdef int current[2]
# 	current[0] = np.int(startPt[0])
# 	current[1] = np.int(startPt[1])
# 	cdef int currentInd = dim2ind(current[0], current[1], shape_[0], shape_[1])
# 	cdef int end[2]

# 	cdef list inds = [dim2ind(startPt[0],startPt[1], shape_[0], shape_[1])]
# 	cdef unsigned int indsCount = 1
	
# 	cdef list costInds = [currentInd]
# 	cdef list gScoreVals = [0]

# 	tmp = np.sqrt((startPt[0]-endPt[0])**2+(startPt[1]-endPt[1])**2)
# 	cdef list fScoreVals = [tmp]

# 	''' Put images into c arrays '''
# 	cdef INT16 dists2_c[720][480][4]#[shape_[0]][shape_[1]]
# 	cdef UINT8 visitMat_c[720][480]#[shape_[0]][shape_[1]]
# 	for x in range(shape_[0]):
# 		for y in range(shape_[1]):
# 			for i in range(4):
# 				dists2_c[x][y][i] = dists2[x,y,i]
# 	for x in range(shape_[0]):
# 		for y in range(shape_[1]):
# 				visitMat_c[x][y] = 0#visitMat[x,y]

# 	''' main '''
# 	while openSet != []:
# 		''' find minimum score '''
# 		lowIndRaw = 0; lowInd = 0;	lowCost = 999
# 		# print 'fScores:', fScoreVals
# 		for i in range(indsCount):
# 			if costInds[i] in openSet and fScoreVals[i] < lowCost:
# 				lowIndRaw = i
# 				lowInd = costInds[i]
# 				lowCost = fScoreVals[i]

# 		tmp = ind2dim(lowInd, shape_[0], shape_[1])
# 		current[0] = tmp[0]
# 		current[1] = tmp[1]
# 		# print 'Current:', lowInd, current[0], current[1]

# 		''' if endpoint then return '''
# 		if current[0] == endPt[0] and current[1] == endPt[1]:
# 			print 1
# 			end[0] = int(endPt[0])
# 			end[1] = int(endPt[1])
# 			print 2
# 			p = ReconstructPath(end, visitMat_c, shape_)
# 			print p
# 			return p

# 		''' Setup neighbors '''
# 		neigh[0][0] = current[0]-1; 	neigh[1][0] = current[1]
# 		neigh[0][1] = current[0]+1;		neigh[1][1] = current[1]
# 		neigh[0][2] = current[0];		neigh[1][2] = current[1]+1
# 		neigh[0][3] = current[0];		neigh[1][3] = current[1]-1

# 		'''' remove current from openSet '''
# 		# print 1
# 		newOpenSet = []
# 		for i in range(len(openSet)):
# 			# print 2
# 			if openSet[i] != lowInd:
# 				newOpenSet.append(openSet[i])
# 		# print 'old', openSet
# 		# print 'new', newOpenSet
# 		openSet = newOpenSet
# 		closedSet.append(int(lowInd))
# 		# print 'closed', closedSet

# 		for i in range(4):
# 			indNeigh = dim2ind(neigh[0][i],neigh[1][i],shape_[0],shape_[1])
# 			for j in range(indsCount):
# 				if costInds[j] == indNeigh:
# 					indNeighRaw = j
# 			''' Check if neighbor is in closedSet '''
# 			if int(indNeigh) in closedSet:
# 				continue

# 			currNeigh = [neigh[0][i], neigh[1][i]]
# 			# print 'neigh', indNeigh, currNeigh[0], currNeigh[1], "f", dists2_c[current[0]][current[1]][i]
# 			tmpGScore = gScoreVals[lowIndRaw] + abs(dists2_c[current[0]][current[1]][i])
# 			# print 2
# 			# added = 1
# 			for j in range(len(openSet)):
# 				if indNeigh == openSet[j]:# or tmpGScore > gScoreVals[indNeigh]:
# 					break
# 			if indNeigh not in openSet or tmpGScore < gScoreVals:
# 				openSet.append(indNeigh)
# 				visitMat_c[currNeigh[0]][currNeigh[1]] = lowInd
# 				print lowInd
# 				indsCount += 1
# 				# cameFrom[i] = current
# 				costInds.append(indNeigh)
# 				gScore[currNeigh[0]][currNeigh[1]] = tmpGScore
# 				fScore[currNeigh[0]][currNeigh[1]] = tmpGScore + np.sqrt((currNeigh[0]-endPt[0])**2+(currNeigh[1]-endPt[1])**2)
# 				gScoreVals.append(tmpGScore)
# 				fScoreVals.append(tmpGScore + np.sqrt((currNeigh[0]-endPt[0])**2+(currNeigh[1]-endPt[1])**2))
# 			# print gScoreVals[-1], fScoreVals[-1]
# 		# print '--'


# 	return -1


# 	# cdef unsigned int currentCost, direc, checkAll







