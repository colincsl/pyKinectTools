import numpy as np


def MinimumSpanningTree(distMatrix, startNode=0):
	#Input 2D NxN numpy array
	nodeList = [startNode]
	edgeList = []
	edgeListDict = {}
	nodes = range(distMatrix.shape[0])
	nodes.remove(startNode)
	currentNode = startNode

	edgeListDict[0] = []
	for i in nodes:
		edgeListDict[i] = []

	while nodes != []:
		minDist = np.inf
		closestNode = -1
		currentNode = -1
		for curr in nodeList:
		# curr = currentNode
		# if 1:
			for n in nodes:
				if distMatrix[n,curr] < minDist:
					minDist = distMatrix[n,curr]
					currentNode = curr
					closestNode = n
		if closestNode >= 0:
			nodes.remove(closestNode)
			nodeList.append(closestNode)
			edgeList.append([currentNode, closestNode])
			edgeListDict[currentNode].append(closestNode)
			edgeListDict[closestNode].append(currentNode)

			# currentNode = closestNode		
			# print minDist
		else:
			print 'Error'
			break	
	return edgeList, edgeListDict



def PruneEdges(edgeDict, maxLength = 2):
	nodeList = edgeDict.keys()
	deletedInds = []
	for n in nodeList:
		if len(edgeDict[n]) == 1:
			length = 1
			subtree = [n]
			n2 = edgeDict[n][0]

			while len(edgeDict[n2]) == 2 and length < maxLength:
				subtree.append(n2)
				# n2 = edgeDict[n2][0]
				n2 = [x for x in edgeDict[n2] if x not in subtree][0]
				print n, n2, subtree
				length += 1
			subtree.append(n2)

			# if len(edgeDict[n2]) > 2:
			if len(subtree) > 1:
				for i in range(0, len(subtree)-1):
					print i, subtree
					edgeDict[subtree[i]].remove(subtree[i+1])
					edgeDict[subtree[i+1]].remove(subtree[i])
					deletedInds.append(subtree[i])

	return edgeDict, deletedInds


def GetLeafLengths(edgeDict):
	nodeList = edgeDict.keys()
	trees = []
	for n in nodeList:
		if len(edgeDict[n]) == 1:
			subtree = [n]
			n2 = edgeDict[n][0]
			# print n2, len(edgeDict[n2])
			while len(edgeDict[n2]) == 2:
				subtree.append(n2)
				nodeList = [x for x in edgeDict[n2] if x not in subtree]
				if len(nodeList) > 0:
					n2 = nodeList[0]
				else:
					break

			subtree.append(n2)
			if len(subtree) >= 1:
				trees.append(subtree)
	return trees



'''----------------------------------------'''


def getClosestConnectedNode(edgeDict, regionLabels, avgColor):
	closestNode = [0]
	for key in edgeDict.keys():
		closestNode.append([-1])
		closestDist = np.inf
		y = regionLabels[key][2][0]
		x = regionLabels[key][2][1]	
		currentDepth = avgColor[y,x,2]
		for item in edgeDict[key]:
			y = regionLabels[item][2][0]
			x = regionLabels[item][2][1]
			if np.abs(avgColor[y,x,2] - currentDepth) < closestDist:
				closestNode[-1] = item
				closestDist = np.abs(avgColor[y,x,2] - currentDepth)

	return closestNode


def checkProfileSpike(profile):
	profile = np.array(profile)
	diff = max(1, np.abs(profile[0] - profile[-1]))
	mean = (profile[0] + profile[-1])/2
	if np.max(np.abs(profile - mean)) > .75*diff:
	# if np.max(np.abs(profile - np.mean(profile))) > .55*():
	# if np.max(np.abs(profile[1::] - profile[:-1])) >= 2:
		print np.max(np.abs(profile - mean)), diff
		return 1
	else:
		return 0

def UnstructuredAStar(start, end, edgeLists, regionXYZ):
	current = start
	nodeCount = len(regionXYZ)
	openSet = [start]
	closedSet = []
	travelDistances = np.zeros(nodeCount)
	prevNode = np.zeros(nodeCount, dtype=int)-1
	# endDistances = np.sum((regionXYZ - regionXYZ[end])**2, 1)

	while openSet != []:
		''' find minimum score '''
		argMin = np.argmin(travelDistances[openSet])#+endDistances[openSet])
		current = openSet[argMin]
		# print current
		if current == end:
			trail = reconstructPath(prevNode, current)
			return trail

		closedSet.append(current)
		openSet.remove(current)

		tmpTravelDist = travelDistances[current] + np.sum((regionXYZ[edgeLists[current]] - regionXYZ[current])**2, 1)
		
		for i in range(len(edgeLists[current])):
			if edgeLists[current][i] in closedSet:
				continue
			if edgeLists[current][i] not in openSet or tmpTravelDist[i] < travelDistances[edgeLists[current][i]]:
				openSet.append(edgeLists[current][i])
				prevNode[edgeLists[current][i]] = current
				travelDistances[edgeLists[current][i]] = tmpTravelDist[i]


	print "Error"
	return []

def reconstructPath(prevNode, currentNode):
	if prevNode[currentNode] != -1:
		p = []
		r = reconstructPath(prevNode, prevNode[currentNode])
		if type(r) == list:
			for i in r:
				p.append(i)
		else:
			p.append(r)
		p.append(currentNode)
		return p
	else:
		return currentNode

