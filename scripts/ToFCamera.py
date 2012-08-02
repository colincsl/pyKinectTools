


import Image 
import sys, os
import cv2
from pyKinectTools.algs.graphAlgs import *
import pyKinectTools.algs.neighborSuperpixels as nsp

def reorder(regions):
	labs = unique(regions)
	for i, lab_i in zip(range(len(labs)), labs):
		regions[regions==lab_i] = i
	return regions

#  Now in pKT.algs.graphAlgs
# def edgeList2Dict(edges):
# 	edgeDict = {}
# 	for e1,e2 in edges:
# 		if e1 not in edgeDict.keys():
# 			edgeDict[e1] = [e2]
# 		else:
# 			edgeDict[e1].append(e2)

# 	return edgeDict

def getRelativeDists(edgeDict):
	relativeDists = []
	for key in edgeDict.keys():
		mean = 0
		for item in edgeDict[key]:
			y = regionLabels[item][2][0]
			x = regionLabels[item][2][1]
			mean += avgColor[y,x,2]
		mean /= len(edgeDict)
		y = regionLabels[key][2][0]
		x = regionLabels[key][2][1]	
		relativeDists.append(avgColor[y,x,2] - mean)
	return relativeDists


# Load and display time of flight images
if 0:
	files = os.listdir('.')
	cv2.namedWindow("win")
	for filename in files:
		tof = np.array(Image.open(filename))
		im = tof[:200,:, 0]
		im8bit = im*(im<150)*(im>50)
		# im8bit = im
		im4d = np.dstack([im8bit>0, im8bit, im8bit, im8bit])
		regions = slic.slic_n(np.array(im4d, dtype=uint8), 300,5)
		regions *= (im8bit>0)
		regions = reorder(regions)
		# regions2 = slic.slic_n(np.array(im4d, dtype=uint8), 50,5)
		# regions2 *= (im8bit>0)
		# regions3 = slic.slic_n(np.array(im4d, dtype=uint8), 20,1)
		# regions3 *= (im8bit>0)
		# regions = slic.slic_n(np.array(im4d, dtype=uint8), 30,5)
		# imshow(regions)

		cv2.imshow("win", regions*1.0/regions.max())
		ret = cv2.waitKey(150)

		if ret >= 0:
			break


# # HOG
# hogSize = 8
# hogIn = np.dstack([regions,regions,regions])
# hog = features.hog(hogIn, hogSize)
# hIm = HOGpicture(hog, hogSize)
# imshow(hIm)
# r2 = overlayHOG(regions>0, hIm)
# imshow(r2)



# touching neighbors
allEdges = nsp.getNeighborEdges(np.ascontiguousarray(regions, dtype=uint8))
edges = []
for i in allEdges:
	if i[0] != 0 and i[1] != 0:
	# if 1:
		if i not in edges:
			edges.append(i)
		if [i[1],i[0]] not in edges:
			edges.append([i[1],i[0]])

edgeDict = edgeList2Dict(edges)
relativeDists = getRelativeDists(edgeDict)
closestNodes = getClosestConnectedNode(edgeDict, regionLabels, avgColor)
regionXYZ = ([x[1] for x in regionLabels if x[0] != 0])
regionPos = ([x[2] for x in regionLabels if x[0] != 0])
regionXYZ.insert(0,[0,0,0])
regionPos.insert(0,[0,0])
edgeProfiles = nsp.getNeighborProfiles(np.ascontiguousarray(im8bit, dtype=np.uint8), edges, regionPos)
edgeProfilesNorm = [x-np.mean(x) for x in edgeProfiles]

''' Draw edges based on closest node '''
imLines = deepcopy(regions)
removeEdges = []
for i, ind in zip(edges, range(len(edges))):
	# if closestNodes[i[0]] == i[1]:
	# if not checkProfileSpike(edgeProfiles[ind]):
	if 1:
		pt1 = (regionLabels[i[0]][2][1],regionLabels[i[0]][2][0])
		pt2 = (regionLabels[i[1]][2][1],regionLabels[i[1]][2][0])
		cv2.line(imLines, pt1, pt2, 40)
	else:
		removeEdges.append(ind)
for i in range(1,regionCount):
	pt1 = (regionLabels[i][2][1],regionLabels[i][2][0])
	cv2.circle(imLines, pt1, radius=0, color=50, thickness=3)
	text(pt1[0]+2, pt1[1], str(i))
# for e in removeEdges:
	# edgeDict
imshow(imLines)

''' Get upper and middle nodes '''
regionXYZ = np.array(regionXYZ)
midpoint = np.argmin(np.sum((regionXYZ[1:] - np.mean(regionXYZ[1:,0]))**2, 1))+1
uppermostPart = np.argmax(regionXYZ[1:,0])+1
upVec = regionXYZ[uppermostPart] - regionXYZ[midpoint]
upVec /= np.sum(upVec)
pt1 = (regionPos[midpoint][1],regionPos[midpoint][0])
pt2 = (regionPos[uppermostPart][1],regionPos[uppermostPart][0])

edgeLists = [edgeDict[x] for x in edgeDict.keys()]
edgeLists.insert(0,[])
trail = UnstructuredAStar(midpoint, uppermostPart, edgeLists, regionXYZ)


for i in range(len(trail)-1):
	pt1 = (regionPos[trail[i]][1], regionPos[trail[i]][0])
	pt2 = (regionPos[trail[i+1]][1], regionPos[trail[i+1]][0])
	cv2.line(imLines, pt1, pt2, color=60)

imshow(imLines)


dists = np.sqrt(np.sum((regionXYZ[1:] - regionXYZ[midpoint])**2, 1))
tmpNodes = np.nonzero((100 < dists) * (dists < 250))[0]+1


upperMidpoint = (regionXYZ[midpoint] + regionXYZ[uppermostPart]) / 2
dists = np.sqrt(np.sum((regionXYZ[1:] - upperMidpoint)**2, 1))
np.nonzero((100 < dists) * (dists < 250))[0]+1




for i in range(regionCount-1):
	# if relativeDists[i] > 0:
	# if np.abs(relativeDists[i]) < np.mean(np.abs(relativeDists)):
	if np.abs(relativeDists[i]) > 20:
		color_ = 50
	else:
		color_ = 30
	pt1 = (regionLabels[i+1][2][1],regionLabels[i+1][2][0])
	cv2.circle(imLines, pt1, radius=0, color=color_, thickness=3)

imshow(imLines)






# edgeLists = [edgeDict[x] for x in edgeDict.keys()]
# edgeLists.insert(0,[])




## -----  Overlay shapes -----
im = posMat[:,:,2]
template = np.ones([10,50])*2-1
template -= nd.binary_erosion(template, iterations=2)
o = nd.convolve(im, template)

func = lambda x: np.max(np.diff(x))
grad = nd.generic_filter(im, func, size=(2,2)) # x-axis
grad *= grad < 100

func = lambda x: np.max(np.diff(x,2))
grad0 = nd.generic_filter(posMat[:,:,0], func, size=(2,2)) # x-axis
grad1 = nd.generic_filter(posMat[:,:,1], func, size=(2,2)) # x-axis
grad2 = nd.generic_filter(posMat[:,:,2], func, size=(2,2)) # x-axis
grad0 *= grad0 > 10
grad1 *= grad1 > 10
grad2 *= grad2 > 10

im -= im.mean()
im /= np.abs(im).max()

im[100:100+template.shape[0],10:10+template.shape[1]]*template*(im[100:100+template.shape[0],10:10+template.shape[1]]>0)
# templates = [template]


resultIm = np.zeros(np.array([im.shape[0]/6, im.shape[1]/7,18], dtype=int))
# resultIm = np.zeros(np.array([im.shape[0]/7, im.shape[1]/7,9], dtype=int))
for iya in range(22):
	iy = iya*5
	for ixa in range(16):
		templates = []
		ix = ixa*5
		for ang,i in zip(range(0,180, 10), range(18)):
			# print ang
			templates.append(nd.rotate(template, ang))
			t = im[iy:iy+templates[-1].shape[0],ix:ix+templates[-1].shape[1]]
			resultIm[iya,ixa,i] = (np.sum(np.abs(t - t.mean())*templates[-1]*(t>0))) / np.sum((t>0))
# interpolation='nearest'
figure(3); imshow(np.argmin(resultIm, 2), interpolation='nearest')

# im8bit[100:100+template.shape[0],10:10+template.shape[1]]*template*(im8bit[100:100+template.shape[0],10:10+template.shape[1]]>0)
conv = []
for i in range(len(templates)):
	conv.append(nd.convolve(im8bit[shoulderPx[0][0]-template.shape[1]/2:shoulderPx[0][0]+template.shape[1]/2-1,shoulderPx[0][1]-template.shape[1]/2:shoulderPx[0][1]+template.shape[1]/2-1,2], templates[i]))
conv = np.array(conv)

## --- Do PCA on each region
im = regions
imPCA = deepcopy(im)

for i in range(1, regions.max()):
	tmpPx = np.nonzero(regions==i)
	tmpXYZ = posMat[tmpPx[0],tmpPx[1],:]
	tmpXYZ -= tmpXYZ.mean(0)
	_,_,vT = svd(tmpXYZ, full_matrices=0)
	variance = np.array(var(np.asmatrix(tmpXYZ)*np.asmatrix(vT.T), 0))[0]
	v = vT[0,:]

	# figure(4); plot([0, v[0]], [0,v[1]])
	start = (int(tmpPx[1].mean()), int(tmpPx[0].mean()))
	end = (int(start[0]+v[1]*10), int(start[1]+v[0]*10))
	cv2.line(imPCA, start, end, (.0))
	cv2.ellipse(imPCA, start, (int(variance[0]/variance.max()*10), int(variance[1]/variance.max()*10)), np.arctan2(v[0], v[1])*180/3.1415, 360, 0.0, 1)


## --- Head regions
headSize = 150
headXYZ = regionXYZ[uppermostPart]
headPos = regionPos[uppermostPart]
dists = np.sqrt(np.sum((regionXYZ[1:] - regionXYZ[uppermostPart])**2, 1))
headRegions = np.nonzero(dists<headSize)[0]+1
head = np.zeros_like(regions)
classifyBody = np.zeros_like(regions)
for i in headRegions:
	head += regions==i
	classifyBody[regions==i] = 1

headPos = 
headPtsXYZ = posMat[(head>0)]


## --- RANSAC for torso

# Initialize with orientation from head to torso
torsoXYZ = regionXYZ[midpoint]
torsoPos = regionPos[midpoint]
headXYZ = regionXYZ[uppermostPart]
neckXYZ = (torsoXYZ+headXYZ)/2
neckPos = np.sum([regionPos[midpoint],regionPos[uppermostPart]], 0)/2
_,_,vT = svd(t-t.mean(0))
# initMean = neckXYZ
initMean = torsoXYZ
initSlope = vT[:,0]
xyz -= xyz.mean(0)

minError = 300
bestSlope = initSlope
bestRotMat = vT
bestSet = []
bestError = np.inf

for iIter in range(10):
	tmpInds = np.random.randint(0, len(xyz), 100)
	tmpPts = xyz[tmpInds]
	_,_,vT = svd(tmpPts-tmpPts.mean(0))
	tmpSlope = vT[:,0]

	# Calculate error
	relVec = (xyz - initMean)
	relMag = np.sqrt(np.sum(relVec**2, 1))
	projMag = np.sqrt(np.sum((tmpSlope*relVec*tmpSlope)**2, 1))
	error = np.sqrt(relMag**2-projMag**2)
	# print "-", tmpSlope

	consensusInds = np.nonzero(error < minError)[0]
	if len(consensusInds) > 500:
		_,_,vT = svd(xyz[consensusInds]-xyz[consensusInds].mean(0))
		newSlope = vT[:,0]
		relVec = (xyz[consensusInds] - initMean)
		relMag = np.sqrt(np.sum(relVec**2, 1))
		projMag = np.sqrt(np.sum((tmpSlope*relVec*tmpSlope)**2, 1))
		newError = np.sum(np.sqrt(relMag**2-projMag**2))

		if newError < bestError:
			bestError = newError
			bestSlope = newSlope
			bestRotMat = vT
			bestConsensus = consensusInds
			# print newError, newSlope

im = deepcopy(regions)
cv2.circle(im, (neckPos[1], neckPos[0]), 5, 30)
cv2.line(im, (neckPos[1], neckPos[0]), (int(neckPos[1]+bestSlope[0]*20), int(neckPos[0]+bestSlope[1]*20)), 30, 2)
imshow(im)

# look in box surrounding neck
rInds = np.nonzero(np.sqrt(np.sum((np.asarray((np.asmatrix(bestRotMat)*(regionXYZ[:,:]-neckXYZ[:]).T).T)[:,[1]]**2), 1)) < 100)[0]
regionXYZ[rInds,:]
regionPos[rInds,:]
# fit box with pivot at neck node


relVec = regionXYZ[1:] - initMean
relMag = np.sqrt(np.sum(relVec**2, 1))
projMag = np.sqrt(np.sum((bestSlope*relVec*bestSlope)**2, 1))
dists = np.sqrt(relMag**2-projMag**2)


bodyError = 150
bodyRegions = np.nonzero(dists<bodyError)[0]+1
body = np.zeros_like(regions)
for i in bodyRegions:
	body += regions==i
	classifyBody[regions==i] = 2
figure(5); imshow(classifyBody)


## Plot 3D
fig = figure(4)
ax = fig.add_subplot(111,projection='3d')
ax.scatter3D(headXYZ[0], headXYZ[2], zs=headXYZ[1])
ax.scatter3D(neckXYZ[0], neckXYZ[2], zs=neckXYZ[1])
ax.scatter3D(xyz[::50,0],xyz[::50,2],zs=xyz[::50,1])
ax.scatter3D(regionXYZ[:,0],regionXYZ[:,2],zs=regionXYZ[:,1], c=255*np.array(range(len(regionXYZ)), dtype=float)/len(regionXYZ))
# ax.scatter3D(x, y, zs=z, c=colorAxis)
i=2
# ax.plot3D([0, 30*vT[i,0]], [0,30*vT[i,2]], zs=[0,30*vT[i,1]], c=np.array([1,1]))
ax.plot3D([0, 100*vT[i,0]], [0,100*vT[i,2]], zs=[0,100*vT[i,1]])

bestSlope = bestRotMat[:,2]
ax.plot3D([neckXYZ[0], neckXYZ[0]+bestSlope[0]*100], [neckXYZ[2], neckXYZ[2]+bestSlope[2]*100], zs=[neckXYZ[1], neckXYZ[1]+bestSlope[1]*100])

xlabel('X')
ylabel('Y')
axis('equal')
draw()





if 1:
	# tmpXYZ = posMat[[(regions<9)*(regions>np.max(headRegions))]]
	tmpXYZ = posMat[(regions<=midpoint)*(regions>np.max(headRegions))]
	
	tmpXYZ -= tmpXYZ.mean(0)
	U,_,vT=svd((tmpXYZ-tmpXYZ.mean(0)), full_matrices=0)
	# vT = vT.T

	fig = figure(4)
	ax = fig.add_subplot(111,projection='3d')
	# ax.scatter3D(tmpXYZ[:,0],tmpXYZ[:,1],zs=tmpXYZ[:,2])
	for i in range(2,3):
		ax.plot3D([0, 200*vT[i,0]], [0,200*vT[i,1]], zs=[0,200*vT[i,2]])

	xlabel('X')
	ylabel('Y')
	axis('equal')
	draw()

if 1:
	tmpXYZ = xyz
	tmpXYZ -= tmpXYZ.mean(0)
	_,_,vT2=svd(tmpXYZ-tmpXYZ.mean(0))
	# vT = vT.T

	# fig = figure(4)
	# ax = fig.add_subplot(111,projection='3d')
	# ax.scatter3D(tmpXYZ[:,0],tmpXYZ[:,1],zs=tmpXYZ[:,2])
	# i=0
	for i in range(2,3):
		ax.plot3D([0, 200*vT2[i,0]], [0,200*vT2[i,1]], zs=[0,200*vT2[i,2]])

	xlabel('X')
	ylabel('Y')
	axis('equal')
	draw()

	angleDiff = np.arccos(np.dot(bodyAxis,shoulderAxis))*180/np.pi
	print "Angle diff:", angleDiff


figure(5)
min_ = 1.0*posMat[:,:,1][posMat[:,:,2]>0].min()
max_ = 1.0*posMat[:,:,1].max() - min_
scatter(xyz[:,0],xyz[:,2], c=1.0*np.array([(posMat[:,:,1][posMat[:,:,2]>0]-min_)*1.0/max_], dtype=float))
scatter(tmpXYZ[:,0],tmpXYZ[:,2], c='r')

i=2
plot([0, 200*vT[i,0]], [0,200*vT[i,2]], c='g')
plot([0, 200*vT2[i,0]], [0,200*vT2[i,2]], c='k')

figure(6)
tmpXYZ2 = (np.asmatrix(vT)*np.asmatrix(tmpXYZ.T)).T
tmpXYZ3 = (np.asmatrix(vT)*np.asmatrix(xyz.T)).T
tmpXYZ3 = (np.asmatrix(vT2.T)*np.asmatrix(xyz.T)).T
# tmpXYZ2 = tmpXYZ2.T
# plot(tmpXYZ2[:,0],tmpXYZ2[:,1], c='b')
plot(tmpXYZ2[:,0],tmpXYZ2[:,1], 'b.')
plot(tmpXYZ2[:,0],tmpXYZ2[:,2], 'g.')
plot(tmpXYZ2[:,1],tmpXYZ2[:,2], 'r.')
# scatter(tmpXYZ2[:,0:2], c='b')
# plot(tmpXYZ2[:,1],tmpXYZ2[:,2], c='b')

im2 = deepcopy(regions)
cv2.circle(im2, (headPos[1], headPos[0]), 10, 5) #Head
cv2.circle(im2, (neckPos[1], neckPos[0]), 5, 5) #Neck
cv2.circle(im2, (torsoPos[1], torsoPos[0]), 5, 5) #Torso
shoulders = [np.argmin(tmpXYZ2[:,0]),np.argmax(tmpXYZ2[:,0])]
shoulderPx = []
for i in shoulders:
 	shoulderPx.append([np.nonzero((posMat[:,:,2]*(regions<=midpoint)*(regions>np.max(headRegions)))>0)[0][i], np.nonzero((posMat[:,:,2]*(regions<=midpoint)*(regions>np.max(headRegions)))>0)[1][i]])
	# shoulderPx.append([np.nonzero(posMat[:,:,2]>0)[0][i], np.nonzero(posMat[:,:,2]>0)[1][i]]) 	
 	cv2.circle(im2, (shoulderPx[-1][1], shoulderPx[-1][0]), 5, 5)


# Neck:
figure(6)
# text(.5, .95, 'Upper body points', horizontalalignment='center') 
subplot(2,2,1); axis('equal'); xlabel('X'); ylabel('Y')
plot(-tmpXYZ2[:,0],-tmpXYZ2[:,1], 'b.')
subplot(2,2,2); axis('equal'); xlabel('Z'); ylabel('Y')
plot(-tmpXYZ2[:,2],-tmpXYZ2[:,1], 'b.')
subplot(2,2,3); axis('equal'); xlabel('X'); ylabel('Z')
plot(-tmpXYZ2[:,0],tmpXYZ2[:,2], 'b.')
# Whole body:
figure(7)
subplot(2,2,1); axis('equal'); xlabel('X'); ylabel('Y')
plot(-tmpXYZ3[:,0],-tmpXYZ3[:,1], 'b.')
subplot(2,2,2); axis('equal'); xlabel('Z'); ylabel('Y')
plot(-tmpXYZ3[:,2],-tmpXYZ3[:,1], 'b.')
subplot(2,2,3); axis('equal'); xlabel('X'); ylabel('Z')
plot(-tmpXYZ3[:,0],tmpXYZ3[:,2], 'b.')

