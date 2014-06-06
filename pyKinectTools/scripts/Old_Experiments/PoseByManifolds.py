
from pyKinectTools.algs.manifolds import *


# xyz = featureExt1.xyz[ind]
xyzInds = np.nonzero(posMat[:,:,2]!=0)
xyz = posMat[xyzInds[0], xyzInds[1]]
xyz -= xyz.mean(0)

ptInterval=1
data = xyz[::ptInterval,:]
# data = xyz[::ptInterval,:][100::]
posVecs = LaplacianEigenmaps(data)

x= posVecs[:,0]; y= posVecs[:,1]; z= posVecs[:,2]
x2= posVecs[:,4]; y2= posVecs[:,5]; z2= posVecs[:,5]


''' Color segments '''
maxDim = 4
colorAxis = np.zeros_like(y)+maxDim
for i in range(1,maxDim):
	colorAxis[posVecs[:,i]>0] = np.minimum(colorAxis[posVecs[:,i]>0], i)
colorAxis *= 255.0/colorAxis.max()



''' Visualize '''
# Viz components 1-3
if 0:
	fig = figure(9)
	ax = fig.add_subplot(111,projection='3d')
	ax.scatter3D(x, y, zs=z, c=colorAxis)
	xlabel('X')
	ylabel('Y')
	# axis('equal')

# Viz components 4-6
if 0:
	fig = figure(10)
	ax = fig.add_subplot(111,projection='3d')
	ax.scatter3D(x2, y2, zs=z2, c=colorAxis)
	xlabel('X')
	ylabel('Y')		

# Viz colors on original structure
if 0:
	fig = figure(12)		
	ax = fig.add_subplot(111,projection='3d')
	ax.scatter3D(data[:,0], data[:,1], zs=data[:,2], c=colorAxis)
	xlabel('X')
	ylabel('Y')
	axis('equal')
	draw()

# Viz components on single axis
if 1:
	fig = figure(13)
	cla()
	for i in range(1,4):
		plot(posVecs[:,i])

# Paint original image
if 1:
	figure(14)
	colorMat = np.zeros_like(posMat)
	inds = xyzInds
	colorMat[inds[0][::ptInterval][::],inds[1][::ptInterval][::],0] = colorAxis
	imshow(colorMat[:,:,0])

