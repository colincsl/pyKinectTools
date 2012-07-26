
# xyz = featureExt1.xyz[ind]
xyzInds = np.nonzero(posMat[:,:,2]!=0)
xyz = posMat[xyzInds[0], xyzInds[1]]
xyz -= xyz.mean(0)

def LaplacianEigenmaps(data, numNeigh=10):
	''' W is weight/distance, D is diagonal, L is the Laplacian '''
	W = distance.squareform(distance.pdist(data))
	inds = np.argsort(W, 1)
	# W[inds == 0 ] = 0
	W[inds > numNeigh] = 0
	W = inds <= numNeigh
	W[W>0] = 1
	W = np.maximum(W, W.T)

	diag = np.diag(np.sum(W,1))

	# W = W**2
	# W /= np.max(np.max(W))
	# W = np.maximum(W, W.T)
	# sigma = 1.0
	# kernel[kernel!=0] = np.exp(-W[kernel!=0] / (2*sigma**2))
	# kernel = W
	# diag = np.diag(np.sum(kernel,1))

	#Calc Laplacian
	L = diag-W
	vals, vecs = eigh(L)
	# Only keep positive eigenvals
	posInds = np.nonzero(vals>0)[0]
	posVecs = vecs[:,posInds]

	return posVecs

ptInterval=10
data = xyz[::ptInterval,:][100::]
posVecs = LaplacianEigenmaps(data)
	
x= posVecs[:,1]; y= posVecs[:,2]; z= posVecs[:,3]
x2= posVecs[:,4]; y2= posVecs[:,5]; z2= posVecs[:,6]


''' Color segments '''
maxDim = 2
colorAxis = np.zeros_like(y)+maxDim
for i in range(1,maxDim):
	colorAxis[posVecs[:,i]>0] = np.minimum(colorAxis[posVecs[:,i]>0], i)
colorAxis *= 255/colorAxis.max()



''' Visualize '''
# Viz components 1-3
if 1:
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
if 1:
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
	colorMat = np.zeros_like(posMat)
	inds = xyzInds
	colorMat[inds[0][::ptInterval][100::],inds[1][::ptInterval][100::],0] = colorAxis
	figure(14); imshow(colorMat[:,:,0])

