''' 
Implements the Laplacian Eigenmap manifold technique

Colin Lea
pyKinectTools
2012
'''
import numpy as np
from sklearn import neighbors
# import scikits.learn as sklearn
from sklearn.metrics import *
from scipy.spatial import distance

def LaplacianEigenmaps(data, numNeigh=5, heatKernel=False, heatSigma=1.0):
	''' W is weight/distance/kernel, D is diagonal, L is the Laplacian '''

	if 1:
		W = distance.squareform(distance.pdist(data))
		inds = np.argsort(W, 1)
		W[inds > numNeigh] = 0
		W = inds <= numNeigh

	if 0:
		W = np.zeros([len(data), len(data)])
		Ball = neighbors.BallTree(data)
		dists, nn = Ball.query(data, numNeigh)

		k=numNeigh
		for di in range(len(dists)):
			for ki in range(k):
				# W[di, nn[di,ki]] = dists[di,ki]
				W[di, nn[di,ki]] = 1.0#dists[di,ki]




	if not heatKernel:
		# Binary representation
		W = np.maximum(W, W.T)
	else:
		# Heat kernel based on distances. Ranges between 0-1
		W = W**2
		W /= np.max(np.max(W))
		W = np.maximum(W, W.T)
		W[W!=0] = np.exp(-W[W!=0] / (2*heatSigma**2))

	diag_ = np.diag(np.sum(W,1))

	#Calc Laplacian
	L = diag_-W
	vals, vecs = np.linalg.eigh(L)
	# Only keep positive eigenvals
	posInds = np.nonzero(vals>0)[0]
	posVecs = vecs[:,posInds]

	return posVecs









