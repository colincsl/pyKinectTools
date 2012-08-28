import numpy as np
import scipy.misc
import os, sys
import scipy.ndimage as nd
from scikits.learn import neighbors


## Gabor filter
def generateGabors(angles, size=[20,20], rho=1):
	# angles in degrees
	if type(angles) != list:
		angles = [angles]
	width = size[0]
	height = size[1]

	xsbase = np.array(range(width)).T
	xs = np.array(xsbase, dtype=float)
	for i in range(height-1):
		xs = np.vstack([xs, xsbase])
	xs -= 1.0*width/2
	xs /= width

	ysbase = np.array(range(height)).T
	ys = np.array(ysbase, dtype=float)
	for i in range(width-1):
		ys = np.vstack([ys, ysbase])
	ys = ys.T
	ys -= 1.0*height/2
	ys /= height

	ux = 1.0/(2*rho)
	uy = 1.0/(2*rho)

	# Gaussian envelope
	gauss = np.exp(-0.5*(xs**2/rho**2 + ys**2/rho**2))
	gauss -= gauss.min() 
	gauss / gauss.max()

	if len(angles) > 1:
		gabors = np.empty([size[0], size[1], len(angles)])
	else:
		gabors = np.empty([size[0], size[1]])

	for a in range(len(angles)):
		theta = (angles[a])*np.pi/180
		s = np.cos(2*np.pi*(ux*(np.cos(theta)*xs+np.sin(theta)*ys) +uy*(np.sin(theta)*ys+np.cos(theta)*xs)))
		if len(angles) > 1:
			gabors[:,:,a] = s*gauss
		else:
			gabors[:,:] = s*gauss

	return gabors




def adaptiveNonMaximalSuppression(pts, vals, radius=1):
	# tree = neighbors.BallTree(pts.T)
 # 	nn = tree.query_radius(pts.T, radius)

	tree = neighbors.NearestNeighbors()
	tree.fit(pts)
	nn = tree.radius_neighbors(pts, radius, return_distance=False)

 	outputPts = []
 	for i in range(len(pts)):
 		
 		if vals[i] >= vals[nn[i]].max():
 			outputPts.append(pts[i])
 			# print vals[i],vals[nn[i]].max(), nn[i], i

 	outputPts = np.array(outputPts)
 	return outputPts





