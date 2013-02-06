import numpy as np
from scipy import sqrt, pi, arctan2, cos, sin
from scipy.ndimage import uniform_filter
import scipy.ndimage as nd
import scipy.misc as sm
import skimage
import skimage.transform

im0 = imread('color_304_21_38_46_6097.jpg').mean(-1).astype(np.uint8)
im1 = imread('color_304_21_38_46_6480.jpg').mean(-1)

im0 = imread('/media/Data/icu_test_current/color/12/16/56/device_1/color_12_16_56_54_11_66.jpg').mean(-1).astype(np.uint8)
im1 = imread('/media/Data/icu_test_current/color/12/16/56/device_1/color_12_16_56_54_12_71.jpg').mean(-1)



'''
Farneback technique for dense optical flow

Based on the paper "Two-Frame Motion Estimation Based on Polynomial Expansion"


http://lsvn.lysator.liu.se/svnroot/spatial_domain_toolbox/trunk/

Displacement Estimation

f1 is starting pixel value. f2 is new pixel value. d is the displacement 
f2(x) = f1(x-d) = (x-d)' A0 (x-d) + b0' (x-d) + c1
	  = x'A0x + (b0 - 2*A0*d)'x + d'A0d - b0'd + c1
	  = x'A1x + b1'x + c2
Thus:
A1 = A0
b1 = b0 - 2*A0*d
c2 = d'*A0*d - b'T*d + c1

Reformat the second equation if A0 is non-singular
2*A0d = b0 + b1
d = 1/2 inv(A0)*(b0-b1)

A = 0.5 * (A0 + A1)
del_b = 0.5*(b0 - b1)
Constraint: A*d = del_b

Using least squares we minimize:
	sum_{del_x in Neigh} w(del_x) * ||A(x+del_x)*d(x) - del_b(x+del_x)||^2
And get:
	d(x) = inv(sum w*A'A) * (sum w*A'*del_b)
min = (sum w*del_b'*del_b) - d(x)'*sum(w*A'*del_b)
'''

'''
Motion model:
dx(x,y) = a0 + a1*x + a3*y + a7*x^2 + a8*x*y
dy(x,y) = a4 + a5*x + a6*y + a7*x*y + a8*y^2


Thus our least squares becomes:
min sum_i wi || Ai*Si*p - del_bi||^2
p = inv(sum_i wi*Si'*Ai'*Ai*S')*(sum_i si*Si'*Ai'*del_bi)
'''

'''
r = 
(x,y) A (x,y)' + b'(x,y)' = r1 + r2*x + r3*y + r4*x^2 + r5*y^2 +r6*x*y

'''

def DisplacementEstimation(displacement, A0, b0, A1, b1, neighbors=9):
	'''
	---Steps---
	1) Find average A and delta b for each index
	2) 

	x is the pixel index
	x_new is the new estimated pixel index
	x_new = x + displacement(x)

	8-parameter motion model:
	displacement = S*p
	S = [1, x, y, 0, 0, 0, x^2, xy;
		 0, 0, 0, 1, x, y, xy, y^2]
	p = (A0, A1, a3, a4, a5, a6, a7, a8)'
	'''

	dims = displacement.shape
	neigh = np.floor(np.sqrt(neighbors)/2).astype(np.int)

	''' Find indices based on disparity '''
	tmpInds = np.mgrid[:dims[0], :dims[1]]
	x_new = np.array([tmpInds[0].ravel() + displacement[:,:,1].ravel(), 
					tmpInds[1].ravel() + displacement[:,:,0].ravel()]).astype(np.int).T

	x_new[:,0] = np.maximum(np.minimum(x_new[:,0], dims[0]-1), 0)
	x_new[:,1] = np.maximum(np.minimum(x_new[:,1], dims[1]-1), 0)

	''' Pre-cache S matrix '''
	S = np.empty([2,8, neighbors], dtype=np.float)
	x, y = np.mgrid[-neigh:neigh+1, -neigh:neigh+1]
	i=0
	for xi, yi in zip(x.ravel(), y.ravel()):
		S[:,:,i] = np.array([[1, xi, yi, 0, 0, 0, xi**2, xi*yi],
							[0, 0, 0, 1, xi, yi, xi*yi, yi**2]])
		i += 1

	''' Pre-cache G, h '''
	G = np.zeros([dims[0], dims[1], 8, 8])
	h = np.zeros([dims[0], dims[1], 8])
	# A = 0.5 * (A0.reshape([dims[0]*dims[1], 2,2])+A1[x_new[:,0],x_new[:,1]])
	A = 0.5 * (A0+A1).reshape([dims[0]*dims[1], 2, 2])

	for i in range(dims[0]):
		for j in range(dims[1]):
			ind = i*dims[1] + j
			del_b = 0.5*(b0[i,j]-b1[i,j]) + np.dot(A[ind], displacement[x_new[ind,0],x_new[ind,1]])

			# displacement[i,j] = .5 * np.linalg.solve(A,del_b)

			Si = 0
			for xi, yi in zip(x.ravel(), y.ravel()):
				tmp = np.dot(S[:,:,Si].T, A[ind].T)
				G[i,j] += np.dot(tmp, tmp.T)
				h[i,j] += np.dot(tmp, del_b.T)
				Si += 1
			
	'''Spatial averaging component-wise'''
	G_avg = G / float(neighbors)
	h_avg = h / float(neighbors)

	''' Calculate disparity '''
	for i in range(dims[0]):
		for j in range(dims[1]):

			try:
				displacement[i,j] = np.dot(S[:,:,0], np.linalg.solve(G_avg[i,j],h_avg[i,j]))
				# print np.linalg.solve(G_avg[i,j],h_avg[i,j])
			except:
				print 'Error in disparity'
				displacement[i,j] = np.array([0,0])

	return displacement


def imnorm(im):
	return (im - im.min()) / (im.max() - im.min())

def OpticalFlow(im0, im1, levels=3, pyramid_scale=2., iterations=1, smoothness_sigma=1.1):
	'''
	---Parameters---
	im0 : first image frame
	im1 : second image frame
	levels : number of layers in the pyramid
	pyramid_scale : ratio of the size of layer N and N-1 in the pyramid
	smoothness_sigma : the value of the smoothing kernel

	---Steps---
	0) Intialize displacement/flow with zero vectors
	1) Create smoothed (N layer) pyramid
	For each pyramid (top to bottom):
		2) Apply polynomial expansion to each layer of each image
		3) Estimate displacement
		4) Feed updated displacement to next layer

	See figure 7.10 in Farneback's PhD thesis for an overview.
	'''

	''' 
	Step 1: Get image pyramids
	'''
	# im0 = imnorm(im0)
	# im1 = imnorm(im1)
	# im0 = skimage.transform.image_as_float(im0)

	# im0_pyr_generator = skimage.transform.pyramid_gaussian(im0, sigma=1.0, downscale=pyramid_scale)
	# im1_pyr_generator = skimage.transform.pyramid_gaussian(im1, sigma=1.0, downscale=pyramid_scale)	
	im0_pyr = []
	im1_pyr = []
	shape_ = im0.shape
	''' Use scipy.misc.imresize instead!?  error in skimage.resize?? '''
	for i in range(levels):
		# tmp0 = im0_pyr_generator.next()
		# tmp1 = im1_pyr_generator.next()
		# im0_pyr.append((tmp0))
		# im1_pyr.append((tmp1))

		shape_ = [shape_[0]/2, shape_[1]/2]
		t = sm.imresize(im0, shape_, interp='nearest')
		t1 = sm.imresize(im1, shape_, interp='nearest')
		t = nd.gaussian_filter(t,2)
		t1 = nd.gaussian_filter(t1,2)
		im0_pyr.append(t)
		im1_pyr.append(t1)		

	''' 
	Go through pyramid top to bottom
	'''
	''' Step 0: Intitialize displacement '''
	displacement = np.zeros([im0_pyr[-1].shape[0], im0_pyr[-1].shape[1], 2], dtype=np.float)

	
	for i in range(levels-1, -1, -1):
		''' Step 4) Feed updated displacement to next layer'''
		displacement = np.dstack([sm.imresize(displacement[:,:,0], im0_pyr[i].shape),
								sm.imresize(displacement[:,:,1], im0_pyr[i].shape)]).astype(np.float)		
		for k in range(iterations):
			''' Step 2) Apply polynomial expansion to each layer'''
			A0, b0, c0 = PolynomialExpansion(im0_pyr[i], spatial_size=25, sigma=.05)
			A1, b1, c1 = PolynomialExpansion(im1_pyr[i], spatial_size=25, sigma=.05)
			''' Step 3) Estimate displacement'''
			displacement = DisplacementEstimation(displacement, A0, b0, A1, b1)

		figure(i)
		imshow(displacement[:,:,0])


	return displacement		    


if 0:
	figure(0)
	imshow(A0[:,:,0,0]); clim([-11, 11])
	figure('0b')
	imshow(A1[:,:,0,0]); clim([-11, 11])
	figure('A combined')
	imshow(A.reshape([dims[0], dims[1],2,2])[:,:,0,0]); clim([-11, 11])

	figure(1)
	imshow(A0[:,:,0,1])

	figure(4)
	imshow(b0[:,:,0])
	figure(5)
	imshow(b0[:,:,1])

	figure(6)
	imshow(c0)




def PolynomialExpansion(im, spatial_size=9, sigma=0.15):
	'''
	---Parameters---
	sigma : height of gaussian kernel

	This model assumes a locally quadratic model:
	f(x) = x'Ax + b'x + c
	Perhaps more intuitively it can be written as:
	f(i2,j2) = (i2-i, j2-j)' A (i2-i, j2-j) + b' (i2-i, j2-j) + f(i,j)

	This means that b and A are comparable to the gradient and hessian respectively and c is the value at x

	In the original formulation they use "certainty" as a measure of confidence in the signal. In practice Farneback sets this to all ones so we just remove it from the equations.
	The term applicability is used for a gaussian kernel

	'''
	sigma = sigma * (spatial_size - 1)

	n = (spatial_size-1)/2
	a = np.exp(-(np.arange(-n,n+1))**2/(2*(sigma**2))).T;

	applicability = np.outer(a, a)
	x, y = np.mgrid[-n:n+1, -n:n+1]
	b = np.dstack([np.ones(x.shape), x, y, x**2, y**2, x*y])
	nb = b.shape[2]

	''' Compute gaussian metric '''
	Q = np.zeros([nb, nb])
	for i in range(nb):
		for j in range(nb):
			if Q[i,j] == 0:
				Q[i,j] = np.sum(b[:,:,i] * b[:,:,j] * applicability)
				Q[j,i] = Q[i,j]

	Qinv = np.linalg.inv(Q)

	''' Convolutions '''
	kernel_0 = a
	kernel_1 = np.arange(-n, n+1) * a.T
	kernel_2 = np.arange(-n, n+1)**2 * a.T

	conv_y0 = nd.convolve1d(im, kernel_0)
	conv_y1 = nd.convolve1d(im, kernel_1)
	conv_y2 = nd.convolve1d(im, kernel_2)

	conv_results = np.zeros([im.shape[0], im.shape[1], 6])
	conv_results[:,:,0] = nd.convolve1d(conv_y0, kernel_0)
	conv_results[:,:,1] = nd.convolve1d(conv_y0, kernel_1)
	conv_results[:,:,3] = nd.convolve1d(conv_y0, kernel_2)
	conv_results[:,:,2] = nd.convolve1d(conv_y1, kernel_0)
	conv_results[:,:,5] = nd.convolve1d(conv_y1, kernel_1)
	conv_results[:,:,4] = nd.convolve1d(conv_y2, kernel_0)

	tmp = Qinv[0,0]*conv_results[:,:,0] + Qinv[0,3]*conv_results[:,:,3] + Qinv[0,5]*conv_results[:,:,5]

	''' Apply the inverse metric.'''
	conv_results[:,:,0] = tmp
	conv_results[:,:,1] *= Qinv[1,1]
	conv_results[:,:,2] *= Qinv[2,2]
	conv_results[:,:,3] = Qinv[3,3]*conv_results[:,:,3] + Qinv[3,0]*conv_results[:,:,0]
	conv_results[:,:,4] = Qinv[4,4]*conv_results[:,:,4] + Qinv[4,0]*conv_results[:,:,0]
	conv_results[:,:,5] *= Qinv[5,5]

	''' 
	Build A,b,c
	r is the convolution result.

	c = [r0]
	b = [r1,r2]'	
	A = [r3, r5/2]
		[r5/2, r4]
	'''

	c = conv_results[:,:,0]

	b = np.zeros([im.shape[0], im.shape[1], 2])
	b[:,:,0] = conv_results[:,:,1]
	b[:,:,1] = conv_results[:,:,2]

	A = np.zeros([im.shape[0], im.shape[1], 2, 2])
	A[:,:,0,0] = conv_results[:,:,3]
	A[:,:,1,1] = conv_results[:,:,4]
	A[:,:,0,1] = conv_results[:,:,5] / 2
	A[:,:,1,0] = conv_results[:,:,5] / 2

	return A, b, c


