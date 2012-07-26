import numpy as np
import scipy.misc
import os, sys
import ndimage as nd

from pyKinectTools.algs.STIP import generateGabors

# angles = range(0, 108, 90/5)
angles = range(0, 108*2, 90/5)
gabors = generateGabors(angles, [10,10], 1)


''' --- Clouds of STIP --- '''

time = 100

O_ratio = []
O_speed = []
C_interests = []

for time in range(100,200,10):
	''' Object features '''
	tSpan = 5
	#T_new
	labeledImg, slices, sliceIndices = extractPeople_2(imgs[time])
	O_new_height = slices[0][0].stop-slices[0][0].start
	O_new_width = slices[0][1].stop-slices[0][1].start
	O_new_ratio = 1.0*O_new_height/O_new_width
	O_new_pos = np.array([slices[0][0].start+height/2, slices[0][1].start+width/2, imgs[time][slices[0]].mean()])
	#T_prev
	labeledImg, slices, sliceIndices = extractPeople_2(imgs[time-tSpan])
	height = slices[0][0].stop-slices[0][0].start
	width = slices[0][1].stop-slices[0][1].start
	O_old_ratio = 1.0*height/width
	O_old_pos = np.array([slices[0][0].start+height/2, slices[0][1].start+width/2, imgs[time-tSpan][slices[0]].mean()])

	O_ratio.append((O_old_ratio+O_new_ratio)/2)
	O_speed.append(np.sqrt(np.sum((O_new_pos-O_old_pos)**2))/tSpan)


	''' Temporal features '''
	
	S_interests = []
	for t in range(0,20,4):
		tSpan = t
		imTmp = imgs[time]-imgs[time-tSpan]
		imTmp = imTmp*(imTmp<200)#*(imTmp>100)
		imTmp = nd.binary_erosion(imTmp, iterations=5)*(imTmp)
		convs = []
		for i in range(len(angles)):
			convs.append(nd.convolve(imTmp, gabors[:,:,i]))
		convs = np.array(convs)
		fused = convs.max(0)*imTmp
		interests = np.nonzero(fused>200)
		if len(interests[0]) > 0:
			s_height = interests[0].max() - interests[0].min()
			s_width = interests[1].max() - interests[1].min()

			s_ratio = 1.0*s_height/s_width
			s_pos = [interests[0].mean(), interests[1].mean()]#, imgs[time][interests[0].mean(), interests[1].mean()]]
			s_density = 1.0*len(interests[0]) / (s_height*s_width)

			# find speed not pos!
			overlap = O_new_pos[0]+O_new_height/2 - 
			S_interests.append([s_height, s_width, s_pos, s_density, s_pos[0]-O_new_pos[0], s_pos[1]-O_new_pos[1], 1.0*s_height/O_new_height, 1.0*s_width/O_new_width])



	C_interests.append(S_interests)
imshow(fused>200)


if 0:
	###
	# angles = range(0, 108, 90/5)
	ii=10

	imTmp = imgs[ii][70::,0:300]
	# imTmp[imTmp>0].min()
	mask = nd.binary_erosion(imTmp>0, iterations=5)
	angles = range(0, 180, 45)
	gabors = generateGabors(angles, [20,20], 1)
	convs = []
	for i in range(len(angles)):
		convs.append(nd.convolve(imTmp, gabors[:,:,i]))
	convs = np.array(convs)
	imshow(convs.max(0)*mask)
	ii+=5




