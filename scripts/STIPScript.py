import numpy as np
import scipy.misc
import os, sys
import scipy.ndimage as nd
from pyKinectTools.algs import backgroundSubtract
from pyKinectTools.algs.STIP import *
import time


angles = range(0, 180, 45)
# angles = range(0, 108*2, 90/5)
gabors = generateGabors(angles, [20,20], 1)


''' --- Clouds of STIP --- '''

i_time = 100

O_ratio = []
O_speed = []
C_interests = []

# for i_time in range(100,200,10):
for i_time in range(0,len(imgs),5):
	''' Object features '''
	tSpan = 5
	#T_new
	labeledImg, slices, sliceIndices = extractPeople(imgs[i_time])
	labeledImg2, slices2, sliceIndices2 = extractPeople(imgs[i_time-tSpan])
	t1 = time.time()
	S_interests = []
	if len(sliceIndices)>0 and len(sliceIndices2)>0:
		O_new_height = slices[0][0].stop-slices[0][0].start
		O_new_width = slices[0][1].stop-slices[0][1].start
		O_new_ratio = 1.0*O_new_height/O_new_width
		O_new_pos = np.array([slices[0][0].start+O_new_height/2, slices[0][1].start+O_new_width/2, imgs[i_time][slices2[0]].mean()])
		#T_prev
		height = slices2[0][0].stop-slices2[0][0].start
		width = slices2[0][1].stop-slices2[0][1].start
		O_old_ratio = 1.0*height/width
		O_old_pos = np.array([slices2[0][0].start+height/2, slices2[0][1].start+width/2, imgs[i_time-tSpan][slices2[0]].mean()])

		O_ratio.append((O_old_ratio+O_new_ratio)/2)
		O_speed.append(np.sqrt(np.sum((O_new_pos-O_old_pos)**2))/tSpan)


		''' Temporal features '''
		for t in range(0,20,8):
			tSpan = t
			imTmp = imgs[i_time]-imgs[i_time-tSpan]
			imTmp = imTmp*(imTmp<200)#*(imTmp>100)
			imTmp = nd.binary_erosion(imTmp, iterations=5)*(imTmp)
			imTmp = np.asarray(imTmp, dtype=float)
			convs = []
			for i in range(len(angles)):
				convs.append(nd.convolve(imTmp, gabors[:,:,i]))
			convs = np.array(convs)
			fused = convs.max(0)*(imTmp>0)
			
			interests = np.nonzero(fused>0)
			# interests = np.nonzero(fused>.5*fused.max())			
			# interests = localMaximums(fused)
			pts = np.array(interests).T
			vals = fused[pts[:,0],pts[:,1]]
			interests = adaptiveNonMaximalSuppression(pts, vals)
			# fused2 = (deepcopy(fused)>0)*1.0
			# fused2[interests[:,0], interests[:,1]] = fused2.max()*10
			if len(interests) > 0:
				s_height = interests[0].max() - interests[0].min()
				s_width = interests[1].max() - interests[1].min()

				s_ratio = float(s_height)/s_width
				s_pos = [interests[0].mean(), interests[1].mean()]#, imgs[i_time][interests[0].mean(), interests[1].mean()]]
				s_density = float(len(interests[0])) / (s_height*s_width)

				# find speed not pos!
				# overlap = O_new_pos[0]+O_new_height/2 - 
				# overlap = np.sum(fused>0 == imgs[i_time-tSpan]>0)

				S_interests.append([s_height, s_width, s_pos[0],s_pos[1], s_density, s_pos[0]-O_new_pos[0], s_pos[1]-O_new_pos[1], float(s_height)/O_new_height, float(s_width)/O_new_width])

		print "Next frame:", time.time()-t1
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




