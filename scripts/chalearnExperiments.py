
import scipy.io as io
import cv2
from copy import deepcopy

''' Load Data '''
import pyKinectTools.data as ktData
dataFolder = ktData.__file__
dataFolder = dataFolder[:dataFolder.find("__init__")]
dep = io.loadmat(dataFolder+'bodyPart_DepthImgs.mat')
feat = io.loadmat(dataFolder+'bodyPart_HOGFeatures.mat')


'''Play with image'''
# im = dep['l_hand'][0][0]
## Sphere test
figure(1)
interval = 10
tIms = []
for s in range(10,70, 10):
	tSize = [s,s]
	template = np.zeros(tSize)
	cv2.circle(template, (s/2,s/2), 0, 1, s)
	# cv2.ellipse(template, center=(s/2,s/2), axes=(s/2,s/4), angle=0, startAngle=0, endAngle=360, color=1, thickness=s/4)
	im2 = np.array(deepcopy(im), dtype=np.float)
	im2 = np.maximum(im2[1:,1:] - im2[:-1,1:],im2[1:,1:] - im2[1:,:-1])
	# im2[np.abs(im2) > 20] = 0
	im2[im2!=0] -= im2.min()
	im2 /= im2.max()
	# im2[im2!=0] = np.log(im2[im2!=0])

	# imOut = np.zeros([240,320])
	imOut = np.zeros([(240-tSize[0])/interval, (320-tSize[1])/interval])
	# imOut = np.zeros([(240-100)/interval, (320-100)/interval])
	for i,ii in zip(range(tSize[0]/2,240-tSize[0]/2, interval), range((240-tSize[0])/interval)):
		for j,jj in zip(range(tSize[1]/2,320-tSize[1]/2,interval), range((320-tSize[0])/interval)):
			# p = [60,60]
			p = [i,j]
			imOut[ii,jj] = np.sum(im2[p[0]-tSize[0]/2:p[0]+tSize[0]/2, p[1]-tSize[1]/2:p[1]+tSize[1]/2, 2] * template)

	subplot(3,3,s/10)
	title(s)
	imshow(imOut)
	tIms.append(deepcopy(imOut))
figure(2); imshow(im2)
figure(3); imshow(im)
figure(4); imshow(tIms[5]/tIms[5].max()-cv2.resize(tIms[2], (tIms[5].shape[1],tIms[5].shape[0]))/tIms[2].max())
# figure(4)


