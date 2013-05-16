
# Patient movement

import cv, cv2
import scipy.ndimage as nd
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy import interpolate

''' #------------ Spline-based ---------------# '''

for fi in range(0,190, 5):
	# fi=114
	fi += 5
	im = imgs[fi]
	mask = (im<90)*(im>10)
	mask = nd.binary_opening(mask)
	im *= mask


	posMat = posImage2XYZ(im, 500, 2000) 
	for i in range(3):
		posMat[:,:,i] *= mask
	inds = np.nonzero(mask)
	xyz = np.vstack([inds[0], inds[1], posMat[inds[0],inds[1],2]]).T

	xyzMean = xyz.mean(0)
	xyz -= xyzMean
	# Get top and bottom
	U,_,vT = np.linalg.svd(xyz, full_matrices=False)
	tmp = np.asmatrix(vT)*np.asmatrix(xyz.T)
	tmp = np.asarray(tmp.T)

	nodeTail = posMat[280, 55]
	nodeHead = posMat[360,460]#[235, 625]
	# nodeTail = np.asarray(np.asmatrix(vT)*np.asmatrix(nodeTail).T).T[0]
	# nodeHead = np.asarray(np.asmatrix(vT)*np.asmatrix(nodeHead).T).T[0]

	n_est=5
	# tckp,u = interpolate.splprep(xyz[::100,:].T, s=3,k=1,nest=n_est, ue=nodeTail,ub=nodeHead)
	tckp,u = interpolate.splprep(xyz[::100,:].T, s=3,k=1,nest=n_est)
	xn,yn,zn=interpolate.splev(linspace(0,1,100),tckp)
	figure(3)
	subplot(2,2,1); axis('equal'); cla(); 
	plot(xn,yn, c=[fi/200.0, 0, 0])
	subplot(2,2,2); axis('equal'); cla(); 
	plot(xn,zn, c=[fi/200.0, 0, 0])
	subplot(2,2,3); axis('equal'); cla(); 
	plot(yn,zn, c=[fi/200.0, 0, 0])

	figure(2)
	imshow(im)

# spl = interpolate.splrep(xyz[:,0], xyz[:,1], xb=nodeHead[:2], xe=nodeTail[:2], s=0)

fig = figure(1);
ax = Axes3D(fig)
xlabel('X'); ylabel('Y'); axis('equal')
ax.plot(xn,yn,zn, c='g')
# ax.scatter(tmp[::100,0],tmp[::100,1],tmp[::100,2])

ax.scatter(xyz[::50,0],xyz[::50,1],xyz[::50,2])

figure(2)
imshow(posMat[:,:,2])

# calculate 3D flow

# cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevPts[, nextPts[, status[, err[, winSize[, maxLevel[, criteria[, flags[, minEigThreshold]]]]]]]]) → nextPts, status, err



''' #------------ FLOW ---------------# '''

cv2.namedWindow("flow")
cv2.namedWindow("im")

totalMag = np.zeros_like(imgs[0], dtype=float)
indsStart = np.nonzero(totalMag==0)
magSums = []
for i in range(5,len(imgs), 1):
# for i in range(150,len(imgs)):
	i2 = i-1
	im1 = imgs[i] 
	mask = (im1<150)*(im1>10)
	mask = nd.binary_opening(mask)
	# im1 *= mask

	im2 = imgs[i2]
	mask = (im2<150)*(im2>10)
	mask = nd.binary_opening(mask)
	# im2 *= mask

	# im2 *= (im2 < 90) * (im2 > 0)
	# poly_n is the pixel neighborhood. 5 or 7 suggested. for 5, use poly_sigma=1.1, for 7 use ps=1.5
	flow = cv2.calcOpticalFlowFarneback(im1, im2, pyr_scale=.5, levels=3, winsize=9, iterations=3, poly_n=5, poly_sigma=1.1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
	mag = np.sqrt(flow[:,:,0]**2+flow[:,:,1]**2)
	mag *= (mag>10)

	# indsEnd_x = flow[:,:,0].flatten()
	# indsEnd_y = flow[:,:,1].flatten()

	totalMag += mag
	magSums.append(mag.sum())
	if magSums[-1] > 200000:
		print magSums[-1]
	# im = totalMag
	im = mag
	cv2.imshow("im", im1)
	cv2.imshow("flow", im/im.max())
	ret = cv2.waitKey(30)
	if ret >= 0:
		break

print totalMag.sum()

imshow(totalMag)


''' #------------ gabor ---------------# '''
from pyKinectTools.algs.STIP import generateGabors

im = im1
mask = (im<90)*(im>10)
angles = range(0, 180, 45)
# gabors = generateGabors(angles, [20,20], 1)
gabors = generateGabors(angles, [30,30], 1)
convs = []
for i in range(len(angles)):
	convs.append(nd.convolve(im, gabors[:,:,i]))
convs = np.array(convs)
imOut = convs.max(0)#*mask
grad = np.gradient(np.asarray(imOut, dtype=float), 1)
mag = np.sqrt(grad[0]**2+grad[1]**2)
mag *= mask

mask *= (mag<.05*mag.max())
mask = nd.binary_opening(mask, iterations=2)
imshow(im*mask)
# imshow(imOut)


