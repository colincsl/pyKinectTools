
import numpy as np
import cv2
import scipy.misc as sm

from pyKinectTools.algs.IterativeClosestPoint import IterativeClosestPoint, PointcloudRegistration
from pyKinectTools.utils.DepthUtils import world2depth

# from IPython import embed
# embed()


def pts2coords(pts, rez, centers=[300, 0, -500], span=np.ones(3, dtype=np.float)*6000):
	xs = np.minimum(np.maximum(rez[0]+((pts[:,2]+centers[2])/span[2]*rez[0]).astype(np.int), 0),rez[0]-1)
	ys = np.minimum(np.maximum(((pts[:,0]+centers[0])/span[0]*rez[1]).astype(np.int), 0),rez[1]-1)

	return xs, ys

def coords2pts(coords, rez, span=np.ones(3, dtype=np.float)*6000):
	xs = (coords[:,2] - rez[0])/rez[0]*span[2] - centers[2]
	ys = coords[:,0]/rez[1]*span[0] - centers[0]

	return np.hstack([xs, ys])



def map_points(pts, mapIm=None, mapRez=None, color=255, centers=[300, 0, -500], span=np.ones(3, dtype=np.float)*6000):
	
	color = np.array(color)
				
	if mapRez is None:
		if mapIm is None:
			mapRez = [200,200, 3]
		else:
			mapRez = mapIm.shape

	if mapIm is None:
		mapIm = 0*np.ones(mapRez)*255

	''' Convert com to xy indcies '''
	xs, ys = pts2coords(pts, mapRez, centers, span)
  	mapIm[xs, ys,:] = color

	return mapIm
	
def list2time(time):
	''' Multiple time by 100 to account for sequential frames/second (which can be up to ~15/sec)'''
	time = [int(x) for x in time]
	return time[0]*(60*60*24*10) + time[1]*(60*60*10) + time[2]*(60*10)+ time[3]*10 + time[4]
	
def plotCamera(cameraPos, im, centers=[3000,0,-500], span=np.ones(3, dtype=np.float)*6000):
	rez = im.shape
	# x = -((cameraPos[2]+centers[2])/span[2]*rez[0]).astype(np.int)
	# y = (rez[0]-(cameraPos[0]+centers[0])/span[0]*rez[1]).astype(np.int)       
	x,y = pts2coords(np.array([cameraPos]), rez, centers, span) 
	cv2.circle(im, (y,x), 5, (255,255,255), thickness=-1)

def viewImage(cam_index, filetime, coms=None):
	
	URIs = ['/media/Data/ICU_Dec2012/ICU_Dec2012_r40_c1/depth/'+filetime[0]+'/'+filetime[1]+'/'+filetime[2]+'/device_1/depth_'+'_'.join(filetime)+'.png',
			'/media/Data/ICU_Dec2012/ICU_Dec2012_r40_c2/depth/'+filetime[0]+'/'+filetime[1]+'/'+filetime[2]+'/device_1/depth_'+'_'.join(filetime)+'.png',
			'/media/Data/ICU_Dec2012/ICU_Dec2012_r40_c2/depth/'+filetime[0]+'/'+filetime[1]+'/'+filetime[2]+'/device_2/depth_'+'_'.join(filetime)+'.png']
	im = sm.imread(URIs[cam_index])
	if coms is not None:
		# for c in coms:
		c = coms
		cv2.circle(im, (c[1],c[0]), 5, 5000, thickness=-1)

	cv2.imshow(str(cam_index), im/im.max().astype(np.float))	
	ret = cv2.waitKey(10)

	return im



data1 = np.load('/media/Data/r40_c1.npy')
data2 = np.load('/media/Data/r40_c2_d1.npy')
data3 = np.load('/media/Data/r40_c2_d2.npy')

coms1_r = np.array([x['com'] for x in data1])
coms2_r = np.array([x['com'] for x in data2])
coms3_r = np.array([x['com'] for x in data3])
cameraCenter1_r = [0,0,0,1]
cameraCenter2_r = [0,0,0,1]
cameraCenter3_r = [0,0,0,1]


'''
Transform12 = np.array([-0.8531195226064485, -0.08215320378328564, 0.5152066878990207, 1661.2299809410998, \
						0.3177589268248827, 0.7014041249433673, 0.6380137286418792, 1427.5420972165339, \
						-0.4137829679564377, 0.7080134918351199, -0.5722766383564786, -3399.696025885259, \
						0.0, 0.0, 0.0, 1.0]).reshape([4,4])
'''
Transform12 = np.array([[ -8.53119523e-01,  -8.21532038e-02,   5.15206688e-01,	1.66222998e+03],
						[  1.57073609e-01,   9.01259156e-01,   4.03806655e-01, 	1.79686253e+02],
						[ -4.97508755e-01,   4.25420714e-01,  -7.55977681e-01,	-3.68191742e+03],
						[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,	1.00000000e+00]])

Transform32 = np.array([[  9.55378205e-01,  -9.69196766e-02,   2.79032365e-01, 5.93818783e+02],
					   [ -8.66849294e-03,   9.35032463e-01,   3.54456133e-01,1.78054818e+02],
	 					[ -2.95258094e-01,  -3.41058454e-01,   8.92469489e-01,-1.89910900e+02],
	  				 	[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,1.00000000e+00]])

#Original:
Transform32 = np.array([0.9553782053802112, -0.09691967661345026, 0.27903236545178867, -392.81878278215254,0.09283849668727677, 0.9952919671849423, 0.02783726980083738, 231.6724797545669, -0.2804166511056782, -0.0006901755293638524, 0.9598781305147085, -118.84124965680712, 0.0, 0.0, 0.0, 1.0]).reshape([4,4])
#Transform12 = np.array([-0.8531195226064485, -0.08215320378328564, 0.5152066878990207, 761.2299809410998, 0.3177589268248827, 0.7014041249433673, 0.6380137286418792, 1427.5420972165339, -0.4137829679564377, 0.7080134918351199, -0.5722766383564786, -3399.696025885259, 0.0, 0.0, 0.0, 1.0]).reshape([4,4])

coms1 = np.array([np.dot(Transform12, np.hstack([x,1]))[:3] for x in coms1_r])
cameraCenter1 = np.dot(Transform12, np.array(cameraCenter1_r))
coms2 = coms2_r
cameraCenter2 = np.array(cameraCenter2_r[:3])
coms3 = np.array([np.dot(Transform32, np.hstack([x,1]))[:3] for x in coms3_r])
cameraCenter3 = np.dot(Transform32, np.array(cameraCenter3_r))

if 0:
	plot(coms1[:5000,0], coms1[:5000,2], '.b')
	plot(coms2[:5000,0], coms2[:5000,2], '.g')
	plot(coms3[:5000,0], coms3[:5000,2], '.r')

#Refine registration
R, t = IterativeClosestPoint(coms1[:4000], coms2, max_iters=100)
Transform12 = np.eye(4)
Transform12[:3,:3] = R
Transform12[:3,3] = t
coms1 = np.array([np.dot(Transform12, np.hstack([x,1]))[:3] for x in coms1])
cameraCenter1 = np.dot(Transform12, np.array(cameraCenter1))[:3]

R, t = IterativeClosestPoint(coms3, coms2, max_iters=100)
# R, t = PointcloudRegistration(coms3[:2000], coms2[:2000])
Transform32 = np.eye(4)
Transform32[:3,:3] = R
Transform32[:3,3] = t
coms3 = np.array([np.dot(Transform32, np.hstack([x,1]))[:3] for x in coms3])
cameraCenter3 = np.dot(Transform32, np.array(cameraCenter3))[:3]



colors = [[255,0,0],
		[0,255,0],
		[0,0,255],
		[255,255,0],
		[255,0,255],
		[0,255,255],
		[255,255,255],
		]
cameraCount = 3
centers=[3000, 0, -600]

temporal_offsets = np.array([-int(80.5*10),0, -5])
times1 = np.array([list2time(x['time']) for x in data1])+temporal_offsets[0]
times2 = np.array([list2time(x['time']) for x in data2])+temporal_offsets[1]
times3 = np.array([list2time(x['time']) for x in data3])+temporal_offsets[2]
timeStart = np.max([times1[0], times2[0], times3[0]])

allTimes = [times1, times2, times3]
allComs  = [coms1, coms2, coms3]
allData  = [data1, data2, data3]

inds = [[]]*cameraCount
indsPrev = [-1]*cameraCount
timesPrev = [0]*cameraCount
currentUsers = []
closedUsers = []
userCount = 0
t = timeStart

mapIm = np.zeros([600,600,3])
ims = []
while 1:
	
	
	''' Find which people are in the current timestep '''
	newComs = [[]]*cameraCount
	for cam_index in range(cameraCount):
		inds[cam_index] = np.where(allTimes[cam_index]==t)[0]

		if inds[cam_index].shape[0] > 0:
			indsPrev[cam_index] = inds[cam_index][-1]
			timesPrev[cam_index] = t#allTimes[cam_index][inds[cam_index][-1]]
			for i in inds[cam_index]:
				newComs[cam_index].append({'com':allComs[cam_index][i], 'time':allTimes[cam_index][i], 'camera':cam_index, 'data':allData[cam_index][i]})
				mapIm = map_points(np.array([allComs[cam_index][i]]), mapIm=mapIm, color=colors[cam_index], centers=centers)
			# print cam_index

	''' Add the new people to the set of current users '''
	''' Start by looking through the previous users and finding the closest new user '''
	for u in currentUsers:
		for cam_index in range(cameraCount):
			#Skip if no entries
			if len(newComs[cam_index]) == 0:
				continue

			dists = [np.linalg.norm(u['coms'][-1] - x['com']) for x in newComs[cam_index]]
			closestPerson = np.argmin(dists)

			if dists[closestPerson] < 500:
				u['coms'].append(newComs[cam_index][closestPerson]['com'])
				u['times'].append(newComs[cam_index][closestPerson]['time'])
				u['cameras'].append(newComs[cam_index][closestPerson]['camera'])
				u['data'].append(newComs[cam_index][closestPerson]['data'])

				# centroid = world2depth(np.array([[-u['data'][-1]['com'][1], -u['data'][-1]['com'][0], u['data'][-1]['com'][2]]])).T[0]/2 #rez=[320,240]
				centroid = world2depth(np.array([u['data'][-1]['com']])).T[0]/2 #rez=[320,240]
				print centroid

				try:
					print u['data'][-1]['time']
					ims.append(viewImage(u['cameras'][-1], u['data'][-1]['time'], coms=centroid))
				except:
					print "Error viewing index:" + str(cam_index) + " time:" + " ".join(newComs[cam_index][closestPerson]['data']['time'])

				newComs[cam_index].pop(closestPerson)


	''' If there isn't a relevant user then add a new user to the list '''
	for cam_index in range(cameraCount):
		for u in newComs[cam_index]:
			userCount += 1
			user = {'coms':[u['com']], 'times':[u['time']], 'cameras':[cam_index], 'data':[u['data']], 'id':userCount}
			currentUsers.append(user)

	removeUsers = []
	for i in range(len(currentUsers)):
		u = currentUsers[i]

		if len(u) > 10: # At least in 2 frames
			''' kill current users who left '''
			if t - u['times'][-1] > 10*10:
				closedUsers.append(u)
				removeUsers.append(i)

		if len(u) > 2: # At least in 2 frames
			''' Visualize users '''
			# im = map_points(np.array([c[-1][0]]), mapIm=im, color=colors[i%len(colors)], centers=centers)
			# im = map_points(np.array(u['coms']), mapIm=im, color=colors[u['id']%len(colors)], centers=centers)
			pass

	# Remove dead users
	removeUsers.reverse()
	for i in removeUsers:
		print "Remove ", i
		closedUsers.append(currentUsers.pop(i))

	#Add time to bottom left of image
	mapIm[530:,:200] = np.array([0,0,0]) #Flush previous time
	cv2.putText(mapIm, str(timesPrev[0]), (10,550), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0,0))
	cv2.putText(mapIm, str(timesPrev[1]), (10,570), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255,0))
	cv2.putText(mapIm, str(timesPrev[2]), (10,590), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0,255))
	
	plotCamera(cameraCenter1, mapIm, centers=centers)
	plotCamera(cameraCenter2, mapIm, centers=centers)
	plotCamera(cameraCenter3, mapIm, centers=centers)
	cv2.imshow("coms", mapIm)
	
	# print "Current users: ", len(currentUsers)
	# print "Total users: ", userCount

	t = np.min([times1[indsPrev[0]+1],times2[indsPrev[1]+1],times3[indsPrev[2]+1]])

	ret = cv2.waitKey(100)
	# if ret > 0:
		# break

if 0:
	# 2,3,5
	imsUV = [ims[-2],ims[-3],ims[-5]]
	# imsXYZ = [depthIm2PosIm(x) for x in imsUV]
	imsXYZ = [depthIm2XYZ(x) for x in imsUV]
	imsXYZ = [np.hstack([imsXYZ[x], np.ones((imsXYZ[x].shape[0], 1))]) for x in range(cameraCount)]

	imsXYZ[0] = np.dot(Transform12, imsXYZ[0].T)[:3]
	imsXYZ[1] = imsXYZ[1].T[:3]
	imsXYZ[2] = np.dot(Transform32, imsXYZ[2].T)[:3]

	# transform and show top down view
	imTop = np.ones([600,600,3])*0
	imsColor = [x[np.nonzero(x>0)].ravel() for x in imsUV]
	for i in range(len(imsXYZ)):
		xs, ys = pts2coords(imsXYZ[i].T, imTop.shape, centers=centers)
		imTop[xs, ys, i] = imsColor[i]/5000.