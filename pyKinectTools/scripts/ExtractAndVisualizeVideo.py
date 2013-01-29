'''
Main file for displaying depth/color/skeleton information and extracting features
'''

import os
import optparse
from time import time
import cPickle as pickle
import cv2
import numpy as np
import scipy.misc as sm
import scipy.ndimage as nd
import skimage
from skimage import feature, color

from pyKinectTools.utils.Utils import createDirectory
from pyKinectTools.utils.DepthUtils import world2depth, depthIm2XYZ
from pyKinectTools.utils.MultiCameraUtils import multiCameraTimeline, formatFileString
from pyKinectTools.utils.FeatureUtils import saveFeatures, loadFeatures, learnICADict, learnNMFDict, displayComponents
from pyKinectTools.algs.HistogramOfOpticalFlow import getFlow, hof, splitIm
from pyKinectTools.algs.BackgroundSubtraction import AdaptiveMixtureOfGaussians, fillImage, extractPeople
from pyKinectTools.algs.FeatureExtraction import calculateBasicPose, plotUsers, computeUserFeatures, computeFeaturesWithSkels

''' 3D visualization '''
if 0:
	from mayavi import mlab
	figure = mlab.figure(1, bgcolor=(0,0,0), fgcolor=(1,1,1))
	mlab.clf()
	figure.scene.disable_render = True

''' Debugging '''
from IPython import embed
import cProfile

np.seterr(all='ignore')

# -------------------------MAIN------------------------------------------
# @profile
def main(get_depth, get_color, get_skeleton, get_mask, calculate_features, visualize, save_anonomized):

	ret = 0
	backgroundTemplates = np.empty([1,1,1])
	backgroundModel = None
	backgroundCount = 20
	bgPercentage = .05
	prevDepthIm = None
	prevDepthIms = []
	prevColorIms = []

	dayDirs = os.listdir('depth/')
	dayDirs = [x for x in dayDirs if x[0]!='.']
	dayDirs.sort(key=lambda x: int(x))

	allFeatures = []
	coms = []
	orns = []

	for dayDir in dayDirs:
		hourDirs = os.listdir('depth/'+dayDir)
		hourDirs = [x for x in hourDirs if x[0]!='.']
		hourDirs.sort(key=lambda x: int(x))


		for hourDir in hourDirs: # Hours
			minuteDirs = os.listdir('depth/'+dayDir+'/'+hourDir)
			minuteDirs = [x for x in minuteDirs if x[0]!='.']
			minuteDirs.sort(key=lambda x: int(x))
			for minuteDir in minuteDirs: # Minutes

				if minuteDir[0] == '.': # Prevent from reading hidden files
					continue

				depthFiles = []
				skelFiles = []

				# For each available device:
				devices = os.listdir('depth/'+dayDir+'/'+hourDir+'/'+minuteDir)
				devices = [x for x in devices if x[0]!='.' and x.find('tmp')<0]
				devices.sort()

				for deviceID in ['device_1']:
					if not os.path.isdir('depth/'+dayDir+'/'+hourDir+'/'+minuteDir+'/'+deviceID):
						continue

					''' Sort files '''
					if get_depth:
						depthTmp = os.listdir('depth/'+dayDir+'/'+hourDir+'/'+minuteDir+'/'+deviceID)
						tmpSort = [int(x.split('_')[-3])*100 + int(formatFileString(x.split('_')[-2])) for x in depthTmp]
						depthTmp = np.array(depthTmp)[np.argsort(tmpSort)].tolist()
						depthFiles.append([x for x in depthTmp if x.find('.png')>=0])
					if get_skeleton:
						skelTmp = os.listdir('skel/'+dayDir+'/'+hourDir+'/'+minuteDir+'/'+deviceID)
						tmpSort = [int(x.split('_')[-4])*100 + int(formatFileString(x.split('_')[-3])) for x in skelTmp]
						skelTmp = np.array(skelTmp)[np.argsort(tmpSort)].tolist()
						skelFiles.append([x for x in skelTmp if x.find('.dat')>=0])

				if len(depthFiles) == 0:
					continue
				for dev, depthFile in multiCameraTimeline(depthFiles):
					if deviceID == 'device_2':
						dev = 1
					else:
						dev = 0
					# try:
					if 1:
						print depthFile

						''' Load Depth '''
						if get_depth:
							depthIm = sm.imread('depth/'+dayDir+'/'+hourDir+'/'+minuteDir+'/'+devices[dev]+'/'+depthFile)
							depthIm = np.array(depthIm, dtype=np.uint16)
						''' Load Color '''
						if get_color:
							colorFile = 'color_'+depthFile[6:-4]+'.jpg'
							colorIm = sm.imread('color/'+dayDir+'/'+hourDir+'/'+minuteDir+'/'+devices[dev]+'/'+colorFile)
							# colorIm_g = colorIm.mean(-1, dtype=np.uint8)
							colorIm_g = skimage.img_as_ubyte(skimage.color.rgb2gray(colorIm))
							# colorIm_lab = skimage.color.rgb2lab(colorIm).astype(np.uint8)
						# ''' Load Mask '''
						# if get_mask:
						# 	maskIm = sm.imread('depth/'+dayDir+'/'+hourDir+'/'+minuteDir+'/'+devices[dev]+'/'+depthFile[:-4]+"_mask.jpg") > 100
						# 	depthIm = depthIm*(1-maskIm)+maskIm*5000

						''' Load Skeleton Data '''
						if get_skeleton:
							skelFile = 'skel_'+depthFile[6:-4]+'_.dat'
							if os.path.isfile('skel/'+dayDir+'/'+hourDir+'/'+minuteDir+'/'+devices[dev]+'/'+skelFile):
								with open('skel/'+dayDir+'/'+hourDir+'/'+minuteDir+'/'+devices[dev]+'/'+skelFile, 'rb') as inFile:
									users = pickle.load(inFile)				
							else:
								print "No user file:", skelFile

							coms = [users[x]['com'] for x in users.keys() if users[x]['com'][2] > 0.0]

							jointCount = 0
							for i in users.keys():
								user = users[i]
								# if user['jointPositions'][user['jointPositions'].keys()[0]] != -1:
									# print user['jointPositionsConfidence']
									# jointCount = 1

						timestamp = depthFile[:-4].split('_')[1:] # Day, hour, minute, second, millisecond, Frame number in this second
						depthIm = np.minimum(depthIm.astype(np.float), 5000)
						fillImage(depthIm)


						'''Background model'''
						if backgroundModel is None:
							bgSubtraction = AdaptiveMixtureOfGaussians(depthIm, maxGaussians=3, learningRate=0.01, decayRate=0.02, variance=300**2)
							backgroundModel = bgSubtraction.getModel()
							if get_color:
								prevColorIm = colorIm_g.copy()
							continue
						else:
							bgSubtraction.update(depthIm)

						backgroundModel = bgSubtraction.getModel()
						foregroundMask = bgSubtraction.getForeground(thresh=50)

						''' Find people '''
						if get_skeleton:
							ret = plotUsers(depthIm, users, device=devices[dev], vis=True)
						if get_mask:
							foregroundMask, userBoundingBoxes, userLabels = extractPeople(depthIm, foregroundMask, minPersonPixThresh=1500, gradientFilter=True, gradThresh=100)
						
						''' Calculate user features '''
						if calculate_features:
							''' Color Optical Flow '''
							flow = getFlow(prevColorIm, colorIm_g)
							prevColorIm = colorIm_g.copy()
							
							userCount = len(userBoundingBoxes)
							for i in xrange(userCount):
								userBox = userBoundingBoxes[i]
								userMask = foregroundMask==i+1
								allFeatures.append(computeUserFeatures(colorIm, depthIm, flow, userBox, time=timestamp, mask=userMask, windowSize=[96,72], visualise=False))
						''' Or get CoM + orientation '''
						if get_mask and not calculate_features:
							userCount = len(userBoundingBoxes)
							for i in xrange(userCount):
								userBox = userBoundingBoxes[i]
								userMask = foregroundMask==i+1
								com, ornBasis = calculateBasicPose(depthIm, userMask)
								coms.append(com)
								orns.append(ornBasis[1])
								allFeatures.append({'com':com, "orn":ornBasis, 'time':timestamp})

						''' Visualization '''
						if visualize:
							if get_depth:
								# " Dev#"+str(dev)
								cv2.putText(depthIm, "Day "+dayDir+" Time "+hourDir+":"+minuteDir+":"+depthFile.split("_")[-3], (5,220), cv2.FONT_HERSHEY_DUPLEX, 0.6, 5000)					
								cv2.imshow("Depth", depthIm/5000.)
							if get_color:
								# cv2.putText(colorIm, "Day "+dayDir+" Time "+hourDir+":"+minuteDir+" Dev#"+str(dev), (10,220), cv2.FONT_HERSHEY_DUPLEX, 0.6, 5000)					
								cv2.imshow("I_orig", colorIm)
								if get_mask:
									# cv2.imshow("I", colorIm*foregroundMask[:,:,np.newaxis])
									cv2.imshow("I_masked", colorIm + (255-colorIm)*(((foregroundMask)[:,:,np.newaxis])))
							if get_mask:
								cv2.imshow("Mask", foregroundMask.astype(np.float)/float(foregroundMask.max()))
								# cv2.imshow("BG Model", backgroundModel.astype(np.float)/float(backgroundModel.max()))


							''' Multi-camera map '''
							if len(coms) > 0:
								mapRez = [200,200]
								mapIm = np.zeros(mapRez)
								# embed()
								coms_np = np.array(coms)
								xs = np.minimum(np.maximum(mapRez[0]+((coms_np[:,2]+500)/3000.*mapRez[0]).astype(np.int), 0),mapRez[0]-1)
								ys = np.minimum(np.maximum(((coms_np[:,0]+500)/1500.*mapRez[0]).astype(np.int), 0), mapRez[1]-1)
								mapIm[xs, ys] = 255
								cv2.imshow("Map", mapIm)
								# scatter(coms_np[:,0], -coms_np[:,2])


							'''3D Vis'''
							if 0:
								# figure = mlab.figure(1, fgcolor=(1,1,1), bgcolor=(0,0,0))
								# from pyKinectTools.utils.DepthUtils import *
								pts = depthIm2XYZ(depthIm).astype(np.int)
								interval = 25
								figure.scene.disable_render = True
								mlab.clf()
								# ss = mlab.points3d(-pts[::interval,0], pts[::interval,1], pts[::interval,2], colormap='Blues', vmin=1000., vmax=5000., mode='2dvertex')
								ss = mlab.points3d(pts[::interval,0], pts[::interval,1], pts[::interval,2], 5.-(np.minimum(pts[::interval,2], 5000)/float((-pts[:,2]).max()))/1000., scale_factor=25., colormap='Blues')#, mode='2dvertex')
								# , scale_factor=25.
								mlab.view(azimuth=0, elevation=0, distance=3000., focalpoint=(0,0,0), figure=figure)#, reset_roll=False)
								# mlab.roll(90)
								currentView = mlab.view()
								figure.scene.disable_render = False
								mlab.draw()
								# mlab.show()
								# ss = mlab.points3d(pts[::interval,0], pts[::interval,1], pts[::interval,2], color=col, scale_factor=5)
								# ss = mlab.points3d(pts[:,0], pts[:,1], pts[:,2], color=(1,1,1), scale_factor=5)

								# from IPython import embed
								# embed()

								# ss = mlab.points3d(pts[:,0], pts[:,1], pts[:,2])

						ret = cv2.waitKey(10)

						if ret > 0:
							break

					if save_anonomized and get_mask:
						saveDir = 'color_masked/'+dayDir+'/'+hourDir+'/'+minuteDir+'/'+devices[dev]+'/'
						createDirectory(saveDir)
						sm.imsave(saveDir+'colorM_'+depthFile[6:-4]+'.jpg', colorIm*(1-foregroundMask))
					# except:
						# print "Erroneous frame"
						# if visualize:
						# 	cv2.imshow("D", depthIm.astype(np.float)/5000)
						# 	ret = cv2.waitKey(10)

				if ret > 0:
					break
			if ret > 0:
				break
		if ret > 0:
			# embed()
			break

	np.save("/media/Data/r40_cX_", allFeatures)
	embed()

if 0:
	coms1 = np.load('../../ICU_Dec2012_r40_c1_coms_partial.npy')
	T = np.array([-0.8531195226064485, -0.08215320378328564, 0.5152066878990207, 761.2299809410998, 0.3177589268248827, 0.7014041249433673, 0.6380137286418792, 1427.5420972165339, -0.4137829679564377, 0.7080134918351199, -0.5722766383564786, -3399.696025885259, 0.0, 0.0, 0.0, 1.0])
	T = T.reshape([4,4])
	coms12 = np.array([np.dot(np.asarray(T), np.array([x[0], x[1], x[2], 1])) for x in coms1])

if __name__=="__main__":

	parser = optparse.OptionParser()
	parser.add_option('-s', '--skel', dest='skel', action="store_true", default=False, help='Enable skeleton')	
	parser.add_option('-d', '--depth', dest='depth', action="store_true", default=False, help='Enable depth images')		
	parser.add_option('-c', '--color', dest='color', action="store_true", default=False, help='Enable color images')	
	parser.add_option('-m', '--mask', dest='mask', action="store_true", default=False, help='Enable enternal mask')
	parser.add_option('-a', '--anonomize', dest='save', action="store_true", default=False, help='Save anonomized RGB image')
	parser.add_option('-f', '--calcFeatures', dest='bgSubtraction', action="store_true", default=False, help='Enable feature extraction')		
	parser.add_option('-v', '--visualize', dest='viz', action="store_true", default=False, help='Enable visualization')
	(opt, args) = parser.parse_args()

	if opt.bgSubtraction or opt.save_anonomized:
		opt.mask = True

	if len(args) > 0:
		print "Wrong input argument"
	elif opt.depth==False and opt.color==False and opt.skel==False:
		print "You must supply the program with some arguments."
	else:
		main(get_depth=opt.depth, get_skeleton=opt.skel, get_color=opt.color, get_mask=opt.mask, calculate_features=opt.bgSubtraction, visualize=opt.viz, save_anonomized=opt.save)

	'''Profiling'''
	# cProfile.runctx('main()', globals(), locals(), filename="ShowSkeletons.profile")



if 0:
	hogIms = np.vstack([allFeatures[i]['hogIm'] for i in range(len(allFeatures))])
	hogs = np.vstack([allFeatures[i]['hog'] for i in range(len(allFeatures))])