'''
Main file for displaying depth/color/skeleton information and extracting features
'''

import os
import optparse
from time import time
import cPickle as pickle
import numpy as np
import scipy.misc as sm
import scipy.ndimage as nd
import skimage
from skimage import feature, color

from pyKinectTools.utils.Utils import createDirectory
from pyKinectTools.utils.VideoViewer import VideoViewer
from pyKinectTools.utils.DepthUtils import world2depth, depthIm2XYZ
from pyKinectTools.utils.MultiCameraUtils import multiCameraTimeline, formatFileString
from pyKinectTools.utils.FeatureUtils import saveFeatures, loadFeatures, learnICADict, learnNMFDict, displayComponents
from pyKinectTools.algs.HistogramOfOpticalFlow import getFlow, hof, splitIm
from pyKinectTools.algs.BackgroundSubtraction import AdaptiveMixtureOfGaussians, fill_image, extract_people
from pyKinectTools.algs.FeatureExtraction import calculateBasicPose, computeUserFeatures, computeFeaturesWithSkels

vv = VideoViewer()

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

''' Keyboard keys (using Video Viewer)'''
keys_ESC = 27
keys_left_arrow = 314
keys_right_arrow = 316
keys_down_arrow = 317
keys_space = 32
keys_i = 105
keys_help = 104
keys_frame_left = 314
keys_frame_right = 316
''' Using OpenCV
keys_ESC = 1048603
keys_right_arrow = 1113939
keys_left_arrow = 1113937
keys_down_arrow = 1113940
keys_space = 1048608
keys_i = 1048681
keys_help = 1048680
keys_frame_left = 1048673
keys_frame_right = 1048691
'''

# -------------------------MAIN------------------------------------------
# @profile
def main(get_depth, get_color, get_skeleton, get_mask, calculate_features, visualize, save_anonomized, device):

	dev = device
	ret = 0
	backgroundTemplates = np.empty([1,1,1])
	backgroundModel = None
	backgroundCount = 20
	bgPercentage = .05
	prevDepthIm = None

	day_dirs = os.listdir('depth/')
	day_dirs = [x for x in day_dirs if x[0]!='.']
	day_dirs.sort(key=lambda x: int(x))
	hour_index = 0
	minute_index=0

	allFeatures = []
	coms = []
	orns = []

	play_speed = 1
	new_date_entered = False
	framerate = 0
	frame_prev = 0
	frame_prev_time = time()

	day_index = 0
	while day_index < len(day_dirs):

		if new_date_entered:
			try:
				day_index = day_dirs.index(day_new)
			except:
				print "Day not found"
				day_index = 0

		dayDir = day_dirs[day_index]

		hour_dirs = os.listdir('depth/'+dayDir)
		hour_dirs = [x for x in hour_dirs if x[0]!='.']
		hour_dirs.sort(key=lambda x: int(x))

		'''Hours'''
		''' Check for new Hours index '''
		if not new_date_entered:
			if play_speed >= 0  and ret != keys_frame_left:
				hour_index = 0
			else:
				hour_index = len(hour_dirs)-1
		else:
			try:
				hour_index = hour_dirs.index(hour_new)
			except:
				print "Hour was not found"
				hour_index = 0

		while hour_index < len(hour_dirs):
			hourDir = hour_dirs[hour_index]

			minute_dirs = os.listdir('depth/'+dayDir+'/'+hourDir)
			minute_dirs = [x for x in minute_dirs if x[0]!='.']
			minute_dirs.sort(key=lambda x: int(x))

			'''Minutes'''
			''' Check for new minute index '''
			if not new_date_entered:
				if play_speed >= 0  and ret != keys_frame_left:
					minute_index = 0
				else:
					minute_index = len(minute_dirs)-1
			else:
				try:
					minute_index = minute_dirs.index(minute_new)
				except:
					print "Minute was not found"
					minute_index = 0

			''' Loop through this minute '''
			while minute_index < len(minute_dirs):
				minute_dir = minute_dirs[minute_index]

				# Prevent from reading hidden files
				if minute_dir[0] == '.':
					continue

				depth_files = []
				skelFiles = []

				# For each available device:
				devices = os.listdir('depth/'+dayDir+'/'+hourDir+'/'+minute_dir)
				devices = [x for x in devices if x[0]!='.' and x.find('tmp')<0]
				devices.sort()

				deviceID = "device_{0:d}".format(dev+1)

				if not os.path.isdir('depth/'+dayDir+'/'+hourDir+'/'+minute_dir+'/'+deviceID):
					continue

				''' Sort files '''
				if get_depth:
					depthTmp = os.listdir('depth/'+dayDir+'/'+hourDir+'/'+minute_dir+'/'+deviceID)
					tmpSort = [int(x.split('_')[-3])*100 + int(formatFileString(x.split('_')[-2])) for x in depthTmp]
					depthTmp = np.array(depthTmp)[np.argsort(tmpSort)].tolist()
					depth_files.append([x for x in depthTmp if x.find('.png')>=0])
				if get_skeleton:
					skelTmp = os.listdir('skel/'+dayDir+'/'+hourDir+'/'+minute_dir+'/'+deviceID)
					tmpSort = [int(x.split('_')[-4])*100 + int(formatFileString(x.split('_')[-3])) for x in skelTmp]
					skelTmp = np.array(skelTmp)[np.argsort(tmpSort)].tolist()
					skelFiles.append([x for x in skelTmp if x.find('.dat')>=0])

				if len(depth_files) == 0:
					continue

				if play_speed >= 0 and ret != keys_frame_left:
					frame_id = 0
				else:
					frame_id = len(depth_files[dev])-1


				while frame_id < len(depth_files[0]):
				# while frame_id < len(depth_files[dev]):

					depthFile = depth_files[0][frame_id]
					# try:
					if 1:
						''' Load Depth '''
						if get_depth:
							depthIm = sm.imread('depth/'+dayDir+'/'+hourDir+'/'+minute_dir+'/'+deviceID+'/'+depthFile)
							depthIm = np.array(depthIm, dtype=np.uint16)
						''' Load Color '''
						if get_color:
							colorFile = 'color_'+depthFile[6:-4]+'.jpg'
							colorIm = sm.imread('color/'+dayDir+'/'+hourDir+'/'+minute_dir+'/'+deviceID+'/'+colorFile)
							# colorIm_g = colorIm.mean(-1, dtype=np.uint8)
							colorIm_g = skimage.img_as_ubyte(skimage.color.rgb2gray(colorIm))
							# colorIm_lab = skimage.color.rgb2lab(colorIm).astype(np.uint8)

						''' Load Skeleton Data '''
						if get_skeleton:
							skelFile = 'skel_'+depthFile[6:-4]+'_.dat'
							if os.path.isfile('skel/'+dayDir+'/'+hourDir+'/'+minute_dir+'/'+deviceID+'/'+skelFile):
								with open('skel/'+dayDir+'/'+hourDir+'/'+minute_dir+'/'+deviceID+'/'+skelFile, 'rb') as inFile:
									users = pickle.load(inFile)
							else:
								print "No user file:", skelFile
							coms = [users[x]['com'] for x in users.keys() if users[x]['com'][2] > 0.0]
							jointCount = 0
							for i in users.keys():
								user = users[i]

						timestamp = depthFile[:-4].split('_')[1:] # Day, hour, minute, second, millisecond, Frame number in this second
						depthIm = np.minimum(depthIm.astype(np.float), 5000)
						fill_image(depthIm)

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
						foregroundMask = bgSubtraction.get_foreground(thresh=50)

						''' Find people '''
						if get_skeleton:
							ret = plotUsers(depthIm, users, device=deviceID, vis=True)
						if get_mask:
							foregroundMask, userBoundingBoxes, userLabels = extract_people(depthIm, foregroundMask, minPersonPixThresh=1500, gradientFilter=True, gradThresh=100)

						''' Calculate user features '''
						if calculate_features and get_color:
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
							tmpSecond = depthFile.split("_")[-3]
							if len(tmpSecond) == 0:
								tmpSecond = '0'+tmpSecond
							if get_depth:
								vv.imshow("Depth", depthIm/6000.)
								vv.putText("Depth", "Day "+dayDir+" Time "+hourDir+":"+minute_dir+":"+tmpSecond, (5,220), size=15)
								vv.putText("Depth", "Play speed: "+str(play_speed)+"x", (5,15), size=15)
								vv.putText("Depth", str(int(framerate))+" fps", (275,15), size=15)

							if get_color:
								vv.putText(colorIm, "Day "+dayDir+" Time "+hourDir+":"+minute_dir+" Dev#"+str(dev), (10,220))
								vv.imshow("I_orig", colorIm)
								if get_mask:
									# vv.imshow("I", colorIm*foregroundMask[:,:,np.newaxis])
									vv.imshow("I_masked", colorIm + (255-colorIm)*(((foregroundMask)[:,:,np.newaxis])))
							if get_mask:
								vv.imshow("Mask", foregroundMask.astype(np.float)/float(foregroundMask.max()))
								# vv.imshow("BG Model", backgroundModel.astype(np.float)/float(backgroundModel.max()))


							''' Multi-camera map '''
							if 0 and len(coms) > 0:
								mapRez = [200,200]
								mapIm = np.zeros(mapRez)
								coms_np = np.array(coms)
								xs = np.minimum(np.maximum(mapRez[0]+((coms_np[:,2]+500)/3000.*mapRez[0]).astype(np.int), 0),mapRez[0]-1)
								ys = np.minimum(np.maximum(((coms_np[:,0]+500)/1500.*mapRez[0]).astype(np.int), 0), mapRez[1]-1)
								mapIm[xs, ys] = 255
								vv.imshow("Map", mapIm)
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

								# ss = mlab.points3d(pts[:,0], pts[:,1], pts[:,2])


						''' Playback control: Look at keyboard input '''
						ret = vv.waitKey()

						if frame_id - frame_prev > 0:
							framerate = (frame_id - frame_prev) / (time() - frame_prev_time)
						frame_prev = frame_id
						frame_prev_time = time()

						new_date_entered = False
						if ret > 0:
							# player_controls(ret)
							# print "Ret is",ret

							if ret == keys_ESC:
								break
							elif ret == keys_space:
								print "Enter the following into the command line"
								tmp = raw_input("Enter date: ")
								day_new = tmp
								tmp = raw_input("Enter hour: ")
								hour_new = tmp
								tmp = raw_input("Enter minute: ")
								minute_new = tmp

								print "New date:", day_new, hour_new, minute_new

								new_date_entered = True
								break

							elif ret == keys_down_arrow:
								play_speed = 0
							elif ret == keys_left_arrow:
								play_speed -= 1
							elif ret == keys_right_arrow:
								play_speed += 1
							elif ret == keys_i:
									embed()
							elif ret == keys_frame_left:
								frame_id -= 1
							elif ret == keys_frame_right:
								frame_id += 1
							elif ret == keys_help:
								display_help()

						frame_id += play_speed

					if save_anonomized and get_mask:
						save_dir = 'color_masked/'+dayDir+'/'+hourDir+'/'+minute_dir+'/'+devices[dev]+'/'
						createDirectory(save_dir)
						sm.imsave(save_dir+'colorM_'+depthFile[6:-4]+'.jpg', colorIm*(1-foregroundMask))
					# except:
						# print "Erroneous frame"
						# if visualize:
						# 	vv.imshow("D", depthIm.astype(np.float)/5000)
						# 	ret = vv.waitKey(10)


					# End seconds
					if ret == keys_ESC or new_date_entered:
						break
					if frame_id >= len(depth_files[0]):
						minute_index += 1
					elif frame_id < 0:
						minute_index -= 1
						break

				# End hours
				if ret == keys_ESC or new_date_entered:
					break

				if minute_index >= len(minute_dirs):
					hour_index += 1
				elif minute_index < 0:
					hour_index -= 1
					break

			# End days
			if ret == keys_ESC:
				break
			if new_date_entered:
				break

			if hour_index >= len(hour_dirs):
				day_index += 1
			elif hour_index < 0:
				day_index -= 1

			if day_index < 0:
				day_index = 0


		if ret == keys_ESC or day_index > len(day_dirs):
			break


	np.save("/media/Data/r40_cX_", allFeatures)
	embed()

if 0:
	coms1 = np.load('../../ICU_Dec2012_r40_c1_coms_partial.npy')
	T = np.array([-0.8531195226064485, -0.08215320378328564, 0.5152066878990207, 761.2299809410998, 0.3177589268248827, 0.7014041249433673, 0.6380137286418792, 1427.5420972165339, -0.4137829679564377, 0.7080134918351199, -0.5722766383564786, -3399.696025885259, 0.0, 0.0, 0.0, 1.0])
	T = T.reshape([4,4])
	coms12 = np.array([np.dot(np.asarray(T), np.array([x[0], x[1], x[2], 1])) for x in coms1])

def display_help():
	print ""
	print "Playback commands: enter these in the image viewer"
	print "--------------------"
	print "h 				help menu"
	print "i				interupt with debugger"
	print "a 				previous frame"
	print "s 				next frame"
	print "spacebar 			pick new time/date [enter in terminal]"
	print "left arrow key			rewind faster"
	print "right arrow key			fast forward faster"
	print "down arrow key			pause"
	print "escape key			exit"




if __name__=="__main__":

	parser = optparse.OptionParser()
	parser.add_option('-s', '--skel', dest='skel', action="store_true", default=False, help='Enable skeleton')
	parser.add_option('-d', '--depth', dest='depth', action="store_true", default=False, help='Enable depth images')
	parser.add_option('-c', '--color', dest='color', action="store_true", default=False, help='Enable color images')
	parser.add_option('-m', '--mask', dest='mask', action="store_true", default=False, help='Enable enternal mask')
	parser.add_option('-a', '--anonomize', dest='save', action="store_true", default=False, help='Save anonomized RGB image')
	parser.add_option('-f', '--calcFeatures', dest='bgSubtraction', action="store_true", default=False, help='Enable feature extraction')
	parser.add_option('-v', '--visualize', dest='viz', action="store_true", default=False, help='Enable visualization')
	parser.add_option('-i', '--dev', dest='dev', type='int', default=0, help='Device number')
	(opt, args) = parser.parse_args()

	if opt.bgSubtraction or opt.save:
		opt.mask = True

	if opt.viz:
		display_help()

	if len(args) > 0:
		print "Wrong input argument"
	elif opt.depth==False and opt.color==False and opt.skel==False:
		print "You must supply the program with some arguments."
	else:
		main(get_depth=opt.depth, get_skeleton=opt.skel, get_color=opt.color, get_mask=opt.mask, calculate_features=opt.bgSubtraction, visualize=opt.viz, save_anonomized=opt.save, device=opt.dev)

	'''Profiling'''
	# cProfile.runctx('main()', globals(), locals(), filename="ShowSkeletons.profile")



if 0:
	hogIms = np.vstack([allFeatures[i]['hogIm'] for i in range(len(allFeatures))])
	hogs = np.vstack([allFeatures[i]['hog'] for i in range(len(allFeatures))])
