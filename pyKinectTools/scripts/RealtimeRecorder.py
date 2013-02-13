
import os, time, sys, optparse
import numpy as np
import scipy.misc as sm
import Image
from pyKinectTools.utils.RealtimeReader import *
from pyKinectTools.utils.Utils import createDirectory
import cPickle as pickle
# import cProfile

# from multiprocessing import Pool, Process, Queue

DIR = '/Users/colin/Data/icu_test/'
# DIR = '/home/clea/Data/WICU_12Feb2012/'
# DIR = '/media/Data/icu_test_color/'

# DIR = '/media/Data/CV_class/'


# @profile
def save_frame(depthName=None, depth=None, colorName=None, color=None, userName=None, users=None, maskName=None, mask=None):

	print depthName
	''' Depth '''
	if depthName is not None:
		im = Image.fromarray(depth.astype(np.int32), 'I')
		im = im.resize([320,240])
		im.save(depthName)
		
	'''Mask'''
	if mask is not None and maskName is not None:
			mask = sm.imresize(mask, [240,320], 'nearest')
			sm.imsave(maskName, mask)

	'''Color'''
	if colorName is not None:
		color = sm.imresize(color, [240,320,3], 'nearest')
		sm.imsave(colorName, color)

	'''User'''
	if userName is not None:
		usersOut = {}
		for k in users.keys():
				usersOut[k] = users[k].toDict()

		with open(userName, 'wb') as outfile:
				pickle.dump(usersOut, outfile, protocol=pickle.HIGHEST_PROTOCOL)


# @profile
def main(deviceID, record, baseDir, frameDifferencePercent, getSkel, anonomize, viz, motionLagTime=10, min_fps=0.5):
		'''
		---parameters---
		deviceID
		record
		baseDir
		frameDifferencePercent
		getSkel
		anonomize
		viz
		motionLagTime
		min_fps
		'''

		'''------------ Setup Kinect ------------'''
		''' Physical Kinect '''
		depthDevice = RealTimeDevice(device=deviceID, getDepth=True, getColor=True, getSkel=getSkel)
		depthDevice.start()

		maxFramerate = 30
		minFramerate = min_fps
		recentMotionTime = time.clock()
		imgStoreCount = 100

		''' ------------- Main -------------- '''

		''' Setup time-based stuff '''
		prevTime = 0
		prevFrame = 0
		prevFrameTime = 0
		currentFrame = 0;
		ms = time.clock()
		diff = 0

		prevSec = 0;
		secondCount = 0
		prevSecondCountMax = 0

		''' Ensure base folder is there '''
		if not os.path.isdir(baseDir):
				os.mkdir(baseDir)        

		depthOld = []
		colorOld = []
		backgroundModel = None

		if viz:
			import cv2
			from pyKinectTools.utils.DepthUtils import world2depth
			cv2.namedWindow("image")        


		while 1:
				# try:
				if 1:

						depthDevice.update()
						colorRaw = depthDevice.colorIm
						depthRaw8 = depthDevice.depthIm
						users = depthDevice.users
						skel = None


						if len(depthOld) == imgStoreCount:
								depthOld.pop(0)

						''' If framerate is too fast then skip '''
						''' Keep this after update to ensure fast enough kinect refresh '''
						if (time.clock() - float(ms))*1000 < 1000.0/maxFramerate:
								continue                

						if viz and 0:
								for i in depthDevice.user.users:
										tmpPx = depthDevice.user.get_user_pixels(i)

										if depthDevice.skel_cap.is_tracking(i):
												brightness = 50
										else:
												brightness = 150
										depthRaw8 = depthRaw8*(1-np.array(tmpPx).reshape([480,640]))
										depthRaw8 += brightness*(np.array(tmpPx).reshape([480,640]))

						d = None
						d = np.array(depthRaw8)

						d /= (np.nanmin([d.max(), 2**16])/256.0)
						d = d.astype(np.uint8)

						''' Get new time info '''
						currentFrame += 1
						time_ = time.localtime()
						day = str(time_.tm_yday)
						hour = str(time_.tm_hour)
						minute = str(time_.tm_min)
						second = str(time_.tm_sec)
						ms = str(time.clock())
						ms_str = str(ms)[str(ms).find(".")+1:]


						''' Look at how much of the image has changed '''
						if depthOld != []:
								diff = np.sum(np.logical_and((depthRaw8 - depthOld[0]) > 200, (depthRaw8 - depthOld[0]) < 20000)) / 307200.0 * 100

								''' We want to watch all video for at least 5 seconds after we seen motion '''
								''' This prevents problems where there is small motion that doesn't trigger the motion detector '''
								if diff > frameDifferencePercent:
										recentMotionTime = time.clock()

						depthOld.append(depthRaw8)                                

						if anonomize:
							'''Background model'''
							if backgroundModel is None:
								bgSubtraction = AdaptiveMixtureOfGaussians(depthRaw8, maxGaussians=3, learningRate=0.2, decayRate=0.9, variance=100**2)
								backgroundModel = bgSubtraction.getModel()
								continue
							else:
								bgSubtraction.update(depthRaw8)

							backgroundModel = bgSubtraction.getModel()
							cv2.imshow("BG Model", backgroundModel/backgroundModel.max())
							foregroundMask = bgSubtraction.getForeground(thresh=100)
							''' Find people '''
							foregroundMask, _, _ = extract_people(depthRaw8, foregroundMask, minPersonPixThresh=5000, gradientFilter=True, gradThresh=15)
						else: 
							foregroundMask = None

						''' Write to file if there has been substantial change. '''
						if diff > frameDifferencePercent or time.clock()-prevFrameTime > 1/minFramerate or time.clock()-recentMotionTime < motionLagTime:
								if depthRaw8 != []:

										''' Logical time '''
										if second != prevSec:
												prevSecondCountMax = secondCount                                
												secondCount = 0
												prevSec = second
										else:
												secondCount = int(secondCount) + 1

										secondCount = str(secondCount)
										if len(ms_str) == 1:
												ms_str = '0' + ms_str
										if len(secondCount) == 1:
												secondCount = '0' + secondCount


										''' Keep track of framerate '''
										if prevTime != second:
												prevTime = second
												# print "FPS: "+ str(prevSecondCountMax) + " Diff: " + str(diff)[:4] + "%"
												print "FPS: {0:.1f} Diff: {1:.1f}%".format((1./(time.clock()-prevFrameTime)), diff)
												# print "FPS: "+ str(1./(time.clock()-prevFrameTime)) + " Diff: " + str(diff)[:4] + "%"
												prevFrame = currentFrame


										''' Create folder/file names '''
										if record:
											depthDir = baseDir+'depth/'+day+"/"+hour+"/"+minute+"/device_"+str(deviceID)
											depthName = depthDir + "/depth_"+day+"_"+hour+"_"+minute+"_"+second+"_"+secondCount+"_"+ms_str+".png"
											
											colorDir = baseDir+'color/'+day+"/"+hour+"/"+minute+"/device_"+str(deviceID)
											colorName = colorDir + "/color_"+day+"_"+hour+"_"+minute+"_"+second+"_"+secondCount+"_"+ms_str+".jpg"
											
											if getSkel:
												skelDir = baseDir+'skel/'+day+"/"+hour+"/"+minute+"/device_"+str(deviceID)
												usersName = skelDir + "/skel_"+day+"_"+hour+"_"+minute+"_"+second+"_"+secondCount+"_"+ms_str+"_.dat"
											else:
												skelDir = None
												usersName = None

											if anonomize:
												maskDir = baseDir+'mask/'+day+"/"+hour+"/"+minute+"/device_"+str(deviceID)										
												maskName = maskDir + "/mask_"+day+"_"+hour+"_"+minute+"_"+second+"_"+secondCount+"_"+ms_str+".jpg"
											else:
												maskDir = None
												maskName = None

											''' Create folders if they doesn't exist '''
											createDirectory(depthDir)
											createDirectory(colorDir)
											if getSkel:
												createDirectory(skelDir)
											if anonomize:
												createDirectory(maskDir)

											''' Save data '''
											save_frame(depthName, depthRaw8, colorName, colorRaw, usersName, users, maskName=maskName, mask=foregroundMask)

										prevFrameTime = time.clock()



								''' Display skeletons '''
								if viz and getSkel:
										for u_key in users.keys():
												u = users[u_key]
												pt = world2depth(u.com)
												w = 10
												d[pt[0]-w:pt[0]+w, pt[1]-w:pt[1]+w] = 255
												w = 3
												if u.tracked:
														print "Joints: ", len(u.jointPositions)
														for j in u.jointPositions.keys():
																pt = world2depth(u.jointPositions[j])
																d[pt[0]-w:pt[0]+w, pt[1]-w:pt[1]+w] = 200                                                        


								if viz:
										if 1:
												cv2.imshow("imageD", d)
												# if len(depthOld) > 10:
													# cv2.imshow("imageD10", depthOld[10]/float(depthOld[10].max()))
													# cv2.imshow("imageD0", depthOld[0]/float(depthOld[0].max()))
										if 0:
												# cv2.imshow("imageM", mask/float(mask.max()))
												cv2.imshow("imageM", colorRaw + (255-colorRaw)*(foregroundMask>0)[:,:,np.newaxis] + 50*(((foregroundMask)[:,:,np.newaxis])))
										if 1:
												cv2.imshow("imageC", colorRaw)
												# cv2.imshow("image", colorRaw + (255-colorRaw)*(foregroundMask>0)[:,:,np.newaxis] + 50*(((foregroundMask)[:,:,np.newaxis])))

										ret = cv2.waitKey(10)
										if ret >= 0:
												break






if __name__ == "__main__":

	parser = optparse.OptionParser(usage="Usage: python %prog [devID] [view toggle] [frameDifferencePercent]")
	parser.add_option('-d', '--device', dest='dev', type='int', default=1, help='Device# (eg 1,2,3)')
	parser.add_option('-v', '--view', dest='viz', action="store_true", default=False, help='View video while recording')	
	parser.add_option('-f', '--framediff', type='int', dest='frameDiffPercent', default=3, help='Frame Difference percent for dynamic framerate capture')
	parser.add_option('-s', '--skel', action="store_true", dest='skel', default=False, help='Turn on skeleton capture')
	parser.add_option('-a', '--anonomize', dest='anonomize', action="store_true", default=False, help='Turn on anonomization')
	parser.add_option('-i', '--dir', dest='dir', default=DIR, help='Save directory')
	parser.add_option('-r', '--stoprecording', dest='record', action="store_false", default=True, help="Don't record data")
	(options, args) = parser.parse_args()


	main(
		deviceID=options.dev, viz=options.viz,
	 	frameDifferencePercent=options.frameDiffPercent,
	 	baseDir=options.dir, getSkel=options.skel,
	 	anonomize=options.anonomize, record=options.record
	 	)



''' Multiprocessing '''

## Multiprocessing
# pool = Pool(processes = 1)
# # queue = SimpleQueue()
# processList = []
# processCount = 2



# Update process thread counts
# removeProcesses = []
# for i in xrange(len(processList)-1, -1, -1):
# 		if not processList[i].is_alive():
# 				removeProcesses.append(i)
# for i in removeProcesses:
# 		processList.pop(i)


''' Have it compress/save on another processor '''
# p = Process(target=save_frame, args=(depthName, depthRaw8, colorName, colorRaw, usersName, users))
# p.start()
# processList.append(p)
# print depthName, 1, depthRaw8.dtype

# queue.put((target=save_frame, args=(depthName, depthRaw8, colorName, colorRaw, usersName, users))
# print "Size: ", queue.qsize()
# pool.apply_async(save_frame, args=(depthName=depthName, depth=depthRaw8, colorName=colorName, color=colorRaw, usersName=usersName, users=users))
# pool.apply_async(save_frame, args=(depthName, depthRaw8, colorName, colorRaw, usersName, users))
# pool.apply_async(save_frame(depthName, depthRaw8, colorName, colorRaw, usersName, users))
# pool.join()

# if len(processList) < processCount:
#         p.start()
# else:
#         processList.append(p)

