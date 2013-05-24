
import os, time, sys, optparse
import numpy as np
import scipy.misc as sm
import Image
import cv2
from pyKinectTools.utils.RealtimeReader import *
from pyKinectTools.utils.Utils import createDirectory
from pyKinectTools.utils.SkeletonUtils import display_skeletons
import cPickle as pickle


DIR = os.path.expanduser('~')+'/Data/Kinect_Recorder/'


# @profile
def save_frame(depth_filename=None, depth=None, color_filename=None, color=None, userName=None, users=None, maskName=None, mask=None):
	''' Depth '''
	if depth_filename is not None:
		depth = depth[::2,::2]
		im = Image.fromarray(depth.astype(np.int32), 'I')
		# im = im.resize([320,240])
		im.save(depth_filename)

	'''Mask'''
	if mask is not None and maskName is not None:
			mask = sm.imresize(mask, [240,320], 'nearest')
			sm.imsave(maskName, mask)

	'''Color'''
	if color_filename is not None:
		# color = sm.imresize(color, [240,320,3], 'nearest')
		sm.imsave(color_filename, color[:,:,[2,1,0]])

	'''User'''
	if userName is not None:
		usersOut = {}
		for k in users.keys():
				usersOut[k] = users[k].toDict()

		with open(userName, 'wb') as outfile:
				pickle.dump(usersOut, outfile, protocol=pickle.HIGHEST_PROTOCOL)


# @profile
def main(device_id, record, base_dir, frame_difference_percent, get_skeleton, anonomize, viz, motion_lag_time=10, min_fps=0.5):
		'''
		---parameters---
		device_id
		record
		base_dir
		frame_difference_percent
		get_skeleton
		anonomize
		viz
		motion_lag_time
		min_fps
		'''

		'''------------ Setup Kinect ------------'''
		''' Physical Kinect '''
		depthDevice = RealTimeDevice(device=device_id, get_depth=True, get_color=True, get_skeleton=get_skeleton)
		depthDevice.start()

		maxFramerate = 100
		minFramerate = min_fps
		recentMotionTime = time.time()
		imgStoreCount = 100

		''' ------------- Main -------------- '''

		''' Setup time-based stuff '''
		prevTime = 0
		prevFrame = 0
		prevFrameTime = 0
		currentFrame = 0
		ms = time.time()
		prevFPSTime = time.time()
		diff = 0

		prevSec = 0;
		secondCount = 0
		prevSecondCountMax = 0

		''' Ensure base folder is there '''
		createDirectory(base_dir)
		skel_dir = None
		skel_filename = None
		maskDir = None
		maskName = None

		depthOld = []
		backgroundModel = None

		while 1:
				# try:
				if 1:
						depthDevice.update()
						colorRaw = depthDevice.colorIm
						depthRaw = depthDevice.depthIm
						users = depthDevice.users
						skel = None

						if len(depthOld) == imgStoreCount:
								depthOld.pop(0)

						''' If framerate is too fast then skip '''
						''' Keep this after update to ensure fast enough kinect refresh '''
						if (time.time() - float(ms))*1000 < 1000.0/maxFramerate:
								continue

						''' Get new time info '''
						time_ = time.localtime()
						day = str(time_.tm_yday)
						hour = str(time_.tm_hour)
						minute = str(time_.tm_min)
						second = str(time_.tm_sec)
						ms = str(time.time())
						ms_str = ms

						''' Look at how much of the image has changed '''
						if depthOld != []:
								mask = (depthRaw != 0) * (depthOld[0] != 0)
								diff = (((depthRaw - depthOld[0]).astype(np.int16) > 200)*mask).sum() / (float(mask.sum()) * 100.)

								''' We want to watch all video for at least 5 seconds after we seen motion '''
								''' This prevents problems where there is small motion that doesn't trigger the motion detector '''
								if diff > frame_difference_percent:
										recentMotionTime = time.time()
						depthOld.append(depthRaw)

						if anonomize:
							'''Background model'''
							if backgroundModel is None:
								bgSubtraction = AdaptiveMixtureOfGaussians(depthRaw, maxGaussians=3, learningRate=0.2, decayRate=0.9, variance=100**2)
								backgroundModel = bgSubtraction.getModel()
								continue
							else:
								bgSubtraction.update(depthRaw)

							backgroundModel = bgSubtraction.getModel()
							cv2.imshow("BG Model", backgroundModel/backgroundModel.max())
							foregroundMask = bgSubtraction.get_foreground(thresh=100)
							''' Find people '''
							foregroundMask, _, _ = extract_people(depthRaw, foregroundMask, minPersonPixThresh=5000, gradientFilter=True, gradThresh=15)
						else:
							foregroundMask = None

						''' Write to file if there has been substantial change. '''
						if diff > frame_difference_percent or time.time()-prevFrameTime > 1/minFramerate or time.time()-recentMotionTime < motion_lag_time:
								currentFrame += 1
								if depthRaw != []:

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
												print "FPS: {0:.1f} Diff: {1:.1f}%".format((currentFrame-prevFrame)/(time.time()-prevFPSTime), diff)
												prevFrame = currentFrame
												prevFPSTime = time.time()

										''' Create folder/file names '''
										if record:
											if 1:
												''' Version 1. 2012 '''
												ms_str = str(ms)[str(ms).find(".")+1:]
												depth_dir = base_dir+'depth/'+day+"/"+hour+"/"+minute+"/device_"+str(device_id)
												depth_filename = depth_dir + "/depth_"+day+"_"+hour+"_"+minute+"_"+second+"_"+secondCount+"_"+ms_str+".png"

												color_dir = base_dir+'color/'+day+"/"+hour+"/"+minute+"/device_"+str(device_id)
												color_filename = color_dir + "/color_"+day+"_"+hour+"_"+minute+"_"+second+"_"+secondCount+"_"+ms_str+".jpg"

												if get_skeleton:
													skel_dir = base_dir+'skel/'+day+"/"+hour+"/"+minute+"/device_"+str(device_id)
													skel_filename = skel_dir + "/skel_"+day+"_"+hour+"_"+minute+"_"+second+"_"+secondCount+"_"+ms_str+"_.dat"
												if anonomize:
													maskDir = base_dir+'mask/'+day+"/"+hour+"/"+minute+"/device_"+str(device_id)
													maskName = maskDir + "/mask_"+day+"_"+hour+"_"+minute+"_"+second+"_"+secondCount+"_"+ms_str+".jpg"
											else:
												''' Version 2. April 2013 '''
												base_sub_dir = "{:s}device_{:d}/{:s}/{:s}/{:s}".format(base_dir,device_id,day,hour,minute)
												depth_dir = "{:s}/depth".format(base_sub_dir)
												color_dir = "{:s}/color".format(base_sub_dir)
												depth_filename = "{:s}/depth_{:s}_{:s}_{:s}_{:s}_{:s}_{:s}_.png".format(depth_dir,day,hour,minute,second,secondCount,ms_str)
												color_filename = "{:s}/color_{:s}_{:s}_{:s}_{:s}_{:s}_{:s}_.jpg".format(color_dir,day,hour,minute,second,secondCount,ms_str)

												if get_skeleton:
													skel_dir = "{:s}/skel".format(base_sub_dir)
													skel_filename = "{:s}/skel_{:s}_{:s}_{:s}_{:s}_{:s}_{:s}_.dat".format(skel_dir,day,hour,minute,second,secondCount,ms_str)

												if anonomize:
													maskDir = "{:s}/mask".format(base_sub_dir)
													maskName = "{:s}/mask_{:s}_{:s}_{:s}_{:s}_{:s}_{:s}_.jpg".format(skel_dir,day,hour,minute,second,secondCount,ms_str)

											''' Create folders if they doesn't exist '''
											createDirectory(depth_dir)
											createDirectory(color_dir)
											if get_skeleton:
												createDirectory(skel_dir)
											if anonomize:
												createDirectory(maskDir)

											''' Save data '''
											save_frame(depth_filename, depthRaw, color_filename, colorRaw, skel_filename, users, maskName=maskName, mask=foregroundMask)

										prevFrameTime = time.time()

								''' Display skeletons '''
								if viz:
									d = np.array(depthRaw)
									d /= (np.nanmin([d.max(), 2**16])/256.0)
									d = d.astype(np.uint8)

									if get_skeleton:
										for u_key in users.keys():
												u = users[u_key]
												pt = skel2depth(np.array([u.com]))[0]
												w = 10
												d[pt[1]-w:pt[1]+w, pt[0]-w:pt[0]+w] = 255
												w = 3
												if u.tracked:
														pts = skel2depth(np.array(u.jointPositions.values()), d.shape)
														d = display_skeletons(d, pts, (100,0,0), skel_type='Kinect')


									if 1:
											cv2.imshow("imageD", d)
									if 0:
											cv2.imshow("imageM", colorRaw + (255-colorRaw)*(foregroundMask>0)[:,:,np.newaxis] + 50*(((foregroundMask)[:,:,np.newaxis])))
									if 1:
											cv2.imshow("imageC", colorRaw)

									ret = cv2.waitKey(10)
									if ret >= 0:
											break


if __name__ == "__main__":

	parser = optparse.OptionParser(usage="Usage: python %prog [devID] [view toggle] [frame_difference_percent]")
	parser.add_option('-d', '--device', dest='dev', type='int', default=1, help='Device# (eg 1,2,3)')
	parser.add_option('-v', '--view', dest='viz', action="store_true", default=False, help='View video while recording')
	parser.add_option('-f', '--framediff', type='int', dest='frameDiffPercent', default=0, help='Frame Difference percent for dynamic framerate capture')
	parser.add_option('-s', '--skel', action="store_true", dest='skel', default=False, help='Turn on skeleton capture')
	parser.add_option('-a', '--anonomize', dest='anonomize', action="store_true", default=False, help='Turn on anonomization')
	parser.add_option('-i', '--dir', dest='dir', default=DIR, help='Save directory')
	parser.add_option('-r', '--stoprecording', dest='record', action="store_false", default=True, help="Don't record data")
	(options, args) = parser.parse_args()

	main(
		device_id=options.dev, viz=options.viz,
	 	frame_difference_percent=options.frameDiffPercent,
	 	base_dir=options.dir, get_skeleton=options.skel,
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
# p = Process(target=save_frame, args=(depth_filename, depthRaw, color_filename, colorRaw, skel_filename, users))
# p.start()
# processList.append(p)
# print depth_filename, 1, depthRaw.dtype

# queue.put((target=save_frame, args=(depth_filename, depthRaw, color_filename, colorRaw, skel_filename, users))
# print "Size: ", queue.qsize()
# pool.apply_async(save_frame, args=(depth_filename=depth_filename, depth=depthRaw, color_filename=color_filename, color=colorRaw, skel_filename=skel_filename, users=users))
# pool.apply_async(save_frame, args=(depth_filename, depthRaw, color_filename, colorRaw, skel_filename, users))
# pool.apply_async(save_frame(depth_filename, depthRaw, color_filename, colorRaw, skel_filename, users))
# pool.join()

# if len(processList) < processCount:
#         p.start()
# else:
#         processList.append(p)

