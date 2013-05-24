'''
Main file for displaying depth/color/skeleton information and extracting features
'''

import os
from time import time
import cPickle as pickle
import numpy as np
import scipy.misc as sm
import scipy.ndimage as nd
import traceback

from pyKinectTools.utils.Utils import createDirectory
from pyKinectTools.utils.SkeletonUtils import  kinect_to_msr_skel, plotUsers
from pyKinectTools.utils.VideoViewer import VideoViewer
from pyKinectTools.utils.DepthUtils import CameraModel, get_kinect_transform
from pyKinectTools.utils.MultiCameraUtils import multiCameraTimeline, formatFileString
from pyKinectTools.algs.BackgroundSubtraction import AdaptiveMixtureOfGaussians, MedianModel, fill_image, extract_people, StaticModel
from pyKinectTools.dataset_readers.BasePlayer import BasePlayer
import pyKinectTools.configs

import matplotlib as mp
from pylab import *
colormap = mp.cm.jet
# colormap = mp.cm.hot
colormap._init()

# import cv2
vv = VideoViewer()

''' Debugging '''
from IPython import embed

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

class KinectPlayer(BasePlayer):

	def __init__(self, device=0, **kwargs):

		super(KinectPlayer, self).__init__(**kwargs)

		self.dev = device
		self.deviceID = "device_{0:d}".format(self.dev)

		# Get calibration
		self.camera_model = CameraModel(pyKinectTools.configs.__path__[0]+"/Kinect_Color_Param.yml")
		self.kinect_transform = get_kinect_transform(pyKinectTools.configs.__path__[0]+"/Kinect_Transformation.txt")
		self.camera_model.set_transform(self.kinect_transform)

		self.ret = 0
		self.day_dirs = os.listdir(self.base_dir+'depth/')
		self.day_dirs = [x for x in self.day_dirs if x[0]!='.']
		self.day_dirs.sort(key=lambda x: int(x))
		self.hour_index = 0
		self.minute_index = 0
		self.second_new = 0

		self.play_speed = 1
		self.new_date_entered = False
		self.framerate = 0
		self.frame_prev = 0
		self.frame_prev_time = time()

		self.player = self.run()
		self.next()

	def update_background(self):
		'''Background model'''
		self.bgSubtraction.update(self.depthIm)
		self.backgroundModel = self.bgSubtraction.getModel()
		self.foregroundMask = self.bgSubtraction.get_foreground(thresh=50)
		self.mask = self.bgSubtraction.get_foreground(thresh=50)


	def next(self, frames=1):
		'''
		frames : skip (this-1) frames
		'''
		try:
		# if 1:
			for i in xrange(frames):
				self.player.next()
				if self.enable_bg_subtraction:
					self.update_background()
			return True
		except:
			traceback.print_exc(file=sys.stdout)
			print 'Error getting next frame. Could be end of file'
			return False

	# ''' Or get CoM + orientation '''
	# def calculate_basic_features(self):
		# userCount = len(userBoundingBoxes)
		# for i in xrange(userCount):
		# 	userBox = userBoundingBoxes[i]
		# 	userMask = foregroundMask==i+1
		# 	com, ornBasis = calculateBasicPose(depthIm, userMask)
		# 	coms.append(com)
		# 	orns.append(ornBasis[1])
		# 	allFeatures.append({'com':com, "orn":ornBasis, 'time':timestamp})



	def visualize(self, color=True, depth=True, skel=False, text=False, colorize=False, depth_bounds=[1000,4000]):
		# depth, color, mask
		self.tmpSecond = self.depthFile.split("_")[-3]
		if len(self.tmpSecond) == 0:
			self.tmpSecond = '0'+self.tmpSecond

		# ''' Find people '''
		if skel:
			self.ret = plotUsers(self.depthIm, self.users)
		if depth and self.get_depth is not None:
			depthIm = self.depthIm
			if depth_bounds is not None:
				depthIm = (self.depthIm-depth_bounds[0])/float(depth_bounds[1]-depth_bounds[0])
			if colorize:
				tmp = depthIm.reshape(-1)
				# Normalize by min/max
				if depth_bounds is None:
					min_ = tmp[tmp>0].min()
					max_ = float(tmp.max())
					tmp[tmp>0] = (tmp[tmp>0]-min_) / (max_-min_)
				else:
					tmp *= 255.
				# Recolor
				tmp = colormap._lut[tmp.astype(np.uint8)]
				depthIm = tmp.reshape([self.depthIm.shape[0], self.depthIm.shape[1], 4])[:,:,:3]
				depthIm[self.depthIm==0] *= 0
			vv.imshow("Depth "+self.deviceID, depthIm)
			if text:
				text_tmp = "Day {0} Time {1}:{2:02d}:{3:02d}".format(self.day_dir, self.hour_dir, int(self.minute_dir), int(self.tmpSecond))
				vv.putText("Depth "+self.deviceID, text_tmp, (5,self.depthIm.shape[0]-20), size=15)
				vv.putText("Depth "+self.deviceID, "Play speed: "+str(self.play_speed)+"x", (5,15), size=15)


		if color and self.get_color is not None:
			vv.imshow("Color "+self.deviceID, self.colorIm)
			if text:
				text_tmp = "Day {0} Time {1}:{2:02d}:{3:02d}".format(self.day_dir, self.hour_dir, int(self.minute_dir), int(self.tmpSecond))
				vv.putText("Color "+self.deviceID, text_tmp, (5,self.colorIm.shape[0]-20), size=15)
				vv.putText("Color "+self.deviceID, "Play speed: "+str(self.play_speed)+"x", (5,15), size=15)

		self.playback_control()

	def sync_cameras(self, camera):
		'''
		Syncs multiple cameras together (only accurate within sub-second)
		---Parameters---
		camera: should be instance of KinectPlayer
		'''
		# Sync the first and second videos
		if int(camera.second) > int(self.second) or camera.day_dir != self.day_dir or camera.hour_dir != self.hour_dir or camera.minute_dir != self.minute_dir:
			self.new_date_entered = True
			self.day_new = camera.day_dir
			self.hour_new = camera.hour_dir
			self.minute_new = camera.minute_dir
			self.second_new = camera.second
			self.frame_id = camera.frame_id
			self.next()
		elif int(camera.second) < int(self.second):
			pass
		else:
			self.next()

		if camera.play_speed > self.play_speed:
			self.play_speed = camera.play_speed
		elif camera.play_speed < self.play_speed:
			camera.play_speed = self.play_speed

	def playback_control(self):
		''' Playback control: Look at keyboard input '''
		self.ret = vv.waitKey()

		if self.frame_id - self.frame_prev > 0:
			self.framerate = (self.frame_id - self.frame_prev) / (time() - self.frame_prev_time)
		self.frame_prev = self.frame_id
		self.frame_prev_time = time()

		self.new_date_entered = False
		if self.ret > 0:
			# player_controls(ret)
			# print "Ret is",self.ret

			if self.ret == keys_ESC:
				exit
			elif self.ret == keys_space:
				print "Enter the following into the command line"
				tmp = raw_input("Enter date: ")
				self.day_new = tmp
				tmp = raw_input("Enter hour: ")
				self.hour_new = tmp
				tmp = raw_input("Enter minute: ")
				self.minute_new = tmp

				print "New date:", self.day_new, self.hour_new, self.minute_new

				self.new_date_entered = True
				# break

			elif self.ret == keys_down_arrow:
				self.play_speed = 0
				print "Pause"
			elif self.ret == keys_left_arrow:
				self.play_speed -= 1
				print "Decrease speed"
			elif self.ret == keys_right_arrow:
				self.play_speed += 1
				print "Increase speed"
			elif self.ret == keys_i:
					embed()
			elif self.ret == keys_frame_left:
				self.frame_id -= 1
			elif self.ret == keys_frame_right:
				self.frame_id += 1
			elif self.ret == keys_help:
				display_help()


	def run(self):

		self.day_index = 0
		while self.day_index < len(self.day_dirs):

			if self.new_date_entered:
				try:
					self.day_index = self.day_dirs.index(self.day_new)
				except:
					print "Day not found"
					self.day_index = 0

			self.day_dir = self.day_dirs[self.day_index]

			hour_dirs = os.listdir(self.base_dir+'depth/'+self.day_dir)
			hour_dirs = [x for x in hour_dirs if x[0]!='.']
			hour_dirs.sort(key=lambda x: int(x))

			'''Hours'''
			''' Check for new Hours index '''
			if not self.new_date_entered:
				if self.play_speed >= 0  and self.ret != keys_frame_left:
					self.hour_index = 0
				else:
					self.hour_index = len(hour_dirs)-1
			else:
				try:
					self.hour_index = hour_dirs.index(self.hour_new)
				except:
					print "Hour was not found"
					self.hour_index = 0

			while self.hour_index < len(hour_dirs):
				self.hour_dir = hour_dirs[self.hour_index]

				self.minute_dirs = os.listdir(self.base_dir+'depth/'+self.day_dir+'/'+self.hour_dir)
				self.minute_dirs = [x for x in self.minute_dirs if x[0]!='.']
				self.minute_dirs.sort(key=lambda x: int(x))

				'''Minutes'''
				''' Check for new minute index '''
				if not self.new_date_entered:
					if self.play_speed >= 0  and self.ret != keys_frame_left:
						self.minute_index = 0
					else:
						self.minute_index = len(self.minute_dirs)-1
				else:
					try:
						self.minute_index = self.minute_dirs.index(self.minute_new)
					except:
						print "Minute was not found"
						self.minute_index = 0

				''' Loop through this minute '''
				while self.minute_index < len(self.minute_dirs):
					self.minute_dir = self.minute_dirs[self.minute_index]

					# Prevent from reading hidden files
					if self.minute_dir[0] == '.':
						continue

					depth_files = []
					skelFiles = []

					# Make sure it's a real directory
					if not os.path.isdir(self.base_dir+'depth/'+self.day_dir+'/'+self.hour_dir+'/'+self.minute_dir+'/'+self.deviceID):
						continue

					''' Sort files '''
					if self.get_depth:
						depthTmp = os.listdir(self.base_dir+'depth/'+self.day_dir+'/'+self.hour_dir+'/'+self.minute_dir+'/'+self.deviceID)
						depthTmp = [x for x in depthTmp if x[0]!='.']
						# embed()
						tmpSort = [int(x.split('_')[-3])*100 + int(formatFileString(x.split('_')[-2])) for x in depthTmp]
						depthTmp = np.array(depthTmp)[np.argsort(tmpSort)].tolist()
						depth_files.append([x for x in depthTmp if x.find('.png')>=0])
					if self.get_skeleton:
						skelTmp = os.listdir(self.base_dir+'skel/'+self.day_dir+'/'+self.hour_dir+'/'+self.minute_dir+'/'+self.deviceID)
						tmpSort = [int(x.split('_')[-4])*100 + int(formatFileString(x.split('_')[-3])) for x in skelTmp]
						skelTmp = np.array(skelTmp)[np.argsort(tmpSort)].tolist()
						skelFiles.append([x for x in skelTmp if x.find('.dat')>=0])

					if len(depth_files) == 0:
						continue

					if self.play_speed >= 0 and self.ret != keys_frame_left:
						self.frame_id = 0
					else:
						self.frame_id = len(depth_files[0])-1

					# Seconds
					if self.new_date_entered:
						fileSeconds = [x.split("_")[4] for x in depth_files[0]]
						self.frame_id = next(i for i in xrange(len(fileSeconds)) if fileSeconds[i] == self.second_new)

					while self.frame_id < len(depth_files[0]):
						self.depthFile = depth_files[0][self.frame_id]
						self.second = self.depthFile.split("_")[4]
						''' Load Depth '''
						if self.get_depth:
							self.depthIm = sm.imread(self.base_dir+'depth/'+self.day_dir+'/'+self.hour_dir+'/'+self.minute_dir+'/'+self.deviceID+'/'+self.depthFile)
							self.depthIm = np.array(self.depthIm, dtype=np.uint16)
						''' Load Color '''
						if self.get_color:
							colorFile = 'color_'+self.depthFile[6:-4]+'.jpg'
							self.colorIm = sm.imread(self.base_dir+'color/'+self.day_dir+'/'+self.hour_dir+'/'+self.minute_dir+'/'+self.deviceID+'/'+colorFile)
							# self.colorIm = self.colorIm[:,:,[1,0,2]]
							# self.colorIm_g = skimage.img_as_ubyte(skimage.color.rgb2gray(self.colorIm))

						''' Load Skeleton Data '''
						if self.get_skeleton:
							skelFile = 'skel_'+self.depthFile[6:-4]+'_.dat'
							if os.path.isfile(self.base_dir+'skel/'+self.day_dir+'/'+self.hour_dir+'/'+self.minute_dir+'/'+self.deviceID+'/'+skelFile):
								skel_filename = self.base_dir+'skel/'+self.day_dir+'/'+self.hour_dir+'/'+self.minute_dir+'/'+self.deviceID+'/'+skelFile
								try:
									# with open(skel_filename, 'rb') as inFile:
										# self.users = pickle.load(inFile)
									# self.users = [x for x in pickle.load(open(skel_filename)).values()]
									self.users = np.array([x['jointPositions'].values() for x in pickle.load(open(skel_filename)).values()])
								except:
									print 'Error loading skeleton. PyOpenNI may not be installed'
							else:
								print "No user file:", skelFile
							# coms = [x['com'] for x in self.users if x['com'][2] > 0.0]
							# coms = [self.users[x]['com'] for x in self.users.keys() if self.users[x]['com'][2] > 0.0]
							jointCount = 0
							for i in self.users:
								user = i#self.users[i]
						timestamp = self.depthFile[:-4].split('_')[1:] # Day, hour, minute, second, millisecond, Frame number in this second
						self.depthIm = self.depthIm.astype(np.float).clip(0,4500)
						# self.depthIm = np.minimum(self.depthIm.astype(np.float), 5000)
						if self.fill_images:
							fill_image(self.depthIm)

						yield

						self.frame_id += self.play_speed

						# End seconds
						if self.ret == keys_ESC or self.new_date_entered:
							break
						if self.frame_id >= len(depth_files[0]):
							self.minute_index += 1
						elif self.frame_id < 0:
							self.minute_index -= 1
							break

					# End hours
					if self.ret == keys_ESC or self.new_date_entered:
						break

					if self.minute_index >= len(self.minute_dirs):
						self.hour_index += 1
					elif self.minute_index < 0:
						self.hour_index -= 1
						break

				# End days
				if self.ret == keys_ESC:
					break
				if self.new_date_entered:
					break

				if self.hour_index >= len(hour_dirs):
					self.day_index += 1
				elif self.hour_index < 0:
					self.day_index -= 1

				if self.day_index < 0:
					self.day_index = 0


			if self.ret == keys_ESC or self.day_index > len(self.day_dirs):
				break



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

