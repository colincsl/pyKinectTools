'''
Main file for displaying depth/color/skeleton information and extracting features
'''

import os
# import optparse
from time import time
import cPickle as pickle
import numpy as np
import scipy.misc as sm
import scipy.ndimage as nd
import skimage
# from skimage import feature, color

from pyKinectTools.utils.Utils import createDirectory
from pyKinectTools.utils.SkeletonUtils import  kinect_to_msr_skel, plotUsers
from pyKinectTools.utils.VideoViewer import VideoViewer
# from pyKinectTools.utils.DepthUtils import world2depth, depthIm2XYZ
from pyKinectTools.utils.MultiCameraUtils import multiCameraTimeline, formatFileString
# from pyKinectTools.utils.FeatureUtils import saveFeatures, loadFeatures, learnICADict, learnNMFDict, displayComponents
from pyKinectTools.algs.BackgroundSubtraction import AdaptiveMixtureOfGaussians, MedianModel, fillImage, extract_people, StaticModel
from pyKinectTools.algs.FeatureExtraction import calculateBasicPose, computeUserFeatures, computeFeaturesWithSkels

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

class KinectPlayer:

	def __init__(self, base_dir='./', device=0, get_depth=True, get_color=False, 
				get_skeleton=False, bg_subtraction=False, fill_images=False):
		
		self.enable_bg_subtraction = bg_subtraction
		self.fill_images = fill_images
		self.base_dir = base_dir
		self.dev = device
		self.deviceID = "device_{0:d}".format(self.dev)

		self.get_depth = get_depth
		self.get_color = get_color
		self.get_skeleton =get_skeleton

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

		self.depthIm = None
		self.colorIm = None
		self.users = None
		self.backgroundModel = None
		self.foregroundMask = None
		self.prevcolorIm = None

		self.player = self.run()
		self.next()

	def update_background(self):
		'''Background model'''
		if self.backgroundModel is None:
			self.bgSubtraction = StaticModel(self.depthIm)
			# self.bgSubtraction = AdaptiveMixtureOfGaussians(self.depthIm, maxGaussians=5, learningRate=0.01, decayRate=0.001, variance=300**2)
			# self.bgSubtraction = MedianModel(self.depthIm)
			self.backgroundModel = self.bgSubtraction.getModel()
			return
		
		self.bgSubtraction.update(self.depthIm)
		self.backgroundModel = self.bgSubtraction.getModel()
		self.foregroundMask = self.bgSubtraction.getForeground(thresh=500)

	def next(self, frames=1):
		'''
		frames : skip (this-1) frames
		'''
		try:
			for i in xrange(frames):
				self.player.next()
				if self.enable_bg_subtraction:
					self.update_background()
			return True
		except:
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


	def get_person(self, edge_thresh=200):
		depth = self.depthIm.astype(np.int16)
		mask = self.foregroundMask

		labelIm, boxes, labels,_ = extract_people(depth*mask, minPersonPixThresh=5000, gradThresh=edge_thresh)
		label_sizes = [np.sum(labelIm[boxes[i]]==i+1) for i in range(len(labels))]
		labels_c = [l+1 for l,lab in zip(range(len(labels)), labels) if 5000 < label_sizes[l]]
		labels_sizes_c = [label_sizes[l] for l,lab in zip(range(len(labels)), labels) if 5000 < label_sizes[l]]

		if len(labels_sizes_c) > 0:
			max_ind = np.argmax(labels_sizes_c)
			mask_new = labelIm==labels_c[max_ind]
			depth[-mask_new] = 0
			return depth
		else:
			return -1


	def get_n_skeletons(self, n):
		'''
		In MSR format
		'''
		if n == -1:
			n = np.inf

		depthIms = []
		skels_world = []
		
		try:
			while depthIms == [] or len(depthIms) < n:
				self.next()
				self.update_background()
				if len(self.users.keys()) > 0:
					s = self.users.keys()[0]
					pts = np.array(self.users[s]['jointPositions'].values())
					if np.all(pts[0] != -1):
						pts = kinect_to_msr_skel(pts)
						skels_world += [pts]

						# depth = self.depthIm.astype(np.int16)
						# mask = self.foregroundMask
						depth = self.get_person()
						depthIms += [depth]
						# colorIms += [color]
		except:
			print "No skeletons remaining."
		
		return depthIms, skels_world

	def visualize(self, show_skel=False):
		# depth, color, mask
		self.tmpSecond = self.depthFile.split("_")[-3]
		if len(self.tmpSecond) == 0:
			self.tmpSecond = '0'+self.tmpSecond

		# ''' Find people '''
		if show_skel:
			self.ret = plotUsers(self.depthIm, self.users)

		if self.get_depth:
			vv.imshow("Depth "+self.deviceID, (self.depthIm-1000)/5000.)
			vv.putText("Depth "+self.deviceID, "Day "+self.day_dir+" Time "+self.hour_dir+":"+self.minute_dir+":"+self.tmpSecond, (5,220), size=15)					
			vv.putText("Depth "+self.deviceID, "Play speed: "+str(self.play_speed)+"x", (5,15), size=15)													
			# vv.putText("Depth "+self.deviceID, str(int(self.framerate))+" fps", (275,15), size=15)													
			# vv.imshow("Depth", self.depthIm/6000.)
			
		if self.get_color:
			vv.imshow("Color "+self.deviceID, self.colorIm)
			# vv.putText("Color "+self.deviceID, self.colorIm, "Day "+self.day_dir+" Time "+self.hour_dir+":"+self.minute_dir+" Dev#"+str(self.dev), (10,220))					
			# vv.imshow("Color", self.colorIm)
			# vv.imshow("Color", self.colorIm)
			# if self.get_mask:
			# vv.imshow("I", self.colorIm*self.foregroundMask[:,:,np.newaxis])
			# vv.imshow("I_masked", self.colorIm + (255-self.colorIm)*(((self.foregroundMask)[:,:,np.newaxis])))

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
							self.colorIm = self.colorIm[:,:,[2,1,0]]
							# self.colorIm_g = skimage.img_as_ubyte(skimage.color.rgb2gray(self.colorIm))
							# self.colorIm_lab = skimage.color.rgb2lab(self.colorIm).astype(np.uint8)

						''' Load Skeleton Data '''
						if self.get_skeleton:
							skelFile = 'skel_'+self.depthFile[6:-4]+'_.dat'
							if os.path.isfile(self.base_dir+'skel/'+self.day_dir+'/'+self.hour_dir+'/'+self.minute_dir+'/'+self.deviceID+'/'+skelFile):
								with open(self.base_dir+'skel/'+self.day_dir+'/'+self.hour_dir+'/'+self.minute_dir+'/'+self.deviceID+'/'+skelFile, 'rb') as inFile:
									self.users = pickle.load(inFile)				
							else:
								print "No user file:", skelFile
							coms = [self.users[x]['com'] for x in self.users.keys() if self.users[x]['com'][2] > 0.0]
							jointCount = 0
							for i in self.users.keys():
								user = self.users[i]
						timestamp = self.depthFile[:-4].split('_')[1:] # Day, hour, minute, second, millisecond, Frame number in this second
						self.depthIm = self.depthIm.astype(np.float).clip(0,3500)
						# self.depthIm = np.minimum(self.depthIm.astype(np.float), 5000)
						
						if self.fill_images:
							fillImage(self.depthIm)

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

