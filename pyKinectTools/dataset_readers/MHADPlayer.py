import os
import cv2
import numpy as np
import scipy.misc as sm
import png
import itertools as it
from skimage.morphology import closing, erosion, opening
from skimage.draw import circle
from pyKinectTools.utils.DepthUtils import CameraModel, skel2depth, depthIm_to_colorIm, world2depth, world2rgb, get_kinect_transform #depthIm2XYZ, depth2world
from pyKinectTools.utils.SkeletonUtils import mhad_to_kinect_skel, display_skeletons
from pyKinectTools.dataset_readers.BasePlayer import BasePlayer
from pyKinectTools.algs.BackgroundSubtraction import fill_image, StaticModel, extract_people
# from pyKinectTools.utils.VideoViewer import VideoViewer
# vv = VideoViewer()

from IPython import embed

'''
For more information about this dataset see: http://tele-immersion.citris-uc.org/berkeley_mhad/
'''

# def get_kinect_transform(filename):
# 	transform = np.eye(4)
# 	with open(filename, 'r') as f:
# 		# Rotation
# 		line = f.readline().split()
# 		line[0] = line[0][3:]
# 		transform[:3,:3] = np.array(line, np.float).reshape([3,3])
# 		# Translation
# 		line = f.readline().split()
# 		line[0] = line[0][3:]
# 		transform[:3,-1] = np.array(line, np.float)
# 	return transform

def create_folder_names(device=1, subjects=[1], actions=[1], reps=[1]):
	folders = []
	for s in subjects:
		for a in actions:
			for r in reps:
				folders += ["Kin{:02d}/S{:02d}/A{:02d}/R{:02d}/".format(device,s,a, r)]
	return folders

def create_mocap_filenames(subjects=[1], actions=[1], reps=[1]):
	folders = []
	for s in subjects:
		for a in actions:
			for r in reps:
				folders += ["moc_s{:02d}_a{:02d}_r{:02d}.txt".format(s,a,r)]
	return folders

def read_pnm(filename):
   fd = open(filename,'rb')
   format, width, height, samples, maxval = png.read_pnm_header( fd )
   pixels = np.fromfile( fd, dtype='u1' if maxval < 256 else '>u2' )
   return pixels.reshape(height,width,samples)

def read_depth_ims(folder):
	files = os.listdir(folder)
	files = [f for f in files if f.find('depth')>=0]
	framecount = len(files)

	ims = np.empty([480,640,framecount], dtype=np.int16)
	rows, cols = [480,640]

	for i,filename in enumerate(files):
		im = read_pnm(folder+filename).reshape([480,640])
		ims[:,:,i] = im

	return ims

def read_color_ims(folder):
	''' Extracts color images from the MSR Daily Activites dataset
	---Parameters---
	folder : folder name for video
	data_type : 'depth' or 'color'
	'''
	files = os.listdir(folder)
	files = [f for f in files if f.find('color')>=0]
	framecount = len(files)

	ims = np.empty([480,640,3,framecount], dtype=np.uint8)
	rows, cols, depth = [480,640,3]

	for i,filename in enumerate(files):
		im = sm.imread(folder+filename)
		ims[:,:,:,i] = im

	return ims

def read_depth_timestamps(filename):

	data = np.loadtxt(filename)
	frames = data[:,0]
	if data.shape[1] == 2:
		times = data[:,1]
	else:
		times = data[:,1] + data[:,2]/1000000
	# timestamps = np.vstack([frames, times])
	timestamps = np.array(times)

	return timestamps

def read_mocap(filename):
	# files = os.listdir(folder)

	data = np.loadtxt(filename)
	frames = data[:,129]
	times = data[:,130]
	data = data[:,:129]
	markers = data.reshape([-1, 43,3])
	mocap = {'frames':frames, 'times':times, 'markers':markers}

	return mocap


class MHADPlayer(BasePlayer):

	def __init__(self, kinect=1, subjects=[1], actions=[1], reps=[1], **kwargs):
		super(MHADPlayer, self).__init__(**kwargs)

		# Settings
		self.deviceID = "Kinect: {:d}".format(kinect)
		self.repetitions = reps
		# Get data filenames
		self.kinect_folder_names = create_folder_names(kinect, subjects, actions, reps)
		self.mocap_filenames = create_mocap_filenames(subjects, actions, reps)
		# Get calibration
		print self.base_dir
		self.camera_model = CameraModel(self.base_dir+"Calibration/camcfg_k{:02d}.yml".format(kinect))
		self.kinect_transform = get_kinect_transform(self.base_dir+"Calibration/RwTw_k{:02d}.txt".format(kinect))
		self.camera_model.set_transform(self.kinect_transform)
		# Setup background model
		self.background_model = StaticModel()
		self.set_background(sm.imread(self.base_dir+"Kinect/Kin{:02d}/background_model.png".format(kinect)).clip(0, 4500))
		self.mask = 1
		# Initialize
		self.player = self.run()
		self.next(1)

	def next(self, frames=1):
		'''
		frames : skip (this-1) frames
		'''
		# Update frame
		try:
		# if 1:
			for i in range(frames):
				self.player.next()
			return True
		except:
			print "Done playing video"
			return False

	def run(self):

		# Read data from new file
		while len(self.kinect_folder_names) > 0:
			print 'New video:', self.kinect_folder_names[-1]
			# Load videos
			self.depth_stack = read_depth_ims(self.base_dir + 'Kinect/' + self.kinect_folder_names[-1])
			self.color_stack = read_color_ims(self.base_dir + 'Kinect/' + self.kinect_folder_names[-1])
			# embed()
			self.mocap_stack = read_mocap(self.base_dir+'Mocap/OpticalData/'+self.mocap_filenames[-1])
			self.skel_stack = self.mocap_stack['markers']
			# Load timestamps
			time_tmp = self.kinect_folder_names[-1].split('/')
			if time_tmp[0] == '':
				time_tmp = time_tmp[1:]
			kinect_timestamp_name = "time_stamps_{:s}_{:s}_{:s}_{:s}.txt".format('kin01', time_tmp[1], time_tmp[2], time_tmp[3])
			# print kinect_timestamp_name, time_tmp
			# kinect_timestamp_name = "time_stamps_{:s}_{:s}_{:s}_{:s}.txt".format(time_tmp[1], time_tmp[2], time_tmp[3], time_tmp[4])
			self.depth_timestamps = read_depth_timestamps(self.base_dir + 'Kinect/Time_stamps/' + kinect_timestamp_name.lower())
			self.mocap_timestamps = self.mocap_stack['times']

			self.mocap_filenames.pop()
			self.kinect_folder_names.pop()
			framecount = self.depth_stack.shape[-1]

			for i in xrange(framecount):
				self.depthIm = self.depth_stack[:,:,i].clip(0,4500)
				self.colorIm = self.color_stack[:,:,[2,1,0],i]
				# self.colorIm = self.color_stack[:,:,:,i]
				self.depth_timestamp = self.depth_timestamps[i]
				# Get mocap data at correct time
				while self.mocap_timestamps[0] < self.depth_timestamp:
					self.mocap_timestamps = self.mocap_timestamps[1:]
					self.skel_stack = self.skel_stack[1:]

				# Transform skeleton to kinect space
				self.users = [self.skel_stack[0]]

				# from pylab import *
				# figure(i)
				# for ii,i in enumerate(self.users[0]):
				# 	scatter(i[0], i[1])
				# 	annotate(str(ii), (i[0], i[1]))
				# axis('equal')
				# show()

				# Convert to Kinect format
				self.users_msr_tmp = [mhad_to_kinect_skel(self.users[0])]
				# self.users_msr[0] *= (self.users_msr[0][:,2]!=0)[:,None]
				# Transform to Kinect world space
				tmp_pts = np.hstack([self.users_msr_tmp[0],np.ones_like(self.users_msr_tmp[0])])[:,:4]
				self.users_msr = [np.dot(self.kinect_transform, tmp_pts.T).T[:,:3]]
				# Transform to Kinect image space
				self.users_uv_msr = [ self.camera_model.world2im(self.users_msr[0], [480,640]) ]
				self.users_uv_msr *= (self.users_msr_tmp[0][:,2]!=0)[:,None]
				self.users = self.users_msr
				self.users_uv = self.users_uv_msr

				# self.colorIm = display_skeletons(self.colorIm, self.users_uv_msr[0], skel_type='Kinect')
				self.update_background()

				if type(self.mask) is not int:
					self.mask = opening(self.mask, np.ones([3,3], np.uint8))
				if self.fill_images:
					self.depthIm = fill_image(self.depthIm)
				self.foregroundMask = self.mask

				yield




