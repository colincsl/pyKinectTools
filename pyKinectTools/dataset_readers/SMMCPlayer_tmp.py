import cv2
import numpy as np
import scipy.misc as sm
import itertools as it
from pyKinectTools.utils.DepthUtils import skel2depth, depthIm_to_colorIm#world2depth, depthIm2XYZ, depth2world
from pyKinectTools.utils.SkeletonUtils import msr_to_kinect_skel
from IPython import embed
from pyKinectTools.dataset_readers.BaseReader import BaseReader

class SMMCPlayer(BasePlayer):

	def __init__(self, base_dir='./', get_depth=True, get_color=True, 
				get_skeleton=True, bg_subtraction=False, fill_images=False,
				actions=[1], subjects=[1], positions=[2]):
		
		self.enable_bg_subtraction = bg_subtraction
		self.fill_images = fill_images
		self.base_dir = base_dir
		self.deviceID = ""#Action {0:d}, Subject {1:d}, Instance {2:d}".format(0, 0, 0)

		self.get_depth = get_depth
		self.get_color = get_color
		self.get_skeleton =get_skeleton

		self.depth_stack = None
		self.mask_stack = None
		self.color_stack = None
		self.skel_stack = None

		self.filenames = create_MSR_filenames(actions, subjects, positions)

		self.player = self.run()
		self.next(1)

	def set_background(self, im):
		self.backgroundModel = im

	def update_background(self):
		'''Background model'''
		# self.backgroundModel = self.depthIm*(-self.mask)
		# self.foregroundMask = self.mask
		pass

	def next(self, frames=1):
		'''
		frames : skip (this-1) frames
		'''
		# Update frame
		try:
			for i in range(frames):
				self.player.next()
			return True
		except:
			return False

	def run(self):
		
		# Read data from new file
		while len(self.filenames) > 0:		
			if len(self.filenames) > 0 and self.depth_stack is None:
					print 'New video'
					name = self.filenames.pop()
					depth_file = self.base_dir + name + "depth.bin"
					color_file = self.base_dir + name + "rgb.avi"
					skeleton_file = self.base_dir + name + "skeleton.txt"
					self.depth_stack, self.mask_stack = read_MSR_depth_ims(depth_file)
					self.color_stack = read_MSR_color_ims(color_file)
					self.skel_stack,_ = read_MSR_skeletons(skeleton_file)
					# Offset!
					self.skel_stack[:,:,1] -= 75 
					framecount = np.min([self.depth_stack.shape[-1],self.color_stack.shape[-1]])

			for i in xrange(framecount):
				mask = self.mask_stack[:,:,i]
				if self.enable_bg_subtraction:
					self.depthIm = depthIm_to_colorIm(self.depth_stack[:,:,i]*mask)
				else:
					self.depthIm = depthIm_to_colorIm(self.depth_stack[:,:,i])
				# self.depthIm = self.depth_stack[:,:,i]
				self.colorIm = self.color_stack[:,:,:,i]			
				self.users = [msr_to_kinect_skel(self.skel_stack[i])]
				
				# tmp = depthIm_to_colorIm(self.depthIm * mask)
				self.mask = self.depthIm > 0
				self.update_background()
				yield


	def get_person(self, edge_thresh=200):
		return self.mask

	def visualize(self, show_skel=False):

		# ''' Find people '''
		if show_skel:
			self.ret = plotUsers(self.depthIm, self.users)

		if self.get_depth:
			cv2.imshow("Depth", (self.depthIm-1000)/2000.)
			# cv2.putText(self.deviceID, (5,220), (255,255,255), size=15)
		cv2.waitKey(10)
