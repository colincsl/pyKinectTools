import os
import cv2
import numpy as np
import scipy.misc as sm
import scipy.ndimage as nd
import itertools as it
from pyKinectTools.utils.DepthUtils import skel2depth, depthIm_to_colorIm#world2depth, depthIm2XYZ, depth2world
from pyKinectTools.utils.SkeletonUtils import msr_to_kinect_skel
from pyKinectTools.algs.BackgroundSubtraction import extract_people
from pyKinectTools.dataset_readers.BasePlayer import BasePlayer
from IPython import embed


def get_EVAL_filenames(base_dir):
	'''
	---Returns---
	list of filenames
	'''
	filenames = os.listdir(base_dir)
	filenames = [f for f in filenames if f[:4]=='seq_' and f[-4:]=='.bin']
	
	return filenames

def read_EVAL_depth_ims(data_file):
	file_ = open(data_file, 'rb')
	frame_size = 930704
	marker_count = 32
	joint_count = 32
	marker_dtype = np.dtype([("name", 'S256'), ("pos", np.float32, 3)])
	joint_dtype = np.dtype([("id", np.int32), ("pos", np.float32, 3)])
	frame_dtype = np.dtype([('magic', np.int32), ('frame', np.int32), \
				('image',np.float32, [240*320,3]), ("marker_count", np.int32), \
				("markers", marker_dtype, marker_count),("joints", joint_dtype, joint_count)])

	frames_stack = []
	markers_stack = []
	skeleton_stack = []
	i=0
	while 1:
		try:
			frame = file_.read(frame_size)
			magic = np.frombuffer(frame[0:4], dtype=np.int32)[0]
			frame_num = np.frombuffer(frame[4:8], dtype=np.int32)[0]
			byte_index = 8

			pts = np.frombuffer(frame[byte_index:byte_index+4*320*240*3], dtype=np.float32).reshape([240,320,3])			
			frames_stack += [np.array(pts)[:,:,:,None]]
			byte_index += 4*320*240*3

			m_count = np.frombuffer(frame[byte_index:byte_index+4], dtype=np.int32)[0]
			byte_index = 8+3*4*320*240+4
			
			markers = np.frombuffer(frame[byte_index:byte_index+marker_count*(256+3*4)], dtype=marker_dtype)
			markers_stack = markers[:m_count]
			byte_index += marker_count*(256+3*4)

			j_count = np.frombuffer(frame[byte_index:byte_index+4], dtype=np.int32)[0]
			byte_index += 4

			joints = np.frombuffer(frame[byte_index:byte_index+joint_count*(4*4)], dtype=joint_dtype)
			skeleton_stack = joints[:j_count]
			i += 1
		except:
			break
	xyz_stack = np.concatenate(frames_stack, -1)
	markers_stack = np.hstack(markers_stack)
	skeleton_stack = np.hstack(skeleton_stack)

	# Convert to my coordinate system
	xyz_stack *= 1000
	xyz_stack[:,:,:] *= -1
	xyz_stack = xyz_stack[:,:,[1,0,2]]

	return xyz_stack, skeleton_stack



class EVALPlayer(BasePlayer):

	def __init__(self, base_dir='./', get_depth=True, 
				get_skeleton=True, bg_subtraction=False, fill_images=False,
				actions=[1], subjects=[1], positions=[2]):
		
		self.enable_bg_subtraction = bg_subtraction
		self.fill_images = fill_images
		self.base_dir = base_dir
		self.deviceID = ""#Action {0:d}, Subject {1:d}, Instance {2:d}".format(0, 0, 0)

		self.get_depth = get_depth
		self.get_skeleton =get_skeleton

		self.xyz_stack = None
		self.skel_stack = None
		self.background_model = sm.imread(base_dir+"background_model.png")		

		self.filenames = get_EVAL_filenames(base_dir)

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
			if len(self.filenames) > 0 and self.xyz_stack is None:
					data_file = self.filenames.pop()				
					print 'New video:', data_file
					self.deviceID = data_file
					self.xyz_stack, self.skeleton_stack = read_EVAL_depth_ims(self.base_dir+data_file)
					framecount = self.xyz_stack.shape[-1]

			for i in xrange(framecount):
				self.depthIm = self.xyz_stack[:,:,2,i].copy()
				self.posIm = self.xyz_stack[:,:,:,i]

				self.mask = (np.abs(self.depthIm - self.background_model) > 500) * (self.depthIm != 0)
				self.mask = extract_people(self.depthIm, self.mask, 1000, 500)[0]

				if self.enable_bg_subtraction:
					self.depthIm *= self.mask
					self.posIm *= self.mask[:,:,None]

				self.users = []#[msr_to_kinect_skel(self.skel_stack[i])]				

				yield


	def get_person(self):
		labelIm, maxLabel = nd.label(self.mask)
		connComps = nd.find_objects(labelIm, maxLabel)
		px_count = [nd.sum(labelIm[c]==l) for c,l in zip(connComps,range(1, maxLabel+1))]
		max_box = np.argmax(px_count)

		return labelIm == max_box+1

	def visualize(self, show_skel=False):

		# ''' Find people '''
		# if show_skel:
			# self.ret = plotUsers(self.depthIm, self.users)
		cv2.imshow("Depth", (self.depthIm-2000)/2000.)
		# cv2.putText(self.deviceID, (5,220), (255,255,255), size=15)
		cv2.waitKey(10)
