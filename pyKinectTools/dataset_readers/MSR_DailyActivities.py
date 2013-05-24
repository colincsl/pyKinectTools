import cv2
import numpy as np
import scipy.misc as sm
import itertools as it
from skimage.morphology import closing, erosion
from pyKinectTools.utils.DepthUtils import skel2depth, depthIm_to_colorIm#world2depth, depthIm2XYZ, depth2world
from pyKinectTools.utils.SkeletonUtils import msr_to_kinect_skel
# from IPython import embed
# from pyKinectTools.utils.VideoViewer import VideoViewer
from pyKinectTools.dataset_readers.BasePlayer import BasePlayer
from pyKinectTools.algs.BackgroundSubtraction import fill_image
# vv = VideoViewer()

from IPython import embed

'''
These functions read in the depth images, color images, and skeletons from the MSR Daily Activity dataset.
http://research.microsoft.com/en-us/um/people/zliu/ActionRecoRsrc/default.htm
'''

def read_MSR_labels():
	return ['drink', 'eat', 'read book', 'call cellphone', 'write on a paper', 'use laptop', 'use vacuum cleaner', 'cheer up', 'sit still', 'toss paper', 'play game', 'lie down on sofa', 'walk', 'play guitar', 'stand up', 'sit down']


def create_MSR_filenames(actions, subjects, positions):
	'''
	---Parameters---
	actions : list of numbers 1-16
	subjects: list of numbers 1-10
	positions: list of numbers netween 1-2 (1=sitting, 2=standing)
	---Returns---
	list of filenames
	'''
	filenames = []
	indicies = [i for i in it.product(actions, subjects, positions)]
	for i in indicies:
		filenames += ["a{0:02d}_s{1:02d}_e{2:02d}_".format(i[0],i[1],i[2])]

	return filenames

def read_MSR_depth_ims(depth_file, resize='VGA'):
	''' Extracts depth images and masks from the MSR Daily Activites dataset
	---Parameters---
	depth_file : filename for set of depth images (.bin file)
	'''

	file_ = open(depth_file, 'rb')

	''' Get header info '''
	frames = np.fromstring(file_.read(4), dtype=np.int32)[0]
	cols = np.fromstring(file_.read(4), dtype=np.int32)[0]
	rows = np.fromstring(file_.read(4), dtype=np.int32)[0]

	''' Get depth/mask image data '''
	data = file_.read()

	'''
	Depth images and mask images are stored together per row.
	Thus we need to extract each row of size n_cols+n_rows
	'''
	dt = np.dtype([('depth', np.int32, cols), ('mask', np.uint8, cols)])

	''' raw -> usable images '''
	frame_data = np.fromstring(data, dtype=dt)
	depthIms = frame_data['depth'].astype(np.uint16).reshape([frames, rows, cols])
	maskIms = frame_data['mask'].astype(np.uint16).reshape([frames, rows, cols])

	if resize == 'VGA':
		# embed()
		depthIms = np.dstack([cv2.resize(depthIms[d,:,:], (640,480)) for d in xrange(len(depthIms))])
		maskIms = np.dstack([cv2.resize(maskIms[d,:,:], (640,480)) for d in xrange(len(maskIms))])

	return depthIms, maskIms

def read_MSR_color_ims(color_file, resize='VGA'):
	''' Extracts color images from the MSR Daily Activites dataset
	---Parameters---
	color_file : filename for color video (.avi file)
	resize : reduces the image size from 640x480 to 320x240
	'''

	colorCapture = cv2.VideoCapture(color_file)
	framecount = int(colorCapture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	if resize=='QVGA':
		colorIms = np.empty([framecount, 240, 320, 3], dtype=np.uint8)
		rows, cols, depth = [240,320,3]
	else:
		colorIms = np.empty([480, 640, 3, framecount], dtype=np.uint8)
		rows, cols, depth = [480,640,3]

	for f in xrange(framecount):
		valid, color = colorCapture.read()
		if not valid:
			break

		if resize == 'QVGA':
			color = cv2.resize(color, (cols,rows))

		colorIms[:,:,:,f] = color

	colorCapture.release()

	return colorIms



def read_MSR_skeletons(skeleton_file, world_coords=True, im_coords=True, resolution=[240,320]):
	''' Extracts skeletons from the MSR Daily Activites dataset
	---Parameters---
	skeleton_file : filename for color video (.avi file)
	resize : reduces the image size from 640x480 to 320x240
	'''

	assert world_coords or im_coords, "Error: requires at least world or image coordinates to be true"

	data_raw = np.fromfile(skeleton_file, sep='\n')

	frameCount = int(data_raw[0])
	joint_count = int(data_raw[1])
	assert joint_count == 20, "Error: joint count is %i not 20" % joint_count

	data = np.zeros([frameCount, joint_count*4*2])

	for i in range(0,frameCount):
		ind = i*(joint_count*2*4+1) + 2
		data[i,:] = data_raw[ind+1:ind+20*4*2+1]

	''' Get rid of confidence variable (it's useless for this data)	'''
	data = data.reshape([frameCount, 40, 4])
	data = data[:,:,:3]

	if world_coords:
		skels_world = data[:,::2,:]
		''' Put in millimeters instead of meters'''
		skels_world *= 1000.
		# skels_world[:,:,2] *= 1000.
	if im_coords:
		skels_im = data[:,1::2,:].astype(np.float)
		''' These coords are normalized, so we must rescale by the image size '''
		skels_im *= np.array(resolution+[1])
		''' The depth values in the image coordinates doesn't make sense (~20,000!).
			So replace them with the values from the world coordinates'''
		skels_im[:,:,2] = skels_world[:,:,2]
		skels_im = skels_im.astype(np.int16)

	if world_coords and im_coords:
		return skels_world, skels_im
	elif world_coords:
		return skels_world
	elif im_coords:
		return skels_im

	return -1




class MSRPlayer(BasePlayer):

	def __init__(self, base_dir='./', get_depth=True, get_color=True,
				get_skeleton=True, bg_subtraction=False, fill_images=False,
				actions=[1], subjects=[1], positions=[2]):

		self.enable_bg_subtraction = bg_subtraction
		self.fill_images = fill_images
		self.base_dir = base_dir
		self.deviceID = "Action {0:d}, Subject {1:d}, Instance {2:d}".format(0, 0, 0)

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
			print "Done playing video"
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

				self.mask = (self.depthIm > 0).astype(np.uint8)
				self.mask = closing(self.mask, np.ones([3,3], np.uint8))
				self.mask = erosion(self.mask, np.ones([3,3], np.uint8))
				self.depthIm = fill_image(self.depthIm)*(self.mask)

				self.update_background()

				yield


	def get_person(self, edge_thresh=200):
		return self.mask


	# def get_n_skeletons(self, n):
	# 	'''
	# 	In MSR format
	# 	'''
	# 	if n == -1:
	# 		n = np.inf

	# 	depthIms = []
	# 	colorIms = []
	# 	skels_world = []

	# 	try:
	# 		while depthIms == [] or len(depthIms) < n:
	# 			self.next()
	# 			self.update_background()
	# 			if len(self.users.keys()) > 0:
	# 				s = self.users.keys()[0]
	# 				pts = np.array(self.users[s]['jointPositions'].values())
	# 				if np.all(pts[0] != -1):
	# 					pts = kinect_to_msr_skel(pts)
	# 					skels_world += [pts]

	# 					depth = self.get_person()
	# 					if self.get_depth:
	# 						depthIms += [depth]
	# 					if self.get_color:
	# 						colorIms += [color]
	# 	except:
	# 		print "No skeletons remaining."

	# 	return depthIms, colorIms, skels_world

	def visualize(self, show_skel=False):

		# ''' Find people '''
		if show_skel:
			self.ret = plotUsers(self.depthIm, self.users)

		if self.get_depth:
			cv2.imshow("Depth", (self.depthIm-1000)/2000.)
			# cv2.putText(self.deviceID, (5,220), (255,255,255), size=15)
			# vv.imshow("Depth", self.depthIm/6000.)

		if self.get_color:
			cv2.imshow("Color "+self.deviceID, self.colorIm)
			# vv.putText("Color "+self.deviceID, self.colorIm, "Day "+self.day_dir+" Time "+self.hour_dir+":"+self.minute_dir+" Dev#"+str(self.dev), (10,220))
			# vv.imshow("Color", self.colorIm)
			# vv.imshow("Color", self.colorIm)
			# if self.get_mask:
			# vv.imshow("I", self.colorIm*self.foregroundMask[:,:,np.newaxis])
			# vv.imshow("I_masked", self.colorIm + (255-self.colorIm)*(((self.foregroundMask)[:,:,np.newaxis])))

		cv2.waitKey(10)


