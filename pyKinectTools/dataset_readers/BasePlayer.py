
import numpy as np
import cv2
from skimage.morphology import closing, erosion, opening
from pyKinectTools.algs.BackgroundSubtraction import *


class BasePlayer(object):

	depthIm = None
	colorIm = None
	users = None
	backgroundModel = None
	foregroundMask = None
	mask = None
	prevcolorIm = None

	def __init__(self, base_dir='./', get_depth=True, get_color=False,
				get_skeleton=False, bg_subtraction=False, fill_images=False,
				background_model=None, background_param=None):

		self.base_dir = base_dir
		self.deviceID = '[]'

		if bg_subtraction and background_model is None:
			raise Exception, "Must specify background_model"

		self.get_depth = get_depth
		self.get_color = get_color
		self.get_skeleton =get_skeleton

		self.enable_bg_subtraction = bg_subtraction
		self.fill_images = fill_images

		if background_model is not None:
			print 'Setting up background model:', background_model
			self.set_bg_model(background_model, background_param)


	def update_background(self):
		try:
			self.background_model.update(self.depthIm)
			self.mask = self.background_model.get_foreground()
			# self.mask = self.get_person()
		except:
			self.mask = None

	def set_background(self, im):
		self.background_model.backgroundModel = im

	def set_bg_model(self, bg_type='box', param=None):
		'''
		Types:
			'box'[param=max_depth]
			'static'[param=background]
			'mean'
			'median'
			'adaptive_mog'
		'''
		self.enable_bg_subtraction = True
		if bg_type == 'box':
			self.bgSubtraction = BoxModel(param)
		elif bg_type == 'static':
			if param==None:
				param = self.depthIm
			self.bgSubtraction = StaticModel(depthIm=param)
		elif bg_type == 'mean':
			self.bgSubtraction = MeanModel(depthIm=self.depthIm)
		elif bg_type == 'median':
			self.bgSubtraction = MedianModel(depthIm=self.depthIm)
		elif bg_type == 'adaptive_mog':
			self.bgSubtraction = AdaptiveMixtureOfGaussians(self.depthIm, maxGaussians=5, learningRate=0.01, decayRate=0.001, variance=300**2)
		else:
			print "No background model added"

		self.backgroundModel = self.bgSubtraction.getModel()

	def next(self, frames=1):
		pass

	def get_person(self, edge_thresh=200):
		mask, _, _, _ = extract_people(self.foregroundMask, minPersonPixThresh=5000, gradThresh=None)
		# mask, _, _, _ = extract_people(self.mask, minPersonPixThresh=5000, gradThresh=None)
		# mask, _, _, _ = extract_people(self.mask, minPersonPixThresh=5000, gradThresh=edge_thresh)
		mask = erosion(mask, np.ones([3,3], np.uint8))
		return mask

	def visualize(self, color=True, depth=True, show_skel=False):
		# ''' Find people '''
		if show_skel:
			plotUsers(self.depthIm, self.users)

		if self.get_depth and depth:
			cv2.imshow("Depth", (self.depthIm-1000)/float(self.depthIm.max()))
			# cv2.putText(self.deviceID, (5,220), (255,255,255), size=15)
			# vv.imshow("Depth", self.depthIm/6000.)

		if self.get_color and color:
			cv2.imshow("Color "+self.deviceID, self.colorIm)
			# vv.putText("Color "+self.deviceID, self.colorIm, "Day "+self.day_dir+" Time "+self.hour_dir+":"+self.minute_dir+" Dev#"+str(self.dev), (10,220))
			# vv.imshow("Color", self.colorIm)

		cv2.waitKey(10)

	def run(self):
		pass
