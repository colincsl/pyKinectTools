

from pyKinectTools.algs.BackgroundSubtraction import *
from pyKinectTools.utils.DepthUtils import *
from openni import *
import time

class rtDepth:
	ctx = []
	depth = []
	rgb = []
	depthIm = []
	depthIm8 = []

	constrain = []
	maxDist = np.inf

	bgModel = []
	bgModel8 = []

	def __init__(self, ctx=[]):
		if ctx != []:
			self.ctx = ctx
		else:
			self.ctx = Context()
			self.ctx.init()
			print "New context created for depth device."			

	def addDepth(self, constrain=[500, 2000]):
		self.constrain = constrain

		try:
			self.depth = DepthGenerator()
			self.depth.create(self.ctx)
			self.depth.set_resolution_preset(RES_VGA)
			self.depth.fps = 30			
		except:
			print "Depth module can not load."

	def addColor(self):

		try:
			self.depth = DepthGenerator()
			self.depth.create(self.ctx)
			self.depth.set_resolution_preset(RES_VGA)
			self.depth.fps = 30			
		except:
			print "Color module can not load."			

	def setMaxDist(self, dist):
		pass

	def start(self):
		self.ctx.start_generating_all()

	def stop(self):
		self.ctx.shutdown()

	def update(self):
		ret = self.ctx.wait_any_update_all()
		assert ret == None, "Error updating depth device."

		if self.depth != []:
			depthRawT = self.depth.get_tuple_depth_map()
			im = np.array(depthRawT).reshape([self.depth.res[1],self.depth.res[0]])

			im[im > self.maxDist] = 0
			self.depthIm = im
			self.depthIm8 = constrain(im, self.constrain[0], self.constrain[1])


	def generateBackgroundModel(self):
		# Get set of 5 frames and create background model
		depthImgs = []
		depthStackInd = 0
		for i in xrange(5):
			ret = self.ctx.wait_one_update_all(self.depth)
			assert ret == None, "Error getting depth map"

			depthRawT = self.depth.get_tuple_depth_map()
			im = np.array(depthRawT).reshape([self.depth.res[1],self.depth.res[0]])
			depthImgs.append(im)
			time.sleep(.2)

		depthImgs = np.dstack(depthImgs)

		self.bgModel = getMeanImage(depthImgs)

		self.bgModel8 = constrain(self.bgModel, self.constrain[0], self.constrain[1])
		self.bgModel8[self.bgModel8==self.bgModel8.max()] = 0

		# return self.depthIm


