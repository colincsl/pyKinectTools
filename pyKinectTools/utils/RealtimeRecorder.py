

# import os, time, sys, cPickle
import numpy as np
import cv, cv2
# Data readers
from pyKinectTools.utils.DepthReader import DepthReader
from pyKinectTools.utils.RealtimeReader import *
# Utils
from pyKinectTools.utils.DepthUtils import *

'''------------ Setup Kinect ------------'''
depthConstraint = [500, 5000]
''' Physical Kinect '''
depthDevice = RealTimeDevice()
depthDevice.addDepth(depthConstraint)
depthDevice.addColor()
depthDevice.start()
depthDevice.setMaxDist(depthConstraint[1])
depthDevice.generateBackgroundModel()

''' Visualization '''
cv2.namedWindow("Color")
cv2.namedWindow("Depth")

''' ------------- Main -------------- '''

while 1:
	depthDevice.update()
	depthRaw = depthDevice.depthIm
	colorRaw = depthDevice.colorIm
	depthRaw[depthRaw > depthConstraint[1], :] = 0
	depthRaw8 = depthDevice.depthIm8
	# posMatTop = getTopdownMap(depthRaw, sceneRotation, centroid=sceneCentroid, rez=rezNewPos, bounds=topBounds)

	cv2.imshow("Depth", depthRaw8)
	cv2.imshow("Color", colorRaw)


	'''Display Image'''
	ret = cv2.waitKey(10)
	if ret >= 0:
		break
