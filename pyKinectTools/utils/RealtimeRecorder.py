

import os, time, sys
import numpy as np
import cv, cv2
from pyKinectTools.utils.RealtimeReader import *

dir_ = '/Users/colin/data/ICU_oct_test/'

'''------------ Setup Kinect ------------'''
depthConstraint = [500, 5000]
''' Physical Kinect '''
depthDevice = RealTimeDevice()
depthDevice.addDepth(depthConstraint)
depthDevice.addColor()
depthDevice.start()
depthDevice.setMaxDist(depthConstraint[1])

''' Visualization '''
cv2.namedWindow("Color")
cv2.namedWindow("Depth")

''' ------------- Main -------------- '''

while 1:
	depthDevice.update()
	colorRaw = depthDevice.colorIm
	depthRaw8 = depthDevice.depthIm8

	time_ = time.localtime()
	day = str(time_.tm_yday)
	hour = str(time_.tm_hour)
	minute = str(time_.tm_min)
	second = str(time_.tm_sec)
	ms = str(time.clock())

	if depthRaw8 != []:
		if not os.path.isdir(dir_+day):
			os.mkdir(dir_+day)

		if not os.path.isdir(dir_+day+"/"+hour):
			os.mkdir(dir_+day+"/"+hour)

		if not os.path.isdir(dir_+day+"/"+hour+"/"+minute):
			os.mkdir(dir_+day+"/"+hour+"/"+minute)
			os.mkdir(dir_+day+"/"+hour+"/"+minute+"/"+"depth")
			os.mkdir(dir_+day+"/"+hour+"/"+minute+"/"+"color")
			# os.mkdir(dir_+day+"/"+hour+"/"+minute+"/"+"skel")

		depthName = dir_+day+"/"+hour+"/"+minute+"/"+"depth"+"/"+\
					"depth_"+day+"_"+hour+"_"+minute+"_"+second+"_"+ms
		colorName = dir_+day+"/"+hour+"/"+minute+"/"+"color"+"/"+\
					"color_"+day+"_"+hour+"_"+minute+"_"+second+"_"+ms
		cv2.imwrite(depthName, depthRaw8)
		cv2.imwrite(colorName, colorRaw)

	print second
	'''Display Image'''
	# cv2.imshow("Depth", depthRaw8)
	# cv2.imshow("Color", colorRaw)	
	ret = cv2.waitKey(10)
	if ret >= 0:
		break
