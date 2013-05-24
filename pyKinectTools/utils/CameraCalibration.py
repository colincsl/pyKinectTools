'''
Modified considerably from https://github.com/abidrahmank/OpenCV-Python/blob/master/Other_Examples/camera_calibration.py

Usage:	python CameraCalibration.py num_board_cols num_board_rows number_of_views
'''

################################################################################################

import time
import sys
import cv
import cv2
import numpy as np
from pyKinectTools.utils.RealtimeReader import *

n_boards=0
board_w=int(sys.argv[1])
board_h=int(sys.argv[2])
n_boards=int(sys.argv[3])
# no of total corners
board_n=board_w*board_h
#size of board
board_sz=(board_w,board_h)

# Init point arrays
image_points = np.empty([n_boards*board_n, 2], np.float32)
object_points = np.empty([n_boards*board_n, 3], np.float32)
point_counts=np.empty([n_boards, 1], np.float32)

#	capture frames of specified properties and modification of matrix values
i=0
z=0		# to print number of frames
successes=0
# capture=cv2.VideoCapture(0)
# capture = RealTimeDevice(0,get_skeleton=False, get_infrared=True,
						# get_depth=False, get_color=False,
						# config_file=pyKinectTools.configs.__path__[0]+'/InfraredConfig.xml')
capture = RealTimeDevice(0,get_skeleton=False)
capture.start()


#	capturing required number of views
while(successes<n_boards):
	found=0

	capture.update()
	if capture.colorIm is not None:
		image = capture.colorIm
	else:
		image = capture.infraredIm

	if len(image.shape)==3:
		gray_image=image.mean(-1, dtype=np.uint8)
	else:
		gray_image=image

	(found,corners)=cv2.findChessboardCorners(gray_image,board_sz,
		cv.CV_CALIB_CB_ADAPTIVE_THRESH | cv.CV_CALIB_CB_FILTER_QUADS)

	# Draw board
	if found==1:
		print "Found frame number {0}".format(z+1)
		term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
		cv2.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), term)
		cv2.drawChessboardCorners(image,board_sz,corners,1)
		corner_count=len(corners)
		z=z+1

	# if got a good image, add to matrix
	if found==1 and len(corners)==board_n:
		step=successes*board_n
		k=step
		# embed()
		for j in range(board_n):
			image_points[k] = corners[j]
			object_points[k] = [float(j)/float(board_w), float(j)%float(board_w), 0.]
			k=k+1
		point_counts[successes] = board_n
		successes=successes+1
		time.sleep(2)
		print "-------------------------------------------------"
		print "\n"
	cv2.imshow("Test Frame",image)
	cv2.waitKey(33)

cv.DestroyWindow("Test Frame")

# camera calibration
h,w = image.shape[:2]
retval,cameraMatrix,distCoeffs,rvecs,tvecs = cv2.calibrateCamera([object_points],[image_points],(w,h))
print " checking camera calibration.........................OK	"
embed()

# storing results in xml files
cv.Save("Intrinsics.xml",cameraMatrix)
cv.Save("Distortion.xml",distCoeffs)
# Loading from xml files
intrinsic = cv.Load("Intrinsics.xml")
distortion = cv.Load("Distortion.xml")
print " loaded all distortion parameters"

mapx = cv.CreateImage( cv.GetSize(image), cv.IPL_DEPTH_32F, 1 );
mapy = cv.CreateImage( cv.GetSize(image), cv.IPL_DEPTH_32F, 1 );
cv.InitUndistortMap(intrinsic,distortion,mapx,mapy)
cv.NamedWindow( "Undistort" )
print "all mapping completed"
print "Now relax for some time"
time.sleep(8)

print "now get ready, camera is switching on"
while(1):
	image=cv.QueryFrame(capture)
	t = cv.CloneImage(image);
	cv.ShowImage( "Calibration", image )
	cv.Remap( t, image, mapx, mapy )
	cv.ShowImage("Undistort", image)
	c = cv.WaitKey(33)
	if(c == 1048688):		# enter 'p' key to pause for some time
		cv.WaitKey(2000)
	elif c==1048603:		# enter esc key to exit
		break

print "Everything is fine"

###############################################################################################
