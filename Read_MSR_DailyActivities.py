import cv2
import numpy as np
color_file = "/home/freewym/a01_s01_e01_rgb.avi"
depth_file = 'a01_s01_e01_depth.bin'

c = cv2.VideoCapture(color_file)
cv2.namedWindow("RGB")

while(1):
  f, im = c.read()
	if not f:
		break
	cv2.imshow("RGB", im)
	ret = cv2.waitKey(10)

	if ret >= 0:
		break


file_ = open(depth_file, 'rb')

frames = np.fromstring(file_.read(4), dtype=int)[0]
cols = np.fromstring(file_.read(4), dtype=int)[0]
rows = np.fromstring(file_.read(4), dtype=int)[0]

depthIm = np.empty([rows, cols])
depthID = np.empty([rows, cols])
for f in range(frames):
	for i in range(rows):
		depthIm[i,:] = np.fromstring(file_.read(4*cols), dtype=int)
		depthID[i,:] = np.fromstring(file_.read(1*cols), dtype=np.uint8)

		cv2.imshow("Depth", depthIm/depthIm.max())
		ret = cv2.waitKey(10)

		if ret >= 0:
			break
