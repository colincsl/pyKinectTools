import cv, cv2
import numpy as np
import scipy.misc as sm
import os
from pyKinectTools.utils.SkeletonUtils import display_MSR_skeletons
from pyKinectTools.utils.DepthUtils import world2depth
from pyKinectTools.algs.BackgroundSubtraction import extract_people
from pyKinectTools.algs.HistogramOfOpticalFlow import getFlow, hof
from skimage.color import rgb2gray
from skimage.feature import hog


def read_MSR_depth_ims(depth_file):
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
	depthIms = frame_data['depth'].reshape([frames, rows, cols])
	maskIms = frame_data['mask'].reshape([frames, rows, cols])

	return depthIms, maskIms

def read_MSR_color_ims(color_file, resize=True):
	''' Extracts color images from the MSR Daily Activites dataset 
	---Parameters---
	color_file : filename for color video (.avi file)
	resize : reduces the image size from 640x480 to 320x240
	'''

	colorCapture = cv2.VideoCapture(color_file)
	framecount = int(colorCapture.get(cv.CV_CAP_PROP_FRAME_COUNT))
	if resize:
		colorIms = np.empty([framecount, 240, 320, 3], dtype=np.uint8)
		rows, cols, depth = [240,320,3]
	else:
		colorIms = np.empty([framecount, 480, 640, 3], dtype=np.uint8)

	for f in xrange(framecount):
		valid, color = colorCapture.read()
		if not valid: 
			break

		if resize:
			color = sm.imresize(color, [rows, cols, 3], interp='nearest')

		colorIms[f] = color

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
	assert joint_count == 20, "Error: joint count is " +str(joint_count) + " not 20."

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



''' 
-----------------------------------
--------------MAIN ----------------
-----------------------------------
'''



''' ------- Display depth/mask/color data for the videos in the dataset -------- '''

''' Get all appropriate files in this folder '''
files = os.listdir('.')
base_names = [f[:12] for f in files]
#Remove bad filenames
base_names = [f for f in base_names if f[0]!='.']
base_names = np.unique(base_names)
name = base_names[0]


''' Initialize feature vectors '''
dataset_features = {}

''' Play data '''
for name in base_names:
	''' Get filenames '''
	depth_file = name + "depth.bin"
	color_file = name + "rgb.avi"
	skeleton_file = name + "skeleton.txt"
	''' Read data from each video/sequence '''
	depthIms, maskIms = read_MSR_depth_ims(depth_file)
	colorIms = read_MSR_color_ims(color_file)
	skels_world, skels_im = read_MSR_skeletons(skeleton_file)

	dataset_features[name] = {'hog':[], 'hog':[], 'skel_image':[], 'skel_world':[]}
	framecount = np.minimum(depthIms.shape[0], colorIms.shape[0])
	grayIm_prev = None

	''' View all data'''
	for frame in xrange(framecount):
		depth = depthIms[frame]
		mask = maskIms[frame]
		color = colorIms[frame]
		# Skeleton in world (w) and image (i) coordinates
		skel_w = skels_world[frame]
		skel_i = world2depth(skel_w, rez=[240,320])

		''' Calculate hogs '''
		grayIm = (rgb2gray(color) * 255).astype(np.uint8)
		hogIm = np.zeros_like(depth)
		person_mask, bounding_boxes, labels = extract_people(grayIm, mask>0)
		hogData, hogImBox = hog(grayIm[bounding_boxes[0]], orientations=4, visualise=True)
		hogIm[bounding_boxes[0]] = hogImBox
		hogIm *= person_mask

		''' Calculate HOF '''
		hofIm = np.zeros_like(depth)
		if grayIm_prev is not None:
			flow = getFlow(grayIm_prev[bounding_boxes[0]], grayIm[bounding_boxes[0]])
			hofData, hofImBox = hof(flow, orientations=5, visualise=True)
			hofIm[bounding_boxes[0]] = hofImBox
			hofIm *= person_mask

			cv2.imshow("HOF", hofIm/float(hofIm.max()))

		''' Add features '''
		dataset_features[name]['hog'] += hogData
		dataset_features[name]['hof'] += hofData
		dataset_features[name]['skel_image'] += skel_i
		dataset_features[name]['skel_world'] += skel_w

		''' Plot skeletons on color image'''
		color = display_MSR_skeletons(color, skel_i)

		''' Visualization '''
		cv2.imshow("Depth", depth/float(depth.max()))
		cv2.imshow("HOG", hogIm/float(hogIm.max()))
		cv2.imshow("RGB", color)
		cv2.imshow("RGB masked", color*(mask[:,:,None]>0))
		ret = cv2.waitKey(10)

		grayIm_prev = np.copy(grayIm)

		if ret >= 0:
			break
	if ret >= 0:
		break

with open('MSR_Features_hog-hof-skel.dat', 'wb') as outfile:
    pickle.dump(dataset_features, outfile, protocol=pickle.HIGHEST_PROTOCOL)

from IPython import embed
embed()

# if 0:
# 	''' Move MSR data '''

# 	'''
# 	mkdir ../MSR_Skels_3
# 	'''

# 	import shutil, os

# 	dir_ = '/Users/Colin/Data/MSR_DailyAct3D_pack3/'
# 	dir_new = '/Users/Colin/Data/MSR_Skels_3/'

# 	files = os.listdir('.')
# 	sFiles = [x for x in files if x.find('.txt') >= 0]
# 	for f in sFiles:
# 		shutil.copyfile(f,dir_new+f)


