import cv2
import numpy as np
import scipy.misc as sm

'''
These functions read in the depth images, color images, and skeletons from the MSR Daily Activity dataset.
http://research.microsoft.com/en-us/um/people/zliu/ActionRecoRsrc/default.htm
'''

def read_MSR_labels():
	return ['drink', 'eat', 'read book', 'call cellphone', 'write on a paper', 'use laptop', 'use vacuum cleaner', 'cheer up', 'sit still', 'toss paper', 'play game', 'lie down on sofa', 'walk', 'play guitar', 'stand up', 'sit down']

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
	framecount = int(colorCapture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
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






