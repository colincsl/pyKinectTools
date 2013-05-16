"""
Pose Estimation + Tracking
"""

import numpy as np
import cv2
import scipy.misc as sm
import scipy.ndimage as nd
import skimage
from skimage import color
from skimage.draw import line, circle, circle_perimeter
from skimage.color import rgb2gray,gray2rgb, rgb2lab
from skimage.feature import hog, local_binary_pattern, match_template, peak_local_max

from sklearn.metrics import pairwise_distances

import pyKinectTools.configs
from pyKinectTools.utils.DepthUtils import *
from pyKinectTools.utils.SkeletonUtils import *
from pyKinectTools.algs.GeodesicSkeleton import *

from IPython import embed
np.seterr(all='ignore')

# from joblib import Parallel, delayed
def display_markers(im, markers, box, color):
	if im.shape[0] == 480:
		pt_radius = 10
	else:
		pt_radius = 5
	height, width,_ = im.shape
	for i,o in enumerate(markers):
		joint = np.array(o) + [box[0].start, box[1].start]
		circ = np.array(circle(joint[0],joint[1], pt_radius)).T
		circ = circ.clip([0,0], [height-1, width-1])
		im[circ[:,0], circ[:,1]] = color
	return im

def feature_joint_displacements(skel_current, im_pos, feature_inds, features_joints, distance_thresh=500):
	'''
	Find the distances between each feature and the corresponding joint
	'''
	joint_displacement = np.zeros([len(skel_current), 3])
	if len(feature_inds) == 0:
		return joint_displacement
	feature_inds = np.array(feature_inds)

	distances = pairwise_distances(skel_current[features_joints], im_pos[feature_inds[:,0], feature_inds[:,1]], 'l2')
	for _ in range(len(features_joints)):
		# Find closest feature to closest joint
		joint_index, feature_index = np.unravel_index(np.argmin(distances), distances.shape)
		# Remove this joint/feature from open set:
		distances[joint_index,:] = np.inf
		distances[:,feature_index] = np.inf
		joint = features_joints[joint_index]
		# Set displacement for the joint:
		error_xy = np.sqrt(np.sum(((skel_current[features_joints[joint_index]][[0,1,2]] - im_pos[feature_inds[feature_index][0], feature_inds[feature_index][1]]))**2, 0))
		if error_xy < distance_thresh:
			joint_displacement[features_joints[joint_index]] = (skel_current[features_joints[joint_index]] - im_pos[feature_inds[feature_index][0], feature_inds[feature_index][1]])

	return joint_displacement



def link_length_constraints(skeleton_xyz, constraint_links, constraint_values, alpha=0.1):
	'''
	alpha : (0.,1.) How strong the constraint is. (0 = Don't Apply, 1 Full weight)
	'''
	# embed()
	for i,c in enumerate(constraint_links):
		pt1 = skeleton_xyz[c[0]]
		pt2 = skeleton_xyz[c[1]]
		link_actual = np.linalg.norm(pt2-pt1)
		link_vector = (pt2-pt1)/np.linalg.norm(pt2-pt1)
		link_ideal = constraint_values[i]
		error = link_ideal - link_actual
		if link_actual > 0:
			skeleton_xyz[c[0]] = alpha*skeleton_xyz[c[0]] + (1-alpha)*(skeleton_xyz[c[0]]-link_vector*error)
			skeleton_xyz[c[1]] = alpha*skeleton_xyz[c[1]] + (1-alpha)*(skeleton_xyz[c[1]]+link_vector*error)

	# embed()
	return skeleton_xyz

def link_length_probability(skeleton_xyz, constraint_links, constraint_values, variance):
	'''
	alpha : (0.,1.) How strong the constraint is. (0 = Don't Apply, 1 Full weight)
	'''
	prob = []
	for i,c in enumerate(constraint_links):
		pt1 = skeleton_xyz[c[0]]
		pt2 = skeleton_xyz[c[1]]
		link_actual = np.linalg.norm(pt2-pt1)
		link_vector = (pt2-pt1)/np.linalg.norm(pt2-pt1)
		link_ideal = constraint_values[i]
		error = link_ideal - link_actual
		if link_actual > 0:
			prob += [np.exp(-error**2/variance**2)]

	return np.array(prob)


def geometry_constraints(skeleton_xyz, joint_variance, alpha=0.5):
	'''
	alpha : (0.,1.) How strong the constraint is. (1 = Don't Apply, 0 Full weight)
	'''

	## Check radius around each point
	distances = pairwise_distances(skeleton_xyz)
	# map(lambda x: joint_variance[x[0]]*joint_variance[x[1]], np.repeat(np.arange(0, 10)[:,None], 2, -1))
	for i in xrange(len(skeleton_xyz)):
		for j in xrange(len(skeleton_xyz)):
			if i < j and distances[i,j] < joint_variance[i]+joint_variance[j]:
				pt1 = skeleton_xyz[i]
				pt2 = skeleton_xyz[j]
				link_vector = (pt2-pt1)/np.linalg.norm(pt2-pt1)
				error = -(distances[i,j] - (joint_variance[i]+joint_variance[j]))
				skeleton_xyz[i] = alpha*skeleton_xyz[i] + (1-alpha)*(skeleton_xyz[i]-link_vector*error)
				skeleton_xyz[j] = alpha*skeleton_xyz[j] + (1-alpha)*(skeleton_xyz[j]+link_vector*error)
				# print i,j, link_vector
	return skeleton_xyz


from scipy.spatial import cKDTree
def collision_constraints(skeleton_xyz, constraint_links, resolution=10):
	'''
	resolution : distance between points
	'''

	## Check connections
	# Sample each connection. Check if any

	# embed()
	# Sample all connections
	points = []
	parts_links = []
	for i,c in enumerate(constraint_links):
		pt1 = skeleton_xyz[c[0]]
		pt2 = skeleton_xyz[c[1]]
		link_length = np.linalg.norm(pt2-pt1)
		link_vector = (pt2-pt1)/np.linalg.norm(pt2-pt1)
		dist = resolution
		while dist < link_length:
			points += [pt1 + dist*link_vector]
			parts_links += [c]
			dist += resolution
	points = np.array(points)

	# joint_parts = []
	# for i,_ in enumerate(skeleton_xyz):
	# 	joint_parts += [parts_links[:,0]] + [parts_links[:,1]]

	# Find nearest neighbors
	tree = cKDTree(points)
	distances,indices = tree.query(points, 2, distance_upper_bound=resolution*3/4)
	# distances = np.argwhere(distances[np.isfinite(distances[:,1]),1])
	indices_a = np.argwhere(np.isfinite(distances[:,1]))
	indices_b = indices[indices_a,1]

	for i,ind in enumerate(indices_a):
		points[ind]


	print len(indices), "colisions"

	return skeleton_xyz


def ray_cast_constraints(skeleton_xyz, skeleton_img, im_pos, surface_map=None, joint_variance=None, mm_per_px=5):
	'''
	These should all be in the masked/bounding box coordinates.
	--Params--
	skeleton_xyz :
	skeleton_img :
	im_pos :
	surface_map :
	--Return--
	skeleton_xyz
	skeleton_img
	'''

	if surface_map is None:
		surface_map = nd.distance_transform_edt(im_pos[:,:,2]==0, return_distances=False, return_indices=True)
	if joint_variance is None:
		joint_variance = np.ones(len(skeleton_xyz))

	## Silhouette: Ensure the points are within the mask
	# Look at density around each joint
	# tt = skeleton_img.copy()
	for i,s in enumerate(skeleton_img):
		radius = int(joint_variance[i]/float(mm_per_px))
		tmp = np.clip(np.transpose(circle_perimeter(s[0],s[1], radius)), [0,0], [im_pos.shape[0]-1, im_pos.shape[1]-1])
		tmp_diff = surface_map[:, tmp[:,0], tmp[:,1]].T - tmp
		mean_diff = np.array([np.mean(tmp_diff[tmp_diff[:,0]!=0, 0]).astype(np.int), np.mean(tmp_diff[tmp_diff[:,1]!=0, 1]).astype(np.int)])
		if -radius < mean_diff[0] < radius:
			skeleton_img[i,0] += mean_diff[0]
		if -radius < mean_diff[1] < radius:
			skeleton_img[i,1] += mean_diff[1]
	skeleton_img = skeleton_img.clip([0,0,0], [im_pos.shape[0]-1, im_pos.shape[1]-1, 99999])

	# print 'D', skeleton_img - tt

	# Only look at point estimate
	# print skeleton_img[:,:2] - surface_map[:, skeleton_img[:,0], skeleton_img[:,1]].T
	skeleton_img[:,:2] = surface_map[:, skeleton_img[:,0], skeleton_img[:,1]].T

	## Z-surface: Ensure the points lie on or behind the surface
	z_surface = im_pos[skeleton_img[:,0], skeleton_img[:,1], 2]
	# skeleton_img[:,2] = skeleton_img[:,2] = z_surface
	skeleton_img[:,2] = np.maximum(skeleton_img[:,2], z_surface)

	skeleton_xyz[:,:2] = im_pos[skeleton_img[:,0],skeleton_img[:,1],:2]

	return skeleton_xyz, skeleton_img


class FaceDetector:
	min_threshold = 0.
	max_threshold = 100.
	face_box = None
	face_position = []
	# cascades = []

	def __init__(self, rez=[480,640]):
		# self.cascades += [cv2.CascadeClassifier('/Users/colin/libs/opencv/opencv/data/haarcascades/haarcascade_frontalface_alt_tree.xml')]
		self.cascade = cv2.CascadeClassifier('/Users/colin/libs/opencv/opencv/data/haarcascades/haarcascade_frontalface_alt_tree.xml')
		self.rez = rez

	def run(self, im_color):
		im_gray = (rgb2gray(im_color)*255).astype(np.uint8)
		if self.rez[0] == 240:
			faces = self.cascade.detectMultiScale(im_gray, scaleFactor=1.1, minNeighbors=1, minSize=(10,10), maxSize=(40,40), flags=cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT)
		else:
			faces = self.cascade.detectMultiScale(im_gray, scaleFactor=1.1, minNeighbors=1, minSize=(40,40), maxSize=(80,80), flags=cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT)

		if len(faces) > 0:
			self.face_box = (slice(faces[0][1], faces[0][1]+faces[0][2])), slice(faces[0][0], faces[0][0]+faces[0][3])
			self.face_position = [[faces[0][1]+faces[0][2]/2, faces[0][0]+faces[0][3]/2]]
			face_lab = rgb2lab(im_color[self.face_box])[:,:,1]
			hist = skimage.exposure.histogram(face_lab)
			# Get rid of background
			hist[0][hist[1]<0] = 0

			total_count = float(np.sum(hist[0]))
			mean = np.sum(hist[0]*hist[1]) / total_count
			var = np.sum((hist[0]/total_count)*(hist[1]**2)) - mean**2
			# self.min_threshold = np.maximum(mean-var, 0)
			self.max_threshold = mean+var
			# print 'Thresh:', self.min_threshold, self.max_threshold
		else:
			self.face_position = []

class HandDetector:

	def __init__(self, rez=[240,320]):
		self.hand_template = sm.imread(pyKinectTools.configs.__path__[0]+'/fist.png')[:,:,2]
		self.hand_template = (255 - self.hand_template)/255.
		if rez[0] == 240:
			self.hand_template = cv2.resize(self.hand_template, (10,10))
		else:
			self.hand_template = cv2.resize(self.hand_template, (20,20))

	def run(self, im, skin_thresh=[-1,1], n_peaks=3):
		'''
		im : color image
		'''
		im_skin = rgb2lab(im.astype(np.int16))[:,:,2]
		self.im_skin = im_skin
		# im_skin = skimage.exposure.equalize_hist(im_skin)
		# im_skin = skimage.exposure.rescale_intensity(im_skin, out_range=[0,1])
		im_skin *= im_skin > skin_thresh[0]
		im_skin *= im_skin < skin_thresh[1]

		skin_match_c = nd.correlate(-im_skin, self.hand_template)
		self.skin_match = skin_match_c

		# Display Predictions - Color Based matching
		optima = peak_local_max(skin_match_c, min_distance=20, num_peaks=n_peaks, exclude_border=False)
		# Visualize
		if len(optima) > 0:
			optima_values = skin_match_c[optima[:,0], optima[:,1]]
			optima_thresh = np.max(optima_values) / 2
			optima = optima.tolist()

			for i,o in enumerate(optima):
				if optima_values[i] < optima_thresh:
					optima.pop(i)
					break
		self.markers = optima

		return self.markers

class CurveDetector:

	def __init__(self, rez=[240,320]):
		self.template = sm.imread(pyKinectTools.configs.__path__[0]+'/fist.png')[:,:,2]
		self.template = (255 - self.template)/255.
		if rez[0] == 240:
			self.template = cv2.resize(self.template, (10,10))
		else:
			self.template = cv2.resize(self.template, (20,20))

	def run(self, im, skin_thresh=[-1,1], n_peaks=3):
		'''
		im : color image
		'''
		im_skin = im
		self.im_skin = im_skin
		skin_match_c = match_template(im_skin, self.template, pad_input=True)*(im>0)
		self.skin_match = skin_match_c
		# cv2.matchTemplate(im_skin, self.template, cv2.cv.CV_TM_SQDIFF_NORMED)
		# imshow(cv2.matchTemplate(im_skin.astype(np.float32), self.template.astype(np.float32), cv2.cv.CV_TM_CCOEFF_NORMED))

		# Display Predictions - Color Based matching
		optima = peak_local_max(skin_match_c, min_distance=20, num_peaks=n_peaks, exclude_border=False)
		# Visualize
		if len(optima) > 0:
			optima_values = skin_match_c[optima[:,0], optima[:,1]]
			optima_thresh = np.max(optima_values) / 2
			optima = optima.tolist()

			for i,o in enumerate(optima):
				if optima_values[i] < optima_thresh:
					optima.pop(i)
					break
		self.markers = optima

		return self.markers


from scipy.spatial import cKDTree
import os
import cPickle as pickle
import pyflann
from copy import deepcopy
class PoseDatabase:

	def __init__(self, filename, learn=True, append=False, search_joints=None, flann=False, scale=1.0):
		'''
		search_joints :
		'''
		self.filename = filename
		self.keys = None
		# embed()
		if os.path.exists(filename) and (append or not learn):
			data_raw = pickle.load(open(filename))
			if len(data_raw) == 2:
				self.keys = np.array([x.reshape(-1) for x in data_raw[1] if x.shape[0]==9])
				data = np.array([x*scale for x,y in zip(data_raw[0], data_raw[1]) if y.shape[0]==9])
			else:
				data = np.array(data_raw)*scale
			self.database = data

			if self.keys is not None:
				# db_tmp=np.array(self.database).reshape(-1, 14*3)
				# self.db_flann = pyflann.FLANN()
				# self.db_flann.build_index(self.keys)
				self.db_flann = cKDTree(self.keys)
				# self.db_flann.build_index(self.keys)
			if 1:
				self.trees = []
				for d in data:
					if np.all(d!=0):
						if search_joints is not None:
							# print d, d.shape
							self.trees += [cKDTree(d[search_joints])]
						else:
							self.trees += [cKDTree(d)]
			self.count = len(data)
			self.error = np.zeros(self.count)
		else:
			self.database = []
			self.keys = []
			self.count = 0
			self.error = np.zeros(0)

	def update(self, pose, keys=None):
		self.database += [deepcopy(pose)]
		if keys is not None:
			self.keys += [deepcopy(keys)]

	# Dimensionality must be constant for flann
	def query_flann(self, pose, knn=1, return_error=False):
		inds, error = self.db_flann.nn_index(pose.reshape([1,-1]), knn)
		output = self.database[inds]
		if return_error:
			return np.array(output, copy=True), error
		else:
			return np.array(output, copy=True)

	def query_tree(self, pose, knn=1, return_error=False):
		# embed()
		error,inds = self.db_flann.query(pose.reshape([1,-1]), knn)
		output = self.database[inds]
		if return_error:
			return np.array(output, copy=True)[0], error[0]
		else:
			return np.array(output, copy=True)[0]

	def query(self, pose, knn=1, return_error=False):
		# for i,tree in enumerate(self.trees):
			# self.error[i] = tree.query(pose)[0].sum()
		self.error = np.array([tree.query(pose)[0].sum() for tree in self.trees])

		error_sorted = np.argsort(self.error)
		output = self.database[error_sorted]
		# output = []
		# for i in xrange(knn):
			# output += [self.database[error_sorted[i]]]
		if return_error:
			return np.array(output, copy=True), self.error#error_sorted
			# return np.array(output, copy=True), error_sorted[:knn]
		else:
			return np.array(output, copy=True)

	def weighted_query(self, markers, marker_types=None, knn=1):

		self.error = np.array([tree.query(markers)[0].sum() for i,tree in enumerate(self.trees)])
		# for i,tree in enumerate(self.trees):
		# 	self.error[i] = tree.query(markers)[0].sum()

		# output = []
		# error_sorted = np.argsort(self.error)
		# for i in xrange(knn):
		# 	output += [self.database[error_sorted[i]]]
		# if knn > 0:
		# 	output = np.mean(output,0)
		# output = deepcopy(self.database[np.argmin(self.error)])
		return deepcopy(np.array(output))


	def reverse_query(self, pose):
		tree = cKDTree(pose)
		for i,d in enumerate(self.database):
			self.error[i] = tree.query(d)[0].sum()
			# self.error[i] = tree.query(pose[:,:2])[0].sum()
		# print np.sort(self.error)[:5]
		return deepcopy(self.database[np.argmin(self.error)])


	def save(self):
		# pickle.dump(self.database, open(self.filename, 'w'))
		pickle.dump([self.database, self.keys], open(self.filename, 'w'))



# class HierarchicalBoundingBoxes:

# 	n_levels = 1
# 	bounding_boxes = []
# 	def __init__(self):
# 		pass



# skel_init = np.array([
# 	[-650,0,0], # head
# 	[-425,0,0], # neck
# 	[-150,0,0],# torso
# 	[-425,-150,0],# l shoulder
# 	[-150,-250,0],# l elbow
# 	[50,-350,0],# l hand
# 	[-425,150,0],# r shoulder
# 	[-150,250,0],# r elbow
# 	[50,350,0],# r hand
# 	[000,-110,0],# l hip
# 	[450,-110,0],# l knee
# 	[600,-110,0],# l foot
# 	[000,110,0],# r hip
# 	[450,110,0],# r knee
# 	[600,110,0],# r foot
# 	])

HEAD,NECK,TORSO,L_SHOUL,L_ELBOW,L_HAND,R_SHOUL,R_ELBOW,R_HAND,\
L_HIP,L_KNEE,L_FOOT,R_HIP,R_KNEE,R_FOOT = range(15)

pose_library = [
	np.array([
			[-650,0,0], # head
			[-525,0,0], # neck
			[-300,0,0],# torso
			[-525,-150,0],# l shoulder
			[-225,-250,0],# l elbow
			[0,-350,0],# l hand
			[-425,150,0],# r shoulder
			[-200,250,0],# r elbow
			[0,350,0],# r hand
			[-50,-110,0],# l hip
			[350,-110,0],# l knee
			[650,-200,0],# l foot
			[-50,110,0],# r hip
			[350,110,0],# r knee
			[650,200,0],# r foot
			])
]

pose_variance = np.array([
		50, # head
		50, # neck
		50,# torso
		50,# l shoulder
		50,# l elbow
		50,# l hand
		50,# r shoulder
		50,# r elbow
		50,# r hand
		50,# l hip
		50,# l knee
		50,# l foot
		50,# r hip
		50,# r knee
		50,# r foot
		])*2

def get_15_joint_properties():
	joints = [HEAD,NECK,TORSO,L_SHOUL,L_ELBOW,L_HAND,R_SHOUL,R_ELBOW,\
			R_HAND,L_HIP,L_KNEE,L_FOOT,R_HIP,R_KNEE,R_FOOT]

	skel_init = pose_library[0][:,joints]
	joint_variance = pose_variance[:,joints]
	constraint_links = np.array([
		[0,1],[1,2],[3,6],[0,3],[0,6],#Head to neck, neck to torso, shoulders, head to shoulders
		[1,3],[3,4],[4,5], # Left arm
		[3,9],[6,12], # shoudlers to hips
		[1,6],[6,7],[7,8], # Right arm
		[2,9],[9,10],[10,11], #Left foot
		[2,12],[12,13],[13,14], #Right foot
		[9,12] #Bridge hips
		])
	features_joints = [[0], [0,5,8], range(len(joints))]
	convert_to_kinect = j15_to_kinect_skel
	return skel_init, joint_variance, constraint_links, features_joints, convert_to_kinect

def get_14_joint_properties():
	joints = [HEAD,TORSO,L_SHOUL,L_ELBOW,L_HAND,R_SHOUL,R_ELBOW,\
			R_HAND,L_HIP,L_KNEE,L_FOOT,R_HIP,R_KNEE,R_FOOT]

	skel_init = pose_library[0][joints]
	joint_variance = pose_variance[joints]
	# constraint_links = np.array([
	# 	[0,1],[0,2],[0,5], [2,5],#Head to torso, shoulders, head to shoulders
	# 	[2,3],[3,4], # Left arm
	# 	[2,8],[5,11],[2,1],[5,1], # shoudlers to hips
	# 	[5,6],[6,7], # Right arm
	# 	[1,8],[8,9],[9,10], #Left foot
	# 	[1,11],[11,12],[12,13], #Right foot
	# 	[8,11] #Bridge hips
	# 	])
	constraint_links = np.array([
		[0,1], [2,5],#Head to torso, shoulders, head to shoulders
		[2,3],[3,4], # Left arm
		[2,1],[5,1], # shoudlers to hips
		[5,6],[6,7], # Right arm
		[8,9],[9,10], #Left foot
		[11,12],[12,13], #Right foot
		[8,11] #Bridge hips
		])
	skel_parts = np.array([[0,1], [2,3,4], [5,6,7],[8,9,10],[11,12,13]])
	features_joints = [[HEAD], [HEAD,L_HAND,R_HAND], range(len(joints))]
	convert_to_kinect = j14_to_kinect_skel
	return skel_init, joint_variance, constraint_links, features_joints, skel_parts, convert_to_kinect

def get_13_joint_properties():
	joints = [HEAD,L_SHOUL,L_ELBOW,L_HAND,R_SHOUL,R_ELBOW,\
			R_HAND,L_HIP,L_KNEE,L_FOOT,R_HIP,R_KNEE,R_FOOT]

	skel_init = pose_library[0][joints]
	joint_variance = pose_variance[joints]
	constraint_links = np.array([
		[0,1],[0,4],[1,4],#Head to shoulders, shoulders together
		[1,2],[2,3], # Left arm
		[1,7],[4,10],[1,10],[4,7], # shoudlers to hips
		[4,5],[5,6], # Right arm
		[7,8],[8,9], #Left foot
		[10,11],[11,12], #Right foot
		[7,10] #Bridge hips
		])
	skel_parts = np.array([[0], [1,2,3], [4,5,6],[7,8,9],[10,11,12]])
	# features_joints = [[0], [0,3,6], range(len(joints))]
	features_joints = [[0], [0,3,6], [0,1,3,4,6,9,12]]
	convert_to_kinect = j13_to_kinect_skel
	return skel_init, joint_variance, constraint_links, features_joints, skel_parts, convert_to_kinect


def get_11_joint_properties():
	joints = [HEAD,L_SHOUL,L_ELBOW,L_HAND,R_SHOUL,R_ELBOW,R_HAND,\
				L_KNEE,L_FOOT,R_KNEE,R_FOOT]

	skel_init = pose_library[0][joints]
	joint_variance = pose_variance[joints]
	constraint_links = np.array([
		[0,1],[0,4],[1,4],#Head to shoulders, shoulders together
		[1,2],[2,3], # Left arm
		[4,5],[5,6], # Right arm
		[7,8], #Left foot
		[9,10], #Right foot
		])
	features_joints = [[0], [0,3,6], range(11)]
	convert_to_kinect = j11_to_kinect_skel
	return skel_init, joint_variance, constraint_links, features_joints, convert_to_kinect

