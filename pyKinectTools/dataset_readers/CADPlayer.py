''' Load Cornel_Action_Dataset 120 Dataset '''

import numpy as np
import os
import scipy.misc as sm
import itertools as it
import cv2

import pyKinectTools
from pyKinectTools.utils.DepthUtils import CameraModel, skel2depth, depthIm_to_colorIm, world2depth, world2rgb, get_kinect_transform #depthIm2XYZ, depth2world
# from pyKinectTools.utils.SkeletonUtils import *
from pyKinectTools.dataset_readers.BasePlayer import BasePlayer
# from pyKinectTools.algs.BackgroundSubtraction import fill_image, StaticModel, extract_people
# from pyKinectTools.utils.VideoViewer import VideoViewer
# vv = VideoViewer()

from IPython import embed

'''
Each action is done 3 times. [except making cereal which has 4]

CAD120 Filestructure
--SubjectX_rgbd_images
	--activity_names
		--iteration_timestamp
			Depth_X.png
			RGB_255.png
--SubjectX_annotations
	--activity_names
		timestamp_globalTransform.txt
		timestamp_globalTransform.bag
		timestamp_obj1.txt		
		timestamp.txt
		labeling.txt
		activityLabel.txt		
--features_cad120_ground_truth_segmentation
	--features_binary_svm_format
		timestamp.txt
	--segments_svm_format
		timestamp.txt

'''


''' Load Labels '''

def read_labels(base_dir='.', subjects=[1], actions=[0], instances=[0]):	

	action_names = os.listdir(base_dir+'/Subject1_rgbd_images')
	action_names = filter(lambda x:x[0]!='.', action_names)

	for s in subjects:
		for a in actions:
			label_folder = "{}/Subject{}_annotations/{}/".format(base_dir,s,action_names[a])
			instance_names = os.listdir(label_folder)
			instance_names = [x[:-4] for x in instance_names if x[:-4].isdigit()]

			filename = "{}/activityLabel.txt".format(label_folder)
			activity_labels = read_activity_labels(filename)

			filename = "{}/labeling.txt".format(label_folder)
			subactivity_labels = read_subaction_labels(filename)
			
			for i in instances:
				objects = []
				object_filenames = os.listdir(label_folder)
				object_filenames = [x for x in object_filenames if x.find(instance_names[i])>=0 and x.find("obj")>=0]
				for o_name in object_filenames:
					filename = "{}/{}".format(label_folder, o_name)
					tmp = read_object_labels(filename)
					objects += [tmp]
				
				filename = "{}/{}_globalTransform.txt".format(label_folder, instance_names[i])
				global_transform = read_global_transform(filename)

				activity = activity_labels[int(instance_names[i])]
				subactivity = subactivity_labels[int(instance_names[i])]

				yield activity, subactivity, objects, s


def read_object_labels(filename="."):
	'''
	e.g. 0510173051_obj1.txt

	A file which provides the object annotations for object (obj_id) in activity #
	format: frame_number,object_id,x1,y1,x2,y2,t1,t2,t3,t4,t5,t6

		x1,y1: upper left corner of the object bounding box	
		x2,y2: lower right corner of the object bounding bos	
		t1-t6: transform matrix matching the SIFT features to the previous frame.	
	'''
	# filename = '0510175411_obj1.txt'
	data = np.fromfile(filename, sep=',')
	end = np.floor(len(data) / 12.)*12
	data = data[:end].reshape([-1,12])

	# data = np.loadtxt(filename, str, delimiter=",")
	# [:,:-1].astype(np.float)
	frames = data[:,0].astype(np.int)
	object_ids = data[:,1].astype(np.int)
	corner_topleft = np.array([data[:,2], data[:,3]], dtype=np.int).T
	corner_bottomright = np.array([data[:,4], data[:,5]], dtype=np.int).T
	sift_rotation = data[:,6:10].reshape([-1,2,2])
	sift_translation = data[:,10:]

	objects = []
	for i,_ in enumerate(frames):
		objects += [{'ID':object_ids[i], 'topleft':corner_topleft[i], 'bottomright':corner_bottomright[i],
					'rotation':sift_rotation[i], "translation":sift_translation[i]}]

	return objects

def read_global_transform(filename="."):
	'''
	e.g. 0510173051_globalTransform.txt

	The transform to be applied to the poinclouds of activity # to make z axis vertical.
	'''
	# filename = '0510175411_globalTransform.txt'
	transform = np.loadtxt(filename, delimiter=",")
	return transform

def read_activity_labels(filename="."):
	'''
	e.g. activityLabel.txt
	
	format : id,activity_id,subject_id,object_id:object_type,object_id:object_type,..
		id:		ten-digit string (e.g. 0510175411, etc.)
		activity_id:	high-level activity identifier (e.g. placing, eating, etc.)
		object_id:	object identifier (e.g. 1, 2, etc.)
		obejct_type:	type of object (e.g. cup, bowl, etc.)	
	'''
	# filename = 'activityLabel.txt'
	# data = np.loadtxt(filename, str, delimiter=",")
	# data = np.fromfile(filename, sep=',')
	data = open(filename).read().split("\n")
	data = [x.split(",") for x in data if len(x)>0]
	ids = [int(x[0]) for x in data]
	activity = [x[1] for x in data]
	subject = [x[2] for x in data]
	objects = [{}]*len(ids)
	for f,_ in enumerate(ids):
		for i in range(2, np.shape(data[f])[0]-1):
			obj_id, obj = data[f][3].split(":")
			objects[f].update({obj_id:obj})


	output = {}
	for i,ID in enumerate(ids):
		output.update({ID:{'ID':ids[i], 'activity':activity[i], 'subject':subject[i], 'objects':objects[i]}})

	return output


def read_subaction_labels(filename="."):
	'''
	e.g. labeling.txt
	
	A file which provides the activity and affordance annotations
	format:  id,start_frame,end_frame,sub-activity_id,affordance_1_id,affordance_2_id,...
		
		id:		ten-digit string (e.g. 0510175411, etc.)
		start_frame:	frame number corresponding to the begining of sub-activity
		end_frame:	frame number corresponding to the ending of sub-activity
		sub-activity_id: sub-activity identifer
		affordance_1_id: affordance identifier of object 1
		affordance_2_id: affordance identifier of object 2 

	'''	
	# filename = 'labeling.txt'
	# data = np.loadtxt(filename, str, delimiter=",")
	data = open(filename).read().split("\n")
	data = [x.split(",") for x in data if len(x)>0]
	ids = [int(x[0]) for x in data]
	start_frame = [int(x[1]) for x in data]
	end_frame = [int(x[2]) for x in data]
	sub_actions = [x[3] for x in data]

	object_affordances = []
	for f,_ in enumerate(ids):
		object_affordances += [{}]
		for i in range(4, np.shape(data[f])[0]):
			obj = data[f][i]
			object_affordances[f].update({i-3:obj})

	output = {x:{} for x in np.unique(ids)}
	sequence_ids = {x:0 for x in np.unique(ids)}
	for i,ID in enumerate(ids):
		sequence_ids[ID]+=1
		output[ID].update({sequence_ids[ID]:{'sequence':sequence_ids[ID], 'ID':ID,
		 				'start':start_frame[i], 'stop':end_frame[i],
		 				'subaction':sub_actions[i], 'objects':object_affordances[i]}})

	return output

''' Load Data '''

def read_data(base_dir='.', subjects=[1], actions=[0], instances=[0], \
			get_depth=True, get_rgb=True, get_skel=True):
	'''
	single channel 16-bit PNG
	'''
	
	rgb_ims = None
	depth_ims = None
	skels = None

	action_names = os.listdir(base_dir+'/Subject1_rgbd_images/')
	action_names = filter(lambda x:x[0]!='.', action_names)

	for s in subjects:
		for a in actions:
			action_folder = "{}/Subject{}_rgbd_images/{}/".format(base_dir,s,action_names[a])
			label_folder = "{}/Subject{}_annotations/{}/".format(base_dir,s,action_names[a])
			instance_filenames = os.listdir(action_folder)
			instance_filenames = filter(lambda x: x.isdigit(), instance_filenames)
			
			for i in instances:
				# Read image files
				files = os.listdir(action_folder+instance_filenames[i])
				rgb_files = filter(lambda x: x.find('RGB')>=0, files)
				rgb_files = sorted(rgb_files, key=lambda x: int(x[:-4].split("_")[1]))
				depth_files = filter(lambda x: x.find('Depth')>=0, files)
				depth_files = sorted(depth_files, key=lambda x: int(x[:-4].split("_")[1]))

				im_count = len(rgb_files)
				if get_rgb:
					rgb_ims = np.zeros([im_count,480,640,3], dtype=np.uint8)
					for ii,f in enumerate(rgb_files):
						filename = action_folder+instance_filenames[i]+"/"+f					
						rgb_ims[ii] = sm.imread(filename)

				im_count = len(depth_files)
				if get_depth:
					depth_ims = np.zeros([im_count,480,640], dtype=np.float)
					for ii,f in enumerate(depth_files):
						filename = action_folder+instance_filenames[i]+"/"+f					
						depth_ims[ii] = sm.imread(filename) * 100. * 0.8
						# tmp = 1091.5 - sm.imread(filename)
						# depth_ims[ii] = (3480000./tmp)
						

				# Read skeleton files
				if get_skel:
					filename = label_folder + instance_filenames[i] + ".txt"
					skel_raw = np.fromfile(filename, sep=',')
					skel = skel_raw[:-1].reshape([im_count,-1])
					skel_frames = skel[:,0]
					# 15 joints. 11 have orientation+position, 4 have just position
					# skel_tmp = skel[:,1:].reshape([-1,14])
					skel_orn = []
					skel_pos = []
					for ii in xrange(11):
						skel_orn += [skel[:,1+ii*14:1+ii*14+10]]
						skel_pos += [skel[:,11+ii*14:11+ii*14+4]]
					for ii in xrange(4):
						skel_pos += [skel[:,155+ii*4:155+ii*4+4]]
					skel_orn = np.array(skel_orn)
					
					skel_orn_conf = skel_orn[:,:,-1].T.astype(np.int)
					skel_orn = skel_orn[:,:,:-1].reshape([-1,11,3,3])

					skel_pos = np.array(skel_pos)
					skel_pos_conf = skel_pos[:,:,-1].T.astype(np.int)
					skel_pos = skel_pos[:,:,:3]
					skel_pos = np.array([skel_pos[:,i] for i in range(skel_pos.shape[1])])
					# skel_pos = skel_pos[:,:,:-1].reshape([15, -1])
					
					skels = {'pos':skel_pos, 'pos_conf':skel_pos_conf,
							'orn':skel_orn, 'orn_conf':skel_orn_conf}

				yield rgb_ims, depth_ims, skels


# def create_filenames(base_dir, subjects, actions, instances):
	
# 	# Get action names
# 	data_folder = "{}/Subject{}_rgbd_images/".format(base_dir,1)
# 	action_folders = os.listdir(base_dir+"/"+data_folder)
# 	action_folders = [x for x in tmp_folders if x[0]!='.']
# 	action_names = {x:action_folders[x] for x in np.arange(len(action_folders))}

# 	for s in subjects:
# 		label_folder = "{}/Subject{}_annotations/".format(base_dir,s)
# 		data_folder = "{}/Subject{}_rgbd_images/".format(base_dir,s)

# 		action_folders = os.listdir(base_dir+"/"+data_folder)
# 		action_folders = [x for x in tmp_folders if x[0]!='.']
		
# 		actions = {x:action_folders[x] for x in np.arange(len(action_folders))}
# 		action_folders = {x:data_folder+action_folders[x] for x in np.arange(len(action_folders))}




if 0:
	player = CADPlayer()


class CADPlayer(BasePlayer):
	def __init__(self, subjects=[1], actions=[0], instances=[0], **kwargs):
		'''
		instances: which instances of an action to use e.g. [0,2]
		'''
		super(CADPlayer, self).__init__(**kwargs)

		# Settings
		self.deviceID = "CAD120 Player"
		self.instances = instances
		# Get data filenames
		self.data = read_data(self.base_dir, subjects, actions, instances, get_depth=self.get_depth, get_rgb=self.get_color, get_skel=self.get_skeleton)
		self.labels = read_labels(self.base_dir, subjects, actions, instances)
		self.data_count = len(list(it.product(subjects, actions, instances)))
		self.data_index = 0

		# Get calibration
		# self.camera_model = CameraModel(pyKinectTools.configs.__path__[0]+"/Kinect_Depth_Param.yml".format(1))
		self.camera_model = CameraModel(pyKinectTools.configs.__path__[0]+"/Kinect_Color_Param.yml".format(1))
		# self.camera_model = CameraModel("/Users/colin/code/pyKinectTools/pyKinectTools/configs/Kinect_Color_CAD_Param.yml")
		self.kinect_transform = get_kinect_transform(pyKinectTools.configs.__path__[0]+"/Kinect_Transformation.txt".format(1))

		self.camera_model.set_transform(self.kinect_transform)
		self.color_stack, self.depth_stack, self.skel_stack = None,None,None
		self.mask = 1
		# Initialize
		self.player = self.run()
		# self.next(1)

	def next(self, frames=1):
		'''
		frames : skip (this-1) frames
		'''
		# Update frame
		try:
		# if 1:
			for i in range(frames):
					self.player.next()			
			return True
		except:
			print "Done playing video"
			return False

	def next_sequence(self):
		'''
		'''
		# Update sequence
		self.frame = self.framecount
		# self.next()

	def run(self):
		'''
		'''
		# Read data from new file
		while self.data_index < self.data_count:
			self.data_index += 1

			# Load videos
			del self.color_stack, self.depth_stack, self.skel_stack
			self.color_stack, self.depth_stack, self.skel_stack = self.data.next()
			# self.depth_stack *= 100.
			self.activity, self.subactivity, self.scene_objects, self.subject = self.labels.next()
			self.action = self.activity['activity']

			if self.depth_stack is not None:
				self.framecount = self.depth_stack.shape[0]
			elif self.color_stack is not None:
				self.framecount = self.color_stack.shape[0]
			else: 
				self.framecount = self.skel_stack['pos'].shape[0]

			print 'Starting video #{} of {}'.format(self.data_index, self.data_count)#, self.kinect_folder_names[-1]
			print "Action:", self.action

			self.subactions = np.zeros(self.framecount, dtype=np.int)
			# self.objects = []
			current = 1			
			for i in xrange(self.framecount):			
				for ii,v in self.subactivity.items():
					if v['start'] <= i < v['stop']:
						current = ii
				self.subactions[i] = current
				# self.objects

			self.frame = 0
			while self.frame < self.framecount:
				self.depthIm = self.depth_stack[self.frame,:,:].clip(0,4500) if self.depth_stack is not None else None
				self.colorIm = self.color_stack[self.frame] if self.color_stack is not None else None
				self.colorIm = self.colorIm[:,:,[2,1,0]] if self.color_stack is not None else None
				self.subaction = self.subactivity[self.subactions[self.frame]]['subaction']
				self.objects = []
				for s in self.scene_objects:
					self.objects += [s[self.frame]]
				self.object_names = self.subactivity[self.subactions[self.frame]]['objects']

				# Transform skeleton to kinect space
				skels_pos = self.skel_stack['pos'][self.frame]
				self.skels_orn = self.skel_stack['orn'][self.frame]
				self.skels_pos_conf = self.skel_stack['pos_conf'][self.frame]
				self.skels_orn_conf = self.skel_stack['orn_conf'][self.frame]
				self.users = [skels_pos]

				# self.users[0][:,2] *= -1
				self.users_uv = [ self.camera_model.world2im(self.users[0], [480,640]) ]
				self.users_uv[0][:,0] = 480 - self.users_uv[0][:,0]
				# if self.depth_stack is not None:
				# 	self.depthIm = display_skeletons(self.depthIm, self.users_uv[0], skel_type='CAD')
				

				# if 0:
				# 	from pylab import *
				# 	for ii,i in enumerate(self.users_uv[0]):
				# 		scatter(i[1], -i[0])
				# 		annotate(str(ii), (i[1], -i[0]))
				# 	axis('equal')
				# 	show()

				# self.colorIm = display_skeletons(self.colorIm, self.users_uv_msr[0], skel_type='Kinect')
				# self.update_background()

				self.frame += 1
				yield



