
"""
Felzenszwalb w/ 1 channel 133 ms per loop
Felzenszwalb w/ 3 channels 420 ms per loop
Adaptive 55ms per loop
HOG on person bounding box: 24 ms
HOG on person whole body: 101 ms for 4x4 px/cell and 70 ms for 8x8 px/cell for 24*24 px boxes
HOG per extrema 2-3 ms

It's not computationally efficient to compute hogs everywhere? What about multi-threaded? gpu?
"""


if 0:
	# Chi Squared Kernel
	svm = SVC(kernel=chi2_kernel)
	svm.fit(features['Gray_HOGs'], labels)
	svm.score(features['Gray_HOGs'], labels)

class MultiChannelClassifier:
	channel_kernels = []
	channel_names = []
	channel_means = []
	channel_data = []

	cl = None #classifier
	kernel = None

	def add_kernel(self, name, kernel, data):
		"""
		kernel : (e.g. AdditiveChi2Sampler(sample_steps=1))
		"""
		data = kernel.fit_transform(data)
		self.channel_data += [data]
		kernel_mean = data.mean()

		self.channel_names += [name]
		self.channel_kernels += [kernel]
		self.channel_means += [kernel_mean]

	def fit(self, classifier, labels, kernel=None):
		"""
		classifier : (e.g. SGDClassifier(alpha=.0001, n_jobs=-1))
		kernel : (e.g. RBFSampler())
		"""
		self.cl = classifier
		data = np.sum()

		# for i in channel_kernels
		# if kernel != None:


	def predict(self, data):
		"""
		data : (e.g. {'Gray_HOGs':data, 'Color_LBPs':data}
		"""
		output = []
		data_transform = 0
		for i in len(data):
			output += [self.chi_kernels[i].transform(data[i])]


	def fit_transform(self, classifier, labels, kernel=None):
		self.fit(classifier, labels, kernel)
		self.predict()

def recenter_image(im):
	"""
	"""
	n_height, n_width = im.shape
	com = nd.center_of_mass(im)
	if any(np.isnan(com)):
		return im

	im_center = im[(com[0]-n_height/2):(com[0]+n_height/2)]
	offset = [(n_height-im_center.shape[0]),(n_width-im_center.shape[1])]

	if offset[0]%2 > 0:
		h_odd = 1
	else:
		h_odd = 0
	if offset[1]%2 > 0:
		w_odd = 1
	else:
		w_odd = 0

	im[offset[0]/2:n_height-offset[0]/2-h_odd, offset[1]/2:n_width-offset[1]/2-w_odd] = im_center

	return im

def visualize_top_ims(ims, labels, count=25):
	"""
	ims : (e.g. ims_rgb, ims_gray)
	"""
	count = 25
	ims = ims_rgb
	n_classes = len(np.unique(labels))
	for j in xrange(n_classes):
		figure(j)
		i=0
		i2=0
		try:
			while i < count:
				i2 += 1#*ims.shape[0]/count/n_classes
				if labels[i2] == j:
					subplot(5,5,i);
					imshow(ims[i2])
					i += 1
		except:
			print 'Error'
			pass
	show()

def visualize_filters(filters):
	# Viz filters
	for i,im in enumerate(filters):
		ii = i*2
		subplot(3,2,ii+1)
		imshow(im*(im>0))
		subplot(3,2,ii+2)
		imshow(im*(im<0))
	show()

if 0:
	# Hard Mining
	X = features['Gray_HOGs']
	for joint_class in range(2):
		# Train on all data
		svm_head = SGDClassifier(n_iter=100, alpha=.0001)
		labels_head = labels == joint_class
		svm_head.fit(X, labels_head.astype(np.int))
		svm_head.score(X, labels_head.astype(np.int))
		filters = [hog2image(c, [height,width], orientations=5, pixels_per_cell=hog_size, cells_per_block=[3,3]) for c in svm_head.coef_]

		# Find false positives in negative data
		neg_data = X[-labels_head]
		pred = svm_head.predict(neg_data)
		# neg_data =
		print pred.sum(), "errors in class", joint_class
		# Retrain on all of the positives and the false positives in the negative samples
		svm_head = SGDClassifier(n_iter=100, alpha=.0001)



def train(ims_rgb, ims_depth, labels):
	"""
	"""

	# --------- Setup --------

	# Joint options: head, shoulders, hands, feat
	joint_set = ['head', 'hands']
	n_classes = len(joint_set)
	# Image options: gray, lab
	image_set = ['gray', 'lab']
	# Feature options: Gray_HOGs, Color_LBPs, Color_Histogram
	feature_set = ['Gray_HOGs']
	features = {}

	model_params = {}
	model_params = {'feature_set':feature_set, 'image_set':image_set, \
				'joints':joint_set, 'patch_size':patch_size}

	n_samples = labels.shape[0]
	height, width = ims_rgb[0].shape[:2]

	mcc = MultiChannelClassifier()
	# mcc.add_classifier(SGDClassifier(n_iter=100, alpha=.0001, n_jobs=-1))

	#  --------- Conversions ---------

	print 'Converting images'
	if 'gray' in image_set:
		ims_gray = (np.array([rgb2gray(ims_rgb[i]) for i in range(n_samples)])*255).astype(np.uint8)
	if 'lab' in image_set:
		ims_lab = np.array([rgb2lab(ims_rgb[i]) for i in range(n_samples)])

	print 'Relabeling classes'
	joints = {'head':[0], 'shoulders':[2,5], 'hands':[4,7], 'feat':[10,13]}
	classes = [joints[j] for j in joint_set]

	# Group into classes
	labels_orig = labels.copy()
	for c in xrange(n_samples):
		this_class = n_classes
		for i in xrange(n_classes):
			if labels[c] in classes[i]:
				this_class = i
				break
		labels[c] = this_class
		# ims_c[c] = recenter_image(ims_c[c])

	# visualize_top_ims(ims_rgb, labels)

	# --------- Calculate features ---------
	print ""
	print '--Starting feature calculations--'

	if 'Gray_HOGs' in feature_set:
		print 'Calculating Grayscale HOGs'
		model_params['hog_size'] = (8,8)
		model_params['hog_cells']= (1,1)
		model_params['hog_orientations'] = 5
		hogs_c = Parallel(n_jobs=-1)(delayed(hog)(im, 5, (8,8), (1,1), False, False) for im in ims_gray)
		hogs_c = np.array(hogs_c)
		# Ensure no negative values
		hogs_c[hogs_c<0] = 0
		features['Gray_HOGs'] = hogs_c
		mcc.add_kernel('Gray_HOGs', AdditiveChi2Sampler(1), hogs_c)

	if 'Depth_HOGs' in feature_set:
		print 'Calculating Depth-based HOGs'
		model_params['hog_size'] = (8,8)
		model_params['hog_cells']= (3,3)
		model_params['hog_orientations'] = 9
		hogs_c = Parallel(n_jobs=-1)(delayed(hog)(im, 5, (8,8), (3,3), False, False) for im in ims_depth)
		hogs_c = np.array(hogs_c)
		features['Depth_HOGs'] = hogs_c

	if 'Color_LBPs' in feature_set:
		print 'Calculating Color LBPs'
		model_params['lbp_px'] = 16
		model_params['lbp_radius'] = 2
		lbps_tmp = np.array(Parallel(n_jobs=-1)(delayed(local_binary_pattern)(im, P=16, R=2, method='uniform') for im in ims_gray ))
		lbps_c = np.array(Parallel(n_jobs=-1)(delayed(np.histogram)(lbp, normed=True, bins = 18, range=(0,18)) for lbp in lbps_tmp ))
		lbps_c = np.array([x[0] for x in lbps_c])
		# Get rid of background and ensure no negative values
		lbps_c[:,lbps_c.argmax(1)]=0
		lbps_c = (lbps_c.T/lbps_c.sum(1)).T
		lbps_c = np.nan_to_num(lbps_c)
		features['Color_LBPs'] = lbps_c

	if 'Depth_LBPs' in feature_set:
		print 'Calculating Depth LBPs'
		lbps_tmp = np.array(Parallel(n_jobs=-1)(delayed(local_binary_pattern)(im, P=16, R=2, method='uniform') for im in ims_depth ))
		lbps_z = np.array(Parallel(n_jobs=-1)(delayed(np.histogram)(lbp, normed=True, bins = 18, range=(0,18)) for lbp in lbps_tmp ))
		lbps_z = np.array([x[0] for x in lbps_z])
		# Get rid of background and ensure no negative values
		lbps_z[:,lbps_z.argmax(1)]=0
		lbps_z = (lbps_z.T/lbps_z.sum(1)).T
		lbps_z = np.nan_to_num(lbps_z)
		features['Depth_LBPs'] = lbps_z

	if 'Color_Histograms' in feature_set:
		print 'Calculating Color Histograms'
		color_hists = np.array([np.histogram(ims_lab[i,:,:,0], 10, [1,255])[0] for i in xrange(n_samples)])
		features['Color_Histograms'] = color_hists

	if 'Curvature' in feature_set:
		print 'Calculating Geometric Curvature'
		# TODO
		# features['Geometric_Curvature'] =

	if 'Hand_Template' in feature_set:
		print 'Calculating Hand Template features'
		# TODO
		# features['Hand_Template'] =

	if 'Face_Template' in feature_set:
		print 'Calculating Face Template features'
		# TODO
		# features['Face_Template'] =


	# Transfrom  pca/chi/rbf
	# pca = PCA(13)
	# rbf = RBFSampler()
	data = features['Gray_HOGs']
	labels = np.array([labels[i] for i,x in enumerate(data) if sum(x)!= 0])
	data = np.vstack([x/x.sum() for x in features['Gray_HOGs'] if sum(x)!= 0])
	# data = data / np.repeat(data.sum(-1)[:,None], data.shape[-1], -1)
	# data = np.nan_to_num(data)
	chi2 = AdditiveChi2Sampler(1)
	model_params['Chi2'] = chi2
	training_data = chi2.fit_transform(data)
	training_labels = labels

	# --------- Classification ---------
	print ""
	print 'Starting classification training'
	svm = SGDClassifier(n_iter=100, alpha=.0001, class_weight="auto", l1_ratio=0, fit_intercept=True, n_jobs=-1)
	svm.fit(training_data, training_labels)
	print "Done fitting both SVM. Self score: {0:.2f}%".format(svm.score(training_data, training_labels)*100)
	model_params['SVM'] = svm

	filters = svm.coef_
	# filters = [hog2image(c, [height,width], orientations=5, pixels_per_cell=model_params['hog_size'], cells_per_block=[3,3]) for c in svm.coef_]
	model_params['filters'] = filters
	# filters = [f*(f>0) for f in filters]

	# rf = RandomForestClassifier(n_estimators=50)
	# rf.fit(training_both_kernel, labels)
	# print "Done fitting forest. Self score: {0:.2f}%".format(rf.score(training_both_kernel, labels)*100)
	# model_params['rf'] = rf

	# Grid search for paramater estimation
	if 0:
		from sklearn.grid_search import GridSearchCV
		from sklearn.cross_validation import cross_val_score
		params = {'alpha': [.0001],
				'l1_ratio':[0.,.25,.5,.75,1.]}
		grid_search = GridSearchCV(svm_both, param_grid=params, cv=2, verbose=3)
		grid_search.fit(training_both_kernel, labels)

	# --------- Save Model Information ---------
	model_params['Description'] = ''
	with open('model_params.dat', 'w') as f:
		pickle.dump(model_params, f)
	print 'Parameters saved. Process complete'




''' ---MAIN--- '''

	if learn:
		n_joints = 14
		rez = [480,640]
		ims_depth = np.empty([n_frames*n_joints, patch_size, patch_size], dtype=np.uint16)
		ims_rgb = np.empty([n_frames*n_joints, patch_size, patch_size, 3], dtype=np.uint8)
		labels = np.empty(n_frames*n_joints, dtype=np.int)
	else:
		model_params = pickle.load(open('model_params.dat', 'r'))
		patch_size = model_params['patch_size']
		svm = model_params['SVM']
		filters = model_params['filters']
		filters *= filters > 0
		n_filters = len(filters)

		true_pos = {'hands':0, 'head':0}
		false_pos = {'hands':0, 'head':0}


		if learn:
			# Use offsets surounding joints
			for i,j_pos in enumerate(skel_msr_im):
				x = j_pos[1]
				y = j_pos[0]
				if x-patch_size/2 >= 0 and x+patch_size/2 < height and y-patch_size/2 >= 0 and y+patch_size/2 < width:
					ims_rgb[frame_count*n_joints+i] = im_color[x-patch_size/2:x+patch_size/2, y-patch_size/2:y+patch_size/2]
					ims_depth[frame_count*n_joints+i] = im_depth[x-patch_size/2:x+patch_size/2, y-patch_size/2:y+patch_size/2]
					labels[frame_count*n_joints+i] = i
				else:
					labels[frame_count*n_joints+i] = -1


		else:
			# Ensure the size is a multiple of HOG size
			# box = (slice(box[0].start, ((box[0].stop - box[0].start) / model_params['hog_size'][0])*model_params['hog_size'][0] + box[0].start),
					# slice(box[1].start, ((box[1].stop - box[1].start) / model_params['hog_size'][1])*model_params['hog_size'][1] + box[1].start))


				if 'gray' in model_params['image_set']:
					im_gray = (rgb2gray(cam.colorIm)*255).astype(np.uint8)
					im_gray *= mask
				if 'lab' in model_params['image_set']:
					# im_lab = (rgb2lab(cam.colorIm)*255).astype(np.uint8)
					# im_lab = rgb2lab(cam.colorIm[box][:,:,[2,1,0]].astype(np.uint16))[:,:,1]
					im_skin = rgb2lab(cam.colorIm[box].astype(np.int16))[:,:,1]
					# im_skin = skimage.exposure.equalize_hist(im_skin)
					# im_skin = skimage.exposure.rescale_intensity(im_skin, out_range=[0,1])
					im_skin *= im_skin > face_detector.min_threshold
					im_skin *= im_skin < face_detector.max_threshold
					# im_skin *= face_detector>.068

					hand_template = sm.imread('/Users/colin/Desktop/fist.png')[:,:,2]
					hand_template = (255 - hand_template)/255.
					if height == 240:
						hand_template = cv2.resize(hand_template, (10,10))
					else:
						hand_template = cv2.resize(hand_template, (20,20))
					skin_match_c = nd.correlate(im_skin, hand_template)

					# hand_template = (1. - hand_template)
					# hand_template = cv2.resize(hand_template, (20,20))
					# skin_match_d = nd.correlate(im_depth[box]/256., 0.5-hand_template)

					# Display Predictions - Color Based matching
					optima = peak_local_max(skin_match_c, min_distance=20, num_peaks=3, exclude_border=False)
					if len(optima) > 0:
						optima_values = skin_match_c[optima[:,0], optima[:,1]]
						optima_thresh = np.max(optima_values) / 2
						optima = optima.tolist()

						for i,o in enumerate(optima):
							if optima_values[i] < optima_thresh:
								optima.pop(i)
								break
							joint = np.array(o) + [box[0].start, box[1].start]
							circ = np.array(circle(joint[0],joint[1], 5)).T
							circ = circ.clip([0,0], [height-1, width-1])
							cam.colorIm[circ[:,0], circ[:,1]] = (0,120 - 30*i,0)#(255*(i==0),255*(i==1),255*(i==2))
					markers = optima

					if 0:
						pass
						# Depth-based matching
						# optima = peak_local_max(skin_match_d, min_distance=20, num_peaks=3, exclude_border=False)
						# for i,o in enumerate(optima):
						# 	joint = o + [box[0].start, box[1].start]
						# 	circ = np.array(circle(joint[0],joint[1], 10))
						# 	circ = circ.clip([0,0,0], [height-1, width-1, 999])
						# 	cam.colorIm[circ[0], circ[1]] = (0,0,255-40*i)#(255*(i==0),255*(i==1),255*(i==2))

						# im_pos = depthIm2PosIm(cam.depthIm).astype(np.int16)
						# im_pos = im_pos[box]*mask[box][:,:,None]

						# cost_map = im_depth[box]
						# extrema = geodesic_extrema_MPI(im_pos, iterations=5, visualize=False)
						# if len(extrema) > 0:
						# 	for i,o in enumerate(extrema):
						# 		joint = np.array(o) + [box[0].start, box[1].start]
						# 		circ = np.array(circle(joint[0],joint[1], 10)).T
						# 		circ = circ.clip([0,0], [height-1, width-1])
						# 		cam.colorIm[circ[:,0], circ[:,1]] = (0,0,120-30*i)#(255*(i==0),255*(i==1),255*(i==2))
						# markers = optima
						# trails = []
						# if len(markers) > 1:
						# 	for i,m in enumerate(markers):
						# 		trails_i = connect_extrema(im_pos, markers[i], markers[[x for x in range(len(markers)) if x != i]], visualize=False)
						# 		trails += trails_i
						# for t in trails:
						# 	try:
						# 		cost_map[t[:,0], t[:,1]] = 0
						# 		cam.colorIm[t[:,0]+box[0].start, t[:,1]+box[1].start] = (0,0,255)
						# 	except:
						# 		print 'Error highlighting trail'

						# cv2.imshow('SkinMap, SkinDetect, DepthDetect',
						# 			np.hstack([ im_skin			/float(im_skin.max()),
						# 						skin_match_c	/float(skin_match_c.max())]
						# 						))




				# SVM
				if 0:
					# b_height,b_width = im_gray[box].shape
					# if im_gray[box].shape[0] < filters[0].shape[0] or im_gray[box].shape[1] < filters[0].shape[1]:
						# continue
					model_params['hog_orientations'] = 5
					# hogs_array_c, hogs_im_c = hog(im_gray[box], orientations=model_params['hog_orientations'], pixels_per_cell=model_params['hog_size'], cells_per_block=model_params['hog_cells'], visualise=True, normalise=False)
					# hogs_array_c, hogs_im_c = hog(im_gray[box], orientations=5, pixels_per_cell=(8,8), cells_per_block=(1,1), visualise=True, normalise=False)
					hogs_array_c = hog(im_gray[box], orientations=5, pixels_per_cell=(8,8), cells_per_block=(1,1), visualise=False, normalise=False)
					# hogs_array_c, hogs_im_c = hog(im_gray[box], orientations=model_params['hog_orientations'], pixels_per_cell=model_params['hog_size'], cells_per_block=(3,3), visualise=True, normalise=False)

					chi2 = model_params['Chi2']
					h_height = im_gray[box].shape[0]/model_params['hog_size'][0]#patch_size
					h_width = im_gray[box].shape[1]/model_params['hog_size'][1]#patch_size
					hog_dim = 5# * ((patch_size/model_params['hog_size'][0])*(patch_size/model_params['hog_size'][1]))#float(h_height) / h_width
					hog_patches = hogs_array_c.reshape([h_height, h_width, hog_dim])
					hog_patches[hog_patches<0]=0
					hog_out = np.zeros([h_height, h_width])-1
					for i in xrange(0, h_height):
						for j in xrange(0, h_width):
							patch = hog_patches[i:i+4, j:j+4].flatten()
							if patch.max() != 0 and patch.shape[0]==hog_dim*16:
								patch_c = chi2.transform(patch / patch.max())
								hog_out[i,j] = svm.predict(patch)[0]
								# print svm.predict(patch)[0]


					# cv2.imshow("H", hog_out)
					# cv2.waitKey(10)
					output = hog_out
					predict_class = hog_out
					# hog_patches = chi2.transform(hog_patches)
					# predict_class = svm.predict(hog_patches)
					# matshow(predict.reshape([h_height,h_width]))

					# hog_layers = np.empty([hogs_im_c.shape[0], hogs_im_c.shape[1], n_filters])
					# for i,f in enumerate(filters):
						# hog_layers[:,:,i] = match_template(hogs_im_c, f, pad_input=True)

					# predict
					# mask_tmp = np.maximum((hog_layers.max(-1) == 0), -mask[box])
					# predict_class = hog_layers[0::8,0::8].argmax(-1)*(-mask_tmp[::8,::8]) + ((-mask_tmp[::8,::8])*1)
					# predict_prob = hog_layers[0::8,0::8].max(-1)*(-mask_tmp[::8,::8]) + ((-mask_tmp[::8,::8])*1)

					# output = np.zeros([predict_class.shape[0], predict_class.shape[1], n_filters], dtype=np.float)
					# for i in range(n_filters):
					# 	output[predict_class==i+1,i] = predict_class[predict_class==i+1]+predict_prob[predict_class==i+1]
					# 	if 0 and i != 2:
					# 		for ii,jj in it.product(range(8),range(8)):
					# 			try:
					# 				cam.colorIm[box][ii::8,jj::8][predict_class==i+1] = (255*(i==0),255*(i==1),255*(i==2))
					# 			except:
					# 				pass

					# Display Predictions
					# n_peaks = [1,2,2]
					# for i in range(n_filters-1):
					# 	optima = peak_local_max(predict_prob*(predict_class==i+1), min_distance=2, num_peaks=n_peaks[i], exclude_border=False)
					# 	for o in optima:
					# 		joint = o*8 + [box[0].start, box[1].start]
					# 		circle = circle(joint[0],joint[1], 15)
					# 		circle = np.array([np.minimum(circle[0], height-1),
					# 							np.minimum(circle[1], width-1)])
					# 		circle = np.array([np.maximum(circle[0], 0),
					# 							np.maximum(circle[1], 0)])

					# 		cam.colorIm[circle[0], circle[1]] = (255*(i==0),255*(i==1),255*(i==2))

					# if 0:
					# 	for i in range(n_filters-1):
					# 		figure(i+1)
					# 		imshow(predict_prob*(predict_class==i+1))
					# 	show()

					# tmp = gray2rgb(sm.imresize(predict_class, np.array(predict_class.shape)*10, 'nearest'))
					# tmp[:,:,2] = 255 - tmp[:,:,2]
					tmp = sm.imresize(output, np.array(predict_class.shape)*10, 'nearest')
					# cv2.imshow("O", tmp/float(tmp.max()))
					# cv2.waitKey(10)

				# # Random Forest
				# if 0:
				# 	h_count = hogs_array_c.shape[0] / 9
				# 	hog_height, hog_width = [b_height/8, b_width/8]
				# 	square_size = patch_size/2*2/8
				# 	hogs_square = hogs_array_c.reshape(hog_height, hog_width, 9)
				# 	predict_class = np.zeros([hog_height, hog_width])
				# 	predict_prob = np.zeros([hog_height, hog_width])
				# 	output = predict_class
				# 	for i in range(0,hog_height):
				# 		for j in range(0,hog_width):
				# 			if i-square_size/2 >= 0 and i+square_size/2 < hog_height and j-square_size/2 >= 0 and j+square_size/2 < hog_width:
				# 				predict_class[i,j] = rf.predict(hogs_square[i-square_size/2:i+square_size/2,j-square_size/2:j+square_size/2].flatten())[0]
				# 				predict_prob[i,j] = rf.predict_proba(hogs_square[i-square_size/2:i+square_size/2,j-square_size/2:j+square_size/2].flatten()).max()


			# Calculate HOGs at geodesic extrema. The predict
			if 0:
				n_px = 16
				n_radius = 2

				extrema = geodesic_extrema_MPI(cam.depthIm*mask, iterations=10)
				extrema = np.array([e for e in extrema if patch_size/2<e[0]<height-patch_size/2 and patch_size/2<e[1]<width-patch_size/2])
				# joint_names = ['head', 'torso', 'l_shoulder', 'l_elbow', 'l_hand',\
				# 				'r_shoulder', 'r_elbow', 'r_hand',\
				# 				'l_hip', 'l_knee', 'l_foot', \
				# 				'r_hip', 'r_knee', 'r_foot']
				joint_names = ['head', 'l_hand', 'l_foot', 'other']

				# Color
				if 1:
					hogs_c_pca = [hog(im_gray[e[0]-patch_size/2:e[0]+patch_size/2, e[1]-patch_size/2:e[1]+patch_size/2], 9, (8,8), (3,3), False, True) for e in extrema]

					lbp_tmp = [local_binary_pattern(im_gray[e[0]-patch_size/2:e[0]+patch_size/2, e[1]-patch_size/2:e[1]+patch_size/2], P=n_px, R=n_radius, method='uniform') for e in extrema]
					lbp_hists_c = np.array([np.histogram(im, normed=True, bins = n_px+2, range=(0,n_px+2))[0] for im in lbp_tmp])
					lbp_hists_c[lbp_hists_c.argmax(0)] = 0
					lbp_hists_c = lbp_hists_c.T / lbp_hists_c.max(1)
					lbp_hists_c = np.nan_to_num(lbp_hists_c)
					# names = [joint_names[i] for i in hogs_c_pred]
				# Both
				if 1:
					# data_both_pca = np.array([pca_both.transform(data_both[i]) for i in range(len(data_both))]).reshape([len(extrema), -1])
					data_both_pca = np.hstack([np.array(hogs_c), lbp_hists_c.T, lbp_hists_z.T])
					data_both_pca[data_both_pca<0] = 0
					hogs_both_pred = np.hstack([svm.predict(h) for h in data_both_pca]).astype(np.int)
					names = [joint_names[i] for i in hogs_both_pred]
					skel_predict = hogs_both_pred.astype(np.int)

				# print names
				im_c = cam.colorIm
				d = patch_size/2
				for i in range(len(extrema)):
					color = 0
					if names[i] == 'head':
						color = [255,0,0]
					elif names[i] in ('l_hand', 'r_hand'):
						color = [0,255,0]
					# elif names[i] == 'l_shoulder' or names[i] == 'r_shoulder':
						# color = [0,0,255]
					# elif names[i] == 'l_foot' or names[i] == 'r_foot':
						# color = [0,255,255]
					# else:
						# color = [0,0,0]

					if color != 0:
						im_c[extrema[i][0]-d:extrema[i][0]+d, extrema[i][1]-d:extrema[i][1]+d] = color
					else:
						im_c[extrema[i][0]-d/2:extrema[i][0]+d/2, extrema[i][1]-d/2:extrema[i][1]+d/2] = 0
					# im_c[extrema[i][0]-d:extrema[i][0]+d, extrema[i][1]-d:extrema[i][1]+d] = lbp_tmp[i][:,:,None] * (255/18.)
						# cv2.putText(im_c, names[i], (extrema[i][1], extrema[i][0]), 0, .4, (255,0,0))
				# cv2.imshow("Label_C", im_c*mask[:,:,None])
				# cv2.waitKey(10)

				# Accuracy
				print skel_predict
				extrema_truth = np.empty(len(extrema), dtype=np.int)
				for i in range(len(extrema)):
					ex = extrema[i]
					dist = np.sqrt(np.sum((ex - skel_msr_im[:,[1,0]])**2,-1))
					extrema_truth[i] = np.argmin(dist)

					# print skel_predict[i], extrema_truth[i]
					if skel_predict[i] == 1:
						# if extrema_truth[i] == 4 or extrema_truth[i] == 7:
						if extrema_truth[i] in (11, 7):
							true_pos['hands'] += .5
						else:
							false_pos['hands'] += .5
					elif skel_predict[i] == 0:
						if extrema_truth[i] == 3:
							true_pos['head'] += 1
							print 'h correct'
						else:
							false_pos['head'] += 1
				print "Hands", true_pos['hands'] / float(frame_count)#float(false_pos['hands'])
				print "Head", true_pos['head'] / float(frame_count)##float(false_pos['head'])



if 0:

	###
	figure(4)
	names = [joint_names[i] for i in hogs_z_pred.astype(np.int)]
	labels_resized = sm.imresize(hogs_z_pred.reshape([-1, 5]), im.shape, 'nearest')
	matshow(labels_resized/13/10. + im/float(im.max()))
	im_c = cam.colorIm[box[0]]
	# matshow(labels_resized/13/10. + im/float(im.max()))
	matshow(labels_resized/13/10. + im_c[:,:,0]/float(im_c.max()))

	### Find joint label closest to each extrema
	extrema = geodesic_extrema_MPI(cam.depthIm*(mask>0), iterations=3)
	for i,_ in enumerate(extrema):
		# j_pos = skel_msr_im[i]
		j_pos = extrema[i]
		x = j_pos[0]
		y = j_pos[1]
		if x-patch_size/2 >= 0 and x+patch_size/2 < height and y-patch_size/2 >= 0 and y+patch_size/2 < width:
			ims_rgb += [im_color[x-patch_size/2:x+patch_size/2, y-patch_size/2:y+patch_size/2]]
			ims_depth += [im_depth[x-patch_size/2:x+patch_size/2, y-patch_size/2:y+patch_size/2]]
			dists = np.sqrt(np.sum((j_pos - skel_msr_im[:,[1,0]])**2,-1))
			labels += [np.argmin(dists)]

	### Multi-cam
	# cam2 = KinectPlayer(base_dir='./', device=2, bg_subtraction=True, get_depth=True, get_color=True, get_skeleton=True, fill_images=False)

	# Transformation matrix from first to second camera
	# data = pickle.load(open("Registration.dat", 'r'))
	# transform_c1_to_c2 = data['transform']

	# cam2_skels = transform_skels(cam_skels, transform_c1_to_c2, 'image')

	# Update frames
	# cam2.sync_cameras(cam)


