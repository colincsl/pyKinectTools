
''' ------------------------------------- '''


''' Plot DTW for joint angles '''
for i_key,key in enumerate(skels_subactivity_angles):
	figure(key)
	print "{} iterations of {}".format(len(skels_subactivity_angles[key]), key)
	for i_iter in range(len(skels_subactivity_angles[key])):
		print 'iter', i_iter
		for i,ang in enumerate(skels_subactivity_angles[key][i_iter].T):
			x = skels_subactivity_angles[key][0][:,i]
			y = ang
			y = resample(y, len(x))
			# error, dtw_mat, y_ind = mlpy.dtw.dtw_std(x, y, dist_only=False)			
			error, dtw_mat, y_ind = DynamicTimeWarping(x, y)
			
			subplot(3,4,i+1)
			y_new = y[y_ind[0]]
			x_new = np.linspace(0, 1, len(y_new))
			# poly = polyfit(x_new, y_new, 5)
			# y_spline_ev = poly1d(poly)(x_new)

			nknots = 4
			idx_knots = (np.arange(1,len(x_new)-1,(len(x_new)-2)/np.double(nknots))).astype('int')
			knots = x_new[idx_knots]
			y_spline = splrep(x_new, y_new, t=knots)
			y_spline_ev = splev(np.linspace(0, 1, len(y_new)), y_spline)
			# plot(y_new)
			plot(y_spline_ev)
			y_spline_ev = resample(y_spline_ev, len(ang))
			skels_subactivity_angles[key][i_iter].T[i] = y_spline_ev
			# show()
			# plot(y[y_ind[0]])

			# subplot(3,10,i+1 + i_iter*10)
			# plot(x[y_ind[1]])
			# plot(y[y_ind[0]])
			print i,":", len(ang), len(x), len(y[y_ind[0]])
			title(CAD_JOINTS[i])

			if i == 10:
				break
show()


''' Plot relative object positions wrt to afforance '''
for i_key,key in enumerate(object_affordances):
	print "{} iterations of {}".format(len(object_affordances[key]), key)
	for i_iter in range(len(object_affordances[key])):
		print 'iter', i_iter
		for i,hand in enumerate(object_affordances[key][i_iter].T):
			y = hand
			y[y>2000] = 0
			subplot(2,8,i_key+1 + 8*i)
			plot(y)
			title(key)
show()

''' Plot relative object positions wrt to subaction '''
n_subactions = len(object_subactions.keys())
for i_key,key in enumerate(object_subactions):
	print "{} iterations of {}".format(len(object_subactions[key]), key)
	for i_iter in range(len(object_subactions[key])):
		print 'iter', i_iter
		for i,hand in enumerate(object_subactions[key][i_iter].T):
			y = hand[0]
			y[y>2000] = 0
			if np.all(y==0):
				continue
			subplot(2,n_subactions,i_key+1 + n_subactions*i- n_subactions)
			plot(y.T)
			title(key)
show()


''' Come up with prototypical motion for each subaction using DTW '''
def get_prototype_motions(skels_subactivity_train, smooth=False, nknots=10):
	'''
	Input: a set of skeleton trajectories
	Output: a motif/prototype skeleton trajectory

	This algorithm takes every instance of a class and compares it to every other instance
	in that class using DTW, optionally smooths. Each (pairwise) transformed class instance
	is then averaged to output a motif.

	Todo: currently this is done independently per-joint per-dimension. Should be per skeleton!
	'''
	skels_subactivity_train = apply_user_frame(skels_subactivity_train)

	proto_motion = {}
	for i_key,key in enumerate(skels_subactivity_train):
		n_instances = len(skels_subactivity_train[key])
		n_frames = int(np.mean([len(x) for x in skels_subactivity_train[key]]))
		# Do x,y,z seperately
		proto_motion[key] = np.zeros([n_frames, 15, 3])
		for i_joint in range(15):
			# error_matrix = np.zeros([n_instances, n_instances], np.float)
			y_spline_set = []
			for i in xrange(n_instances):
				for j in xrange(n_instances):
					if i >= j:
						continue
					x = skels_subactivity_train[key][i][:,i_joint]
					y = skels_subactivity_train[key][j][:,i_joint]
					# error, dtw_mat, y_ind = mlpy.dtw.dtw_std(x, y, dist_only=False)
					error, dtw_mat, y_ind = DynamicTimeWarping(x, y)
					# error = mlpy.dtw.dtw_std(x, y, dist_only=True)
					# error_matrix[i,j] = error

					y_new = y[y_ind[1]]
					x_new = np.linspace(0, 1, len(y_new))
					# poly = polyfit(x_new, y_new, 5)
					# y_spline_ev = poly1d(poly)(x_new)

					if smooth:
						# Generate Spline representation
						nknots = np.minimum(nknots, len(x_new)/2)
						idx_knots = (np.arange(1,len(x_new)-1,(len(x_new)-2)/np.double(nknots))).astype('int')
						knots = x_new[idx_knots]
						y_spline = splrep(x_new, y_new, t=knots)
						y_spline_ev = splev(np.linspace(0, 1, len(y_new)), y_spline)
						y_spline_ev = resample(y_spline_ev, n_frames)
						y_spline_set += [y_spline_ev]
					else:
						y_spline_ev = resample(y_new, n_frames)
						y_spline_set += [y_spline_ev]						

				proto_motion[key][:,i_joint] = np.mean(y_spline_set, 0)
	return proto_motion

''' Gaussian mixture model prototype '''
y_spline_set = np.vstack(y_spline_set)
from sklearn import gaussian_process
gp = gaussian_process.GaussianProcess(theta0=1e+1, normalize=False)
x = np.arange(y_spline_set.shape[1])[:,None].repeat(y_spline_set.shape[0], 1).T.astype(np.float)
x += np.random.random(x.shape)/10000
x_test = np.arange(y_spline_set.shape[1])[:,None]
y = y_spline_set - y_spline_set[:,0][:,None]
# gp.fit(y.ravel()[:,None], x.ravel()[:,None])
gp.fit(x.ravel()[:,None], y.ravel()[:,None])

y_pred, MSE = gp.predict(x_test, eval_MSE=True)
sigma = np.sqrt(MSE)
plot(x_test, y_pred, 'b', label=u'Prediction')
fill(np.concatenate([x_test, x_test[::-1]]),
        np.concatenate([y_pred - 1.9600 * sigma,
                       (y_pred + 1.9600 * sigma)[::-1]]),
        alpha=.5, fc='b', ec='None', label='95% confidence interval')
# plot(x_test, y[0], 'r', label=u'Obs')
for i in range(15):
	plot(x_test, y[i], 'r', label=u'Obs_DTW')
for i in range(6):
	plot(np.arange(len(skels_subactivity_train['opening'][i][:,-1,2])), skels_subactivity_train['opening'][i][:,-1,2], 'y', label=u'Obs')	
plot(proto_motion['opening'][:,-1,2], 'g', label=u'Prototype')
legend()
show()


''' Test similarity of samples to prototypical (eval on test data) '''

accuracy_dtw = []
accuracy_lcs = []
training_sets = list(it.combinations([1,3,4], 1))
testing_sets = [tuple([x for x in [1,3,4] if x not in y]) for y in training_sets]
for train_set, test_set in zip(training_sets, testing_sets):
	# Get training/test sets + calculate prototype motions
	skels_subactivity_train, skels_subactivity_test = split_skeleton_data(skels_subactivity, train_set, test_set)
	proto_motion = get_prototype_motions(skels_subactivity_train, nknots=5)
	skels_subactivity_test = apply_user_frame(skels_subactivity_test)
	print "Prototypes generated"

	# Generate precision/recall
	n_subactions = len(skels_subactivity_test.keys())
	max_iterations = max([len(skels_subactivity_test[x]) for x in skels_subactivity_test])
	errors_dtw = np.zeros([n_subactions, n_subactions,max_iterations], np.float)
	errors_lcs = np.zeros([n_subactions, n_subactions,max_iterations], np.float)
	errors_mask = np.zeros([n_subactions, max_iterations], dtype=np.int)
	# Evaluate each test instance
	for i_key,key in enumerate(skels_subactivity_test):
		n_instances = len(skels_subactivity_test[key])
		for i in xrange(n_instances):
			# Evaluate for each prototype
			for i_key2,key2 in enumerate(skels_subactivity_test):
				err_dtw = 0
				err_lcs = 0
				x_skel = proto_motion[key2]
				y_skel = skels_subactivity_test[key][i]

				for i_joint in CAD_ENABLED:
					for i_dim in range(3):
						x = x_skel[:,i_joint,i_dim]
						y = y_skel[:,i_joint,i_dim]
						error_dtw, _, y_ind = mlpy.dtw_std(x, y, dist_only=False, squared=True)
						error_lcs,_ = mlpy.lcs_real(x,y[y_ind[1]], np.std(x), len(x)/2)
						err_dtw += error_dtw
						err_lcs += error_lcs/len(x)
				errors_dtw[i_key, i_key2, i] = err_dtw
				errors_lcs[i_key, i_key2, i] = err_lcs
				errors_mask[i_key,i] = 1

	iterations_per_subaction = np.sum(errors_mask,1).astype(np.float)
	print "Train:{}, Test:{}".format(train_set, test_set) 
	solution = np.arange(n_subactions)[:,None].repeat(errors_dtw.shape[2], 1)
	true_positives = np.sum((errors_dtw.argmin(1) == solution)*errors_mask, 1)
	false_negatives = np.sum((errors_dtw.argmin(1) != solution)*errors_mask, 1)
	accuracy_dtw += [np.mean(true_positives / iterations_per_subaction)]
	print "DTW Precision:", accuracy_dtw[-1]
	# print "DTW Recall:", np.mean(true_positives / (true_positives+false_negatives).astype(np.float))
	# precision = True Positives / (True Positives + False Positives) = TP/TP+FP
	# recall = True Positives / (True Positives + False Negatives) = TP/TP+0

	accuracy_lcs += [np.mean(np.sum((errors_lcs.argmax(1) == solution)*errors_mask, 1) / iterations_per_subaction)]
	# print accuracy_lcs
	print "LCS Precision:", accuracy_lcs[-1]
	print ""

print '---- N-fold accuracy ---'
print "DTW: {:.4}%".format(np.mean(accuracy_dtw)*100)
print "LCS: {:.4}%".format(np.mean(accuracy_lcs)*100)



skels_subactivity_test = apply_user_frame(skels_subactivity_test)
skels_subactivity_train = apply_user_frame(skels_subactivity_train)
proto_motion = get_prototype_motions(skels_subactivity_train, smooth=False)



''' Put together and visualize a new sequence '''
object_position = np.array([0, 1050,3500])
obj_position_uv = cam.camera_model.world2im(np.array([object_position]), [480,640])
# actions = ['null', 'moving', 'cleaning', 'moving', 'null', 'placing']
actions = proto_motion.keys()
a = actions[0]
new_action = proto_motion[a] + object_position
new_action_labels = [a]*len(new_action)
for a in actions:
	new_action = np.vstack([new_action, proto_motion[a] + object_position])
	new_action_labels += [a]*len(proto_motion[a])

from time import time
t0 = time()
ii = 0
sequence_samples = np.random.choice(n_samples, 5, replace=False)
for i,f in enumerate(new_action):
	if i>0 and new_action_labels[i] != new_action_labels[i-1]:
		ii = 0
		im = np.ones([480,640])*255
		n_samples = len(skels_subactivity_train[new_action_labels[i]])
		sequence_samples = np.random.choice(n_samples-1, 5, replace=False)
		cv2.imshow("New action", im)
		cv2.waitKey(1)

	bg_im = np.ones([480,640])
	# cv2.rectangle(bg_im, tuple(obj_position_uv[0][[1,0]]-[30,30]), tuple(obj_position_uv[0][[1,0]]+[30,30]), 2000)
	f_uv = cam.camera_model.world2im(f, [480,640])
	f_uv[:,0] = 480 - f_uv[:,0]
	im = display_skeletons(bg_im, f_uv, skel_type='CAD_Upper', color=2000)
	cv2.putText(im, "Action: "+new_action_labels[i], (20,60), cv2.FONT_HERSHEY_DUPLEX, 1, (2000,0,0), thickness=2)

	cv2.putText(im, "Prototype", (240,160), cv2.FONT_HERSHEY_DUPLEX, 1, (2000,0,0), thickness=2)
	# Plot training samples below the protype action
	for i_iter,i_sample in enumerate(sequence_samples):
		try:
			ii_frame = min(ii, len(skels_subactivity_train[new_action_labels[i]][i_sample])-1)
			skel = skels_subactivity_train[new_action_labels[i]][i_sample][ii_frame] - skels_subactivity_train[new_action_labels[i]][i_sample][ii_frame][2]
			# ii_frame = min(ii, len(skels_subactivity_test[new_action_labels[i]][i_iter])-1)
			# skel = skels_subactivity_test[new_action_labels[i]][i_iter][ii_frame] - skels_subactivity_test[new_action_labels[i]][i_iter][ii_frame][2]
			# skel = normalize_basis(skel)
			skel += [-1400+i_iter*700, 0,3500]
			f_uv = cam.camera_model.world2im(skel, [480,640])
			f_uv[:,0] = 480 - f_uv[:,0]
			im = display_skeletons(bg_im, f_uv, skel_type='CAD_Upper', color=2000)
		except: pass
	cv2.putText(im, "Training Samples: "+str(list(train_set)), (140,320), cv2.FONT_HERSHEY_DUPLEX, 1, (2000,0,0), thickness=2)

	# Plot test samples below the protype action
	for i_iter in range(5):
		try:
			ii_frame = min(ii, len(skels_subactivity_test[new_action_labels[i]][i_iter])-1)
			skel = skels_subactivity_test[new_action_labels[i]][i_iter][ii_frame] - skels_subactivity_test[new_action_labels[i]][i_iter][ii_frame][2]
			# skel = normalize_basis(skel)
			skel += [-1400+i_iter*700, -1000,3500]
			f_uv = cam.camera_model.world2im(skel, [480,650])
			f_uv[:,0] = 480 - f_uv[:,0]
			im = display_skeletons(bg_im, f_uv, skel_type='CAD_Upper', color=2000)
		except: pass
	cv2.putText(im, "Testing Samples: "+str(list(test_set)), (150,470), cv2.FONT_HERSHEY_DUPLEX, 1, (2000,0,0), thickness=2)

	cv2.imshow("New action", (im-1000.)/(im.max()-1000))
	cv2.waitKey(30)
	ii += 1
	print "{} fps".format(i/(time()-t0))







''' Inverse kinematics from hand to torso? '''
''' Add physical/collision constraints '''
''' add symmetries '''
''' break into left hand, right hand, torso, legs '''

for i,f in enumerate(y_skel):
	bg_im = np.ones([480,640])
	# f_uv = cam.camera_model.world2im(f, [480,640])
	f_uv = cam.camera_model.world2im(f+[0,0,3000], [480,640])
	f_uv[:,0] = 480 - f_uv[:,0]
	im = display_skeletons(bg_im, f_uv, skel_type='CAD_Upper', color=2000)
	cv2.putText(im, "Action: "+new_action_labels[i], (20,60), cv2.FONT_HERSHEY_DUPLEX, 1, (2000,0,0), thickness=2)
	cv2.imshow("New action", (im-1000.)/(im.max()-1000))
	cv2.waitKey(1)

