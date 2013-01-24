
import numpy as np, cv, cv2
import time, os
from copy import deepcopy

from pyKinectTools.utils.DepthReader import *
from pyKinectTools.utils.SkeletonUtils import *
from pyKinectTools.algs.PeopleTracker import *

import random as rand


# labels = {1:'Rounds', 2:'Talking', 3:'Observing', 4:'Checkup', 5:'Procedure', 6:'Unrelated'}
# labels = {1:'ventilator', 2:'talking', 3:'observing', 4:'diagnostics',\
#  5:'urine', 6:'other', 7:'documentation', 8:'procedure', 9:'error'}
labels = {1:'ventilator', 3:'observing/talking', 4:'checkup/diagnostics',\
 5:'urine', 6:'other', 7:'documentation', 8:'procedure'}

# data = np.load('ActionData/1_5-9_Wide_labeled_AMIA.npz')
data = np.load('Spring2012Actions/BEST_1_5-9_Wide_labeled.npz')
dir_ = '/Users/colin/data/ICU_7May2012_Wide/'

# labelData("1.npz", '/Users/colin/data/ICU_7March2012_Head/', speed=10)
# playData("1.npz", '/Users/colin/data/ICU_7March2012_Head/', speed=10)
# playData("1.npz", '/Volumes/ICU/ICU_7March2012_Head/', speed=10)
# labelData("1.npz", '/Volumes/ICU/ICU_7March2012_Head/', speed=20)

# labelData("1_5-9_Wide.npz", '/Users/colin/data/ICU_7May2012_Wide/', speed=20)
# labelData("2_5-9_Close.npz", '/Users/colin/data/ICU_7May2012_Close/', speed=20)

# playData("1_5-9_Wide.npz", '/Users/colin/data/ICU_7May2012_Wide/', speed=10)
# playData("1.npz", '/Users/colin/data/ICU_7May2012_Close/', speed=10)


# labelData("2_800s.npz", '/Users/colin/data/ICU_7March2012_Foot/')
# playData("2.npz", '/Users/colin/data/ICU_7March2012_Foot/')
# playData("2.npz", '/Volumes/ICU/ICU_7March2012_Foot/', speed=10)
# labelData("2.npz", '/Volumes/ICU/ICU_7March2012_Foot/', speed=20)

if 0:
	if 0:
		# data = np.load('ActionData/1_5-9_Wide_labeled_good.npz')
		
		# data = np.load('ActionData/1_4-23_labeled.npz')
		# data = np.load('ActionData/1.npz')
		# dir_ = '/Users/colin/data/ICU_7May2012_Wide_jpg/'
		# dir_ = '/Users/colin/data/ICU_7March2012_Head/'
		dir_ = '/Users/colin/data/ICU_7May2012_Wide/'
	else:
		data = np.load('ActionData/2_800s_labeled_good.npz')
		data = np.load("ActionData/2_5-9_Close_labeled.npz")
		
		data = np.load('ActionData/2.npz')
		dir_ = '/Users/colin/data/ICU_7March2012_Foot/'

	pOrig = data['data']
	p = filterEvents(pOrig)
	m = data['meta']	
	p = [x for x in p if type(x['label'])==unicode]
	p = [x for x in p if int(x['label']) != 9] # not erroneous
	p = [x for x in p if int(x['label']) != -1] # not erroneous

	for i in xrange(len(p)):
		if int(p[i]['label']) == 2:
			p[i]['label'] = 3
	
	totalEventCount = len(p) # includes errors / really short events

	# labels = set()
	# truthLabels = np.array([int(x['label']) for x in p if 'label' in x.keys()], dtype=str)
	truthLabels = np.array([x['label'] for x in p if 'label' in x.keys()], dtype=int)
	labelNames = unique(truthLabels)

	# Get counts for each label
	counts = [[], []]
	counts = np.histogram(truthLabels, 9, [.5,9.5])


	# Get indices for each label
	labelInds = []
	for i in xrange(len(labelNames)):
		labelInds.append([y for x,y in zip(p, range(0,len(p))) if 'label' in x.keys() and int(x['label'])==labelNames[i]])

	labeledTimes = []
	for i in xrange(len(labelNames)):
		labeledTimes.append(np.sum([p[x]['elapsed'] for x in labelInds[i]]))


# -------

	# Show label images
	for lab in xrange(len(labelNames)):
		figure(lab)
		for i in xrange(len(labelInds[lab])):
			subplot(2,int(len(labelInds[lab])/2),i)	
			img = DepthReader.getDepthImage(dir_+p[labelInds[lab][i]]['data']['filename'][0])
			imshow(img)
			axis('off')
			title(labelNames[lab])

	################# Time #############
	# Get time for each label
	allTimes = [x['elapsed'] for x in p]	
	labeledTimes = [np.sum([p[x2]['elapsed'] for x2 in x]) for x in labelInds]
	classifiedTimes = [np.sum(allTimes*(classifiedLabels==i)) for i in labelNames]

	bar(np.array(range(len(labeledTimes)))*2, labeledTimes, color='r')
	bar(np.array(range(len(labeledTimes)))*2+1, classifiedTimes, color='b')
	title('Time per task'); xlabel('Tasks'); ylabel('Time (sec)')
	xticks(np.arange(1,len(labels)*2, 2), [x[1] for x in labels.items()]); 

	# Time histogram
	times = [x['elapsed'] for x in p if x['elapsed'] > 10]
	times_hist = np.histogram(times, bins=20, range=[10,250])
	plot(times_hist[0], 'k')
	title('Histogram of action durations (Median: 37 sec)')
	xticks(np.arange(1,len(labels)*2, 2), [x[1] for x in labels.items()]); 
	# xticks(range(.5,len(times_hist[1])*2+1, 2), times_hist[1][::2]); 
	xlabel('Time (sec)'); ylabel('Count')

	# Time plot

	# Average times
	eventCount = np.sum([1 for x in p if x['elapsed'] > 10])	
	totalEventTime = np.sum([x['elapsed'] for x in p])
	totalValidEventTime = np.sum([x['elapsed'] for x in p if x['elapsed'] > 10])
	totalTime = p[-1]['start']+p[-1]['elapsed']
	avgEventTime = totalValidEventTime / eventCount
	avgTime = totalTime / eventCount
	totalDuration = p[-1]['start']+p[-1]['elapsed']
	print "Percent time a nurse is in the room:", (totalEventTime / float(totalDuration))

	# Play segment
	tmp = [y for x,y in zip(p, range(len(p))) if x['elapsed'] > 600] # time outliers
	i = tmp[1]
	# img = DepthReader.getDepthImage(dir_+p[i]['data']['filename'][0])
	# imshow(img)
	speed = 2
	for j in xrange(0, len(p[i]['data']['filename']), speed):
		showLabeledImage(p[i], j, dir_)


if 0:

	# Create time-space
	timeEvents = {}
	for i in xrange(len(p)):
		datum = p[i]
		for j in xrange(len(datum['data']['time'])):
			t = datum['data']['time'][j]
			if t not in timeEvents.keys():
				timeEvents[t] = {i:[j]} #Event, event-time
			else:
				if i not in timeEvents[t].keys():
					timeEvents[t][i] = [j]
				else:
					timeEvents[t][i].append(j)


	max_ = 0
	figure(2)
	for i in timeEvents.keys():
		if len(timeEvents[i]) > max_:
			max_ = len(timeEvents[i])
			# argmax_ = timeEvents[i][0]
			# argmaxs_ = timeEvents[i]
		bar(int(i/60), len(timeEvents[i]))
	totalTime = p[-1]['start']+p[-1]['elapsed']
	axis([0, int(totalTime/60), 0, max_+1])
	title('Person count at each timestep', fontsize=20)
	xlabel('Time (min)', fontsize=18)
	ylabel('# People', fontsize=18)
	xticks(fontsize=16)
	yticks(fontsize=16)

	# *****************************
	# Plot of each action over time
	COLORS = 'krgbcmy'
	figure(3)
	ids = []
	maxEnd = 0
	cumDurations = np.zeros(len(labelNames))
	cumDurationsCorrect = np.zeros(len(labelNames))
	for i in range(len(p)):
		start = p[i]['data']['time'][0]
		end = p[i]['data']['time'][-1]
		
		if end > maxEnd:
			maxEnd = end
		l = int(p[i]['label'])
		if l > 1:
			l -= 1
		cumDurations[l-1] += end-start
		l2 = int(deepcopy(classifiedLabels[i]))
		if l2 > 1:
			l2 -= 1
		cumDurationsCorrect[l-1] += (end-start) * (l2==l)

		# plot([start, end], [l,l], linewidth=2, color=COLORS[mod(l,len(COLORS))])
		plot([start, end], [l+.15,l+.15], linewidth=3, color='r')
		if l2==l:#classifiedLabels[i]-1:
			# plot([start, end], [l2-.15,l2-.15], linewidth=3, color='b')	
			plot([start, end], [l2-.15,l2-.15], linewidth=3, color='b')				
		else:	
			plot([start, end], [l2-.23,l2-.23], linewidth=3, color='k')	
			# plot([start, end], [l2-.15,l2-.15], linewidth=3, color='k')	
	axis([0, maxEnd, 0, 8])
	title('Actions over time')
	xlabel('Time (hours)'); ylabel("Actions")
	yticks(range(1,len(labeledTimes)+1,1), [x[1] for x in labels.items()]); 
	xticks(range(0,maxEnd, 3600), floor(arange(0, maxEnd, 3600)/3600))

	# Show images
	for i in range(len(argmaxs_)):
		showLabeledImage(p[argmaxs_[i]], 0, dir_)
	# Show each target in a frame
	k=0
	for i in range(len(timeEvents[k])):
		for j in range(len(p[timeEvents[k][i]]['data']['time'])):
			showLabeledImage(p[timeEvents[k][i]], j, dir_)
			time.sleep(.5)
	# Show video over time
	for i in xrange(100):
		if i in timeEvents:
			showLabeledImage(p[timeEvents[i][0]], 0, dir_)
			time.sleep(.1)

	# Show overhead view
	figure(4)
	# for i in range(len(argmaxs_)):
		# com = p[argmaxs_[i]]['com']

	# tt=0
	# for i in xrange(30000):
	# 	if i in timeEvents:
	# 		for val,j in zip(timeEvents[i], xrange(len(timeEvents[i]))):
	# 			com = p[timeEvents[i][val][0]]['com']
	# 			plot(-com[0], com[2], 'o', color=COLORS[j%6])
	# 			tt +=1
	t == 0
	for per in p:
		com = per['com']
		plot(-com[0], com[2], 'o', color=colors[j%6])
		tt +=1

	title('Location of people over time', fontsize=20)
	xlabel('X (mm)', fontsize=18); ylabel('Z (mm)', fontsize=18)
	xticks(fontsize=16); yticks(fontsize=16)
	axis([-1050,1050,500, 3100])
	axis('equal')



	# Display jpgs of one activity
	uInds = labelInds[4]
	cv2.namedWindow("win")
	for i in uInds:
		for filename in p[i]['data']['filename']:
			im = getDepthImage(dir_+filename)
			# imshow(im)
			cv2.imshow("win", im.astype(np.float)/im.max())
			cv2.waitKey(1)
		cv2.imshow("win", np.ones([200,200])*255)
		cv2.waitKey(100)			


	# -------------------Features-------------------------------------------------	
	frameCounts = [len(x['data']['time']) for x in p]
	coms = np.array([x['com'] for x in p])
	allCOMs = np.array([x['data']['com'] for x in p])

	## Touch sensors
	touches = [x for x,y in zip(p, range(len(p))) if 'touches' in x['data'].keys()]
	touchInds = [y for x,y in zip(p, range(len(p))) if 'touches' in x['data'].keys()]

	if 0:
		for i in range(1, len(touches)):
			for j in range(0, len(touches[i]['data']['time']), 3):
				showLabeledImage(touches[i], j, dir_)

	touchTmp1 = [y for x,y in zip(np.array(touches[0]['data']['touches'])[:,0], np.array(touches[0]['data']['touches'])[:,1]) if 0 in x]
	touchTmp2 = [y for x,y in zip(np.array(touches[0]['data']['touches'])[:,0], np.array(touches[0]['data']['touches'])[:,1]) if 1 in x]	
	touch0 = np.zeros(len(p))
	touch1 = np.zeros(len(p))
	for i in touchInds:
		touch0[i] = np.sum([1 for x in np.array(p[i]['data']['touches'])[:,0] if 0 in x])
		touch1[i] = np.sum([1 for x in np.array(p[i]['data']['touches'])[:,0] if 1 in x])

	# center = np.array([-346.83551756, -16.10465379, 3475.0]) # new footage
	# comsRad = np.sqrt(np.sum((coms - center)**2, 1))

	''' Patient's head '''
	center = np.array([-146.83551756, -16.10465379, 3475.0])
	touchRad1 = []
	for i in allCOMs:
		touchRad1.append(np.min(np.sqrt(np.sum((i - center)**2, 1))))
	touchRad1 = np.array(touchRad1)
	# touchRad1 = np.sqrt(np.sum((coms - center)**2, 1))	

	''' Patient's foot '''
	center = np.array([-141.13408991, -251.45427198, 2194.0])
	touchRad2 = []
	for i in allCOMs:
		touchRad2.append(np.min(np.sqrt(np.sum((i - center)**2, 1))))
	touchRad2 = np.array(touchRad2)	

	''' Ventilator '''
	# depth2world(np.array([[225, 510, 2754]]))
	center = np.array([-970.07707018, 57.13045534,  2754.])
	touchRad3 = []
	for i in allCOMs:
		touchRad3.append(np.min(np.sqrt(np.sum((i - center)**2, 1))))
	touchRad3 = np.array(touchRad3)	

	''' Computer '''
	# depth2world(np.array([[473,625,2570]]))
	center = np.array([-1402.64381383, -1025.05589005,  1570.0])
	touchRad4 = []
	for i in allCOMs:
		touchRad4.append(np.min(np.sqrt(np.sum((i - center)**2, 1))))
	touchRad4 = np.array(touchRad4)	


	## Find arclengths
	arclengths = np.empty([len(p)])
	for i in xrange(len(p)):
		sum_ = 0
		for j in xrange(1, len(p[i]['data']['com'])):
			sum_ += np.sqrt(np.sum((p[i]['data']['com'][j]-p[i]['data']['com'][j-1])**2))
		arclengths[i] = sum_
	times = np.array([x['elapsed'] for x in p])
	lengthTime = arclengths / times
	if 0:
		subplot(3,1,1); plot(arclengths); title('Arclengths')
		subplot(3,1,2); plot(times); title('Times')
		subplot(3,1,3); plot(lengthTime); title('Length/Times')

	## Radial COMs
	# coms = np.array([x['com'] for x in p])
	# center = np.array([250, -200, 1000]) # Old footage
	center = np.array([-346.83551756, -16.10465379, 3475.0]) # new footage
	comsRad = np.sqrt(np.sum((coms - center)**2, 1))
	comsX = coms[:,0]
	comsY = coms[:,1]
	comsZ = coms[:,2]
	# comsSTD = np.array([np.std(x['data']['com'], 0) for x in p])
	# comsX = comsSTD[:,0]
	# comsY = comsSTD[:,1]
	# comsZ = comsSTD[:,2]

	## Orientation comparison
	ornFeatures = np.array([np.mean(x['data']['ornCompare'], 0) for x in p])

	##  Orientation Historgams
	allBasis = [x['data']['basis'] for x in p]
	ornHists = []
	for i in xrange(len(allBasis)):
		h = allBasis[i]
		h0 = np.array([x[0] for x in allBasis[i]])
		h1 = np.array([x[1] for x in allBasis[i]])
		h2 = np.array([x[2] for x in allBasis[i]])
		ang1 = np.array([np.arctan2(-x[1][0],x[1][2]) for x in h])*180.0/np.pi
		ang2 = np.array([np.arctan2(-x[2][0],x[2][2]) for x in h])*180.0/np.pi
		a1 = np.minimum(ang1, ang2)
		# a2 = np.maximum(ang1, ang2)
		validPoints = np.abs(h1[:,1])>.5 # ensure first vector is pointing up
		ang1 = ang1[validPoints]; a1 = a1[validPoints]
		# ang2 = ang2[validPoints]
		ang1Hist, ang1HistInds = np.histogram(a1, 12, [-180, 180])
		ang1Hist = ang1Hist[:6]+ang1Hist[6:] #Collapse both directions
		ang1Hist = ang1Hist*1.0/np.max(ang1Hist)
		# plot(ang1Hist)
		ornHists.append(ang1Hist)
	ornHists = np.array(ornHists)
	ornHists = np.nan_to_num(ornHists)

	
	# do personCount
	arcMax = np.max(arclengths); arcMin = np.min(arclengths); 
	lengthTimeMax = np.max(lengthTime); lengthTimeMin = np.min(lengthTime)
	touch0Max = np.max(touch0); touch0Min = np.min(touch0)
	touch1Max = np.max(touch1); touch1Min = np.min(touch1)
	touchRad1Max = np.max(touchRad1); touchRad1Min = np.min(touchRad1)
	touchRad2Max = np.max(touchRad2); touchRad2Min = np.min(touchRad2)
	touchRad3Max = np.max(touchRad3); touchRad3Min = np.min(touchRad3)
	touchRad4Max = np.max(touchRad4); touchRad4Min = np.min(touchRad4)	
	frameCountMax = np.max(frameCounts); frameCountMin = np.min(frameCounts)
	comsMax = np.max(comsRad); comsMin = np.min(comsRad)
	comsMaxX = np.max(comsX); comsMinX = np.min(comsX)
	comsMaxY = np.max(comsY); comsMinY = np.min(comsY)
	comsMaxZ = np.max(comsZ); comsMinZ = np.min(comsZ)
	ornMax = 1.0; ornMin = 0.0;

	featureLimits = {
					"arclength":[arcMin, arcMax],
					"lengthTime":[lengthTimeMin, lengthTimeMax],
					"touchRad1":[touchRad1Min, touchRad1Max],
					"touchRad2":[touchRad2Min, touchRad2Max],
					"touchRad3":[touchRad3Min, touchRad3Max],
					"touchRad4":[touchRad4Min, touchRad4Max],
					"frameCount":[frameCountMin, frameCountMax],
					"comRad":[comsMin, comsMax],
					"comX":[comsMinX, comsMaxX],
					"comY":[comsMinY, comsMaxY],
					"comZ":[comsMinZ, comsMaxZ],
					"orn":[ornMin, ornMax]
					}

	# featureNames=['arc', 'lenTime', 'touch0', 'touch1', 'frames', 'COM',
	# 			'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 
	# 			'orn','orn','orn','orn','orn','orn',
	# 			'orn','orn','orn','orn','orn','orn']
	# featureNames=['path', 'velocity', 'touch0', 'touch1', 'frames', 'COM',
	# 			'basis', 'basis', 'basis', 'basis', 'basis',
	# 			'orn','orn','orn','orn','orn','orn',
	# 			'orn','orn','orn','orn','orn','orn']
	featureNames=['Path', 'Velocity', #'frames',
				  #'touch0', 'touch1', 
				  'Touch\n0', 'Touch\n1','Touch\n3','Touch\n4',
				  'CoM\nX', 'CoM\nY', 'CoM\nZ',
				'Inter\n1', 'Inter\n2', #'basis', #'basis', 'basis',
				'Orn','Orn','Orn','Orn','Orn','Orn']
				# 'orn','orn','orn','orn',]#'orn','orn']
	featuresNorm = []
	featuresNorm.append((arclengths-arcMin)/(arcMax-arcMin))
	featuresNorm.append((lengthTime-lengthTimeMin)/(lengthTimeMax-lengthTimeMin))
	# featuresNorm.append((frameCounts-frameCountMin)/(frameCountMax-frameCountMin))	
	# featuresNorm.append((touch0-touch0Min)/(touch0Max-touch0Min))
	# featuresNorm.append((touch1-touch1Min)/(touch1Max-touch1Min))
	if 1:
		featuresNorm.append((touchRad1-touchRad1Min)/(touchRad1Max-touchRad1Min))
		featuresNorm.append((touchRad2-touchRad2Min)/(touchRad2Max-touchRad2Min))
		featuresNorm.append((touchRad3-touchRad3Min)/(touchRad3Max-touchRad3Min))
		featuresNorm.append((touchRad4-touchRad4Min)/(touchRad4Max-touchRad4Min))
	if 0:
		featuresNorm.append((comsRad-comsMin)/(comsMax-comsMin))
	featuresNorm.append((comsX-comsMinX)/(comsMaxX-comsMinX))
	featuresNorm.append((comsY-comsMinY)/(comsMaxY-comsMinY))
	featuresNorm.append((comsY-comsMinZ)/(comsMaxZ-comsMinZ))
	for i in xrange(ornFeatures.shape[1]-3):
		featuresNorm.append(((ornFeatures[:,i]-ornMin)/(ornMax-ornMin)))
	for i in xrange(ornHists.shape[1]):
		featuresNorm.append(ornHists[:,i])		
	featuresNorm = np.array(featuresNorm).T

	X = featuresNorm
	# X = np.nan_to_num(X)
	COLORS = 'rgbcmyk'
	eventLabels = [int(x['label']) for x in p]
	labelColorsTmp = [COLORS[int(x['label'])%len(COLORS)] for x in p]

	# --------------------------------------------------------------------------------
	## Clustering
	if 0:
		from sklearn import manifold
		# Params: # Neigh, # Output dims
		X_iso = manifold.Isomap(3, 2).fit_transform(X)
		# figure(1); scatter(X_iso[:,0], X_iso[:,1], c=labelColorsTmp); title('Isomap') 
		X_lle = manifold.LocallyLinearEmbedding(3, 2).fit_transform(X)
		# figure(2); scatter(X_lle[:,0], X_lle[:,1], c=labelColorsTmp); title('LLE')

		# LLE doesn't seperate as well

		from scipy.spatial import distance
		from sklearn.cluster import DBSCAN
		from sklearn.cluster import SpectralClustering
		from sklearn.cluster import AffinityPropagation as AP
		from sklearn.cluster import MeanShift as MS
		from sklearn.cluster import MiniBatchKMeans as MBK
		from sklearn import metrics

		# Xl = X
		Xl = X_iso
		# Xl = X_lle
		D = distance.squareform(distance.pdist(Xl))
		S = 1 - (D / np.max(D))
		# clust = DBSCAN(min_samples=100, eps=0.85)
		# clust = AP()
		# clust = MS()
		clust = MBK(3)
		clust = clust.fit(S)

		labels = clust.labels_
		n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
		print n_clusters, "clusters"

		if 1:
			for i in range(3):
				plot(-Xl[:,0][labels==i], -Xl[:,1][labels==i], '.')
		

		# Print labeled manifold
		figure(1);
		# for i in range(0, n_clusters):
			# plot(Xl[labels==i,0], Xl[labels==i,1], 'o')
			# scatter(Xl[labels==i,0], Xl[labels==i,1], c=labelColorsTmp[labels==i])
		for i in range(1, 9):
			scatter(Xl[np.nonzero(truthLabels==i),0], Xl[np.nonzero(truthLabels==i),1], c=COLORS[i])

		labelCounts = []
		labelInds = []
		for i in range(n_clusters):
			labelCounts.append(np.sum(labels == i))
			labelInds.append([y for x,y in zip(labels, range(len(labels))) if x == i])

		for i in labelInds[0]:
			for j in xrange(0,len(p[i]['data']['time']), 20):
				showLabeledImage(p[i], j, dir_)

	#  --------------SVM----------------------------------------
	if 0:
		from sklearn import svm as SVM

		Y = np.zeros(len(p))
		for i in touchInds:
			Y[i] = 1
		Ystart = deepcopy(Y)

		svm = SVM.NuSVC(nu=.2, probability=True)
		# svm = SVM.NuSVC(nu=.5, kernel='poly')

		for i in xrange(10):
			svm.fit(X, Y)
			Y = svm.predict(X)

		probs = svm.predict_proba(X)

		changed = [y for x, y in zip(Y!=Ystart, range(len(Y))) if x]
		changedToPos = [x for x in changed if Y[x]]
		changedToNeg = [x for x in changed if not Y[x]]

		if 0:
			for i in changedToNeg:
				for j in xrange(0,len(p[i]['data']['time']), 20):
					showLabeledImage(p[i], j, dir_)
				print i

			for i in range(len(p)):
				if not Y[i]:
					for j in xrange(0,len(p[i]['data']['time']), 20):
						showLabeledImage(p[i], j, dir_)
					print i


		# Seperate second level
		truthLabels = [int(x['label']) for x in p]
		X = np.array([featuresNorm[y] for x,y in zip(Y,range(len(Y))) if x==1])
		L2inds = np.array([y for x,y in zip(Y,range(len(Y))) if x==1])
		labelColorsTmp = [COLORS[int(x['label'])] for x in p]
		labelColorsTmp = [labelColorsTmp[y] for x,y in zip(Y,range(len(Y))) if x==1]
		p2 = [p[y] for x,y in zip(Y,range(len(Y))) if x==1]
		truthLabels2 = [truthLabels[y] for x,y in zip(Y,range(len(Y))) if x==1]
		dominant = argmax(np.histogram(truthLabels2, 7, [0, 7])[0])+1
		relabel = []
		for i in xrange(len(truthLabels2)):
			if truthLabels2[i] == dominant:
				relabel.append(1)
			else:
				relabel.append(0)

		svm = SVM.NuSVC(nu=.2, probability=True)
		# svm = SVM.NuSVC(nu=.5, kernel='poly')
		relabel2 = deepcopy(relabel)
		for i in xrange(10):
			svm.fit(X, relabel2)
			relabel2 = svm.predict(X)

		for i in xrange(len(L2inds)):
			ind = L2inds[i]
			if relabel2[i] == 0:
				for j in xrange(0,len(p[ind]['data']['time']), 20):
					showLabeledImage(p[ind], j, dir_)
				print i			

	# ----------------- SVM one versus all ---------------------
	if 0:
		truthLabels = np.array([int(x['label']) for x in p])
		svm = []
		Yout = []
		Y = []
		score = []
		labelInds = []
		for i in xrange(1,7):
			labelInds.append(np.array([y for x,y in zip(truthLabels,range(len(truthLabels))) if x == i]))
			Y.append(np.zeros(len(p)))
			Y[-1][labelInds[-1]] = 1
			# svm.append(SVM.NuSVC(nu=.1, probability=True, kernel='rbf'))
			svm.append(SVM.SVC(probability=True, kernel='rbf'))
			# poly: 45%, sigmoid: 14%, rbf:74 , linear: 64%
			svm[-1].fit(X, Y[-1])
			Yout.append(svm[-1].predict(X))
			score.append(svm[-1].score(X, Y[-1]))
			print score[-1]
		print "Mean:", np.mean(score[0:-1])
		# 

		score2 = []
		for i in range(6):
			score2.append(svm[i].predict_proba(X)[:,0])
		score2 = np.array(score2).T
		score2ind = np.argmax(score2, 1)+1
		accuracy1 = np.mean(np.equal(score2ind, truthLabels))

	# ----------------- SVM pair-wise ---------------------
	from sklearn import svm as SVM
	truthLabels = np.array([int(x['label']) for x in p])
	classCounts = [len(x) for x in labelInds]
	# truthLabels[np.nonzero(np.equal(truthLabels,7))[0]] = 6

	# classWeights = {}
	# for i in range(0,7):
	# 	classWeights[i] = np.sum(np.equal(truthLabels,i))*1.0 / len(truthLabels)

	# svmAll = SVM.SVC(probability=True, kernel='rbf')
	# svmAll.fit(X, truthLabels, class_weight=classWeights)	
	# for i in range(1,1000,10):
	svmAll = SVM.SVC(C=100, probability=True, kernel='rbf')
	svmAll.fit(X, truthLabels)
	weightedScore  = svmAll.score(X, truthLabels) # 87.17%	
	print weightedScore #80.1% new
	pred = svmAll.predict(X) # 89.2% %old
	import random as rand

	labelCount = len(labels.keys())+2
	scores = []
	classConfusion = np.zeros([labelCount,labelCount])
	ratio=len(p)
	classScores = np.zeros([ratio, labelCount])
	# classScores = np.zeros(labelCount)
	t0 = time.time()

	svmScore = []
	# for i in range(ratio):


	# 	testInds = [x for x in range(len(X)) if rand.randint(0,ratio-1) == 0]
	# 	perLabelCount = [np.sum(1 for x in testInds if x in labelInds[ii]) for ii in range(len(labelInds))]
	# 	for z,zInd in zip(perLabelCount, range(len(perLabelCount))):
	# 		if z == 0:
	# 			testInds.append(labelInds[zInd][rand.randint(0,len(labelInds[zInd])-1)])
	# 	perLabelCount = [np.sum(1 for x in testInds if x in labelInds[ii]) for ii in range(len(labelInds))]
	# 	trainInds = [x for x in range(len(X)) if x not in testInds]			

	# 	'''Train'''
	# 	XTrain = X[trainInds]
	# 	truthTrain = truthLabels[trainInds]
	# 	svmAll = SVM.SVC(C=100, probability=True, kernel='rbf')
	# 	svmAll.fit(XTrain, truthTrain)		
	# 	'''Test'''
	# 	XTest = X[testInds]
	# 	truthTest = truthLabels[testInds]
	# 	weightedScore  = svmAll.score(XTest, truthTest) # 47%
	# 	scores.append(weightedScore)
	# 	fPred = svmAll.predict(XTest)		
	# 	print weightedScore
	# 	"""Store results"""
	# 	for j in range(1, labelCount+1):
	# 		predInds = np.nonzero(np.equal(truthTest, j))
	# 		classScores[i, j-1] = np.sum(np.equal(fPred[predInds], truthTest[predInds]), dtype=float) / np.sum(np.equal(truthTest, j))
	# print 'Mean:', np.mean(scores)

	for i in range(len(X)):

		# testInds = [x for x in range(len(X)) if rand.randint(0,ratio-1) == 0]
		# perLabelCount = [np.sum(1 for x in testInds if x in labelInds[ii]) for ii in range(len(labelInds))]
		# for z,zInd in zip(perLabelCount, range(len(perLabelCount))):
			# if z == 0:
				# testInds.append(labelInds[zInd][rand.randint(0,len(labelInds[zInd])-1)])
		# perLabelCount = [np.sum(1 for x in testInds if x in labelInds[ii]) for ii in range(len(labelInds))]
		# trainInds = [x for x in range(len(X)) if x not in testInds]			
		testInds = [i]
		trainInds = range(len(X))
		trainInds.remove(i)

		'''Train'''
		XTrain = X[trainInds]
		truthTrain = truthLabels[trainInds]
		svmAll = SVM.SVC(C=100, probability=True, kernel='rbf')
		svmAll.fit(XTrain, truthTrain)		
		'''Test'''
		XTest = X[testInds]
		truthTest = truthLabels[testInds]
		weightedScore  = svmAll.score(XTest, truthTest)
		scores.append(weightedScore)
		fPred = svmAll.predict(XTest)		
		print weightedScore
		"""Store results"""
		for j in range(1, labelCount+1):
			predInds = np.nonzero(np.equal(truthTest, j))
			classScores[i, j-1] = np.sum(np.equal(fPred[predInds], truthTest[predInds]), dtype=float) / np.sum(np.equal(truthTest, j))
	print 'Mean:', np.mean(scores)


	print 'Time:', time.time()-t0
	print 'Overall Mean:', np.mean(scores) # ~63%, new 48%

	figure(1)
	imshow(classScores, interpolation='nearest')
	xticks(arange(len(labelNames)), [labels[x] for x in labels.keys()])
	fMean = np.nansum(classScores, 0) / np.sum(-np.isnan(classScores), 0)
	fMean = [fMean[0],fMean[2],fMean[3],fMean[4],fMean[5],fMean[6],fMean[7]]	
	print 'Per-class means', fMean
	print 'Overall per-class mean', np.nansum(fMean) / np.sum(~np.isnan(fMean))

	"""Per-class accuracy"""
	figure(2)
	bar(arange(-.5, labelCount-2.5), fMean, color='k')
	xticks(arange(len(labelNames)+1), [labels[x] for x in labels.keys()], fontsize=18)
	axis([-.75, labelCount-2.5, 0, 1])
	xlabel('Classes', fontsize=20)
	ylabel('Accuracy', fontsize=20)
	title('Per-class accuracy using an SVM', fontsize=26)
	# savefig("/Users/colin/Desktop/PerClassAccuracy_D2.svg", format="svg")

	"""Class Confusion"""
	figure(4)
	for j in range(len(classConfusion)):
		classConfusion[j] /= np.sum(classConfusion, 1)[j]
	imshow(classConfusion[:7,:7], cmap=cm.gray_r, interpolation='nearest')
	xticks(arange(len(labelNames)), [labels[x] for x in labels.keys()], fontsize=14)
	yticks(arange(len(labelNames)), [labels[x] for x in labels.keys()], fontsize=14)
	xlabel('Predicted Class', fontsize=20)
	ylabel('Actual Class', fontsize=20)
	title('Class Confusion Matrix', fontsize=26)
	# savefig("/Users/colin/Desktop/PerClassAccuracy_D2.svg", format="svg")



	# ------------- Random Forest (Extra Trees) ---------------
	# Note, 2 procs is faster than 8
	truthLabels = np.array([int(x['label']) for x in p])
	from sklearn.ensemble import RandomForestClassifier	
	from sklearn.ensemble import ExtraTreesClassifier
	fCount = 5#featuresNorm.shape[1]
	# forest = ExtraTreesClassifier(n_estimators=10, compute_importances=False, n_jobs=4, bootstrap=False, random_state=0, max_features=1)#26)
	forest = ExtraTreesClassifier(n_estimators=100, compute_importances=True, n_jobs=7, bootstrap=True, random_state=0, max_features=fCount)
	# forest = RandomForestClassifier(n_estimators=30, compute_importances=True, n_jobs=4, bootstrap=True, random_state=0, max_features=10)#26)
	t0 = time.time()
	forest.fit(X, truthLabels)
	print "Time:", time.time()-t0
	importances = forest.feature_importances_
	forestScore = forest.score(X, truthLabels) # 100%
	predF = forest.predict(X)
	print forestScore
	if 1:
		figure(3)
		bar(range(fCount), importances, color='k')
		xticks(arange(.5, featuresNorm.shape[1]+.5), featureNames, fontsize=14)
		yticks(fontsize=12)
		title('Importance Weighting of Random Forest Features', fontsize=28)
		xlabel("Features", fontsize=22)
		ylabel("Weighting", fontsize=22)
		axis([-.25, fCount, 0, .2])

	# forest = RandomForestClassifier(n_estimators=200, compute_importances=False, n_jobs=1, bootstrap=True, random_state=0, max_features=fCount) # BEST
	# forest = RandomForestClassifier(n_estimators=100, compute_importances=False, n_jobs=1, bootstrap=True, random_state=0, max_features=fCount)	
	# forest = ExtraTreesClassifier(n_estimators=20, compute_importances=False, n_jobs=1, bootstrap=True, random_state=0, max_features=fCount)	


	forest = ExtraTreesClassifier(n_estimators=20, compute_importances=False, n_jobs=1, bootstrap=True, random_state=0, max_features=5)
	Xorig = deepcopy(X)
	truthLabelsOrig = deepcopy(truthLabels)
	# XWO6 = np.array([x for x, y in zip(Xorig, truthLabelsOrig) if y != 6])
	# truthLabelsWO6 = np.array([y for x, y in zip(Xorig, truthLabelsOrig) if y != 6])
	X = Xorig
	truthLabels = truthLabelsOrig

	mapLab2Ind = {1:0, 2:'e',3:1,4:2,5:3,6:4,7:5,8:6,9:'e'}

	# labelCount = len(labelNames
	labelCount = len(labels.keys())+2
	forestScores = []; classConfusion = np.zeros([labelCount,labelCount])
	forestProbs = []; forestProbGood = 0; forestProbTotal = 0;
	forestIndivScores = []
	ratio=len(p)
	forestClassScores = np.zeros([ratio, labelCount])
	t0 = time.time()
	classifiedLabels = []

	# for i in range(ratio):

	# 	"""Create training/testing sets"""
	# 	testInds = [x for x in range(len(X)) if rand.randint(0,ratio-1) == 0]
	# 	perLabelCount = [np.sum(1 for x in testInds if x in labelInds[ii]) for ii in range(len(labelInds))]
	# 	for z,zInd in zip(perLabelCount, range(len(perLabelCount))):
	# 		if z == 0:
	# 			testInds.append(labelInds[zInd][rand.randint(0,len(labelInds[zInd])-1)])
	# 	perLabelCount = [np.sum(1 for x in testInds if x in labelInds[ii]) for ii in range(len(labelInds))]
	# 	trainInds = [x for x in range(len(X)) if x not in testInds]			

	# 	"""Training Data"""
	# 	XTrain = X[trainInds]
	# 	truthTrain = truthLabels[trainInds]
	# 	forest.fit(XTrain, truthTrain)
	# 	"""Testing Data"""
	# 	XTest = X[testInds]
	# 	truthTest = truthLabels[testInds]
	# 	# Look at probability of scores
	# 	if len(truthTest) > 0:
	# 		fPredProb = forest.predict_proba(XTest).max(1)
	# 		forestProbs.append(fPredProb)
	# 		forestProbTotal += len(truthTest)
	# 		forestProbGood += np.sum(fPredProb > .7)
	# 	else:
	# 		forestProbs.append([])

	# 	weightedScore = forest.score(XTest, truthTest)
	# 	# weightedScore = np.sum(fPred == truthTest, dtype=float) / len(fPred)
	# 	fPred = forest.predict(XTest)	
	# 	predictTest = fPred
	# 	forestIndivScores.append(fPred == truthTest)
	# 	forestScores.append(weightedScore)
	# 	print weightedScore
	# 	"""Confusion matrix"""
	# 	if 0:
	# 		# predictTest = forest.predict(XTest)
	# 		for j in xrange(len(predictTest)):
	# 			classConfusion[mapLab2Ind[truthTest[j]], mapLab2Ind[predictTest[j]]] += 1
	# 	"""Store results"""
	# 	for j in range(1, labelCount+1):
	# 		predInds = np.nonzero(np.equal(truthTest, j))
	# 		forestClassScores[i, j-1] = np.sum(np.equal(fPred[predInds], truthTest[predInds]), dtype=float) / np.sum(np.equal(truthTest, j))

	for i in range(len(X)):

		"""Create training/testing sets"""
		# testInds = [x for x in range(len(X)) if rand.randint(0,ratio-1) == 0]
		# perLabelCount = [np.sum(1 for x in testInds if x in labelInds[ii]) for ii in range(len(labelInds))]
		# for z,zInd in zip(perLabelCount, range(len(perLabelCount))):
			# if z == 0:
				# testInds.append(labelInds[zInd][rand.randint(0,len(labelInds[zInd])-1)])
		# perLabelCount = [np.sum(1 for x in testInds if x in labelInds[ii]) for ii in range(len(labelInds))]
		# trainInds = [x for x in range(len(X)) if x not in testInds]
		testInds = [i]
		trainInds = range(len(X))
		trainInds.remove(i)

		"""Training Data"""
		XTrain = X[trainInds]
		truthTrain = truthLabels[trainInds]
		forest.fit(XTrain, truthTrain)
		"""Testing Data"""
		XTest = X[testInds]
		truthTest = truthLabels[testInds]
		# Look at probability of scores
		fPredProb = forest.predict_proba(XTest).max(1)
		forestProbs.append(fPredProb)
		forestProbTotal += len(truthTest)
		forestProbGood += np.sum(fPredProb > .7)

		weightedScore = forest.score(XTest, truthTest)
		# weightedScore = np.sum(fPred == truthTest, dtype=float) / len(fPred)
		fPred = forest.predict(XTest)	
		classifiedLabels.append(fPred)		
		predictTest = fPred
		forestIndivScores.append(fPred == truthTest)
		forestScores.append(weightedScore)
		print weightedScore
		"""Confusion matrix"""
		if 1:
			# predictTest = forest.predict(XTest)
			for j in xrange(len(predictTest)):
				classConfusion[mapLab2Ind[truthTest[j]], mapLab2Ind[predictTest[j]]] += 1
		"""Store results"""
		for j in range(1, labelCount+1):
			predInds = np.nonzero(np.equal(truthTest, j))
			forestClassScores[i, j-1] = np.sum(np.equal(fPred[predInds], truthTest[predInds]), dtype=float) / np.sum(np.equal(truthTest, j))



	highConfResults = []
	for r1, c1 in zip(forestIndivScores, forestProbs):
		for c2 in range(len(c1)):
			if c1[c2] > .5:
				highConfResults.append(r1[c2])
	print np.mean(highConfResults)

	# print 'Time:', time.time()-t0
	print 'Overall Mean:', np.nansum(forestScores)/(~np.isnan(forestScores)).sum()

	figure(1)
	imshow(forestClassScores, interpolation='nearest')
	xticks(arange(len(labelNames)), [labels[x] for x in labels.keys()])
	fMean = np.nansum(forestClassScores, 0) / np.sum(-np.isnan(forestClassScores), 0)
	fMean = [fMean[0],fMean[2],fMean[3],fMean[4],fMean[5],fMean[6],fMean[7]]	
	# print 'Per-class means', fMean
	print 'Overall per-class mean', np.nansum(fMean) / np.sum(~np.isnan(fMean))

	"""Per-class accuracy"""
	figure(2)
	bar(arange(-.5, labelCount-2.5), fMean, color='k')
	xticks(arange(len(labelNames)+1), [labels[x] for x in labels.keys()], fontsize=18)
	axis([-.75, labelCount-2.5, 0, 1])
	xlabel('Classes', fontsize=20)
	ylabel('Accuracy', fontsize=20)
	title('Per-class accuracy using a Decision Forest', fontsize=26)
	# savefig("/Users/colin/Desktop/PerClassAccuracy_D2.svg", format="svg")

	"""Class Confusion"""
	figure(4)
	for j in range(len(classConfusion)):
		classConfusion[j] /= np.sum(classConfusion, 1)[j]
	imshow(classConfusion[:7,:7], cmap=cm.gray_r, interpolation='nearest')
	xticks(arange(len(labelNames)), [labels[x] for x in labels.keys()], fontsize=14)
	yticks(arange(len(labelNames)), [labels[x] for x in labels.keys()], fontsize=14)
	xlabel('Predicted Class', fontsize=20)
	ylabel('Actual Class', fontsize=20)
	title('Class Confusion Matrix', fontsize=26)
	# savefig("/Users/colin/Desktop/PerClassAccuracy_D2.svg", format="svg")

	#  ---------------------------------------------------------
	timeEvents = {}
	for i in xrange(len(p)):
		datum = p[i]
		for j in xrange(len(datum['data']['time'])):
			t = datum['data']['time'][j]
			if t not in timeEvents.keys():
				timeEvents[t] = {i:[j]} #Event, event-time
			else:
				if i not in timeEvents[t].keys():
					timeEvents[t][i] = [j]
				else:
					timeEvents[t][i].append(j)
	# Fill out rest of times
	maxTimeData = p[timeEvents[timeEvents.keys()[-1]].keys()[0]]
	maxTime = maxTimeData['start']+maxTimeData['elapsed']
	for i in xrange(maxTime):
		if i not in timeEvents.keys():
			timeEvents[i] = {}

	timeEventsFeatures = {}
	for i in xrange(len(timeEvents)):
		event = timeEvents[i]
		dist = np.zeros(len(event.keys()))
		peopleCount = len(event.keys())
		for key, j in zip(event.keys(), range(peopleCount)):
			# key = event[j]
			com = np.sqrt(np.sum(p[key]['data']['com'][event[key][0]]))
			for key2 in event.keys():
				dist[j] += np.sqrt(np.sum(p[key]['data']['com'][event[key2][0]]))
		dist /= peopleCount



		#  --------------HMM--------------------------------------

	# labels = {1:'group', 2:'talking', 3:'observing', 4:'read', 5:'procedure', 6:'unrelated'}
	
	prevStates = np.zeros(len(labelNames), dtype=bool)
	newStates = np.zeros(len(labelNames), dtype=bool)
	labelCounts = np.zeros([len(labelNames), 2,2], dtype=float)

	# Generate A matrix
	maxTime = timeEvents.keys()[-1]
	for secInd in xrange(1, maxTime):
		newStates[:] = False

		for eventInd in timeEvents[secInd]:
			datum = p[eventInd]
			label = int(datum['label'])
			newStates[label-1] = True
		# else: 

		for i in xrange(len(labels)):
			if prevStates[i] and newStates[i]:   # P(t|t)
				labelCounts[i,0,0] += 1
			elif ~prevStates[i] and newStates[i]:  # P(t|~t)
				labelCounts[i,0,1] += 1
			elif prevStates[i] and ~newStates[i]:  # P(~t|t)
				labelCounts[i,1,0] += 1
			elif ~prevStates[i] and ~newStates[i]: # P(~t|~t)
				labelCounts[i,1,1] += 1

		prevStates[:] = newStates[:]

	AMats = labelCounts / labelCounts[0].sum()

	# Generate pi matrix
	piMats = np.empty([len(labels.keys())])
	for i in xrange(len(labels.keys())):
		piMats[i] = Amats[i][0,:].sum() / Amats[i][1,:].sum()

	## Generate B matrix

	# Plot Distributions
	for i in range(6):
		subplot(2,3,i)
		h = hist(featuresNorm[:,i])
		plot(h[1][1:], h[0])

	# SVD of distributions
	_,_,v = svd(featuresNorm, full_matrices=0)
	basis = v[0]
	pcaFeatures = np.dot(featuresNorm, v[0])
	scatter(range(len(pcaFeatures)), pcaFeatures, c=labelColorsTmp)
	scatter(np.array(range(len(pcaFeatures)))*0, pcaFeatures, c=labelColorsTmp)

	from sklearn.mixture import GMM
	labelMeans = np.empty([len(labels.keys())])
	labelCovars = np.empty([len(labels.keys())])
	m1 = GMM(1)
	for i in labels.keys():
		m1.fit(pcaFeatures[np.nonzero(np.equal(eventLabels, i))])
		labelMeans[i-1] = m1.means
		labelCovars[i-1] = m1.covars[0][0][0]


	def gaussProb(data, mean, covar):
		return (1 / np.sqrt(2*np.pi*covar) * np.exp( -(data - mean)**2 / (2*covar))) / 2.124


	#  --------------Test----------------
	current = piMats
	for i in xrange(maxTime):
		val = np.dot(featuresNorm[i,:], basis)
		BMat = gaussProb(1.0, labelMeans, labelCovars)

		current *= BMat


	#### Current feature data is in event space ###
	#!!!#### Create feature data in time space ######!!#


	#  --------------Split events--------------------------------------
	splitLength = 3 #seconds
	tmp = [x for x in p if x['elapsed'] > 200]
	tmpInd = [y for x,y in zip(p,range(len(p))) if x['elapsed'] > 200]


	for j in xrange(0,len(p[tmpInd[0]]['data']['time']), 20):
		showLabeledImage(p[tmpInd[0]], j, dir_)


	sFrames = []
	sCOM = []
	sArcLen = []
	sVel = []
	sOrnComp = []
	sBasis = []

	j = 0; 
	timeTmp=tmp[0]['start']
	prevJ = 0
	while j < len(tmp[0]['data']['com']):
		if tmp[0]['data']['time'][j]-timeTmp <= splitLength:
			j+=1
			continue
		else:
			sCOM.append(tmp[0]['data']['com'][j])
			arcLen = 0
			for k in range(prevJ+1, j):
				arcLen += np.sqrt(np.sum((np.array(tmp[0]['data']['com'][k])-np.array(tmp[0]['data']['com'][k-1]))**2))
			sArcLen.append(arcLen)
			sVel.append(arcLen/9.0)
			sBasis.append(tmp[0]['data']['basis'][j][1])
			sOrnComp.append(tmp[0]['data']['ornCompare'][j])
			sFrames.append(range(prevJ, j))
			prevJ = j
			timeTmp=tmp[0]['data']['time'][j]
			j+=1

	center = np.array([-346.83551756, -16.10465379, 3475.0]) # new footage
	sCOM -= center
	sOrnComp = np.array(sOrnComp)
	sBasis = np.array(sBasis)


	featuresNorm2 = []
	featuresNorm2.append((sArcLen-arcMin)/(arcMax-arcMin))
	featuresNorm2.append((sVel-lengthTimeMin)/(lengthTimeMax-lengthTimeMin))
	featuresNorm2.append((sCOM[:,0]-comsMinX)/(comsMaxX-comsMinX))
	featuresNorm2.append((sCOM[:,1]-comsMinY)/(comsMaxY-comsMinY))
	featuresNorm2.append((sCOM[:,2]-comsMinZ)/(comsMaxZ-comsMinZ))

	for i in xrange(np.shape(sOrnComp)[1]):
		featuresNorm2.append(((sOrnComp[:,i]-ornMin)/(ornMax-ornMin)))
	for i in xrange(np.shape(sBasis)[1]):
		featuresNorm2.append(sBasis[:,i])		
	featuresNorm2 = np.array(featuresNorm2).T

	X2 = featuresNorm2
	X2 = np.nan_to_num(X2)


	# ---------- Manifolds on sets ---------
	from sklearn import manifold, metrics
	from scipy.spatial import distance
	from sklearn.cluster import DBSCAN, AffinityPropagation as AP, MeanShift as MS

	# Params: # Neigh, # Output dims
	X_iso = manifold.Isomap(3, 2).fit_transform(X2)
	figure(1); scatter(X_iso[:,0], X_iso[:,1], c=labelColorsTmp); title('Isomap') 
	X_lle = manifold.LocallyLinearEmbedding(3, 2).fit_transform(X2)
	# figure(2); scatter(X_lle[:,0], X_lle[:,1], c=labelColorsTmp); title('LLE')

	Xl = X_iso #X_lle			
	# D = distance.squareform(distance.pdist(Xl))
	D = distance.squareform(distance.pdist(Xl, metric='chebyshev'))
	S = 1 - (D / np.max(D))
	clust = DBSCAN().fit(S, eps=0.85, min_samples=5)
	# clust = AP(damping=.5).fit(S)
	# clust = MS().fit(S)

	labels = clust.labels_
	n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
	print n_clusters, "clusters"

	# Print labeled manifold
	figure(1);
	for i in range(-1, n_clusters):
		print i
		scatter(Xl[labels==i,0], Xl[labels==i,1], c=COLORS[i])
	# for i in range(1, 9):
	# 	scatter(Xl[np.nonzero(truthLabels==i),0], Xl[np.nonzero(truthLabels==i),1], c=COLORS[i])

	# labelCounts = []
	# labelInds = []
	# for i in range(n_clusters):
	# 	labelCounts.append(np.sum(labels == i))
	# 	labelInds.append([y for x,y in zip(labels, range(len(labels))) if x == i])

	for k in xrange(1, n_clusters):
		ii = 0
		imgD = np.ones([480, 640, 3])*255
		for i in np.nonzero(labels==k)[0]:
			print ii, "of", len(np.nonzero(labels==k)[0]), " sequences"
			ii += 1
			for jj in range(0,len(sFrames[i]),3):
				j = sFrames[i][jj]
				showLabeledImage(p[tmpInd[0]], j, dir_)
			cv2.imshow("a", imgD)
			ret = cv2.waitKey(1)
			time.sleep(.05)
		imgD *= 0
		cv2.imshow("a", imgD)
		ret = cv2.waitKey(1)
		time.sleep(.05)			






	from sklearn import svm as SVM
	Y = np.random.randint(2, size=[np.shape(X2)[0]])
	Ystart = deepcopy(Y)

	svm = SVM.NuSVC(nu=.2, probability=True)
	for i in xrange(10):
		svm.fit(X2, Y)
		Y = svm.predict(X2)

	probs = svm.predict_proba(X)

	changed = [y for x, y in zip(Y!=Ystart, range(len(Y))) if x]
	changedToPos = [x for x in changed if Y[x]]
	changedToNeg = [x for x in changed if not Y[x]]




## -------- Visualize individual data -------

if 0:
	figure(0)
	plot([-x[0] for x in p[1]['data']['com']], [x[1] for x in p[1]['data']['com']])
	axis('equal');figure(1)
	plot([-x[0] for x in p[1]['data']['com']], [x[2] for x in p[1]['data']['com']])
	axis('equal');figure(2)
	plot([x[1] for x in p[1]['data']['com']], [x[2] for x in p[1]['data']['com']])
	axis('equal')
	
	
	fig = figure(9)
	ax = fig.add_subplot(111,projection='3d')
	ax.scatter3D(xs=[-x[0] for x in p[1]['data']['com']], zs=[x[1] for x in p[1]['data']['com']], ys=[x[2] for x in p[1]['data']['com']])
	xlabel('X')
	ylabel('Y')
	axis('equal')
	draw()

