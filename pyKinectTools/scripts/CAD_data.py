''' Load CAD Dataset '''

import numpy as np
import os, sys

dir_ = '/Users/colin/Data/CAD_data1'
txt = '0512164333.txt'

data = []
for d in os.listdir('.'):
	if d.find('.txt') >= 0 and d[0] != 'a':
		print d
		d_raw = np.fromfile(d, sep=',')
		if data == []:
			data = d_raw[:-1].reshape([-1,171])
		else:
			data = np.vstack([data, d_raw[:-1].reshape([-1,171])])

data_norm = (data-data.min(0))/(data.max(0)-data.min(0))

meanSize = 60
data_mean = np.zeros([data.shape[0]/meanSize, 171])
for i in xrange(data.shape[0]/meanSize):
	data_mean[i,:] = np.mean(data_norm[i*meanSize:(i+1)*meanSize, :], 0)

imshow(data_mean)



# ----------------------------------------
''' Load MSR Dataset '''

'''
First row is number of frames and number of joints
	First row per set is number of joint values
	In each pair of rows the first is world coordinates (meters) and the second is image coordinates
'''
import numpy as np
import os, sys


dir_ = '/Users/colin/Data/MSRDailyAct3D_pack1'
os.chdir(dir_)

data_raw = np.fromfile('a01_s01_e01_skeleton.txt', sep='\n')

frameCount = int(data_raw[0])
jointCount = int(data_raw[1])

data = np.zeros([frameCount, jointCount*4*2]) # in world coords

for i in range(0,frameCount):
	ind = i*(jointCount*2*4+1) + 2	
	data[i,:] = data_raw[ind+1:ind+20*4*2+1]

# Get rid of confidence variable (it's useless for this data)
data_world = data.reshape([frameCount, 40, 4])
data_world = data_world[:,:,:3]
data_world = data_world[:,::2,:]
# Center data and divide by standard deviation
data_world -= data_world.mean(0) 
data_world /= data_world.std(0)

data = data_world.reshape([frameCount,-1])


def loadMSRSkel(filename):

	data_raw = np.fromfile(filename, sep='\n')

	frameCount = int(data_raw[0])
	jointCount = int(data_raw[1])

	data = np.zeros([frameCount, jointCount*4*2]) # in world coords

	for i in range(0,frameCount):
		ind = i*(jointCount*2*4+1) + 2	
		data[i,:] = data_raw[ind+1:ind+20*4*2+1]

	# Get rid of confidence variable (it's useless for this data)
	data_world = data.reshape([frameCount, 40, 4])
	data_world = data_world[:,:,:3]
	# Only get the global positions
	data_world = data_world[:,::2,:]
	# Center data and divide by standard deviation
	data_world -= data_world.mean(0) 
	data_world /= data_world.std(0)

	data = data_world.reshape([frameCount,-1])	

	return data


# Size is 61148x60
data = []
files = os.listdir('.')
for f in files:
	if data == []:
		data = loadMSRSkel(f)
	else:
		try:
			data = np.vstack([data, loadMSRSkel(f)])
		except:
			pass



patches_pos = data[:61140,:].reshape([-1, 15, 60])
patches_vel = patches_pos[:,1:,:] - patches_pos[:,:-1,:]
patches_all = np.hstack([patches_pos[:,1:,:],patches_vel])
# Find mean and std over 15 frames
patches_mean = patches_all.mean(1)
patches_std = patches_all.std(1)
patches_sum = np.hstack([patches_mean, patches_std])

''' Compute PCA of feature vector '''
def PCA(data, nComponents=-1, whiten=True):
	# Normalize
	p = (data-data.mean(0))/data.std(0)
	C = np.cov(p.T)

	if whiten:
		e = np.linalg.eigvals(C)
		C /= e

	u,s,vT = svd(C, full_matrices=False)

	# Get top N components
	if nComponents >= 0:
		components = u[:nComponents]
	else:
		components = u

	return components

patchComponents = PCA(patches_sum, whiten=False)
patchComponents2 = PCA(patches_sum)


# ''' LASSO '''
# from sklean.linear_model import Lasso

# dense_lasso = Lasso(1, fit_intercept=False, max_iter=1000)
# dense_lasso.fit(X)

# ----------------------------------------
''' Move MSR data '''

'''
mkdir ../MSR_Skels_3
'''

import shutil, os

dir_ = '/Users/Colin/Data/MSR_DailyAct3D_pack3/'
dir_new = '/Users/Colin/Data/MSR_Skels_3/'

files = os.listdir('.')
sFiles = [x for x in files if x.find('.txt') >= 0]
for f in sFiles:
	shutil.copyfile(f,dir_new+f)



