''' Load CAD Dataset '''

import numpy as np
import os


def read_Cornel_Action_Dataset_skels(folder_name='.'):

	data = []
	for d in os.listdir(folder_name):
		if d.find('.txt') >= 0 and d[0] != 'a':
			print d
			d_raw = np.fromfile(d, sep=',')
			if data == []:
				data = d_raw[:-1].reshape([-1,171])
			else:
				data = np.vstack([data, d_raw[:-1].reshape([-1,171])])

	return data


if 0:
	dir_ = '/Users/colin/Data/CAD_data1'
	skels = read_Cornel_Action_Dataset_skels(dir_)


# data_norm = (data-data.min(0))/(data.max(0)-data.min(0))

# meanSize = 60
# data_mean = np.zeros([data.shape[0]/meanSize, 171])
# for i in xrange(data.shape[0]/meanSize):
# 	data_mean[i,:] = np.mean(data_norm[i*meanSize:(i+1)*meanSize, :], 0)

# imshow(data_mean)



