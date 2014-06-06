import numpy as np
from pyKinectTools_algs_dynamic_time_warping import DynamicTimeWarping_c, dtw_path
'''Dynamic Time Warping'''


def test():
	from scipy.signal import resample
	ref = np.random.random([10,3])
	x = resample(ref, 100)
	y = resample(ref, 200)

	if x.ndim == 1:
		x = np.atleast_2d(x).T
		y = np.atleast_2d(y).T

	x = np.ascontiguousarray(x)	
	y = np.ascontiguousarray(y)	

	DynamicTimeWarping(x,y)

	print "Cost:", cost_matrix[-1,-1]
	print "Path:", path.T


'''
def tmp(x, y):
	cost_matrix = DynamicTimeWarping_c(x,y)
	path = dtw_path(np.ascontiguousarray(cost_matrix))

%timeit tmp(x,y)

Compare to mlpy's implementation (which uses FastDTW?)
import mlpy
%timeit error, dtw_mat, y_ind = mlpy.dtw.dtw_std(x[:,0], y[:,0], dist_only=False, squared=True)

'''


def DynamicTimeWarping(x, y, return_path=True):
	# To allow for multidimensional data, x and y must be at least 2D
	if x.ndim == 1:
		x = np.atleast_2d(x).T
		y = np.atleast_2d(y).T
	
	x = np.ascontiguousarray(x)	
	y = np.ascontiguousarray(y)	

	cost_matrix = DynamicTimeWarping_c(x,y)

	if not return_path:
		return cost_matrix[-1,-1], cost_matrix

	# Compute optimal path from start to end of matrix
	path = dtw_path(cost_matrix)

	return cost_matrix[-1,-1], cost_matrix, path

def euclidian_distance(x, y):
	'''
	Note: don't both taking square root
	'''
	return np.sum((x - y)**2)

# def DynamicTimeWarping_py(x, y, return_path=True, distance_fcn='euclidian'):
# 	'''
# 	Implements vanilla DTW

# 	http://en.wikipedia.org/wiki/Dynamic_time_warping
# 	'''
# 	if distance_fcn == "euclidian":
# 		distance_fcn = euclidian_distance

# 	x_res = len(x)
# 	y_res = len(y)
# 	cost_matrix = np.zeros([x_res, y_res], dtype=np.float)
	
# 	# Caclulate edges first (they only depend on one value)
# 	for i in xrange(1, x_res):
# 		cost_matrix[i,0] = cost_matrix[i-1,0] + distance_fcn(x[i], y[0])
# 	for j in xrange(1, y_res):
# 		cost_matrix[0,j] = cost_matrix[0,j-1] + distance_fcn(x[0], y[j])		

# 	# Calculate distance at each location
# 	for i in xrange(1, len(x)):
# 		for j in xrange(1, len(y)):
# 			cost = distance_fcn(x[i], y[j])
# 			cost_matrix[i,j] = cost + np.min([cost_matrix[i-1,j], cost_matrix[i,j-1], cost_matrix[i-1,j-1]])

# 	# Cost is simply the last element
# 	max_cost = cost_matrix[-1,-1]

# 	if not return_path:
# 		return max_cost, cost_matrix

# 	# Calculate path from start to finish
# 	path = [[0,0]]
# 	candidate_pts = np.array([[0,1],[1,0],[1,1]])
# 	current_pos = [0,0]

# 	while path[-1][0] < x_res-1 or path[-1][1] < y_res-1:
# 		if path[-1][0] == x_res-1:
# 			next = 0
# 		elif path[-1][1] == y_res-1:
# 			next = 1
# 		else:
# 			next = np.argmin(cost_matrix[candidate_pts[:,0], candidate_pts[:,1]])
# 		path += [[candidate_pts[next][0], candidate_pts[next][1]]]

# 		if next==0:
# 			candidate_pts += [0,1]
# 		elif next == 1:
# 			candidate_pts += [1,0]
# 		else:
# 			candidate_pts += [1,1]

# 	path = np.array(path)

# 	return max_cost, cost_matrix, path


def draw_path(cost_matrix, path):
	'''
	Draw path from start to finish of DTW cost matrix
	(for visualization purposes)
	'''
	path = np.array(path)

	max_score = score_matrix[-1,-1]
	for p in path:
		cost_matrix[p[0], p[1]] = max_score

	return cost_matrix






		
