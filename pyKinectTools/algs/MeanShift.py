'''

NOTE: This is adapted from
http://sociograph.blogspot.com/2011/11/scalable-mean-shift-clustering-in-few.html
'''

import numpy as np
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.utils import extmath
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict 

  
def mean_shift(X, bandwidth, n_seeds, kernel_function='gaussian', max_iterations=100, proximity_thresh=5):
    '''
    ---Parameters---
    X : data in form (samples, dims)
    bandwidth : radius of nearest neighbors
    n_seeds : 
    kernel_update_function : can be "gaussian" or "flat" or your own kernel
    proximity_thresh : minimum distance (in pixels) a new cluster must be away from previous ones

    ---Returns---
    cluster_centers : 
    cluster_counts : how many pixels are with the neighborhood of each cluster
    '''

    import numpy as np
    from sklearn.neighbors import BallTree, NearestNeighbors
    from sklearn.utils import extmath
    from sklearn.metrics.pairwise import euclidean_distances
    from collections import defaultdict 

    if kernel_function == 'gaussian':
        kernel_update_function = gaussian_kernel
    elif kernel_function == 'flat':
        kernel_update_function = flat_kernel
    else:
        kernel_update_function = kernel_function


    n_points, n_features = X.shape
    stop_thresh = 1e-2 * bandwidth # when mean has converged                                                                                                               
    cluster_centers = []
    cluster_counts = [] 
    # ball_tree = BallTree(X)# to efficiently look up nearby points
    neighbors = NearestNeighbors(radius=bandwidth).fit(X)

    seeds = X[(np.random.uniform(0,X.shape[0], n_seeds)).astype(np.int)]
 
    # For each seed, climb gradient until convergence or max_iterations                                                                                                     
    for weighted_mean in seeds:
         completed_iterations = 0
         while True:
             points_within = X[neighbors.radius_neighbors([weighted_mean], bandwidth, return_distance=False)[0]]
             old_mean = weighted_mean  # save the old mean                                                                                                                  
             weighted_mean = kernel_update_function(old_mean, points_within, bandwidth)
             converged = extmath.norm(weighted_mean - old_mean) < stop_thresh
             if converged or completed_iterations == max_iterations:
                # Only add cluster if it's different enough from other centers
                if len(cluster_centers) > 0:
                    diff_from_prev = [np.linalg.norm(weighted_mean-cluster_centers[i], 2) for i in range(len(cluster_centers))]
                    if np.min(diff_from_prev) > proximity_thresh:
                        cluster_centers.append(weighted_mean)
                        cluster_counts.append(points_within.shape[0])
                else:
                    cluster_centers.append(weighted_mean)
                    cluster_counts.append(points_within.shape[0])
                break
             completed_iterations += 1
 
    return cluster_centers, cluster_counts

 
def gaussian_kernel(x, points, bandwidth):
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(points, x)
    weights = np.exp(-1 * (distances ** 2 / bandwidth ** 2))
    return np.sum(points * weights, axis=0) / np.sum(weights)
 
def flat_kernel(x, points, bandwidth):
    return np.mean(points, axis=0)

 
def bin_points(X, bin_size, min_bin_freq):
    bin_sizes = defaultdict(int)
    for point in X:
        binned_point = np.cast[np.int32](point / bin_size)
        bin_sizes[tuple(binned_point)] += 1
 
    bin_seeds = np.array([point for point, freq in bin_sizes.iteritems() if freq >= min_bin_freq], dtype=np.float32)
    bin_seeds = bin_seeds * bin_size
    return bin_seeds