'''
Adapted somewhat from skimage Local Binary Pattern implementation

-- This works differently than I implemented! --

'''

#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as cnp
from libc.math cimport sin, cos, abs
# from skimage._shared.interpolation cimport bilinear_interpolation
ctypedef cnp.int16_t INT16
ctypedef cnp.int64_t INT64

cdef inline int _bit_rotate_right(int value, int length):
    """Cyclic bit shift to the right.

    Parameters
    ----------
    value : int
        integer value to shift
    length : int
        number of bits of integer

    """
    return (value >> 1) | ((value & 1) << (length - 1))


# def _local_occupancy_pattern(cnp.ndarray[double, ndim=2] image,
def _local_binary_pattern_depth(cnp.ndarray[INT16, ndim=2] image,
                          int P, float R, char method='D',
                          int px_diff_thresh=1):
    """Gray scale and rotation invariant LBP (Local Binary Patterns).

    LBP is an invariant descriptor that can be used for texture classification.

    Parameters
    ----------
    image : (N, M) double array
        Graylevel image.
    P : int
        Number of circularly symmetric neighbour set points (quantization of the
        angular space).
    R : float
        Radius of circle (spatial resolution of the operator).
    method : {'D', 'R', 'U', 'V'}
        Method to determine the pattern.

        * 'D': 'default'
        * 'R': 'ror'
        * 'U': 'uniform'
        * 'V': 'var'

    Returns
    -------
    output : (N, M) array
        LBP image.
    """

    # texture weights
    cdef cnp.ndarray[int, ndim=1] weights = 2 ** np.arange(P, dtype=np.int32)
    # local position of texture elements
    rp = - R * np.sin(2 * np.pi * np.arange(P, dtype=np.double) / P)
    cp = R * np.cos(2 * np.pi * np.arange(P, dtype=np.double) / P)
    cdef cnp.ndarray[double, ndim=2] coords = np.round(np.vstack([rp, cp]).T, 5)
    # pre allocate arrays for computation
    cdef cnp.ndarray[double, ndim=1] texture = np.zeros(P, np.double)
    cdef cnp.ndarray[char, ndim=1] signed_texture = np.zeros(P, np.int8)
    cdef cnp.ndarray[int, ndim=1] rotation_chain = np.zeros(P, np.int32)

    output_shape = (image.shape[0], image.shape[1])
    cdef cnp.ndarray[double, ndim=2] output = np.zeros(output_shape, np.double)

    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]

    cdef double lbp
    cdef Py_ssize_t r, c, changes, i, r_new, c_new
    for r in xrange(rows):
        for c in xrange(cols):
            for i in xrange(P):
                r_new = int(r + coords[i, 0])
                c_new = int(c + coords[i, 1])
                if r_new < 0 or r_new >= rows or c_new < 0 or c_new >= cols:
                    texture[i] = 0.
                else:
                    texture[i] = image[r_new, c_new]

                # signed / thresholded texture
                if abs(texture[i] - image[r, c]) >= px_diff_thresh:
                    signed_texture[i] = 1
                else:
                    signed_texture[i] = 0

            lbp = 0
            # if method == 'uniform' or method == 'var':
            if method == 'U' or method == 'V':
                # determine number of 0 - 1 changes
                changes = 0
                for i in range(P - 1):
                    changes += abs(signed_texture[i] - signed_texture[i + 1])

                if changes <= 2:
                    for i in range(P):
                        lbp += signed_texture[i]
                else:
                    lbp = P + 1

                if method == 'V':
                    var = np.var(texture)
                    if var != 0:
                        lbp /= var
                    else:
                        lbp = np.nan
            else:
                # method == 'default'
                for i in range(P):
                    lbp += signed_texture[i]# * weights[i]

                # method == 'ror'
                if method == 'R':
                    # shift LBP P times to the right and get minimum value
                    rotation_chain[0] = <int>lbp
                    for i in range(1, P):
                        rotation_chain[i] = \
                            _bit_rotate_right(rotation_chain[i - 1], P)
                    lbp = rotation_chain[0]
                    for i in range(1, P):
                        lbp = min(lbp, rotation_chain[i])


            output[r, c] = lbp

    return output

def _local_occupancy_pattern(cnp.ndarray[INT16, ndim=2] image,\
                            cnp.ndarray[INT16, ndim=1] bin_size,\
                            cnp.ndarray[INT16, ndim=1] grid_size):
    """ Local Occupancy Pattern for RGBD Data

    LOP is an invariant descriptor that can be used for depth-based texture classification.

    Parameters
    ----------
    image : (N, M) double array
        16-bit image
    bin_size : list
        [x,y,z] where each element is the size of the bin
    grid_size : list
        [x,y,z] where each element is the number of bins

    Returns
    -------
    output : (N, M) array
        LOP image.
    """

    # pre allocate arrays for computation
    cdef int grid_count = grid_size[0] + grid_size[1] + grid_size[2]
    cdef cnp.ndarray[INT16, ndim=3] grid = np.zeros([grid_size[0],grid_size[1],grid_size[2]], np.int16)
    cdef cnp.ndarray[INT64, ndim=1] texture = np.zeros(grid_count, np.int64)

    output_shape = (image.shape[0], image.shape[1])
    cdef cnp.ndarray[INT64, ndim=2] output = np.zeros(output_shape, np.int64)
    cdef cnp.ndarray[INT64, ndim=1] weights = 2 ** np.arange(grid_count, dtype=np.int64)
    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]

    # cdef double lop
    cdef int grid_height = bin_size[0]*grid_size[0]
    cdef int grid_width = bin_size[1]*grid_size[1]
    cdef int grid_depth = bin_size[2]*grid_size[2]

    cdef int grid_offset_r = (grid_size[0]-1)/2
    cdef int grid_offset_c = (grid_size[1]-1)/2
    cdef int bin_offset_r = (bin_size[0]-1)/2
    cdef int bin_offset_c = (bin_size[1]-1)/2
    cdef int bin_offset_z = (bin_size[1]-1)/2
    cdef int depth_diff = 0
    cdef int depth_mod = 0

    cdef Py_ssize_t r, c, d
    for r in xrange(grid_height, rows-grid_height):
        for c in xrange(grid_width, cols-grid_width):
            if image[r,c] != 0:
                print 'a'
                # Go through each bin
                for rr in xrange(-grid_offset_r, grid_offset_r+1):
                    for cc in xrange(-grid_offset_c, grid_offset_c+1):
                        print 'b', r+rr-bin_offset_r, r+rr+bin_offset_r+1, c+cc-bin_offset_c,c+cc+bin_offset_c+1, image[r,c], image[r+rr-bin_offset_r:r+rr+bin_offset_r+1, c+cc-bin_offset_c:c+cc+bin_offset_c+1]
                        depth_diff = image[r+rr-bin_offset_r:r+rr+bin_offset_r+1, c+cc-bin_offset_c:c+cc+bin_offset_c+1] - image[r,c]
                        # depth_diff = image[r+rr-bin_offset_r:r+rr+bin_offset_r+1, c+cc-bin_offset_c:c+cc+bin_offset_c+1] - image[r,c]
                        print "diff:", depth_diff
                        if abs(depth_diff) < grid_depth/2:
                            depth_mod = depth_diff%bin_size[2]+bin_offset_z
                            print "Mod:", depth_mod
                            grid[rr,cc,depth_mod] += 1
                        for d in xrange(grid_size[2]):
                            if grid[rr,cc,d] > 0:
                                texture[rr*grid_size[0]+cc*grid_size[1]+d] += 1

                    output[r,c] += texture * weights
                    texture *= 0

    return output
