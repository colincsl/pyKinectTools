"""
Methods to characterize image textures.
"""

import numpy as np
from pyKinectTools_algs_local_occupancy_pattern import _local_occupancy_pattern, _local_binary_pattern_depth

def local_binary_pattern_depth(image, P, R, method='default', px_diff_thresh=10):
    """Gray scale and rotation invariant LBP (Local Binary Patterns).

    LBP is an invariant descriptor that can be used for texture classification.

    Parameters
    ----------
    image : (N, M) array
        Graylevel image.
    P : int
        Number of circularly symmetric neighbour set points (quantization of
        the angular space).
    R : float
        Radius of circle (spatial resolution of the operator).
    method : {'default', 'ror', 'uniform', 'var'}
        Method to determine the pattern.

        * 'default': original local binary pattern which is gray scale but not
            rotation invariant.
        * 'ror': extension of default implementation which is gray scale and
            rotation invariant.
        * 'uniform': improved rotation invariance with uniform patterns and
            finer quantization of the angular space which is gray scale and
            rotation invariant.
        * 'var': rotation invariant variance measures of the contrast of local
            image texture which is rotation but not gray scale invariant.

    Returns
    -------
    output : (N, M) array
        LBP image.

    References
    ----------
    .. [1] Multiresolution Gray-Scale and Rotation Invariant Texture
           Classification with Local Binary Patterns.
           Timo Ojala, Matti Pietikainen, Topi Maenpaa.
           http://www.rafbis.it/biplab15/images/stories/docenti/Danielriccio/\
           Articoliriferimento/LBP.pdf, 2002.
    """

    methods = {
        'default': ord('D'),
        'ror': ord('R'),
        'uniform': ord('U'),
        'var': ord('V')
    }
    image = np.array(image, dtype='double', copy=True)
    output = _local_binary_pattern_depth(image, P, R, methods[method.lower()], px_diff_thresh)
    return output

def local_occupancy_pattern(image, bin_size, grid_size):
    """
    LOP
    """
    image = np.ascontiguousarray(image, dtype=np.int16)
    # from IPython import embed
    # embed()
    output = _local_occupancy_pattern(image, np.array(bin_size, dtype=np.int16), np.array(grid_size, dtype=np.int16))

    return output
