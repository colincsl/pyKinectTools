"""
Convert jpg+png image/depth files to .pcd files to work with PCL
"""

import os
import optparse
import numpy as np
from pyKinectTools.dataset_readers.KinectPlayer import KinectPlayer
from IPython import embed


DIR = os.path.expanduser('~')+'/Data/PCL_Data'

header = '''# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH 76800
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 76800
DATA ascii
'''

# -------------------------MAIN------------------------------------------

def main(visualize=False, save_dir='~/', sparse_pointcloud=False):

	# Create save directory if it doesn't exist
	if not os.path.isdir(save_dir):
		os.mkdir(save_dir)

	cam = KinectPlayer(base_dir='./', device=1, get_depth=True,
						get_color=True, get_skeleton=False)

	framerate = 1
	while cam.next(framerate):
		# Get depth points
		pts = cam.camera_model.im2PosIm(cam.depthIm).reshape([-1,3])

		# Get RGB points in floating point notation
		rgb = cam.colorIm.reshape([-1,3]).astype(np.int)
		rgb_float = (np.left_shift(rgb[:,0],16) + np.left_shift(rgb[:,1],8) + rgb[:,2]).astype(np.float)

		# Merge depth+color data
		assert pts.shape[0] == rgb_float.shape[0] and pts.ndim==2 \
			 and rgb_float.ndim==1, "Wrong number of data dimensions"
		pts_rgbd = np.hstack([pts, rgb_float[:,None]])

		# Filename
		depth_name = cam.depthFile.split("_")
		filename = "pcl_" + "_".join(depth_name[1:-1]) + "_" + depth_name[-1].split(".")[0] + ".pcd"

		# Save to file
		with open(save_dir+"/"+filename, 'w') as f:
			f.write(header)
			for p in zip(pts_rgbd):
				f.write("{} {} {} {}\n".format(*p[0]))
			print "Saved frame: {}/{}".format(save_dir, filename)

		if visualize:
			cam.visualize(color=True, depth=True, text=True, colorize=False, depth_bounds=[500,3500])

	print 'Done'

if __name__=="__main__":

	parser = optparse.OptionParser()
	parser.add_option('-v', '--visualize', dest='viz', action="store_true", default=False, help='Enable visualization')
	parser.add_option('-i', '--dir', dest='dir', default=DIR, help='Save directory')

	(opt, args) = parser.parse_args()

	main(visualize=opt.viz, save_dir=opt.dir)
