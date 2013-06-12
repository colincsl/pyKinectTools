"""
Convert jpg+png image/depth files to .pcd files to work with PCL
"""

import os
import optparse
import numpy as np
import pcl
from pyKinectTools.dataset_readers.KinectPlayer import KinectPlayer, display_help
from IPython import embed


DIR = os.path.expanduser('~')+'/Data/PCL_Data/'


# -------------------------MAIN------------------------------------------

def main(visualize=False, save_dir='~/', sparse_pointcloud=False):

	# Create save directory if it doesn't exist
	if not os.path.isdir(save_dir):
		os.mkdir(save_dir)

	cam = KinectPlayer(base_dir='./', device=1, bg_subtraction=True, get_depth=True,
						get_color=True, get_skeleton=False, background_model='box', background_param=3200)
	
	cloud = pcl.PointCloud()

	framerate = 1
	while cam.next(framerate):

		pts = cam.camera_model.im2PosIm(cam.depthIm).reshape([-1,3])
		cloud.from_array(pts.astype(np.float32))
		depth_name = cam.depthFile.split("_")
		filename = "pcl_" + "_".join(depth_name[1:-1]) + "_" + depth_name[-1].split(".")[0] + ".pcd"

		if sparse_pointcloud:
			nonzero_idx = np.nonzero(pts[:,2]>0)
			cloud.from_array(pts[nonzero_idx].astype(np.float32))

		cloud.to_file(save_dir+filename, ascii=True)

		if visualize:
			cam.visualize(color=True, depth=True, text=True, colorize=False, depth_bounds=[500,3500])

	embed()

	print 'Done'

if __name__=="__main__":

	parser = optparse.OptionParser()
	parser.add_option('-v', '--visualize', dest='viz', action="store_true", default=False, help='Enable visualization')
	parser.add_option('-s', '--sparse', dest='sparse', action="store_true", default=False, help='Save sparse pointcloud')
	parser.add_option('-i', '--dir', dest='dir', default=DIR, help='Save directory')

	(opt, args) = parser.parse_args()

	main(visualize=opt.viz, save_dir=opt.dir, sparse_pointcloud=opt.sparse)