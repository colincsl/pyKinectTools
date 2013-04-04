class BasePlayer:

	depthIm = None
	colorIm = None
	users = None
	backgroundModel = None
	foregroundMask = None
	prevcolorIm = None

	def update_background(self):
		pass

	def next(self, frames=1):
		pass

	def get_person(self, edge_thresh=200):
		pass

	def get_n_skeletons(self, n):
		pass

	def visualize(self, show_skel=False):
		pass

	def run(self):
		pass