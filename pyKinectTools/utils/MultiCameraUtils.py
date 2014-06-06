'''
Used to stream multiple asynchonrous cameras simultaneously
'''

import numpy as np

class multiCameraTimeline:

	def __init__(self, files, temporalCameraOffsets=[0, 18, 13]):
		self.files = files
		self.stamp = np.zeros(len(files), dtype=int)
		self.fileLengths = [len(x) for x in files]
		self.devCount = len(files)
		self.temporalCameraOffsets = temporalCameraOffsets

	def __iter__(self):

		while 1:
			# Get latest file names/times
			tmpFiles = [self.files[i][self.stamp[i]] if self.stamp[i]<self.fileLengths[i] else np.inf for i in xrange(self.devCount)]
			sec = [int(x.split('_')[-4])*100 if type(x)==str else np.inf for x in tmpFiles]
			ms = [int(x.split('_')[-3]) if type(x)==str else np.inf for x in tmpFiles]			
			times = [s*100 + m for s,m in zip(sec, ms)]

			# Account for time difference between the cameras
			for i in xrange(self.devCount):
				times[i] += self.temporalCameraOffsets[i]*100 

			#Find the earliest frame
			dev = np.argsort(times)[0]
			#If there are no more frames 
			if self.stamp[dev] >= self.fileLengths[dev]:
				dev = None
				for d, i in zip(np.argsort(times), range(len(times))):
					if d < self.fileLengths[i]:
						dev = d
						break
				if dev == None:
					raise StopIteration()

			self.stamp[dev] += 1
			# If at the end of the roll
			if tmpFiles[dev] == np.inf:
				raise StopIteration()

			yield dev, tmpFiles[dev]


def formatFileString(x):
	if len(x) == 1:
		return x+'0'
	else:
		return x