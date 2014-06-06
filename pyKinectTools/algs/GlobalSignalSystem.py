import cv2
import numpy as np
import time

class GlobalSignalSystem:
	markers = {}
	signals = {}
	times = []
	signalCount = 0
	markerCount = 0
	radius = 0
	chart = []

	def __init__(self, markers=[], radius=30):

		if markers != []:
			self.addMarkers(markers)

		self.radius = radius


	def addMarkers(self, markers):
		# Markers should be a dictionary with name:[2xfloat position]
		for m in markers.keys():
			self.markers[m] = markers[m]
			self.signals[m] = []
			self.markerCount += 1

	def getMarkerPos(self):
		markers = []
		for m in self.markers.keys():
			markers.append(self.markers[m])

		return markers		

	def update(self, coms=[], curTime=[]):
		if curTime == []:
			curTime = time.time()
		self.times.append(curTime)

		# If no people, add empty signal for all markers
		if coms == []:
			for k in self.signals.keys():
				self.signals[k].append(0)
		else:
			# If people, check if they're within bounds of all markers and add a signal
			for m in self.markers.keys():
				# Default to 0 signal
				self.signals[m].append(0)

				for c in coms:
					if np.sqrt(np.sum((self.markers[m][0:2]-c[0:2])**2)) < self.radius:
						self.signals[m][-1] += 1
		self.signalCount += 1

	def getChart(self):
		# TODO: base this on time, not frames
		# TODO: design chart (inc. resolution, add titles, ...)

		rez = [200,600]
		signalRez = [(rez[0]-50)/self.markerCount-5, rez[1]-40]
		chartIm = np.zeros(rez)

		self.chart = np.empty([self.markerCount, self.signalCount])

		for m,i in zip(self.markers.keys(), range(len(self.markers.keys()))):
			self.chart[i,:] = self.signals[m]

			#								0->2
			signalIm = np.interp(np.arange(0, self.chart.shape[1], self.chart.shape[1]/float(signalRez[1])), range(self.chart.shape[1]), self.chart[i]) >= 0.5
			signalIm = np.vstack(signalIm).repeat(signalRez[0]-5, 1)
			# pdb.set_trace()
			chartIm[i*signalRez[0]+50+5 : (i+1)*signalRez[0]+50, 20:20+signalRez[1]] = signalIm.T*200 + 20

		cv2.putText(chartIm, "Signals:", org=(rez[1]/2-50, 25), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=[100,100,100])

		# cv2.line(charIm, )

		# return self.chart
		return chartIm








