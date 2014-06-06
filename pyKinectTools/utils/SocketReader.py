

from pyKinectTools.algs.BackgroundSubtraction import *
from pyKinectTools.utils.DepthUtils import *
import numpy as np
from socket import *
import time

localIP = [ip for ip in gethostbyname_ex(gethostname())[2] if not ip.startswith("127.")][:1]


class DeviceClient:
	''' Store info about each client '''
	client = -1
	address = -1

	colorIm = []
	depthIm = []
	depthIm8 = []
	constrain = []

	def __init__(self, client, address):
		self.client = client
		self.address = address


class SocketDeviceHost:

	server = []
	clients = []

	host = -1
	port = -1
	address = -1
	bufferSize = -1

	currentCam = 0
	depthIm = []
	depthIm8 = []
	colorIm = []

	constrain = []
	maxDists = []
	timeout=.03

	bgModel = []
	bgModel8 = []


	def __init__(self, port=29770, host='', bufferSize=4096, maxDists=[4000]):
		self.port = self.port
		self.host = self.host
		self.address = (host, port)
		self.bufferSize = bufferSize
		self.maxDists = maxDists

		# Start server
		self.server = socket(AF_INET, SOCK_STREAM)
		self.server.bind((self.address))
		# Set timeout to .1 second
		self.timeout = .01
		self.server.settimeout(self.timeout)

		self.connectDevices()


	def connectDevices(self):
		self.server.listen(10)
		t1 = time.time()
		while time.time() - t1 < self.timeout*3:
			try:
				conn, connAddr = self.server.accept()

				dev = DeviceClient(conn, connAddr)
				dev.client.settimeout(self.timeout)
				self.clients.append(dev)
			except:
				pass


	def setMaxDist(self, dists):
		self.maxDists = dists

	def stop(self):
		for c in xrange(len(self.clients)):
			self.clients[c].client.close()
		self.server.close()

	def close(self):
		self.stop()

	def flushBuffer(self):
		for i in xrange(len(self.clients)):
			try:
				data_str = self.clients[i].client.recv(self.bufferSize)
			except:
				pass

	def update(self):
		imTypes = {'color':921600, 'depth':307200}
		imSize = 99999
		imsRaw = []
		for i in xrange(len(self.clients)):
			data = ''
			time_ = time.time()
			imStarted = 0
			while len(data) < imSize and time.time()-time_ < self.timeout*3:
				data_str = ''
				try:
					data_str = self.clients[i].client.recv(self.bufferSize)
				except:
					# print "Buffer is empty"
					pass

				if len(data_str) > 0:
					# Check for start of image
					if imStarted == 0:
						for type_ in imTypes:
							if type_ in data_str:
								imType = type_
								imSize = imTypes[imType]								
								data = data_str[data_str.find(type_)+len(type_):]
								imStarted = 1
					# Add image packet
					elif imStarted == 1:
						# Make sure there isn't a problem with buffer
						# Restrict to only allow buffersize (unless near the end of the image)
						# if len(data_str) == self.bufferSize or len(data) > imSize-self.bufferSize:
						if 'end' not in data_str:
							data += data_str
						else:
							data += data_str[:data_str.find('end')]
							
							# If image is complete, transform from string->im
							if len(data) == imSize:
								if imType == 'depth':
									d = np.fromstring(str(data), dtype=np.uint8).reshape([480,640])
									self.clients[i].depthIm = d
									self.clients[i].depthIm8 = d
								else:
									d = np.fromstring(str(data), dtype=np.uint8).reshape([480,640, 3])
									self.clients[i].colorIm = d
								# data = ''
								imStarted = 0
								print 'New image:', imType
							elif len(data) > imSize:
								print "Data longer than image size"

							if imStarted == 0:
								for type_ in imTypes:
									if type_ in data_str:
										imType = type_
										imSize = imTypes[imType]								
										data = data_str[data_str.find(type_)+len(type_):]
										imStarted = 1


					time_ = time.time()
			self.data = data


			# if imSize != 99999 and len(data) == imSize:
				# if imType == 'depth':
				# 	d = np.fromstring(str(data), dtype=np.uint8).reshape([480,640])
				# 	self.clients[i].depthIm = d
				# 	self.clients[i].depthIm8 = d
				# else:
				# 	d = np.fromstring(str(data), dtype=np.uint8).reshape([480,640, 3])
				# 	self.clients[i].colorIm = d
				# # imsRaw.append(d)
				# print 'New image'


		# Update im w/ 'current' camera
		self.depthIm = self.clients[self.currentCam].depthIm
		self.depthIm8 = self.clients[self.currentCam].depthIm8
		self.colorIm = self.clients[self.currentCam].colorIm



class SocketDeviceClient():

	client = -1
	server = -1

	host = -1
	port = -1
	address = -1
	bufferSize = -1

	depthIm = []
	depthIm8 = []
	colorIm = []	

	constrain = []
	maxDists = []

	# bgModel = []
	# bgModel8 = []

	def __init__(self, port=29770, host='localhost', bufferSize=2048):
		self.host = host
		self.port = port
		self.address = (host, port)
		self.bufferSize = bufferSize		

		self.client = socket(AF_INET, SOCK_STREAM)
		self.client.connect((self.address))
		self.client.settimeout(.1)

	def connect(self, port=-1, host=-1):
		if port > 0:
			self.port = port
		if host > 0:
			self.host = host
		self.address = (self.host, self.port)

		self.client.connect((self.address))

	def close(self):
		self.client.close()

	def getSample(self):
		import os
		import numpy as np
		import scipy.misc as sm
		import sys

		os.chdir('/Users/colin/Dropbox/Public')

		self.depthIm8 = sm.imread('depthImage.jpg')
		self.colorIm = np.dstack([self.depthIm8, self.depthIm8, self.depthIm8])

		self.client.settimeout(5)

	def setDepth(self, im):
		self.depthIm = im
		if constrain != [] and self.depthIm.max() > 256:
			self.depthIm8 = constrain(im, self.constrain[0], self.constrain[1])
		else:
			self.depthIm8 = self.depthIm

	def setColor(self, im):
		self.colorIm = im

	def sendIm(self, label="depth", im=[]):
		# Turn data into string to send over network
		im_str = label
		im_str += im.tostring()
		im_str += 'end'
		imSize = sys.getsizeof(im_str)

		# Send in bite-size chunks
		i = 0
		
		data = ''
		while i < imSize+self.bufferSize:
			c = self.client.send(im_str[i:i+self.bufferSize])
			# print c
			# if c == self.bufferSize or i >= imSize-self.bufferSize:
			data += im_str[i:i+c]
			i += self.bufferSize

	def sendAll(self):
		if self.depthIm8 != []:
			self.sendIm("depth", self.depthIm8)
		if self.colorIm != []:
			self.sendIm("color", self.colorIm)


