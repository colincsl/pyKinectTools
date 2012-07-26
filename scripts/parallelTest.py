
from multiprocessing import Pool
import random
import numpy as np
import time


def fcn(y): 
	x = random.randint(2,6)**100000
	return x


for i in range(1, 16):

	tStart = time.time()	
	pool = Pool(processes = i)
	result = pool.map(fcn, range(1000))
	tEnd = time.time()
	print "Time ", i, ":", tEnd - tStart

#---------------------------
import numpy as np
import time
from ctypes import *
depthFilename = '/Users/colin/data/ICU_7March2012_Head/10-42-46/10-42-46_870461383000000.depth'

tStart = time.time()
# depthData = np.fromfile(depthFilename, dtype=np.uint16, sep=" ").reshape([480, 640])[:,::-1]	
# depthData = np.fromfile(depthFilename)
x = StringIO.StringIO(open(depthFilename).read())
data = np.genfromtxt(x, dtype=np.int16)
tEnd = time.time()
print "Time 1: ", tEnd - tStart

tStart = time.time()
depthData = np.fromfile(depthFilename, dtype=np.uint16, count=640*480, sep=" ").reshape([480, 640])[:,::-1]	
tEnd = time.time()
print "Time 1: ", tEnd - tStart


depthRaw = open(depthFilename, 'rb').read().split()
newfp = np.memmap(depthFilename, dtype='int16', mode='r', shape=(480,640))

tStart = time.time()
#mode 'c' means mem is writable but not saved to disk
fp = np.memmap(depthFilename, dtype=np.dtype([('a', np.uint16), ('b', np.str, 1)]), mode='r', shape=480*640, offset=3)
fp = np.memmap(depthFilename, dtype=np.dtype([('a', 'u2'), ('b', 'V2')]), mode='c')
fp = np.memmap(depthFilename, dtype=np.dtype([('a', np.uint16), ("b", void, 1)]), mode='c', offset=0)
fp = np.memmap(depthFilename, dtype=np.dtype([('a', c_ushort), ("b", void, 1)]), mode='c', offset=0)
fp = np.memmap(depthFilename, dtype=np.dtype([('a', '>u2'), ("b", void, 1)]), mode='c', offset=0)
im = fp['a'].reshape([480, -1])
imshow(im)
tEnd = time.time()
print "Time 1: ", tEnd - tStart
