''' 
Iterative closest point example

Colin Lea 
pyKinectTools
2012
'''

import numpy as np
from pyKinectTools.algs.IterativeClosestPoint import IterativeClosestPoint


template = np.asmatrix(posMat[np.nonzero((regions <= 14) * (regions > 0))])
template -= template.mean(0)
# fakeRot = np.matrix([[0,1,0], [1,0,0],[0,0,1]])
ang = 5 * np.pi/180
# fakeRot = np.matrix([[1,0,0], [0,1,0],[0,0,1]])
fakeRot = np.matrix([[cos(ang),-sin(ang),0], [sin(ang),cos(ang),0],[0,0,1]])
fakeTrans = np.matrix([0,100,0]).T
pointcloudInit = np.asmatrix(fakeRot*template.T + fakeTrans)
# pointcloudInit = template.T
pointcloudInit += np.random.rand(3, pointcloudInit.shape[1])
pointcloud = np.copy(pointcloudInit)


R, T = IterativeClosestPoint(pointcloud, template, 100, .001)

