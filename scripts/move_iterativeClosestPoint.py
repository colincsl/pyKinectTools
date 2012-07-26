
import numpy as np

# Iterative closest point

template = np.asmatrix(posMat[np.nonzero((regions <= 14) * (regions > 0))])
template -= template.mean(0)
# fakeRot = np.matrix([[0,1,0], [1,0,0],[0,0,1]])
ang = 5 * np.pi/180
# fakeRot = np.matrix([[1,0,0], [0,1,0],[0,0,1]])
fakeRot = np.matrix([[cos(ang),-sin(ang),0], [sin(ang),cos(ang),0],[0,0,1]])
fakeTrans = np.matrix([0,100,0]).T
headStart = np.asmatrix(fakeRot*template.T + fakeTrans)
# headStart = template.T
headStart += np.random.rand(3, headStart.shape[1])
head = np.copy(headStart)

# ICP
# T = np.matrix([.0,.0,.0])
R = np.eye(3)

T = head.mean(1)
head = head.T - T
minDists = np.empty(head.shape[0])
iters = range(100)
residual = np.inf
while iters.pop():
# if 1:
	H = np.asmatrix(np.zeros([3,3]))
	TNew = head.mean(0)# - template.mean(0)
	T = T + TNew

	head -= TNew

	for i, val in zip(range(head.shape[0]),head):
		dists = np.sum(np.asarray(template - val)**2, 1)
		argMin_ = np.argmin(dists)
		minDists[i] = np.asarray(dists[argMin_])
		H += np.asmatrix(head[i,:]).T * template[argMin_,:]
	# print H

	# print "--"
	residualNew = np.abs(minDists.sum() / head.shape[0])
	print "Error: ", residualNew
	if residual - residualNew < .0001:
		break
	residual = residualNew

	U,_,VT = np.linalg.svd(H, full_matrices=0)
	RotNew = (VT.T*U.T)
	R = R*RotNew

	if np.abs(np.linalg.det(RotNew) - 1) > .1:
		print "Error", np.linalg.det(RotNew) - 1


	head = (R*headStart).T - T

	print "Rotated: ", np.arcsin(R[0,1])*180/np.pi
