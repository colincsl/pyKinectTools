


import numpy as np
from pylab import *


class KalmanFilter:
	''' A position-only kalman filter '''

	def __init__(self, data=None, ProcessVariance=.001, MeasurementVariance=.00001):


		dims = data.shape[0]

		self.state = data							# State estimate
		self.covar = np.eye(dims)*ProcessVariance	# Covariance estimate
		self.StateTransition = np.eye(dims)
		self.ObsTransition = np.eye(dims)
		
		self.ProcessVar = np.eye(dims)*ProcessVariance
		self.MeasurementVar = np.eye(dims)*MeasurementVariance

	def update(self, data):
		''' Predict and update '''

		if self.state is None:
			self.state = data
			return

		state_prev = self.state
		cov_prev = self.covar

		''' Prediction '''
		state_est = np.dot(self.StateTransition, state_prev)
		covar_est = np.dot(self.StateTransition, np.dot(self.covar, self.StateTransition.T)) + self.MeasurementVar

		''' Update '''		
		stateError = data - np.dot(self.ObsTransition, state_est)
		covarError = np.dot(self.ObsTransition, np.dot(self.covar, self.ObsTransition.T)) + self.ProcessVar
		weight = np.dot(np.dot(covar_est, self.ObsTransition.T), np.linalg.inv(covarError))
		print weight
		self.state = state_est + np.dot(weight, stateError)

		self.covar = np.dot(np.eye(stateError.shape[0]) - np.dot(self.ObsTransition, self.StateTransition), covar_est)

		return self.state


# com = np.load('../../ICU_Dec2012_r40_c1_')

kalman = KalmanFilter(coms[5,[0,2]], ProcessVariance=50, MeasurementVariance=.000001)
x = []
xn = []
for i in range(6, 50):
    x.append(coms[i, [0,2]])
    xn.append(kalman.update(coms[i, [0,2]]))
    # print kalman.covar

x = np.array(x)
xn = np.array(xn)
plot(x[:,0], x[:,1], 'g', linewidth=1)
plot(xn[:,0], xn[:,1], 'r', linewidth=1)
show()


'''
		# # from IPython import embed
		# # embed()
		# self.weight = cov_prev / (cov_prev + self.MeasurementVar) # K matrix
		# print 'w', self.weight
		# print 'diff', data-state_prev, self.weight, (data - state_prev)
		# print 'data', data
		# print 'cov', cov_prev
		# self.state = state_prev + self.weight * (data - state_prev)
		# self.covar = (1-self.weight)*cov_prev
'''



'''
Based on kalman filter at http://www.scipy.org/Cookbook/KalmanFiltering by Andrew D. Straw

# intial parameters
n_iter = 50
sz = (n_iter,2) # size of array
x = [-0.37727, -.37727] # truth value (typo in example at top of p. 13 calls this z)
z = numpy.random.normal(x,0.1,size=sz) # observations (normal about x, sigma=0.1)

Q = np.ones([1,2])*1e-5 # process variance

# allocate space for arrays
state=numpy.zeros(sz)      # a posteri estimate of x
P=numpy.zeros(sz)         # a posteri error estimate
state_prev=numpy.zeros(sz) # a priori estimate of x
P_prev=numpy.zeros(sz)    # a priori error estimate
K=numpy.zeros(sz)         # gain or blending factor

R = np.ones([1,2])*0.01**2 # estimate of measurement variance, change to see effect

# intial guesses
state[0,:] = 0.0
P[0,:] = 1.0

for k in range(1,n_iter):
    # time update
    state_prev[k] = state[k-1]
    P_prev[k] = P[k-1]+Q

    # measurement update
    K[k] = P_prev[k]/( P_prev[k]+R )
    state[k] = state_prev[k]+K[k]*(z[k]-state_prev[k])
    P[k] = (1-K[k])*P_prev[k]

figure()
# plot(z,'k+',label='noisy measurements')
# plot(state,'b-',label='a posteri estimate')

plot(z[:,0], z[:,1], '-b', linewidth=1, label='noisy measurements')
plot(state[:,0], state[:,1], '-g', linewidth=1, label='a posteri estimate')
plot(x[0], x[1], '+r')
# axhline(x,color='g',label='truth value')
# legend()
# xlabel('Iteration')
# ylabel('Voltage')

# figure()
# valid_iter = range(1,n_iter) # Pminus not valid at step 0
# plot(valid_iter,P_prev[valid_iter],label='a priori error estimate')
# xlabel('Iteration')
# ylabel('$(Voltage)^2$')
# setp( gca(),'ylim',[0,.01])
# show()

'''