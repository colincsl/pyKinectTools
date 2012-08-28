import numpy as np

''' Belief Propagation on tree'''

class Node:
	children = []
	parent = -1
	pos = -1
	index = -1
	depth = 0

	msgsDown = {}
	msgUp = -1
	psi = -1
	belief = -1	

	def __init__(self, parent_=-1, index_=-1, children_=[], pos_=-1, depth_=0):
		self.index = index_
		self.pos = pos_
		self.parent = parent_
		self.children = []
		self.depth = depth_

		if len(children_) > 1:
			for i in children_:
				if self.parent == -1 or i != self.parent.index:
					# print self.index, i
					self.children.append(Node(index_=i, parent_=self, children_=edgeDict[i], pos_=regionLabels[i][1], depth_=self.depth+1))
					self.msgsDown[i] = -1
		self.msgUp = -1; self.psi = -1; self.belief = -1


	def getLeaves(self, leaves=set()):
		if self.children == []:
			leaves.add(self)
		else:
			for c in self.children:
				l = c.getLeaves(leaves)

		if self.parent == -1:
			return list(leaves)
		else:
			return leaves

	def calcPsi(self, hypothesis):
		#psi is the unary potentials/messages
		# hypothesis = guessPose
		self.psi = np.empty([np.shape(hypothesis)[0]])
		for i in xrange(np.shape(hypothesis)[0]):
			self.psi[i] = np.sqrt(np.sum((hypothesis[i] - self.pos)**2))
		# print '1', self.psi
		self.psi = (np.sum(self.psi) - self.psi)/np.sum(self.psi)
		self.psi /= np.sum(self.psi)
		# print '4', self.psi

	def setPsi(self, psi):
		pass

	def calcMessageUp(self):
		for kid in self.children:
			if np.all(kid.msgUp == -1):
				return
		tmpMsg = 1
		# print '-'
		tmpMsg *= self.psi
		# print self.psi
		if self.children != []:
			for kid in self.children:
				# print np.asarray(kid.msgUp).T[0], tmpMsg
				# tmpMsg *= np.asarray(kid.msgUp).T[0]
				# tmpMsg = np.sum(np.asmatrix(tmpMsg).T*kid.msgUp.T, 0)
				tmpMsg = tmpMsg*kid.msgUp
				tmpMsg /= np.sum(tmpMsg)
				# print tmpMsg, np.asarray(kid.msgUp).T[0]
				# pdb.set_trace()

		# pdb.set_trace()
		self.msgUp = np.array(transitionMatrix*np.asmatrix(tmpMsg).T).T
		self.msgUp /= np.sum(self.msgUp)
		
		# print self.msgUp, np.asmatrix(tmpMsg).T

	def calcMessagesDown(self):
		for c, kid in zip(range(len(self.children)), self.children):
			tmpMsg = 1
			tmpMsg *= self.psi
			if self.parent != -1:
				for m in self.parent.msgsDown.keys():
					if m == self.index:
						# tmpMsg = np.sum(np.asmatrix(tmpMsg).T*self.parent.msgsDown[m].T, 0)
						tmpMsg = tmpMsg*self.parent.msgsDown[m]
						tmpMsg /= np.sum(tmpMsg)

						break
		
			for c2_i, kid2 in zip(range(len(self.children)), self.children):
				if kid != kid2:
					# pdb.set_trace()
					# tmpMsg *= np.array(self.children[c2_i].msgUp).T[0]

					# tmpMsg = np.sum(np.asmatrix(tmpMsg).T*kid.msgUp.T, 0)
					tmpMsg = tmpMsg*kid.msgUp
			tmpMsg /= np.sum(tmpMsg)
			
			self.msgsDown[kid.index] = np.array(transitionMatrix*np.asmatrix(tmpMsg).T).T
			self.msgsDown[kid.index] /= np.sum(self.msgsDown[kid.index])
			# pdb.set_trace()
			# print np.array(transitionMatrix*np.asmatrix(tmpMsg).T).T


	def calcBelief(self):
		self.belief = 1
		self.belief *= self.psi

		if self.parent != -1:
			for c in self.parent.msgsDown.keys():
				if c == self.index:
					# self.belief *= np.array(self.parent.msgsDown[c]).T[0]
					self.belief *= self.parent.msgsDown[c][0]
					break

		if self.children != []:
			for kid in self.children:
				if np.any(kid.msgUp == -1):
					self.belief = -1
					break
				else:
					# self.belief *= np.asarray(kid.msgUp).T[0]
					self.belief *= kid.msgUp[0]

		if np.all(self.belief >= 0):
			self.belief /= np.sum(self.belief)


	def updateLeaves(self):
		self.calcMessagesDown()
		self.calcBelief()

		if self.children != []:
			for c in self.children:
				c.updateLeaves()

	def reset(self):
		self.msgUp = -1
		self.psi = -1
		self.belief = -1
		self.msgsDown = {}

		if self.children != []:
			for c in self.children:
				c.reset()

	def calcTotalBelief(self):

		if self.children == []:
			return np.max(self.belief)
		else:
			belief = 1
			for kids in self.children:
				belief *= kids.calcTotalBelief()

		return belief



	def calcAll(self, hypothesis):

		if np.any(self.belief >= 0):
			self.reset()

		leaves = self.getLeaves()
		oldLeaves = set()
		while np.any(self.belief < 0):
			newLeaves = set()
			for l in leaves:
				l.calcPsi(hypothesis)
				l.calcMessageUp()

				if np.any(l.msgUp != -1):
					oldLeaves.add(l)
				if np.any(l.parent != -1) and l.parent not in oldLeaves:
					newLeaves.add(l.parent)
			leaves = newLeaves
			self.calcBelief()

		self.calcMessagesDown()
		# self.calcBelief()

		for c in self.children:
			c.updateLeaves()

	def drawClass(self):
		pt1 = (regionLabels[self.index][2][1],regionLabels[self.index][2][0])
		color_ = int(np.argmax(self.belief)*30)
		cv2.circle(imLines, pt1, radius=10, color=color_, thickness=1)

	def drawAll(self):
		self.drawClass()

		for kid in self.children:
			kid.drawAll()



