from pylab import *
import numpy as np
import time, sys

# S_HEAD 		=1
# S_NECK		=2
# S_TORSO		=3
# S_L_SHOULDER	=6
# S_L_ELBOW	=7
# S_L_HAND		=9
# S_R_SHOULDER	=12
# S_R_ELBOW	=13
# S_R_HAND		=15
# S_L_HIP		=17
# S_L_KNEE		=18
# S_L_FOOT		=20
# S_R_HIP		=21
# S_R_KNEE		=22
# S_R_FOOT		=24
# #SKELETON = [S_HEAD, S_NECK, S_TORSO, S_L_SHOULDER, S_L_ELBOW, S_L_HAND, S_R_SHOULDER, S_R_ELBOW, S_R_HAND,  S_L_HIP, S_L_KNEE, S_L_FOOT, S_R_HIP, S_R_KNEE, S_R_FOOT]
# SKELETON = [S_HEAD, S_L_SHOULDER, S_L_ELBOW, S_L_HAND, S_R_SHOULDER, S_R_ELBOW, S_R_HAND, S_L_KNEE, S_L_FOOT, S_R_KNEE, S_R_FOOT] # removed neck, torso, hip

# wave = range(0,5)
# circle_counter = range(5,10)
# circle_clock = range(10,15)
# push_forward = range(15,20)
# push_left = range(20,25)
# push_right = range(25,30)
# swoosh_right = range(30,35)
# reach_up = range(35,40)
# duck = range(40,45)
# kick = range(45,50)
# bow = range(50,55)

# GESTURES = [wave,
#             circle_counter,
#             circle_clock,
#             push_forward,
#             push_left,
#             push_right,
#             swoosh_right,
#             reach_up,
#             duck,
#             kick,
#             bow]

class pcaGestureRecognizer:
    # def __init__(self, skeletonSet=SKELETON, gestureSet=GESTURES):
    def __init__(self, skeletonSet, gestureSet):
        self.skeletonSet = skeletonSet
        self.trainingSet = gestureSet
        
        self.label = []#np.zeros([len(skeletonSet), 1])
        self.perClassLabel = []#np.zeros([1, len(gestureSet)])
        self.discardPercent = 0.0
    
    def train(self, skelSets, setIndicies, jointType='jointsRel'):
        # jointType can be 'jointsRel', 'jointsCentered', or 'jointsPos'
        # setInd is the set of indices that should be trained on (in form [[g1],[g2]...]
        self.jointType = jointType
        self.skelSets = skelSets
        
        sets = setIndicies
        
        # Train
        self.classCount = len(sets)
        self.sampleCount = len(setIndicies[0])
        self.bestVecs = {}
        self.prior = []
        self.pcaVecs = {}                    
        gestureSets = {}
        for setInd in range(self.classCount):
            for i in sets[setInd]:
                # Discard first %
                discard = int(len(self.skelSets[i][0][self.jointType]) * self.discardPercent) # Get middle points
                # discard = 0
                if gestureSets.has_key(setInd) == 0:
                    gestureSets[setInd] = self.skelSets[i][0][self.jointType][discard:(-1-discard)]
                    # gestureSets[setInd] = self.skelSets[i][0][self.jointType][discard::]
                else:
                    gestureSets[setInd]  = np.concatenate((gestureSets[setInd] , self.skelSets[i][0][self.jointType][discard:(-1-discard)]), axis=0)
                    # gestureSets[setInd]  = np.concatenate((gestureSets[setInd] , self.skelSets[i][0][self.jointType][discard::]), axis=0)
            self.pcaVecs[setInd] = {}
            projError = []
            for joint in self.skeletonSet:
                tmp = gestureSets[setInd][:,joint,:]
                #tmp -= np.mean(tmp, axis=0) #center data
                u,_,v = svd(tmp.T)
                self.pcaVecs[setInd][joint] = u[:,0]
                projError.append(np.sqrt(np.sum((tmp - u[:,0]*tmp)**2)))
            #Keep vecs with greatest variance
            #    self.bestVecs[setInd] = np.argsort(projError)
            self.bestVecs[setInd] = np.argsort(projError)[-3:]
            self.prior.append(np.exp(-1*(projError / np.max(projError))))
            self.prior[-1] /= np.sum(self.prior[-1])    

        self.compareBasis(show=0)
    
    def compareBasis(self, show=0):
        skelCount = len(self.skeletonSet)
        ## Look at similarities between each basis
        self.basisCompare = np.zeros([skelCount, self.classCount,self.classCount])
        for g in range(skelCount):
            for i in range(self.classCount):
                for j in range(self.classCount):
                    self.basisCompare[g,i,j] = np.abs(np.dot(self.pcaVecs[i][self.skeletonSet[g]],(self.pcaVecs[j][self.skeletonSet[g]])))
                    #if i==j:
                    #    self.basisCompare[g,i,j] = 1
                        
        # Create per class prior based on similarity matrix
        self.basisPrior = []
        for i in range(self.classCount):
            p = np.sum(self.basisCompare[:,:,i], axis=1)
            p -= np.mean(p)
            self.basisPrior.append(exp(-p/np.max(p))+.25) #.25 is 'regularizer'    
                        
        self.basisPrior = np.array(self.basisPrior)
        self.basisPrior /= np.max(self.basisPrior, axis=0)
        # Create per joint prior
        self.jointBasisPrior = []
        for i in range(skelCount):
            p = np.sum(self.basisCompare[i,:,:], axis=1)
            p -= np.mean(p)
            self.jointBasisPrior.append(exp(-p/np.max(p))+.25)  #.25 is 'regularizer'    
    
        self.jointBasisPrior = np.array(self.jointBasisPrior).T
        self.jointBasisPrior /= np.max(self.jointBasisPrior, axis=0)

        if show:
            # Display basis comparison
            figure(4)
            imshow(self.basisCompare[0], interpolation="nearest")
            title('Basis comparison')
            basisCompareMag = np.sum(np.sum(self.basisCompare, axis=2), axis=1)
    
    def test(self, testSet = [], d=0):
        if len(testSet) == 0:
            testSet = self.skelSets[0]
            print "No test set defined. Will use default"

#        while self.label.shape[1] < (d+1):
        if self.label == []:
            self.label = np.zeros([len(self.skeletonSet),1], dtype=int)
        else:
            self.label = np.hstack([self.label, np.zeros([len(self.skeletonSet),1], dtype=int)])
#        while self.perClassLabel.shape[0] < (d+1):
        if self.perClassLabel == []:
            # self.perClassLabel = np.zeros([1,len(self.skeletonSet)])
            self.perClassLabel = np.zeros([1,len(self.pcaVecs)]) #(1 x #classes)
        else:
            # self.perClassLabel = np.vstack([self.perClassLabel, np.zeros([1,len(self.skeletonSet)])])
            self.perClassLabel = np.vstack([self.perClassLabel, np.zeros([1,len(self.pcaVecs)])])
        
#        for d in range(self.classCount*5):
#        for d in range(len(testSet)):
#            tmpSet = skelSets[d][0][self.jointType]    
        discard = int(len(testSet[0][self.jointType]) * self.discardPercent) # Discard start/finish
        # discard = 0
        tmpSet = testSet[0][self.jointType][discard:(-1-discard)]
        # tmpSet = testSet[0][self.jointType][discard::]
        
        for i in range(len(self.skeletonSet)): # Compare with each gesture            
            jointT = tmpSet[:,self.skeletonSet[i],:]
            # jointT -= np.mean(jointT, axis=0)  #center data
            u,_,vt = svd(jointT.T)
            vec = u[:,0]
            
            projections = []
            if 1: # normal
                for c in range(self.classCount):
                    projections.append(vec.dot(self.pcaVecs[c][self.skeletonSet[i]]))                
            if 0: # top vectors
                for c in range(self.classCount):
                    if i in self.bestVecs[c]:
                        projections.append(vec.dot(self.pcaVecs[c][self.skeletonSet[i]]))
                    else:
                        projections.append(0.)
            if 0: # prior
                for c in range(self.classCount):
#                    projections.append(self.prior[c][i]*vec.dot(self.pcaVecs[c][self.skeletonSet[i]]))
                    projections.append(self.basisPrior[c][i]*vec.dot(self.pcaVecs[c][self.skeletonSet[i]]))
            if 0: # joint basis prior
                for c in range(self.classCount):
                    projections.append(self.jointBasisPrior[c][i]*vec.dot(self.pcaVecs[c][self.skeletonSet[i]]))                        
            if 0: # projection error
                for c in range(self.classCount):
                    t = jointT - (self.pcaVecs[c][self.skeletonSet[i]]*jointT)
                    err = np.mean(np.sum(t**2, axis=1))
                    projections.append(err) 
            # Get per gesture label
            self.label[i, -1] = np.argmax(projections)
            # self.label[i, -1] = np.argmax(np.abs(projections))    

            # pdb.set_trace()
            # Get per class likelihood
            if 1:
                for c in range(self.classCount):
                    #                t = jointT - (self.pcaVecs[c][self.skeletonSet[i]]*jointT)
                    #                err = np.mean(np.sum(t**2, axis=1))
                    #                projections.append(err)
                    self.perClassLabel[-1,c] += vec.dot(self.pcaVecs[c][self.skeletonSet[i]])*self.basisPrior[c][i]

        
        return np.bincount(self.label[:,-1]).argmax()
    
    def calcAccuracy(self, truth_perSet, testClassCount=4.0, show=1):
        self.actual_perSet = truth_perSet
        self.actual = np.repeat([self.actual_perSet], self.label.shape[0], axis=0)
        self.actual = np.array(self.actual, dtype=int)
        self.label = np.array(self.label, dtype=int)

        # Avg joint accuracy
        self.accuracyImg = np.equal(self.label, self.actual)
        accuracy = float(np.sum(self.accuracyImg)) / float(self.accuracyImg.size)
        
        # Avg max likelihood accuracy
        self.avgLabel = []
        for i in range(self.label.shape[1]):
            self.avgLabel.append(np.bincount(self.label[:,i]).argmax())
        self.avgLabel = np.array(self.avgLabel)
        self.avgAccuracyImg = np.equal(self.avgLabel, self.actual_perSet)
        self.avgAccuracy = float(np.sum(self.avgAccuracyImg)) / float(self.avgAccuracyImg.size)
        
        # Per class, class dist.
        perClassAvgAcc = np.sum(self.actual_perSet==self.perClassLabel.argmax(axis=1)) / (self.classCount*testClassCount)
        
        if show:
            figure(1)
            subplot(2,1,1)
            imshow(self.label, interpolation="nearest")
            title('Gesture Labels')
            print "Avg joint accuracy: " + str(accuracy)
            print "Avg max likelihood accuracy: " + str(self.avgAccuracy)
            print "Avg per class accuracy: " + str(perClassAvgAcc)

    #imshow(self.accuracyImg)
    
    def calcClassAccuracy(self, testCases = 4, show=1):
        ## Per class distribution
        ##PROBLEM: '4' in the next line shouldn't be there. parameterize!
        self.classDist = self.avgLabel.reshape(-1,testCases).T # cols are samples, rows are classes
        ## Per class accuracy
        self.classAccuracy = np.empty([self.classCount, 1])
        for i in range(self.classCount):
            self.classAccuracy[i] = np.sum(self.avgAccuracyImg[i*testCases:(i+1)*testCases])/testCases*1.0
        
        if show:
            figure(2)
            subplot(2,2,2)
            bar(range(self.classCount), self.classAccuracy)
            axis([0,self.classCount-2, 0,1])
            title('Per Gesture Accuracy')
    
    def calcClassConfusion(self, show=1):
        ## Per class confusion matrix
        classConfusion = np.zeros([self.classCount, self.classCount])
        for i in range(self.classCount):
            for j in range(self.classCount):
                classConfusion[i,j] = np.sum(self.classDist[:,j] == i)
        
        if show:
            subplot(2,2,1)
            imshow(classConfusion, interpolation="nearest")
            title('Gesture Confusion')
    
    def calcJointAccuracy(self, show=1):
        ## Per joint distribution
        #find the chance that the joint's label is correct
        jointAccuracy = np.empty([len(self.skeletonSet), 1])
        for i in range(len(self.skeletonSet)):
            jointAccuracy[i] = np.sum(self.accuracyImg[i])/(self.classCount*5.)
        
        if show:
            subplot(2,2,4)
            bar(range(len(self.skeletonSet)), jointAccuracy)
            axis([0,len(self.skeletonSet), 0,1])
            title('Per Joint Accuracy')
    
    def calcJointConfusion(self, show=1):
        jointConfusion = np.zeros([len(self.skeletonSet), len(self.skeletonSet)])
        for i in range(len(self.skeletonSet)):
            for j in range(len(self.skeletonSet)):
                jointConfusion[i,j] = np.sum((self.actual_perSet == i) * (self.label[i] == j))
        
        if show:
            subplot(2,2,3)
            imshow(jointConfusion, interpolation="nearest")
            title('Joint Confusion')



