from pylab import *
import numpy as np
import pdb
import time, sys,os
sys.path.append(os.getcwd() + "/pcaGestures")
import SkelPlay as SkelModule
import gestureRecognizer as GestModule

# Set gestures
wave = range(0,5)
circle_counter = range(5,10)
circle_clock = range(10,15)
push_forward = range(15,20)
push_left = range(20,25)
push_right = range(25,30)
swoosh_right = range(30,35)
reach_up = range(35,40)
duck = range(40,45)
kick = range(45,50)
bow = range(50,55)

GESTURES = [wave,
            circle_counter,
            circle_clock,
            push_forward,
            push_left,
            push_right,
            swoosh_right,
            reach_up,
            duck,
            kick,
            bow]

#Jed Sequences
SEQUENCE_TRUTH = [
[3,6,5],
[0,1,2],
[3,4,5],
[6,7,8],
[9,0,3],
[1,4,6],
[2,5,7],
[8,6,2],
[9,3,0],
[4,8,1]]


if __name__ == "__main__":

    SKELETON = SkelModule.SKELETON    
    print "Start"   
    if len(sys.argv) > 1:
        skelFolder = sys.argv[1]
    else:
        skelFolder = ""

    skelFolder = "/Users/colin/data/Gestures/jed_11Gest"
    skelSetsTrain = SkelModule.getAllSkeletons(skelFolder, normalizeOrientation=0)
    # skelSetsTrain = SkelModule.getAllSkeletons("/Users/colin/data/Gestures/11Gestures_9Feb2012", normalizeOrientation=0)
    # skelSetsTest = SkelModule.getAllSkeletons(skelFolder, normalizeOrientation=0)
    skelSetsTest = SkelModule.getAllSkeletons("/Users/colin/data/Gestures/11Gestures_9Feb2012", normalizeOrientation=0)
    
    # Train on 1, test on 4
    if 0:
        ml_accuracy = []
        for s in range(5):
            trainingSetInds = []
            for i in range(10):
                trainingSetInds.append([GESTURES[i][s]])

            g = GestModule.pcaGestureRecognizer(SKELETON, GESTURES)
            g.train(skelSetsTrain, trainingSetInds, 'jointsCentered')
            tmp = []
            for d in trainingSetInds: tmp.append(d[0])
            testData = set(range(50)).difference(set(tmp))
            for i in testData:
                g.test(skelSetsTest[i], i)
            g.calcAccuracy(np.array(list(testData))/5)
            ml_accuracy.append(g.avgAccuracy)

        print "----------------"
        print "Overall accuracy", np.array(ml_accuracy).mean()
        print ml_accuracy

        g.calcClassAccuracy()
        g.calcClassConfusion()
        g.calcJointAccuracy()
        g.calcJointConfusion()
        figure(1)
        subplot(2,1,2)
        imshow(g.accuracyImg, interpolation="nearest")
        title("Gesture/Joint Accuracy")

        pdb.set_trace()
    


    # Train on 4, test on 1
    if 1:
        ml_accuracy = []
        for s in range(5):
            trainingSetInds = []
            for i in range(10):
                tmpA = []
                for j in range(5):
                    if j != s:
                        tmpA.append(GESTURES[i][j])
                trainingSetInds.append(tmpA)
            
            g = GestModule.pcaGestureRecognizer(SKELETON, GESTURES)
            g.train(skelSetsTrain, trainingSetInds, 'jointsCentered')
            tmp = []
            for d in trainingSetInds: tmp.append(d[0])
            testData = set(range(50)).difference(set(tmp))
            for i in testData:
                g.test(skelSetsTest[i], i)
            g.calcAccuracy(np.array(list(testData))/5)
            ml_accuracy.append(g.avgAccuracy)
        
        print "----------------"
        print "Overall accuracy", np.array(ml_accuracy).mean()
        print ml_accuracy
        g.calcClassAccuracy()
        g.calcClassConfusion()
        g.calcJointAccuracy()
        g.calcJointConfusion()
        pdb.set_trace()

    # Train on 4, test on 1. Vary time.
    # Accuracy appears to stay very close to 'whole' dataset
    if 0:
        import copy
        skelSetShort = copy.deepcopy(skelSetsTrain)
        for i in range(10):
            half_len = int(skelSetsTrain[i][0]['jointsRel'].shape[0]/10)
            skelSetShort[i][0]['jointsRel'] = copy.deepcopy(skelSetsTrain[i][0]['jointsRel'][20:half_len+20])
#            skelSetShort[i][0]['jointsRel'] = copy.deepcopy(skelSetsTrain[i][0]['jointsRel'][0:half_len])        
        ml_accuracy = []
        for s in range(5):
            trainingSetInds = []
            for i in range(10):
                tmpA = []
                for j in range(5):
                    if j != s:
                        tmpA.append(GESTURES[i][j])
                trainingSetInds.append(tmpA)
            
            g = GestModule.pcaGestureRecognizer(SKELETON, GESTURES)
            g.train(skelSetsTrain, trainingSetInds)
            tmp = []
            for d in trainingSetInds: tmp.append(d[0])
            testData = set(range(50)).difference(set(tmp))
            for i in testData:
                g.test(skelSetShort[i], i)
            g.calcAccuracy(np.array(list(testData))/5)
            ml_accuracy.append(g.avgAccuracy)

        print "----------------"
        print "Overall accuracy", np.array(ml_accuracy).mean()
        print ml_accuracy

        g.calcClassAccuracy()
        g.calcClassConfusion()
        g.calcJointAccuracy()
        g.calcJointConfusion()
        pdb.set_trace()        


    # Train on all 5. Test on sequences
    if 0:
        import copy
        skelSetsSeq = SkelModule.getAllSkeletons("/Users/colin/data/Gestures/jed_sequences", 10)
        skelSetShort = copy.deepcopy(skelSetsSeq)

        #Test
        trainingSetInds = []
        for i in range(10):
            tmpA = []
            for j in range(5):
                tmpA.append(GESTURES[i][j])
            trainingSetInds.append(tmpA)
        g = GestModule.pcaGestureRecognizer(SKELETON, GESTURES)
        g.train(skelSetsTrain, trainingSetInds)

        #Test
        gestCount = 6
        seqsLabel = []
        for i in range(len(SEQUENCE_TRUTH)): # For each gesture
            seqLen = int(skelSetsSeq[i][0]['jointsRel'].shape[0])
            gestLen = int(seqLen/gestCount)
            seqLabel = []
            for gestNum in range(gestCount): # for each part of the gesture
                skelSetShort[i][0]['jointsRel'] = copy.deepcopy(skelSetsSeq[i][0]['jointsRel'][gestNum*gestLen : (gestNum+1)*gestLen])
                seqLabel.append(g.test(skelSetShort[i]))
            seqsLabel.append(seqLabel)
        print "Test: ", seqsLabel
        print "Truth: ", SEQUENCE_TRUTH
        # print np.equal(seqsLabel, SEQUENCE_TRUTH)*1

        # ml_accuracy = []
        # tmp = []
        # for d in trainingSetInds: tmp.append(d[0])
        # testData = set(range(50)).difference(set(tmp))
        # for i in testData:
        #     g.test(skelSetShort[i], i)
        # g.calcAccuracy(np.array(list(testData))/5)
        # ml_accuracy.append(g.avgAccuracy)

        # print "----------------"
        # print "Overall accuracy", np.array(ml_accuracy).mean()
        # print ml_accuracy

