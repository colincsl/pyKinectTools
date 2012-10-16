
import os, time, sys
import numpy as np
import scipy.misc as sm
sys.path.append('/home/icu/kinect/pyKinectTools/build/lib.linux-i686-2.7')
from pyKinectTools.utils.RealtimeReader import *


def main():
        dir_ = '/home/icu/data/icu_test/'

        '''------------ Setup Kinect ------------'''
        depthConstraint = [500, 5000]
        ''' Physical Kinect '''
        depthDevice = RealTimeDevice()
        depthDevice.addDepth(depthConstraint)
        depthDevice.addColor()
        depthDevice.start()
        depthDevice.setMaxDist(depthConstraint[1])


        ''' ------------- Main -------------- '''

        while 1:
        #for i in range(100):
                depthDevice.update()
                colorRaw = depthDevice.colorIm
                #depthRaw8 = depthDevice.depthIm8
                depthRaw8 = depthDevice.depthIm

                time_ = time.localtime()
                day = str(time_.tm_yday)
                hour = str(time_.tm_hour)
                minute = str(time_.tm_min)
                second = str(time_.tm_sec)
                ms = str(time.clock())

                if depthRaw8 != []:
                        if not os.path.isdir(dir_+day):
                                os.mkdir(dir_+day)

                        if not os.path.isdir(dir_+day+"/"+hour):
                                os.mkdir(dir_+day+"/"+hour)

                        if not os.path.isdir(dir_+day+"/"+hour+"/"+minute):
                                os.mkdir(dir_+day+"/"+hour+"/"+minute)
                                os.mkdir(dir_+day+"/"+hour+"/"+minute+"/"+"depth")
                                os.mkdir(dir_+day+"/"+hour+"/"+minute+"/"+"color")
                                # os.mkdir(dir_+day+"/"+hour+"/"+minute+"/"+"skel")


                        depthName = dir_+day+"/"+hour+"/"+minute+"/"+"depth"+"/"+\
                                                "depth_"+day+"_"+hour+"_"+minute+"_"+second+"_"+ms+".png"
                        colorName = dir_+day+"/"+hour+"/"+minute+"/"+"color"+"/"+\
                                                "color_"+day+"_"+hour+"_"+minute+"_"+second+"_"+ms+".jpg"

                        sm.imsave(depthName, sm.imresize(depthRaw8, [240,320], 'nearest'))
                        sm.imsave(colorName, sm.imresize(colorRaw, [240,320,3],'nearest'))


                        print "Written", depthName
                        depthRaw8 = None
                        colorRaw = None

                print second