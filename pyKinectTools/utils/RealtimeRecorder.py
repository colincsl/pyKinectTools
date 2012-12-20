
import os, time, sys
import numpy as np
import scipy.misc as sm
import Image
from pyKinectTools.utils.RealtimeReader import *
import cPickle as pickle

from multiprocessing import Pool, Process, Queue

# DIR = '/Users/colin/Data/icu_test/'
# DIR = '/home/clea/Data/tmp/'
# DIR = '/home/clea/Data/ICU_Nov2012/'
DIR = '/media/Data/icu_test/'
# DIR = '/media/Data/CV_class/'

pool = Pool(processes = 1)
# queue = SimpleQueue()
processList = []
processCount = 2

def save_frame(depthName, depth, colorName, color, userName, users, mask=None):

        try:
                # sm.imsave(depthName, depth)
                im = Image.fromarray(depth.astype(np.int32), 'I')
                im = im.resize([320,240])
                im.save(depthName)
                
                if mask != None:
                        mask = sm.imresize(mask, [240,320], 'nearest')
                        sm.imsave(depthName[:-4]+"_mask.jpg", mask)

                color = sm.imresize(color, [240,320,3], 'nearest')
                sm.imsave(colorName, color)

                usersOut = {}
                for k in users.keys():
                        # from IPython import embed
                        # embed()
                        usersOut[k] = users[k].toDict()
                # print usersOut
                # print users.toDict()

                with open(userName, 'wb') as outfile:
                        pickle.dump(usersOut, outfile, protocol=pickle.HIGHEST_PROTOCOL)
                return 0
        except:
                print "Error saving"
                return -1


def main(deviceID=1, dir_=DIR, viz=0, getSkel=True, frameDifferencePercent=5, anonomize=True,  depthStoreCount=5):
# def main(deviceID=1, dir_=DIR, viz=0, frameDifferencePercent=0*5, depthStoreCount=1):

        if viz:
                import cv2
                from pyKinectTools.utils.DepthUtils import world2depth
                cv2.namedWindow("depth")


        '''------------ Setup Kinect ------------'''
        depthConstraint = [500, 5000]
        ''' Physical Kinect '''
        depthDevice = RealTimeDevice(device=deviceID)
        # getSkel=False
        # depthDevice.addDepth(depthConstraint)
        # depthDevice.addColor()
        # depthDevice.addUsers()
        depthDevice.start()
        # depthDevice.setMaxDist(depthConstraint[1])

        depthOld = [] #None
        colorOld = [] #None

        maxFramerate = 30
        minFramerate = 1.0/3.0
        motionLagTime = 5
        recentMotionTime = time.clock()
        

        ''' ------------- Main -------------- '''

        prevTime = 0
        prevFrame = 0
        prevFrameTime = 0
        currentFrame = 0;
        ms = time.clock()
        diff = 0

        prevSec = 0;
        secondCount = 0
        prevSecondCountMax = 0

        if not os.path.isdir(dir_):
                os.mkdir(dir_)        

        while 1:
                try:

                        depthDevice.update()
                        colorRaw = depthDevice.colorIm
                        depthRaw8 = depthDevice.depthIm
                        # print depthRaw8.dtype
                        users = depthDevice.users
                        skel = None

                        # Update process thread counts
                        removeProcesses = []
                        for i in xrange(len(processList)-1, -1, -1):
                                if not processList[i].is_alive():
                                        removeProcesses.append(i)
                        for i in removeProcesses:
                                processList.pop(i)

                        if len(depthOld) == depthStoreCount:
                                depthOld.pop(0)                

                        ''' If framerate is too fast then skip '''
                        ''' Keep this after update to ensure fast enough kinect refresh '''
                        if (time.clock() - float(ms))*1000 < 1000.0/maxFramerate:
                                continue                

                        if viz and 0:
                                for i in depthDevice.user.users:
                                        tmpPx = depthDevice.user.get_user_pixels(i)

                                        if depthDevice.skel_cap.is_tracking(i):
                                                brightness = 50
                                        else:
                                                brightness = 150
                                        depthRaw8 = depthRaw8*(1-np.array(tmpPx).reshape([480,640]))
                                        depthRaw8 += brightness*(np.array(tmpPx).reshape([480,640]))

                        d = None
                        d = np.array(depthRaw8)

                        d /= (np.nanmin([d.max(), 2**16])/256.0)
                        d = d.astype(np.uint8)

                        ''' Get new time info '''
                        currentFrame += 1
                        time_ = time.localtime()
                        day = str(time_.tm_yday)
                        hour = str(time_.tm_hour)
                        minute = str(time_.tm_min)
                        second = str(time_.tm_sec)
                        ms = str(time.clock())
                        ms_str = str(ms)[str(ms).find(".")+1:]


                        ''' Look at how much of the image has changed '''
                        if depthOld != []:
                                diff = np.sum(np.logical_and((depthRaw8 - depthOld[0]) > 200, (depthRaw8 - depthOld[0]) < 20000)) / 307200.0 * 100

                                ''' We want to watch all video for at least 5 seconds after we seen motion '''
                                ''' This prevents problems where there is small motion that doesn't trigger the motion detector '''
                                if diff > frameDifferencePercent:
                                        recentMotionTime = time.clock()

                        depthOld.append(depthRaw8)                                




                        ''' Write to file if there has been substantial change. '''
                        # if 1:
                        if diff > frameDifferencePercent or time.clock()-prevFrameTime > 1/minFramerate or time.clock()-recentMotionTime < motionLagTime:
                                if depthRaw8 != []:

                                        ''' Logical time '''
                                        if second != prevSec:
                                                prevSecondCountMax = secondCount                                
                                                secondCount = 0
                                                prevSec = second
                                        else:
                                                secondCount = int(secondCount) + 1

                                        secondCount = str(secondCount)
                                        if len(ms_str) == 1:
                                                ms_str = '0' + ms_str
                                        if len(secondCount) == 1:
                                                secondCount = '0' + secondCount


                                        ''' Keep track of framerate '''
                                        if prevTime != second:
                                                prevTime = second
                                                # print currentFrame - prevFrame, " fps. Diff = ", str(diff)[:4] + "%" #" #threads = ", len(processList), 
                                                print "FPS: "+prevSecondCountMax + " Diff: " + str(diff)[:4] + "%" #" #threads = ", len(processList), 
                                                prevFrame = currentFrame


                                        ''' Create a folder if it doesn't exist '''

                                        # if not os.path.isdir(dir_+day):
                                        #         try:
                                        #                 os.mkdir(dir_+day)
                                        #         except:
                                        #                 pass
                                        # if not os.path.isdir(dir_+day+"/"+hour):
                                        #         try:
                                        #                 os.mkdir(dir_+day+"/"+hour)
                                        #         except:
                                        #                 pass
                                        # if not os.path.isdir(dir_+day+"/"+hour+"/"+minute):
                                        #         try:
                                        #                 os.mkdir(dir_+day+"/"+hour+"/"+minute)
                                        #         except:
                                        #                 pass
                                        # if not os.path.isdir(dir_+day+"/"+hour+"/"+minute+"/device_"+str(deviceID)):
                                                # try:
                                                        # os.mkdir(dir_+day+"/"+hour+"/"+minute+"/device_"+str(deviceID))
                                                        # os.mkdir(dir_+day+"/"+hour+"/"+minute+"/device_"+str(deviceID)+"/"+"depth")
                                                        # os.mkdir(dir_+day+"/"+hour+"/"+minute+"/device_"+str(deviceID)+"/"+"color")
                                                        # os.mkdir(dir_+day+"/"+hour+"/"+minute+"/device_"+str(deviceID)+"/"+"skel")
                                                # except:
                                                        # pass
                                        depthDir = dir_+'depth/'+day+"/"+hour+"/"+minute+"/device_"+str(deviceID)
                                        colorDir = dir_+'color/'+day+"/"+hour+"/"+minute+"/device_"+str(deviceID)
                                        skelDir = dir_+'skel/'+day+"/"+hour+"/"+minute+"/device_"+str(deviceID)
                                        # print ""
                                        # print depthDir, depthDir.split('/')
                                        if not os.path.isdir(depthDir):
                                                for p in xrange(4, len(depthDir.split("/"))+1):                         
                                                        try:
                                                                # print "/".join(depthDir.split('/')[0:p]), p
                                                                # os.mkdir("/".join(depthDir.split('/'))) 
                                                                os.mkdir("/".join(depthDir.split('/')[0:p])) 
                                                                os.mkdir("/".join(colorDir.split('/')[0:p]))
                                                                os.mkdir("/".join(skelDir.split('/')[0:p]))
                                                                        # os.mkdir(dir_+day+"/"+type_dir+"/"+hour+"/"+minute+"/device_"+str(deviceID)) 

                                                        except:
                                                                # print "error making dir"
                                                                pass


                                        ''' Define filenames '''
                                        # depthName = dir_+day+"/"+hour+"/"+minute+"/device_"+str(deviceID)+"/"+"depth"+"/"+\
                                        #                         "depth_"+day+"_"+hour+"_"+minute+"_"+second+"_"+secondCount+"_"+ms_str+".png"
                                        # colorName = dir_+day+"/"+hour+"/"+minute+"/device_"+str(deviceID)+"/"+"color"+"/"+\
                                        #                         "color_"+day+"_"+hour+"_"+minute+"_"+second+"_"+secondCount+"_"+ms_str+".jpg"
                                        # # The extra "_" at the end makes it easier to read
                                        # usersName = dir_+day+"/"+hour+"/"+minute+"/device_"+str(deviceID)+"/"+"skel"+"/"+\
                                        #                         "skel_"+day+"_"+hour+"_"+minute+"_"+second+"_"+secondCount+"_"+ms_str+"_.dat"
                                        depthName = depthDir + "/depth_"+day+"_"+hour+"_"+minute+"_"+second+"_"+secondCount+"_"+ms_str+".png"
                                        colorName = colorDir + "/color_"+day+"_"+hour+"_"+minute+"_"+second+"_"+secondCount+"_"+ms_str+".jpg"
                                        usersName = skelDir + "/skel_"+day+"_"+hour+"_"+minute+"_"+second+"_"+secondCount+"_"+ms_str+"_.dat"


                                        # print 'c:', secondCount
                                        # print str(ms), str(ms)[str(ms).find(".")+1:]                                            

                                        ''' Save data '''
                                        ''' Anonomize '''
                                        if anonomize:
                                                mask = np.ma.ones([480,640])
                                                mask.mask = True
                                                for i in depthDevice.user.users:
                                                        mask.mask *= np.equal(np.array(depthDevice.user.get_user_pixels(i)).reshape([480,640]), 0)#[:,:,np.newaxis]
                                                save_frame(depthName, depthRaw8, colorName, colorRaw, usersName, users, mask=mask.mask)
                                        else:
                                                save_frame(depthName, depthRaw8, colorName, colorRaw, usersName, users)
                                        # save_frame(depthName, depthRaw8, colorName, colorRaw*mask.mask[:,:,np.newaxis], usersName, users)
                                        # save_frame(depthName, depthRaw8, colorName, colorRaw, usersName, users, mask=mask.mask)

                                        ''' Have it compress/save on another processor '''
                                        # p = Process(target=save_frame, args=(depthName, depthRaw8, colorName, colorRaw, usersName, users))
                                        # p.start()
                                        # processList.append(p)
                                        # print depthName, 1, depthRaw8.dtype

                                        # queue.put((target=save_frame, args=(depthName, depthRaw8, colorName, colorRaw, usersName, users))
                                        # print "Size: ", queue.qsize()
                                        # pool.apply_async(save_frame, args=(depthName=depthName, depth=depthRaw8, colorName=colorName, color=colorRaw, usersName=usersName, users=users))
                                        # pool.apply_async(save_frame, args=(depthName, depthRaw8, colorName, colorRaw, usersName, users))
                                        # pool.apply_async(save_frame(depthName, depthRaw8, colorName, colorRaw, usersName, users))
                                        # pool.join()

                                        # if len(processList) < processCount:
                                        #         p.start()
                                        # else:
                                        #         processList.append(p)

                                        prevFrameTime = time.clock()



                                ''' Display skeletons '''
                                if viz:
                                        # print "Users: ", len(users)
                                        for u_key in users.keys():
                                                u = users[u_key]
                                                pt = world2depth(u.com)
                                                w = 10
                                                d[pt[0]-w:pt[0]+w, pt[1]-w:pt[1]+w] = 255
                                                w = 3
                                                if u.tracked:
                                                        print "Joints: ", len(u.jointPositions)
                                                        for j in u.jointPositions.keys():
                                                                pt = world2depth(u.jointPositions[j])
                                                                d[pt[0]-w:pt[0]+w, pt[1]-w:pt[1]+w] = 200                                                        


                                if viz:
                                        cv2.imshow("depth", d)
                                        # if len(users.keys()) > 0 and users[1].tracked==1:
                                                # from IPython import embed
                                                # embed()                         
                                        # cv2.imshow("depth", colorRaw*mask.mask[:,:,np.newaxis ])
                                        # cv2.imshow("depth", np.logical_and((depthRaw8 - depthOld[0]) > 200, (depthRaw8 - depthOld[0]) < 20000).astype(np.uint8)*255)
                                        r = cv2.waitKey(10)
                                        if r >= 0:
                                                break
                except:
                     print "Error recording frame"


''' there shouldn't be 2 threads when nothing is going on! '''

if __name__ == "__main__":
        if len(sys.argv) > 1:

                ''' Viz? '''
                if len(sys.argv) > 2:
                        viz = sys.argv[2]
                else:
                        viz = 0

                ''' Get frame difference percent '''
                if len(sys.argv) > 3:
                        frameDiffPercent = sys.argv[2]
                else:
                        frameDiffPercent = -1

                if frameDiffPercent < 0:
                        main(deviceID=int(sys.argv[1]), viz=int(viz))
                else:
                        main(deviceID=int(sys.argv[1]), viz=int(viz), frameDifferencePercent = 6)   

        else:
                main(1)

