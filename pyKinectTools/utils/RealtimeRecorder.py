
import os, time, sys
import numpy as np
import scipy.misc as sm
import Image
from pyKinectTools.utils.RealtimeReader import *

from multiprocessing import Pool, Process, Queue

# DIR = '/Users/colin/Data/icu_test/'
# DIR = '/home/clea/Data/icu_test/'
# DIR = '/media/Data/icu_test/'
DIR = '/media/Data/CV_class/'

pool = Pool(processes = 1)
# queue = SimpleQueue()
processList = []
processCount = 2

def save_frame(depthName, depth, colorName, color, usersName, users):

        try:
                # sm.imsave(depthName, depth)
                # depth = sm.imresize(depth, [240,320], 'nearest', 'I')
                # print depth.dtype
                # import pdb
                # pdb.set_trace()
                im = Image.fromarray(depth.astype(np.int32), 'I')
                im.resize([240,320])
                im.save(depthName)
                # print depthName

                color = sm.imresize(color, [240,320,3], 'nearest')
                sm.imsave(colorName, color)
                # sm.imsave(depthName, sm.imresize(depthRaw8, [240,320], 'nearest'))
                # sm.imsave(colorName, sm.imresize(colorRaw, [240,320,3],'nearest'))                
                np.savez(usersName, users=users)
                return 0
        except:
                print "Error saving"
                return -1


def main(deviceID=1, dir_=DIR, viz=0, frameDifferencePercent=8, depthStoreCount=10):

        if viz:
                import cv2
                from pyKinectTools.utils.DepthUtils import world2depth
                cv2.namedWindow("depth")


        '''------------ Setup Kinect ------------'''
        depthConstraint = [500, 5000]
        ''' Physical Kinect '''
        depthDevice = RealTimeDevice(device=deviceID)
        # depthDevice.addDepth(depthConstraint)
        # depthDevice.addColor()
        # depthDevice.addUsers()
        depthDevice.start()
        # depthDevice.setMaxDist(depthConstraint[1])

        depthOld = [] #None
        colorOld = [] #None

        maxFramerate = 30
        # minFramerate = 1.0/3.0
        minFramerate = 30
        recentMotionTime = time.clock()
        

        ''' ------------- Main -------------- '''

        prevTime = 0
        prevFrame = 0
        prevFrameTime = 0
        currentFrame = 0;
        ms = time.clock()
        diff = 0

        if not os.path.isdir(dir_):
                os.mkdir(dir_)        

        while 1:

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

                if viz:
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

                currentFrame += 1

                time_ = time.localtime()
                day = str(time_.tm_yday)
                hour = str(time_.tm_hour)
                minute = str(time_.tm_min)
                second = str(time_.tm_sec)
                ms = str(time.clock())

                # Look at how much of the image has changed
                if depthOld != []:
                        diff = np.sum(np.logical_and((depthRaw8 - depthOld[0]) > 200, (depthRaw8 - depthOld[0]) < 20000)) / 307200.0 * 100

                        ''' We want to watch all video for at least 5 seconds after we seen motion '''
                        ''' This prevents problems where there is small motion that doesn't trigger the motion detector '''
                        if diff > frameDifferencePercent:
                                recentMotionTime = time.clock()

                # depthOld = depthRaw8
                depthOld.append(depthRaw8)                                

                ''' Keep track of framerate '''
                if prevTime != second:
                        prevTime = second
                        print currentFrame - prevFrame, " #threads = ", len(processList), " fps. Diff = ", diff
                        prevFrame = currentFrame


                ''' Write to file if there has been substantial change. '''
                if 1 or diff > frameDifferencePercent or time.clock()-prevFrameTime > 1.0/minFramerate or time.clock()-recentMotionTime < 5:
                        if depthRaw8 != []:
                                ''' Create a folder if it doesn't exist '''
                                if not os.path.isdir(dir_+day):
                                        try:
                                                os.mkdir(dir_+day)
                                        except:
                                                pass
                                if not os.path.isdir(dir_+day+"/"+hour):
                                        try:
                                                os.mkdir(dir_+day+"/"+hour)
                                        except:
                                                pass
                                if not os.path.isdir(dir_+day+"/"+hour+"/"+minute):
                                        try:
                                                os.mkdir(dir_+day+"/"+hour+"/"+minute)
                                        except:
                                                pass
                                if not os.path.isdir(dir_+day+"/"+hour+"/"+minute+"/device_"+str(deviceID)):
                                        try:
                                                os.mkdir(dir_+day+"/"+hour+"/"+minute+"/device_"+str(deviceID))
                                                os.mkdir(dir_+day+"/"+hour+"/"+minute+"/device_"+str(deviceID)+"/"+"depth")
                                                os.mkdir(dir_+day+"/"+hour+"/"+minute+"/device_"+str(deviceID)+"/"+"color")
                                                os.mkdir(dir_+day+"/"+hour+"/"+minute+"/device_"+str(deviceID)+"/"+"skel")
                                        except:
                                                pass

                                ''' Define filenames '''
                                depthName = dir_+day+"/"+hour+"/"+minute+"/device_"+str(deviceID)+"/"+"depth"+"/"+\
                                                        "depth_"+day+"_"+hour+"_"+minute+"_"+second+"_"+str(ms)[str(ms).find(".")+1:]+".png"
                                colorName = dir_+day+"/"+hour+"/"+minute+"/device_"+str(deviceID)+"/"+"color"+"/"+\
                                                        "color_"+day+"_"+hour+"_"+minute+"_"+second+"_"+str(ms)[str(ms).find(".")+1:]+".jpg"
                                usersName = dir_+day+"/"+hour+"/"+minute+"/device_"+str(deviceID)+"/"+"skel"+"/"+\
                                                        "skel_"+day+"_"+hour+"_"+minute+"_"+second+"_"+str(ms)[str(ms).find(".")+1:]    

                                # print str(ms), str(ms)[str(ms).find(".")+1:]                                            

                                ''' Save data '''
                                # sm.imsave(depthName, sm.imresize(depthRaw8, [240,320], 'nearest'))
                                # sm.imsave(colorName, sm.imresize(colorRaw, [240,320,3],'nearest'))

                                # sm.imsave(depthName, depthRaw8)
                                # sm.imsave(colorName, colorRaw)
                                # np.savez(usersName, users=users)

                                save_frame(depthName, depthRaw8, colorName, colorRaw, usersName, users)
                                # ''' Have it compress/save on another processor '''
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


                ''' Display skeletons '''
                if viz:
                        # print "Users: ", len(users)
                        for u_key in users.keys():
                                u = users[u_key]
                                pt = world2depth(u.com)
                                w = 10
                                d[pt[0]-w:pt[0]+w, pt[1]-w:pt[1]+w] = 255
                                w = 7
                                if u.tracked:
                                        print "Joints: ", len(u.jointPositions)
                                        for j in u.jointPositions.keys():
                                                pt = world2depth(u.jointPositions[j])
                                                d[pt[0]-w:pt[0]+w, pt[1]-w:pt[1]+w] = 150                                                        


                if viz:
                        cv2.imshow("depth", d)
                        r = cv2.waitKey(10)
                        if r >= 0:
                                break


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

