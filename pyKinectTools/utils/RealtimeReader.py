
from pyKinectTools.algs.BackgroundSubtraction import *
from pyKinectTools.utils.DepthUtils import *
import pyKinectTools.configs
from openni import *
import time

class User:
        com = []
        userID = -1
        jointPositions = {}
        jointPositionsConfidence = {}
        jointOrientations = {}
        jointOrientationsConfidence = {}
        timestamp = []
        tracked = 0

        joints = [SKEL_HEAD, SKEL_TORSO,\
                SKEL_LEFT_SHOULDER, SKEL_LEFT_ELBOW, SKEL_LEFT_HAND, \
                SKEL_LEFT_HIP, SKEL_LEFT_KNEE, SKEL_LEFT_FOOT, \
                SKEL_RIGHT_SHOULDER, SKEL_RIGHT_ELBOW, SKEL_RIGHT_HAND, \
                SKEL_RIGHT_HIP, SKEL_RIGHT_KNEE, SKEL_RIGHT_FOOT]

        def __init__(self, id_):
                self.userID = id_

                for j in self.joints:
                    self.jointPositions[j] = -1
                    self.jointPositionsConfidence[j] = -1
                    self.jointOrientations[j] = -1
                    self.jointOrientationsConfidence[j] = -1


        def toDict(self):
            d = {}
            d['com'] = self.com
            d['userID'] = self.userID
            d['jointPositions'] = self.jointPositions
            d['jointPositionsConfidence'] = self.jointPositionsConfidence
            d['jointOrientations'] = self.jointOrientations
            d['jointOrientationsConfidence'] = self.jointOrientationsConfidence
            d['timestamp'] = self.timestamp
            d['tracked'] = self.tracked
            d['joints'] = self.joints

            return d

class RealTimeDevice:
        ctx = []
        deviceNumber = None
        depth = None
        depthIm = []
        depthIm8 = []
        color = None
        colorIm = []

        user = None
        users = {}
        userIDs = []
        skel_cap = []
        pose_cap = []

        constrain = []
        maxDist = np.inf

        bgModel = []
        bgModel8 = []

        # def __init__(self, device=-1, ctx=[], get_skeleton=True):
        def __init__(self, device=-1, ctx=None, get_depth=True, get_color=True, get_skeleton=True):

                self.deviceNumber = device
                if ctx is not None:
                    self.ctx = ctx
                else:
                    self.ctx = Context()

                if device == -1:
                    self.ctx.init()
                else:
                    self.ctx.init_from_xml_file_by_device_id(pyKinectTools.configs.__path__[0]+'/SamplesConfig.xml', self.deviceNumber)
                    print "New context created for depth device. (#", self.deviceNumber, ")"

                    if get_depth:
                        self.depth = self.ctx.find_existing_node(NODE_TYPE_DEPTH)
                    if get_color:
                        self.color = self.ctx.find_existing_node(NODE_TYPE_IMAGE)
                    if get_skeleton:
                        self.user = self.ctx.find_existing_node(NODE_TYPE_USER)

                    ''' Change viewpoint if both depth and color are used '''
                    if self.depth is not None and self.color is not None:
                        self.depth.alternative_view_point_cap.set_view_point(self.color)

                    if self.user is not None:
                        self.skel_cap = self.user.skeleton_cap
                        self.pose_cap = self.user.pose_detection_cap

                        ''' Register users '''
                        self.user.register_user_cb(self.new_user, self.lost_user)
                        self.pose_cap.register_pose_detected_cb(self.pose_detected)
                        self.skel_cap.register_c_start_cb(self.calibration_start)
                        self.skel_cap.register_c_complete_cb(self.calibration_complete)

                        ''' Set the profile '''
                        self.skel_cap.set_profile(SKEL_PROFILE_ALL)


        def addDepth(self, constrain=[500, 2000]):
                self.constrain = constrain
                self.maxDist = constrain[1]

                try:
                    self.depth = DepthGenerator()
                    self.depth.create(self.ctx)
                except:
                    print "Depth module can not load."

                if self.color is not None:
                        self.depth.alternative_view_point_cap.set_view_point(depthDevice.color)

        def addColor(self):

                try:
                    self.color = ImageGenerator()
                    self.color.create(self.ctx)
                except:
                    print "Color module can not load."

                if self.depth is not None:
                    self.depth.alternative_view_point_cap.set_view_point(self.color)

        def addUsers(self):

            try:
                self.user = UserGenerator()
                self.user.create(self.ctx)

                self.skel_cap = self.user.skeleton_cap
                self.pose_cap = self.user.pose_detection_cap

                '''Register them'''
                self.user.register_user_cb(self.new_user, self.lost_user)
                self.pose_cap.register_pose_detected_cb(self.pose_detected)
                self.skel_cap.register_c_start_cb(self.calibration_start)
                self.skel_cap.register_c_complete_cb(self.calibration_complete)

                '''Set the profile'''
                self.skel_cap.set_profile(SKEL_PROFILE_ALL)

            except:
                print "Can't add skeleton module"


        def setMaxDist(self, dist):
            self.maxDist = dist

        def start(self):
            self.ctx.start_generating_all()

        def stop(self):
            self.ctx.shutdown()

        # @profile
        def update(self):
            # ret = self.ctx.wait_any_update_all()
            ret = self.ctx.wait_and_update_all()
            assert ret == None, "Error updating depth device."

            if self.depth is not None:
                    self.depthIm = np.frombuffer(self.depth.get_raw_depth_map(), np.uint16).reshape([self.depth.res[1],self.depth.res[0]])
                    #self.depthIm8 = constrain(self.depthIm, self.constrain[0], self.constrain[1])

            if self.color is not None:
                    # from IPython import embed
                    # embed()
                    self.colorIm = np.frombuffer(self.color.get_raw_image_map_bgr(), np.uint8).reshape([self.color.res[1],self.color.res[0], 3])

            if self.user is not None:
                    for i in self.user.users:
                        self.users[i].com = self.user.get_com(i)
                        self.users[i].timestamp = self.user.timestamp

                        if self.skel_cap.is_tracking(i):
                            self.users[i].tracked = 1
                            for j in self.users[i].joints:
                                self.users[i].jointPositions[j] = self.skel_cap.get_joint_position(i, j).point
                                self.users[i].jointPositionsConfidence[j] = self.skel_cap.get_joint_position(i, j).confidence
                                self.users[i].jointOrientations[j] = self.skel_cap.get_joint_orientation(i, j).matrix
                                self.users[i].jointOrientationsConfidence[j] = self.skel_cap.get_joint_orientation(i, j).confidence
                        else:
                            self.users[i].tracked = 0


        def generateBackgroundModel(self):
            '''Get set of 5 frames and create background model'''
            depthImgs = []
            depthStackInd = 0
            for i in xrange(5):
                ret = self.ctx.wait_one_update_all(self.depth)
                assert ret == None, "Error getting depth map"

                depthRawT = self.depth.get_tuple_depth_map()
                im = np.array(depthRawT).reshape([self.depth.res[1],self.depth.res[0]])
                depthImgs.append(im)
                time.sleep(.2)

            depthImgs = np.dstack(depthImgs)

            self.bgModel = getMeanImage(depthImgs)

            self.bgModel8 = constrain(self.bgModel, self.constrain[0], self.constrain[1])
            self.bgModel8[self.bgModel8==self.bgModel8.max()] = 0



        def new_user(self, src, id):
            print "1/4 User {} detected. Looking for pose..." .format(id)
            if id not in self.userIDs:
                self.userIDs.append(id)
                self.users[id] = User(id)
                # self.pose_cap.start_detection("Psi", id)
                self.skel_cap.request_calibration(id, True)
            # else:
                # self.skel_cap.request_calibration(id, True)

        def lost_user(self, src, id):
            print "--- User {} lost." .format(id)
            if id in self.users:
                del self.users[id]
            self.userIDs.remove(id)


        def pose_detected(self, src, pose, id):
            print "2/4 Detected pose {} on user {}. Requesting calibration..." .format(pose,id)
            self.pose_cap.stop_detection(id)
            self.skel_cap.request_calibration(id, True)

        def calibration_start(self, src, id):
            print "3/4 Calibration started for user {}." .format(id)
            if id not in self.userIDs:
                self.userIDs.append(id)
                self.skel_cap.request_calibration(id, True)

        def calibration_complete(self, src, id, status):
            if status == CALIBRATION_STATUS_OK:
                print "4/4 User {} calibrated successfully! Starting to track." .format(id)
                self.skel_cap.start_tracking(id)
            else:
                print "ERR User {} failed to calibrate. Restarting process." .format(id)
                self.new_user(self.user, id)

