import numpy as np
import yaml
import pyKinectTools.configs

'''
---XYZ Convention---
x increases to the right
y increases upwards
z decreases as things get farther away

See this for calibration details:
http://nicolas.burrus.name/index.php/Research/KinectCalibration
'''

def get_kinect_transform(filename):
    transform = np.eye(4)
    with open(filename, 'r') as f:
        # Rotation
        line = f.readline().split()
        line[0] = line[0][3:]
        transform[:3,:3] = np.array(line, np.float).reshape([3,3])
        # Translation
        line = f.readline().split()
        line[0] = line[0][3:]
        transform[:3,-1] = np.array(line, np.float)
    return transform


class CameraModel:
    def __init__(self, filename=None):
        if filename is None:
            filename = pyKinectTools.configs.__path__[0]+'/Kinect_Color_Param.yml'
        text = open(filename).read()
        text = text.replace("!","#")
        text = text.replace("%","#")
        yaml_file = yaml.load(text)
        data = yaml_file['Camera_1']['K']['data']

        self.fx = 1/data[0]
        self.fy = 1/data[4]
        self.cx = data[2]
        self.cy = data[5]

        self.transform = np.eye(4, dtype=np.float)


    def world2im(self, pts, rez=[480, 640]):
        '''
        Convert world coordinates to depth coordinates using the standard Kinect calibration
        ---Parameters---
        x_ : numpy array (Nx3)
        rez : resolution of depth image
        '''
        assert type(pts) == np.ndarray, "world2depth input must be a numpy array"

        # pts = pts[:,[1,0,2]]
        y = np.array(pts[:,1])
        x = np.array(pts[:,0])
        ''' Make sure z coordinates are positive and greater than 0'''
        z = np.array(pts[:,2])
        z = np.maximum(1, z)

        xo = (( x / (z * self.fx)) + self.cx)
        yo = (( y / (z * self.fy)) + self.cy)

        xo *= rez[1]/640.
        yo *= rez[0]/480.

        output = np.round([yo, xo, z]).astype(np.int16).T
        output = output.clip([0,0,0], [rez[0]-1,rez[1]-1, 99999])

        return output

    def im2world(self, pts, rez=(480,640)):
        ''' Convert color coordinates to world coordinates standard Kinect calibration '''
        assert type(pts) == np.ndarray, "Wrong type into rgb2world"

        pts = pts[:,[1,0,2]]
        pts = pts.astype(np.float)
        pts[:,0] *= 480./rez[0]
        pts[:,1] *= 640./rez[1]

        y = np.array(pts[:,1])
        x = np.array(pts[:,0])
        d = np.array(pts[:,2])

        xo =  ((x - self.cx) * d * self.fx)
        yo = ((y - self.cy) * d * self.fy)
        zo = d

        return np.round([xo, yo, zo]).astype(np.float).T

    def set_transform(self, transform):
        '''
        transform : 4x4 numpy array
        '''
        self.transform = transform

    def im2PosIm(self, depthMap):
        imOut = np.zeros([depthMap.shape[0],depthMap.shape[1],3], dtype=np.float)
        imOut[:,:,2] = depthMap

        inds = np.nonzero(imOut[:,:,2]>0)
        xyzVals = self.im2world(np.array([inds[0],inds[1],imOut[inds[0],inds[1],2]], np.float).T, depthMap.shape)

        imOut[inds[0],inds[1],0] = xyzVals[:,0]
        imOut[inds[0],inds[1],1] = xyzVals[:,1]

        return imOut


# Find optimal focal lengths
# focal_lens = range(1, 2000, 100)
# for fx in focal_lens:
    # cam.camera_model.fx = fx
    # cam.camera_model.fy = fx
    # xyz_im = cam.camera_model.im2PosIm(cam.depthIm)
    # print np.sum(xyz_im[skel_orig_uv[:,0], skel_orig_uv[:,1]] - skel_orig)
    # print xyz_im[skel_orig_uv[:,0], skel_orig_uv[:,1]],  skel_orig

# Find optimal multiplier
multiplier = np.arange(.6, 1.0, .05)
for mx in multiplier:
    print "mx", mx
    xyz_im = cam.camera_model.im2PosIm(cam.depthIm*mx)
    print np.sum(xyz_im[skel_orig_uv[:,0], skel_orig_uv[:,1]] - skel_orig)
    # print xyz_im[skel_orig_uv[:,0], skel_orig_uv[:,1]],  skel_orig





# def read_kinect_camera_parameters(filename):
#     filename = '/Users/colin/Data/BerkeleyMHAD/Calibration/camcfg_k01.yml'
#     # import codecs
#     # f = codecs.open(filename, 'r', encoding='utf8')
#     # data = yaml.safe_load(f)
#     # yaml_file = yaml.load(open(filename).read())
#     text = open(filename).read()
#     text = text.replace("!","#")
#     text = text.replace("%","#")
#     yaml_file = yaml.load(text)
#     cam = Camera(**yaml_file['Camera_1'])



''' Default parameters '''
''' Depth camera parameters '''
fx_d = 1.0 / 5.9421434211923247e+02;
fy_d = 1.0 / 5.9104053696870778e+02;
cx_d = 3.3930780975300314e+02;
cy_d = 2.4273913761751615e+02;

''' Color params '''
fx_rgb = 1./5.2921508098293293e+02
fy_rgb = 1./5.2556393630057437e+02
cx_rgb = 3.2894272028759258e+02
cy_rgb = 2.6748068171871557e+02

# k1_d -2.6386489753128833e-01
# k2_d 9.9966832163729757e-01
# p1_d -7.6275862143610667e-041
# p2_d 5.0350940090814270e-03
# k3_d -1.3053628089976321e+00

#Transorm between depth and color
R = np.array([[ 9.9984628826577793e-01, 1.2635359098409581e-03, -1.7487233004436643e-02],
            [-1.4779096108364480e-03, 9.9992385683542895e-01, -1.2251380107679535e-02],
            [1.7470421412464927e-02, 1.2275341476520762e-02,9.9977202419716948e-01 ]])

T = np.array([ 1.9985242312092553e-02, -7.4423738761617583e-04,-1.0916736334336222e-02 ])*1000.

# # dep->3d_d
# P3D.x = (x_d - cx_d) * depth(x_d,y_d) / fx_d
# P3D.y = (y_d - cy_d) * depth(x_d,y_d) / fy_d
# P3D.z = depth(x_d,y_d)

# P2D_rgb.x = (P3D'.x * fx_rgb / P3D'.z) + cx_rgb
# P2D_rgb.y = (P3D'.y * fy_rgb / P3D'.z) + cy_rgb

# 3dc
# P3D' = R.P3D + T
# P3D.x = (x_d - cx_d) * depth(x_d,y_d) / fx_d
# P3D.y = (y_d - cy_d) * depth(x_d,y_d) / fy_d
# P3D.z = depth(x_d,y_d)



def u16_to_u8(im):
    '''
    Convert 16 bit image to 8 bit
    '''
    return ((im-im.min())/(im.max()/255.)).astype(np.uint8)

def depthIm_to_colorIm(depthIm, rez=(480,640)):
    # Convert depth->rgb and ensure points are within bounds
    rgb_pts = world2rgb(depthIm2XYZ(depthIm), rez)
    in_bounds = (0 <= rgb_pts[:,0]) * (rgb_pts[:,0] < rez[0]) * (0 <= rgb_pts[:,1]) * (rgb_pts[:,1] < rez[1])
    im_out = np.zeros(rez)
    im_out[rgb_pts[in_bounds,0], rgb_pts[in_bounds,1]] = rgb_pts[in_bounds,2]

    return im_out

def skel2depth(skel, rez=(480,640)):
    '''
    '''
    if type(skel) != np.ndarray:
        skel = np.array(skel)

    mod_x = rez[0]/480.
    mod_y = rez[1]/640.

    # 3d_d->3d_r
    pts_r = np.dot(R, skel.T).T + T
    # 3d_r->rgb
    rgb_x = pts_r[:,0]/fx_rgb / pts_r[:,2] + cx_rgb
    rgb_y = pts_r[:,1]/fy_rgb / pts_r[:,2] + cy_rgb
    rgb_z = pts_r[:,2]

    rgb_x *= rez[1]/640.
    rgb_y *= rez[0]/480.

    pts = np.round([rgb_x,rez[0]-rgb_y,rgb_z]).astype(np.int16).T

    return pts

def depth2world(pts, rez=(480,640)):
    ''' Convert depth coordinates to world coordinates standard Kinect calibration '''
    assert type(pts) == np.ndarray, "Wrong type into depth2world"

    pts = pts.astype(np.float)
    pts[:,0] *= 480./rez[0]
    pts[:,1] *= 640./rez[1]

    y = np.array(pts[:,1])
    x = np.array(pts[:,0])
    d = np.array(pts[:,2])

    xo =  ((x - cx_d) * d * fx_d)
    yo = ((y - cy_d) * d * fy_d)
    zo = d

    return np.round([xo, yo, zo]).astype(np.int16).T

def rgb2world(pts, rez=(480,640)):
    ''' Convert color coordinates to world coordinates standard Kinect calibration '''
    assert type(pts) == np.ndarray, "Wrong type into rgb2world"

    pts = pts.astype(np.float)
    pts[:,0] *= 480./rez[0]
    pts[:,1] *= 640./rez[1]

    y = np.array(pts[:,1])
    x = np.array(pts[:,0])
    d = np.array(pts[:,2])

    xo =  ((x - cx_rgb) * d * fx_rgb)
    yo = ((y - cy_rgb) * d * fy_rgb)
    zo = d

    return np.round([xo, yo, zo]).astype(np.int16).T

def world2depth(x_, rez=[480, 640]):
    '''
    Convert world coordinates to depth coordinates using the standard Kinect calibration
    ---Parameters---
    x_ : numpy array (Nx3)
    rez : resolution of depth image
    '''
    assert type(x_) == np.ndarray, "world2depth input must be a numpy array"

    y = np.array(x_[:,1])
    x = np.array(x_[:,0])
    ''' Make sure z coordinates are positive and greater than 0'''
    z = np.array(x_[:,2])
    z = np.maximum(1, z)

    xo = (( x / (z * fx_d)) + cx_d)
    yo = ((y / (z * fy_d)) + cy_d)

    yo *= rez[1]/640.
    xo *= rez[0]/480.

    output = np.round([xo, yo, z]).astype(np.int16).T
    output = output.clip([0,0,0], [rez[0]-1,rez[1]-1, 99999])

    return output


def world2rgb(x_, rez=[480, 640]):
    '''
    Convert world coordinates to depth coordinates using the standard Kinect calibration
    ---Parameters---
    x_ : numpy array (Nx3)
    rez : resolution of depth image
    '''
    assert type(x_) == np.ndarray, "world2depth input must be a numpy array"

    y = np.array(x_[:,1])
    x = np.array(x_[:,0])
    ''' Make sure z coordinates are positive and greater than 0'''
    z = np.array(x_[:,2])
    z = np.maximum(1, z)

    xo = (( x / (z * fx_rgb)) + cx_rgb)
    yo = (( y / (z * fy_rgb)) + cy_rgb)

    yo *= rez[1]/640.
    xo *= rez[0]/480.

    output = np.round([xo, yo, z]).astype(np.int16).T
    output = output.clip([0,0,0], [rez[0]-1,rez[1]-1, 99999])

    return output


def depthIm2XYZ(depthMap):
    inds = np.nonzero(depthMap>0)
    xyzVals = depth2world(np.array([inds[0],inds[1],depthMap[inds[0],inds[1]]]).T)

    return xyzVals

def depthIm2PosIm(depthMap):
    imOut = np.zeros([depthMap.shape[0],depthMap.shape[1],3], dtype=float)
    imOut[:,:,2] = depthMap

    inds = np.nonzero(imOut[:,:,2]>0)
    xyzVals = depth2world(np.array([inds[0],inds[1],imOut[inds[0],inds[1],2]]).T, depthMap.shape)

    imOut[inds[0],inds[1],0] = xyzVals[:,0]
    imOut[inds[0],inds[1],1] = xyzVals[:,1]

    return imOut

def rgbIm2PosIm(depthMap):
    imOut = np.zeros([depthMap.shape[0],depthMap.shape[1],3], dtype=float)
    imOut[:,:,2] = depthMap

    inds = np.nonzero(imOut[:,:,2]>0)
    xyzVals = rgb2world(np.array([inds[0],inds[1],imOut[inds[0],inds[1],2]]).T, depthMap.shape)

    imOut[inds[0],inds[1],0] = xyzVals[:,0]
    imOut[inds[0],inds[1],1] = xyzVals[:,1]

    return imOut


def posIm2depth(posIm):

    xyz = np.array([inds[0],inds[1],imOut[inds[0],inds[1],2]]).T, depthMap.shape
    world2depth = world2depth()

def posImage2XYZ(im_in, min_=500, max_=4000):
    # Used to convert 8bit depth image to a
    # 3 dimensional floating point image w/ x,y,z
    imOut = np.zeros([im_in.shape[0],im_in.shape[1],3], dtype=float)
    if im_in.sum() == 0:
        return imOut

    imOut[:,:,2] = im_in
    imOut[:,:,2] *= float((max_-min_)/256.0)
    imOut[:,:,2] += min_
    imOut[:,:,2] *= imOut[:,:,2]>min_

    inds = np.nonzero(imOut[:,:,2]>min_)
    xyzVals = depth2world(np.array([inds[0],inds[1],imOut[inds[0],inds[1],2]]).T)

    # imOut[inds] = xyzVals
    imOut[inds[0],inds[1],0] = xyzVals[:,0]
    imOut[inds[0],inds[1],1] = xyzVals[:,1]
    # imOut[inds[0],inds[1],0] = xyzVals[:,1]
    # imOut[inds[0],inds[1],1] = xyzVals[:,0]
    imOut[inds[0],inds[1],2] = xyzVals[:,2]

    return imOut


