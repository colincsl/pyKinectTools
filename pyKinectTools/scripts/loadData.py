import os, scipy
import scipy.ndimage as nd
from pyKinectTools.utils.DepthUtils import posImage2XYZ
from pyKinectTools.algs.BackgroundSubtraction import extractPeople, removeNoise
dataDir = '/Users/colin/data/ICU_7May2012_Wide_jpg/diffDraw1/'
dataDir = '/Users/colin/data/ICU_7May2012_Close_jpg/diffDraw1/'
dataDir = '/Users/colin/data/ICU_7May2012_Close_jpg/d1c/'
# cd /Users/colin/data/ICU_7May2012_Wide_jpg/d/1c

'''#################### Load Images #########################'''

'''
imgs = array of many images
im = specific image
posMat = 3-dimensional array of XYZ positions at each pixel
xyz = list of points
'''

files = os.listdir(dataDir)
files = [int(x[0:-4]) for x in files if x[0]!='.']
files = np.sort(files)
# sequenceFrameNames = files[410:640] # to 1101
sequenceFrameNames = files[597:620] #400-600#597, 1100
# sequenceFrameNames = files[2000:2300] #400-600#597, 1100
# He moved around 2100-2200
imgs = []
for i in sequenceFrameNames:
    imgs.append(scipy.misc.imread(dataDir+str(i)+'.jpg'))
imgs = np.array(imgs)

''' Get posMat from individual image '''
im = imgs[9]
objectNum = 0
posMatFull = posImage2XYZ(im, 500, 2000)
# posMatFull = posImage2XYZ(im, 500, 2000) 
imLabels, objSlices, objInds = extractPeople(posMatFull[:,:,2], 10000, True)
posMat = posMatFull[objSlices[objectNum]]
for i in range(3):
	posMat[:,:,i] *= (imLabels[objSlices[objectNum]]==objInds[objectNum])

# posMat = removeNoise(posMat, thresh=500)
xyz = posMat[(posMat[:,:,2]>0)*(posMat[:,:,0]!=0),:]



# Edge test
im1 = imgs[0]
im2 = imgs[1]
imgs[0] = np.array(imgs[0], dtype=int16)
imgs[1] = np.array(imgs[1], dtype=int16)
imD = np.array(imgs[0]-imgs[1], dtype=np.int16)
diff = (np.diff(np.abs(imD)>10))
diff = nd.binary_closing(diff, iterations=5)
# diff = nd.binary_dilation(diff, iterations=3)
imshow(im1[:,0:-1]*(1-diff))

im = im1[:,0:-1]*(1-diff)



'''#################### Test HOG #########################'''


''' Load example (uncompressed) img '''
# saved = np.load('tmpPerson_close.npz')['arr_0'].tolist()
# saved = np.load('tmpPerson1.npz')['arr_0'].tolist()
# objects1 = saved['objects']; labelInds1=saved['labels']; out1=saved['out']; d1=saved['d']; com1=saved['com'];featureExt1=saved['features']; posMat=saved['posMat']; xyz=saved['xyz']



'''############### Load Time of Flight ##################'''

# cd /Users/colin/data/TUM/operation_camera_tool/operation_camera_tool/



'''#################### Other stuff #########################'''

'''Other'''
# np.savez('tmpPerson1.npz', {'objects':objects1, 'labels':labelInds1, 'out':out1, 'd':d1, 'com':com1, 'features':featureExt1, 'posMat':posMat, 'xyz':xyz})
