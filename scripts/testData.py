

'''##### ICU #####'''

## Depth - Close


## Depth - Wide
''' Test geodesic '''
from geodesicSkeleton import *
# saved = np.load('tmpPerson_close.npz')['arr_0'].tolist()
saved = np.load('tmpPerson1.npz')['arr_0'].tolist()
objects1 = saved['objects']; labelInds1=saved['labels']; out1=saved['out']; d1=saved['d']; com1=saved['com'];featureExt1=saved['features']; posMat=saved['posMat']; xyz=saved['xyz']

''' Load many images '''
# cd /Users/colin/data/ICU_7May2012_Wide_jpg/d1c
# cd /Users/colin/data/ICU_7May2012_Wide_jpg/diffDraw1
files = os.listdir('.')
files = [int(x[0:-4]) for x in files if x[0]!='.']
files = np.sort(files)
sequenceFrameNames = files[410:1101]
imgs = []
for i in sequenceFrameNames:
    imgs.append(scipy.misc.imread(str(i)+'.jpg'))
imgs = np.array(imgs)


''' Get posMat from individual image '''
im = imgs[35]
objectNum = 0
posMatFull = posImage2XYZ(im, 500, 4000)
# Extract people
labels = nd.label(posMatFull[:,:,2])
objects = nd.find_objects(labels[0], labels[1])
objInds = []
for i in range(0, labels[1]):
	if nd.sum(posMatFull[:,:,2]>0, labels[0], i+1) > 10000:# and (objects[i][0].stop-objects[i][0].start) > 30 and (objects[i][1].stop-objects[i][1].start) > 30:
		objInds.append(i+1)


objSlices = [objects[x-1] for x in objInds]
posMat = posMatFull[objSlices[objectNum]]
for i in range(3):
	posMat[:,:,i] *= (labels[0][objSlices[objectNum]]==objInds[objectNum])

# remove outliers
zAvg = posMat[posMat[:,:,2]>0,2].mean()
zThresh = 500
posMat[posMat[:,:,2]>zAvg+zThresh] = 0
posMat[posMat[:,:,2]<zAvg-zThresh] = 0
posMat[:,:,2] = nd.median_filter(posMat[:,:,2], 3)
xyz = posMat[(posMat[:,:,2]>0)*(posMat[:,:,0]!=0),:]



'''Test HOG'''




##### TOF #####

cd /Users/colin/data/TUM/operation_camera_tool/operation_camera_tool/




'''Other'''
# np.savez('tmpPerson1.npz', {'objects':objects1, 'labels':labelInds1, 'out':out1, 'd':d1, 'com':com1, 'features':featureExt1, 'posMat':posMat, 'xyz':xyz})
