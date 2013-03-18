''' Use this file to hand register multiple depth cameras with the 3D visualizer

Procedure:
1) Modify the scrip below for your files
2) After adding points, click the mayavi button in the window and add Transformation to the scene. Drag the second points to the transformation.
3) Manually match the two scenes
4) Click red button "Start/Stop Script Recording". Transform a tiny bit so that you see the transformation matrix
5) Write down/save the transformation matrix
'''

from pyKinectTools.utils.DepthUtils import *
from scipy.misc import imread
from mayavi import mlab
from mayavi.api import Engine
import cPickle as pickle


''' ---- Transforms ---
ICUDec2012 data:
#3->2:	transform_data.transform.matrix.__setstate__({'elements': [0.9553782053802112, -0.09691967661345026, 0.27903236545178867, -392.81878278215254, 0.09283849668727677, 0.9952919671849423, 0.02783726980083738, 231.6724797545669, -0.2804166511056782, -0.0006901755293638524, 0.9598781305147085, -118.84124965680712, 0.0, 0.0, 0.0, 1.0]})
#1->2:	transform_data2.transform.matrix.__setstate__({'elements': [-0.8531195226064485, -0.08215320378328564, 0.5152066878990207, 761.2299809410998, 0.3177589268248827, 0.7014041249433673, 0.6380137286418792, 1427.5420972165339, -0.4137829679564377, 0.7080134918351199, -0.5722766383564786, -3399.696025885259, 0.0, 0.0, 0.0, 1.0]})

Office 23Feb2013
top->bottom view
#1->2 [0.9955555989899513, 0.03914715069837748, 0.08565366257179756, 240.34720254711863, 
		-0.08535684788048972, 0.7593599156829527, 0.6450478485925556, 1305.7428154935583, 
		-0.039790172651939335, -0.6494921239292868, 0.759326493093817, -237.20556423494145, 
		0.0, 0.0, 0.0, 1.0]})
 '''

'''
ICUDec2012
base_dir1 = '/media/Data/ICU_Dec2012/ICU_Dec2012_r40_c1/depth/356/14/1/'
base_dir23 = '/media/Data/ICU_Dec2012/ICU_Dec2012_r40_c2/depth/356/14/0/'

depthFile1 = base_dir1+'device_1/'+'depth_356_14_1_55_00_95.png'
depthFile2 = base_dir23+'device_1/'+'depth_356_14_0_10_01_44.png'
depthFile3 = base_dir23+'device_2/'+'depth_356_14_0_10_00_48.png'
'''

'''
# Office_25Feb2013
base_dir = '/Users/colin/Data/Office_25Feb2013/depth/56/17/31/'
depthFile1 = base_dir+'device_1/'+'depth_56_17_31_0_00_506677.png'
depthFile2 = base_dir+'device_2/'+'depth_56_17_31_0_00_510469.png'
'''
'''
# CIRL_28Feb2013
base_dir = '/Users/colin/Data/CIRL_28Feb2013/depth/59/13/42/'
depthFile1 = base_dir+'device_1/'+'depth_59_13_42_0_00_364016.png'
depthFile2 = base_dir+'device_2/'+'depth_59_13_42_0_00_645072.png'

result:
T = np.array([[0.857551855717905, 0.11935353392976167, 0.5003594195108932, -1053.586999301418], 
			[0.1430128492517155, 0.8790419590510106, -0.45478847740098743, 1081.8626448851123], 
			[-0.4941175363248235, 0.4615625289885183, 0.736754974282534, 1295.7083313896273], 
			[0.0, 0.0, 0.0, 1.0]])
'''



''' --------------------Main setup-------------------------------- '''

depthIm1 = imread(depthFile1)
depthIm2 = imread(depthFile2)
pts1 = depthIm2XYZ(depthIm1).astype(np.int)
pts2 = depthIm2XYZ(depthIm2).astype(np.int)

engine = Engine()
engine.start()

figure = mlab.figure(1, bgcolor=(0,0,0), fgcolor=(1,1,1))
mlab.clf()
figure.scene.disable_render = True

interval = 20 # Don't show all points (otherwise it's slow!)
pts = np.array([x for x in pts1 if x[2] > -93500])
pts = np.vstack([[0,0,1], pts])
ptsViz1 = mlab.points3d(pts[::interval,0], pts[::interval,1], pts[::interval,2], 2.-(np.minimum(pts[::interval,2], 5000)/float((-pts[:,2]).max()))/1000., scale_factor=30., colormap='summer')
pts = np.array([x for x in pts2 if x[2] > -93500])
pts = np.vstack([[0,0,1], pts])
ptsViz1 = mlab.points3d(pts[::interval,0], pts[::interval,1], pts[::interval,2], 2.-(np.minimum(pts[::interval,2], 5000)/float((-pts[:,2]).max()))/1000., scale_factor=30., colormap='Blues')


# Copy description and transform here as 4x4 matrix
# e.g.
filename = 'Registration.dat'
description = "CIRL_28Feb2013 Bottom (good) cam to top (bad) cam"
T = np.array([[0.857551855717905, 0.11935353392976167, 0.5003594195108932, -1053.586999301418], 
				[0.1430128492517155, 0.8790419590510106, -0.45478847740098743, 1081.8626448851123], 
				[-0.4941175363248235, 0.4615625289885183, 0.736754974282534, 1295.7083313896273], 
				[0.0, 0.0, 0.0, 1.0]])
# save
pickle.dump({'description':description, 'transform':T}, open(filename, 'w'))



''' --------------------------------------------------------- '''
''' ------------------- Fine Tuning ------------------------- '''
''' --------------------------------------------------------- '''
from mayavi.filters.transform_data import TransformData

depthIm1 = imread(depthFile1)
depthIm2 = imread(depthFile2)
depthIm3 = imread(depthFile3)

''' Put all in the frame of 2 '''

pts1 = depthIm2XYZ(depthIm1)#.astype(np.int)
pts2 = depthIm2XYZ(depthIm2)#.astype(np.int)
pts3 = depthIm2XYZ(depthIm3).astype(np.int)

p1 = depthIm2PosIm(depthIm1)
p2 = depthIm2PosIm(depthIm2)
p3 = depthIm2PosIm(depthIm3)

'''3DViz'''

engine = Engine()
engine.start()

figure = mlab.figure(1, bgcolor=(0,0,0), fgcolor=(1,1,1))
mlab.clf()
figure.scene.disable_render = True

interval = 15

'''2'''
pts = np.array([x for x in pts2 if x[2] > -93500])
ptsViz1 = mlab.points3d(pts[::interval,0], pts[::interval,1], pts[::interval,2], 2.-(np.minimum(pts[::interval,2], 5000)/float((-pts[:,2]).max()))/1000., scale_factor=10., colormap='Blues')
'''3'''
pts = np.array([x for x in pts3 if x[2] > -93500])
ptsViz2 = mlab.points3d(pts[::interval,0], pts[::interval,1], pts[::interval,2], 2.-(np.minimum(pts[::interval,2], 5000)/float((-pts[:,2]).max()))/1000., scale_factor=10., colormap='PuOr')

transform_data = TransformData()
engine.add_filter(transform_data, engine.scenes[0].children[1])
transform_data.children = [engine.scenes[0].children[1].children[0]]
# engine.scenes[0].children[1].children[0]=[]

transform_data.transform.matrix.__setstate__({'elements': [0.9553782053802112, -0.09691967661345026, 0.27903236545178867, -392.81878278215254, 0.09283849668727677, 0.9952919671849423, 0.02783726980083738, 231.6724797545669, -0.2804166511056782, -0.0006901755293638524, 0.9598781305147085, -118.84124965680712, 0.0, 0.0, 0.0, 1.0]})
transform_data.widget.set_transform(transform_data.transform)
transform_data.filter.update()
transform_data.widget.enabled = False

'''1'''
pts = np.array([x for x in pts1 if x[2] > -93500])
ptsViz1 = mlab.points3d(pts[::interval,0], pts[::interval,1], pts[::interval,2], 2.-(np.minimum(pts[::interval,2], 5000)/float((-pts[:,2]).max()))/1000., scale_factor=10., colormap='summer')
mlab.view(azimuth=0, elevation=0, distance=3000., focalpoint=(0,0,0), figure=figure)#, reset_roll=False)
figure.scene.disable_render = False

transform_data2 = TransformData()
engine.add_filter(transform_data2, engine.scenes[0].children[2])
transform_data2.children = [engine.scenes[0].children[2].children[0]]
# engine.scenes[0].children[2].children[0]=[]


transform_data2.transform.matrix.__setstate__({'elements': [-0.8531195226064485, -0.08215320378328564, 0.5152066878990207, 761.2299809410998, 0.3177589268248827, 0.7014041249433673, 0.6380137286418792, 1427.5420972165339, -0.4137829679564377, 0.7080134918351199, -0.5722766383564786, -3399.696025885259, 0.0, 0.0, 0.0, 1.0]})
transform_data2.widget.set_transform(transform_data1.transform)
transform_data2.filter.update()
transform_data2.widget.enabled = False


'''
mlab.view(azimuth=0, elevation=0, distance=3000., focalpoint=(0,0,0), figure=figure)#, reset_roll=False)

'''