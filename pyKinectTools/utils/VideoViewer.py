'''
Replacement for OpenCVs imshow and waitkey commands
'''

import numpy as np
import visvis as vv
import time
from IPython import embed


class VideoViewer:
	'''
	Uses the library visvis to write images to an opengl window. 
	It imitates the OpenCV video player commands imshow(), waitKey() and putText().
	'''

	open_windows = []

	def __init__(self):
		pass

	def _getWindow(self, name):
		names = [x['name'] for x in self.open_windows]
		try:
			index = names.index(name)
			return self.open_windows[index]
		except:
			return None

	def _getWindowFromFigure(self, figure):
		figs = [x['canvas'] for x in self.open_windows]
		try:
			index = figs.index(figure)
			return self.open_windows[index]
		except:
			return None			

	def _keyHandler(self, event):
		# embed()
		win = self._getWindowFromFigure(event.owner)
		win['keyEvent'] = event.key
		# print event.text, event.key
		
	def _createWindow(self, name, im, axis):

		vv.figure()
		vv.gca()
		vv.clf()
		fig = vv.imshow(im)
		dims = im.shape

		''' Change color bounds '''
		if im.dtype == np.uint8:
			fig.clim.Set(0, 255)
		else:
			fig.clim.Set(0., 1.)
		
		fig.GetFigure().title = name
		
		''' Show ticks on axes? '''
		if not axis:
			fig.GetAxes().axis.visible = False
			bgcolor = (0.,0.,0.)
		else:
			fig.GetAxes().axis.showBox = False
			bgcolor = (1.,1.,1.)

		''' Set background color '''
		fig.GetFigure().bgcolor = bgcolor
		fig.GetAxes().bgcolor = bgcolor

		''' Setup keyboard event handler '''
		fig.eventKeyDown.Bind(self._keyHandler)

		win = {'name':name, 'canvas':fig, 'shape':dims, 'keyEvent':None, 'text':[]}
		self.open_windows.append(win)

		return win


	def destroyWindow(self, name):
		'''
		Kill a specific window
		'''		
		win = self._getWindow(name)
		
		if win is not None:
			win['canvas'].Destroy()
		else:
			print "No window found"

		vv.update()

	def destroyAll(self):
		'''
		Kill all open windows
		'''
		for win in self.open_windows:
			if win is not None:
				win['canvas'].Destroy()

		vv.update()		

	def imshow(self, name, im, axis=False):
		'''
		Inputs
			name: string with figure name
			im: numpy array
			axis: display axes?
		'''

		win = self._getWindow(name)
		
		if win is None:
			win = self._createWindow(name, im, axis)
		else:	
			''' If new image is of different dimensions we must create new image'''
			if im.shape != win['shape']:
				self.destroyWindow(win['name'])
				win = self._createWindow(name, im)
			else:
				[x.Destroy() for x in win['text']]
				win['text'] = []

		win['canvas'].SetData(im)

	
	def update(self):
		''' Does not capture key inputs '''
		vv.processEvents()

	def putText(self, figureName, text, pos=(0,0), size=10, color=(255,255,255), font='sans'):
		'''
		Fonts: 'mono', 'sans' or 'serif'
		Color: Can be character (e.g. 'r', 'g', 'k') or 3-element tuple

		Notes: 
		*This draws text onto the canvas and not the image.
		*You can use symbols (e.g. \leftarrow) and greek letters (e.g. \alpha, \omega,...)
		*Bold, italics, superscript, and subscript: \b{Text}, \i{Text}, \sup^{Text}, \sub_{Text}

		See here for examples and formatting: 
		*http://code.google.com/p/visvis/wiki/example_text		
		*http://code.google.com/p/visvis/wiki/Text
		'''
		# embed()
		win = self._getWindow(figureName)
		win['canvas'].GetFigure().MakeCurrent()
		win['text'].append(vv.Text(win['canvas'].GetAxes(), text=text, x=pos[0], y=pos[1], fontSize=size, fontName=font, color=color))


	def waitKey(self, duration=0):
		return self.waitAndUpdate(duration)

	def waitAndUpdate(self, duration=.00001):
		'''
		Input:
		*Duration (in milliseconds)
			If negative it will wait until a key is pressed

		Output:
		*The first key pressed
		'''

		keys = []

		''' Wait for set duration '''
		if duration >= 0:
			time.sleep(duration)
			vv.processEvents()
			keys = [x['keyEvent'] for x in self.open_windows if x['keyEvent'] is not None]
			for i in self.open_windows:
				i['keyEvent'] = None
		else:
			''' Wait until key pressed '''
			while(time.time()-start_time < np.inf):
				vv.processEvents()
				keys = [x['keyEvent'] for x in self.open_windows if x['keyEvent'] is not None]
				time.sleep(.000001)
				if len(keys) > 0:
					break

		''' Check for keys pressed '''
		if len(keys) > 0:
			return keys.pop()
		else:
			return 0


if __name__=="__main__":
	w = VideoViewer()
	im = vv.imread('/Users/colin/Desktop/hog.png')
	w.imshow("Hi", np.eye(100, dtype=np.uint8), axis=True)
	w.imshow("im", im, axis=False)

	for j in range(100):
		key = w.waitAndUpdate(1000)
		w.imshow("im", im+j*5, axis=False)
		print key
	


