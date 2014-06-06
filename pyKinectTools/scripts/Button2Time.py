"""
Main file for viewing data
"""

import os
import time
import optparse

""" Debugging """
from IPython import embed

try:
	import serial
	ser = serial.Serial(port='/dev/tty.usbmodemfa131', baudrate=9600)
	# Turn on light
	ser.write('3')
except:
	raise Exception, "Please install PySerial (pip install pyserial)"

# -------------------------MAIN------------------------------------------

FILE = os.path.expanduser('~')+'/Data/Kinect_Recorder/time.txt'

def main(filename):

	recording_enabled = True
	button_pressed = False

	while 1:

		while ser.inWaiting() > 0:
			button_current = ser.readline()
			print button_current, button_pressed
			if button_pressed != button_current and button_current:
				recording_enabled = False
				recording_time = time.time()
				# Turn on light
				ser.write('4')
				print 'On'

				# Write time to file
				with open(filename, 'a') as f:
					f.write(str(time.gmtime())+'\n')

			button_pressed = button_current

		if not recording_enabled:
			if time.time() - recording_time > 2:
				recording_enabled = True
				ser.write('3')
				print 'Off'
			else:
				continue



	# Pause at the end
	embed()


if __name__=="__main__":

	parser = optparse.OptionParser()
	parser.add_option('-f', '--file', dest='file', default=FILE, help='Filename for times')
	(opt, args) = parser.parse_args()

	main(opt.file)

