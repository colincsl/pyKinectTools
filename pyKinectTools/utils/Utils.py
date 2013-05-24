import os

''' Create a folder if it doesn't exist '''
def createDirectory(new_dir):
	if not os.path.isdir(new_dir):
		for p in xrange(2,len(new_dir.split("/"))+1):
			tmp_string = "/".join(new_dir.split('/')[:p])
			if not os.path.isdir(tmp_string):
				try:
					os.mkdir(tmp_string)
				except:
					print "error making dir:", tmp_string

def formatFileString(x):
	if len(x) == 1:
		return x+'0'
	else:
		return x
