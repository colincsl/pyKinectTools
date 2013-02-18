import os

''' Create a folder if it doesn't exist '''
def createDirectory(new_dir):
	if not os.path.isdir(new_dir):
		for p in xrange(4, len(new_dir.split("/"))+1):                         
			try:
					os.mkdir("/".join(new_dir.split('/')[0:p])) 
			except:
					print "error making dir", 
