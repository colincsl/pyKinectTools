

## Example Qt app using python

import sys
from PyQt4 import QtGui


app = QtGui.QApplication(sys.argv)

w = QtGui.QWidget()
w.resize(250, 150)
w.move(300,300)
w.setWindowTitle("Simple")
w.show()


sys.exit(app.exec_())