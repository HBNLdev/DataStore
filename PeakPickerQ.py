''' Qt version of HBNL Peak Picker

    Just run me with python3 in conda 'upgrade' environment.
'''

import os, sys
from PyQt4 import QtGui

class Picker(QtGui.QMainWindow):

    def __init__(s):
        super(Picker,s).__init__()
        s.setGeometry(50,50,900,750)
        s.setWindowTitle("HBNL Peak Picker")

        s.tabs = QtGui.QTabWidget()

        s.navTab = QtGui.QWidget()

        s.navLayout = QtGui.QVBoxLayout()
        s.directoryInput = QtGui.QLineEdit()
        s.startButton = QtGui.QPushButton("Start")

        s.navLayout.addWidget(s.directoryInput)
        s.navLayout.addWidget(s.startButton)
        s.navTab.setLayout(s.navLayout)

        s.pickTab = QtGui.QWidget()

        s.pickLayout = QtGui.QVBoxLayout()

        s.controls_1 = QtGui.QHBoxLayout()

        s.applyButton = QtGui.QPushButton("Apply")
        s.prevButton = QtGui.QPushButton("Prev")
        s.nextButton = QtGui.QPushButton("Next")

        s.controls_1.addWidget(s.applyButton)
        s.controls_1.addWidget(s.prevButton)
        s.controls_1.addWidget(s.nextButton)

        s.plotsHolder = QtGui.QLineEdit()
        s.plotsHolder.resize(500,500)

        s.pickLayout.addLayout(s.controls_1)
        s.pickLayout.addWidget(s.plotsHolder)

        s.pickTab.setLayout(s.pickLayout)

        s.tabs.addTab(s.navTab,"Navigate")
        s.tabs.addTab(s.pickTab,"Pick")

        s.setCentralWidget(s.tabs)

        s.show()

app = QtGui.QApplication(sys.argv)
GUI = Picker()
sys.exit(app.exec_())