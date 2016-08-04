''' Qt version of HBNL Peak Picker

    Just run me with python3 in conda 'upgrade' environment.
'''

import os, sys
from PyQt4 import QtGui
import pyqtgraph as pg

class Picker(QtGui.QMainWindow):


    def __init__(s):
        super(Picker,s).__init__()
        s.setGeometry(50,50,900,750)
        s.setWindowTitle("HBNL Peak Picker")

        s.buttons = {}

        s.tabs = QtGui.QTabWidget()

        s.navTab = QtGui.QWidget()

        s.navLayout = QtGui.QVBoxLayout()
        s.directoryInput = QtGui.QLineEdit()
        s.fileInput = QtGui.QLineEdit()
        s.startButton = QtGui.QPushButton("Start")

        s.navLayout.addWidget(s.directoryInput)
        s.navLayout.addWidget(s.fileInput)
        s.navLayout.addWidget(s.startButton)
        s.navTab.setLayout(s.navLayout)

        s.pickTab = QtGui.QWidget()

        s.pickLayout = QtGui.QVBoxLayout()

        s.controls_1 = QtGui.QHBoxLayout()

        buttons_1 = ['Apply','Prev','Next','Clear',]

        for label in buttons_1:
            s.buttons[label] = QtGui.QPushButton(label)
            s.controls_1.addWidget(s.buttons[label])

        s.plotsGrid = pg.GraphicsLayoutWidget()#QtGui.QGridLayout()
        s.plots = {}
        plot_layout = [['a','b','c'],['d','e','f'],['g','h','i']]
        for rN,prow in enumerate(plot_layout):
            for cN,p_title in enumerate(prow):
                plot = s.plotsGrid.addPlot(rN,cN)
                s.plots[p_title] = plot

        s.pickLayout.addLayout(s.controls_1)
        s.pickLayout.addWidget(s.plotsGrid)

        s.pickTab.setLayout(s.pickLayout)

        s.tabs.addTab(s.navTab,"Navigate")
        s.tabs.addTab(s.pickTab,"Pick")

        s.setCentralWidget(s.tabs)

        s.show()

app = QtGui.QApplication(sys.argv)
GUI = Picker()
sys.exit(app.exec_())