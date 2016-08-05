''' Qt version of HBNL Peak Picker

    Just run me with python3 in conda 'upgrade' environment.
'''

import os, sys
from PyQt4 import QtGui
import pyqtgraph as pg

import numpy as np
import pandas as pd

import organization as O
import EEGdata




class Picker(QtGui.QMainWindow):

    init_files_by_exp = {'ant':'ant_0_a0_11111111_avg.h1', 
                    'vp3':'vp3_0_a0_11111111_avg.h1',
                    'aod':'aod_1_a1_11111111_avg.h1'}

    dir_paths_by_exp = { 'ant':['/processed_data/avg-h1-files/ant/l8-h003-t75-b125/suny/ns32-64/',
                                'ant_5_e1_40143015_avg.h1 ant_5_e1_40146034_avg.h1 ant_5_a1_40026180_avg.h1'],
                        'vp3':['/processed_data/avg-h1-files/vp3/l16-h003-t75-b125/suny/ns32-64/',
                                'vp3_5_a1_40021069_avg.h1 vp3_5_a1_40017006_avg.h1 vp3_5_a1_40026204_avg.h1'],
                        'aod':['/processed_data/avg-h1-files/aod/l16-h003-t75-b125/suny/ns32-64/',
                                'aod_6_a1_40021070_avg.h1 aod_6_a1_40021017_avg.h1 aod_6_a1_40017007_avg.h1']
                        }


    plot_props = {'width':180, 'height':110,
                 'extra_bottom_height':40, # for bottom row
                'min_border':4,
                'line colors':[(221,34,34),(102,221,102),(34,34,221),(34,221,34)]}


    app_data = {}

    app_data['display props'] = {'marker size':8,
                            'pick dash':[4,1],
                            'pick width':2,
                            'current color':'#ffbc00',
                            'picked color':'#886308',
                            'time range':[-10,850]
     } 

    user = ''
    if len(sys.argv) > 1:
        user = sys.argv[1]
    #userName for store path
    if '_' in user:
        userName = user.split('_')[1]
    else: userName = 'default'
    app_data['user'] = userName
    app_data['file paths'] = ['/processed_data/avg-h1-files/ant/l8-h003-t75-b125/suny/ns32-64/ant_5_e1_40143015_avg.h1']#os.path.join(os.path.dirname(__file__),init_files_by_exp['ant'] ) ]
    app_data['paths input'] = []

    app_data['file ind'] = 0

    def __init__(s):
        super(Picker,s).__init__()
        s.setGeometry(50,50,1200,750)
        s.setWindowTitle("HBNL Peak Picker ("+s.user+")")

        s.buttons = {}

        s.tabs = QtGui.QTabWidget()

        ### Navigation ###
        #temporary placeholders
        dir_files = s.dir_paths_by_exp['ant']

        s.navTab = QtGui.QWidget()

        s.navLayout = QtGui.QVBoxLayout()

        s.directoryLayout = QtGui.QHBoxLayout()
        directory_label = QtGui.QLabel('Directory:')
        s.directoryInput = QtGui.QLineEdit(dir_files[0])
        s.directoryLayout.addWidget(directory_label)
        s.directoryLayout.addWidget(s.directoryInput)

        s.filesLayout = QtGui.QHBoxLayout()
        files_label = QtGui.QLabel('Files:')
        s.filesInput = QtGui.QLineEdit(dir_files[1])
        s.filesLayout.addWidget(files_label)
        s.filesLayout.addWidget(s.filesInput)
        s.startButton = QtGui.QPushButton("Start")
        s.startButton.clicked.connect(s.start_handler)

        s.navLayout.addLayout(s.directoryLayout)
        s.navLayout.addLayout(s.filesLayout)
        s.navLayout.addWidget(s.startButton)
        s.navTab.setLayout(s.navLayout)

        ### Picking ###
        s.pickTab = QtGui.QWidget()

        s.pickLayout = QtGui.QVBoxLayout()

        s.controls_1 = QtGui.QHBoxLayout()

        buttons_1 = [('Apply',None),('Prev',s.previous_file),
                    ('Next',s.next_file),('Clear',None)]

        for label,handler in buttons_1:
            s.buttons[label] = QtGui.QPushButton(label)
            s.controls_1.addWidget(s.buttons[label])
            if handler:
                s.buttons[label].clicked.connect(handler)


        s.plotsGrid = pg.GraphicsLayoutWidget()#QtGui.QGridLayout()
        s.plots = {}

        s.load_file(initialize=True)
        #plot_layout = [['a','b','c'],['d','e','f'],['g','h','i']]

        for rN,prow in enumerate(s.plot_desc):
            for cN,p_desc in enumerate(prow):
                if p_desc:
                    elec = p_desc['electrode']
                    plot = s.plotsGrid.addPlot(rN,cN,title=elec)
                    #plot.resize(300,250)
                    s.plots[elec] = plot


        s.pickLayout.addLayout(s.controls_1)
        s.plotsScroll = QtGui.QScrollArea()
        s.plotsGrid.resize(1150,1800)
        #s.plotsScroll.setFixedWidth(1200)
        #s.plotsScroll.setFixedHeight(900)
        s.plotsScroll.setWidget(s.plotsGrid)
        s.pickLayout.addWidget(s.plotsScroll) #s.plotsGrid)

        s.pickTab.setLayout(s.pickLayout)

        s.tabs.addTab(s.navTab,"Navigate")
        s.tabs.addTab(s.pickTab,"Pick")

        s.setCentralWidget(s.tabs)

        s.show()

    def start_handler(s,signal):
        print('Start:',signal)
        directory = s.directoryInput.text().strip()
        
        files = s.filesInput.text().split(' ')
        files = [f for f in files if '.h1' in f]
        paths = [ os.path.join(directory,f) for f in files ]
        if len(s.app_data['paths input']) > 0 and paths == s.app_data['paths input'][-1]:
            return
        s.app_data['paths input'].append(paths)

        s.app_data['file paths'] = paths
        s.app_data['file ind'] = -1

        s.next_file()

    def next_file(s):
        print('Next')
        s.load_file(next=True)

    def previous_file(s):
        print('Previous')
        if s.app_data['file ind'] > 0:
            s.app_data['file ind'] -= 1
            s.load_file()

    def load_file(s,next=False, initialize=False):

        paths = s.app_data['file paths']

   
        if next: 
            if s.app_data['file ind'] < len(paths)-1:
                s.app_data['file ind'] += 1
            else:
                print('already on last file')
                return

        ind = s.app_data['file ind']
        print('load file: #',ind, paths)
        if ind < len(paths):
            eeg = EEGdata.avgh1( paths[ind] )
            experiment = eeg.file_info['experiment']
            print('Load  ', experiment,' n paths, ind: ', len(paths), ind, eeg.file_info)
            s.app_data['current experiment'] = experiment
            #expD = s.app_data[experiment]
            # reversing initialize flag for testing
            data_sourceD, peak_sourcesD = eeg.make_data_sources(empty_flag=initialize, 
                                            time_range=s.app_data['display props']['time range'])
            s.current_data = data_sourceD

            s.plot_desc = eeg.selected_cases_by_channel(mode='server',style='layout')

            if not initialize:
                chan_cases = [ entry for entry in s.current_data if '_' in entry and 'BLANK' not in entry]
                chans = set([cc.split('_')[0] for cc in chan_cases])
                cases = set([cc.split('_')[1] for cc in chan_cases])
                for elec in chans:
                    s.plots[elec].clear()
                    for c_ind,case in enumerate(cases):
                        s.plots[elec].plot(x=s.current_data['times'],
                                y=s.current_data[elec+'_'+case], 
                                pen=s.plot_props['line colors'][c_ind] )


app = QtGui.QApplication(sys.argv)
GUI = Picker()
sys.exit(app.exec_())