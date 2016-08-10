''' Qt version of HBNL Peak Picker

    Just run me with python3 in conda 'upgrade' environment.
'''

import os, sys
from PyQt4 import QtGui, QtCore
Qt = QtCore.Qt
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

    show_only = ('X','Y')
    repick_modes = ('all','single')
    peak_choices = ['P1','P2','P3','P4','N1','N2','N3','N4']


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
    app_data['pick state'] = { 'case':None, 'peak':None,
                                'repick mode':repick_modes[0]}

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

        s.caseChooser = QtGui.QComboBox()
        s.peakChooser = QtGui.QComboBox()
        for peak in s.peak_choices:
            s.peakChooser.addItem(peak)
        s.pickButton = QtGui.QPushButton("Pick")

        s.peakControls = QtGui.QHBoxLayout()
        s.peakControls.addWidget(s.caseChooser)
        s.peakControls.addWidget(s.peakChooser)
        s.peakControls.addWidget(s.pickButton)
        all_single_label = QtGui.QLabel("repick mode:")
        all_single_label.setAlignment(Qt.AlignRight)
        s.pickModeToggle = QtGui.QPushButton(s.app_data['pick state']['repick mode'])
        s.peakControls.addWidget(all_single_label)
        s.peakControls.addWidget(s.pickModeToggle)

        s.pickButton.clicked.connect(s.pick_init)
        s.pickModeToggle.clicked.connect(s.mode_toggle)

        s.plotsGrid = pg.GraphicsLayoutWidget()#QtGui.QGridLayout()
        s.plotsGrid.ci.layout.setContentsMargins(0, 0, 0, 0)
        s.plotsGrid.ci.layout.setSpacing(0) 
        s.plots = {}
        s.plot_labels = {}

        s.load_file(initialize=True)
        s.caseChooser.clear()
        #plot_layout = [['a','b','c'],['d','e','f'],['g','h','i']]
        last_plot = None
        for rN,prow in enumerate(s.plot_desc):
            for cN,p_desc in enumerate(prow):
                if p_desc:
                    elec = p_desc['electrode']
                    plot = s.plotsGrid.addPlot(rN,cN)#,title=elec)

                    #plot.resize(300,250)
                    plot.vb.sigRangeChanged.connect(s.update_ranges)

                    s.plots[elec] = plot
                    if last_plot:
                        plot.setXLink(last_plot)
                        plot.setYLink(last_plot)
                    last_plot = plot


        s.pickLayout.addLayout(s.controls_1)
        s.pickLayout.addLayout(s.peakControls)
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

    def gather_info(s,exp):
        exp.extract_transforms_data()
        exp.extract_case_data()
        filter_info = 'Filter band: '+'{:4.3f}'.format(exp.transforms['hi_pass_filter']) \
                        + ' Hz to '+'{:4.1f}'.format(exp.transforms['lo_pass_filter'])+' Hz'
        case_info = ['cases: trials accepted/total']
        trials_str =''
        for caseN, caseD in exp.cases.items():
            trials_str += caseD['case_type']+': '+str(caseD['n_trials_accepted'])+'/' \
                        +str(caseD['n_trials']) +',   '
        trials_str = trials_str[:-4]
        case_info.append( trials_str )

        s.app_data['info'] = [ filter_info ] + case_info

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
            s.gather_info(eeg)

            print('Load  ', experiment,' n paths, ind: ', len(paths), ind, eeg.file_info)
            s.app_data['current experiment'] = experiment
            s.app_data['experiment cases'] = eeg.case_list
            s.caseChooser.clear()
            for case in eeg.case_list:
                s.caseChooser.addItem(case)
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
                    plot = s.plots[elec]
                    plot.clear()
                    for c_ind,case in enumerate(cases):
                        s.plots[elec].plot(x=s.current_data['times'],
                                y=s.current_data[elec+'_'+case], 
                                pen=s.plot_props['line colors'][c_ind] )
                    label = pg.TextItem(text=elec)
                    plot.addItem(label)                    
                    s.plot_labels[plot.vb] = label
                    s.adjust_label(plot.vb)

        s.peak_regions = {}

    def adjust_label(s,viewbox):

        if viewbox in s.plot_labels:
            label = s.plot_labels[viewbox]
            region = viewbox.getState()['viewRange']
            #print('adjust label',elec,region)
            label.setPos(region[0][0],region[1][1])

    def update_ranges(s):
        '''This scheme leads to a recursive mess.  Need a way to limit
        signals to only the axis being updated, and update afterward
        '''
        s.adjust_label(s.sender())

    def update_regions(s):
        
        region = s.sender().getRegion()
        case = s.app_data['pick state']['case']
        peak = s.app_data['pick state']['peak']

        if s.app_data['pick state']['repick mode'] == 'all':
            for reg in s.peak_regions:
                if reg[1] == case and reg[2] == peak:
                    s.peak_regions[reg].setRegion( region )

    def pick_init(s):

        case = s.caseChooser.currentText()
        peak = s.peakChooser.currentText()
        s.app_data['pick state']['case'] = case
        s.app_data['pick state']['peak'] = peak

        print('Pick init for ',case, peak)
        peak_center_ms = 100*int(peak[1])
        start_range = (peak_center_ms-75,peak_center_ms+75)

        for elec in [ p for p in s.plots if p not in s.show_only ]:
            region = pg.LinearRegionItem(values=start_range,movable=True)
            region.sigRegionChangeFinished.connect(s.update_regions)
            s.peak_regions[(elec,case,peak)] = region 
            s.plots[elec].addItem(region)

    def mode_toggle(s):
        current_mode = s.app_data['pick state']['repick mode']
        current_mode_i = s.repick_modes.index(current_mode)
        s.app_data['pick state']['repick mode'] = \
            s.repick_modes[(current_mode_i+1)%len(s.repick_modes)]
        s.pickModeToggle.setText(s.app_data['pick state']['repick mode'])

app = QtGui.QApplication(sys.argv)
GUI = Picker()
sys.exit(app.exec_())