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

    # def add_buttons(s, layout, labels_slots):
    #     for label, handler in labels_slots:
    #         button = QtGui.QPushButton(label)
    #         layout.addWidget(button)
    #         if handler:
    #             button.connect(handler)
    #         s.buttons[label] = button

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

    ignore = ['BLANK']
    show_only = ['X','Y']
    repick_modes = ('all','single')
    peak_choices = ['P1','P2','P3','P4','N1','N2','N3','N4']


    plot_props = {'width':180, 'height':110,
                 'extra_bottom_height':40, # for bottom row
                'min_border':4,
                'line colors':[(221,34,34),(102,221,102),(34,34,221),(221,34,221)]}


    app_data = {}
    

    app_data['display props'] = {'marker size':8,
                            'pick dash':[4,1],
                            'pick width':2,
                            'current color':'#ffbc00',
                            'picked color':'#886308',
                            'time range':[-10,850],
                            'bar length':np.float64(1),
                            'pick region':(80,80,80,50),
                            'background':(40, 40, 40),
                            'foreground':(135, 135, 135),
                            'main position':(50,50,1200,750),
                            'zoom position':(300,200,750,600)
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
    def dummy_slot(s,ev):
        print('Dummy slot')

    def __init__(s):
        DProps = s.app_data['display props']
        super(Picker,s).__init__()
        s.setGeometry(*DProps['main position'])
        s.setWindowTitle("HBNL Peak Picker ("+s.user+")      ")



        pg.setConfigOption('background', DProps['background'])
        pg.setConfigOption('foreground', DProps['foreground'])

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



        # Display controls
        s.dispLayout = QtGui.QHBoxLayout()
        s.dispLayout.setAlignment(Qt.AlignLeft)
        s.casesLayout = QtGui.QHBoxLayout()
        s.casesLayout.setAlignment(Qt.AlignLeft)

        disp_label = QtGui.QLabel("Display:")
        s.pickRegionToggle = QtGui.QCheckBox("regions")
        s.pickRegionToggle.setChecked(True)
        s.peakMarkerToggle = QtGui.QCheckBox("peaks")
        s.peakMarkerToggle.setChecked(True)
        s.dispLayout.addWidget(disp_label)
        s.dispLayout.addWidget(s.pickRegionToggle)
        s.dispLayout.addWidget(s.peakMarkerToggle)
        disp_cases_label = QtGui.QLabel("cases:")
        s.casesLayout.addWidget(disp_cases_label)
        s.controls_1.addLayout(s.dispLayout)
        s.controls_1.addLayout(s.casesLayout)

        # Picks display
        s.stateLayout = QtGui.QHBoxLayout()
        s.stateLayout.setAlignment(Qt.AlignRight)
        state_label = QtGui.QLabel("Picked:")
        s.stateInfo = QtGui.QLabel()
        s.stateInfo.setAlignment(Qt.AlignLeft)
        s.stateLayout.addWidget(state_label)
        s.stateLayout.addWidget(s.stateInfo)
        s.controls_1.addLayout(s.stateLayout)

        buttons_1 = [('Save',s.save_mt),
                    ('Prev', s.previous_file),
                     ('Next', s.next_file)]

        # s.add_buttons(s.controls_1,buttons_1)

        for label, handler in buttons_1:
            s.buttons[label] = QtGui.QPushButton(label)
            s.controls_1.addWidget(s.buttons[label])
            if handler:
                s.buttons[label].clicked.connect(handler)

        s.case_toggles = {} # checkboxes populated for each file

        s.pickRegionToggle.stateChanged.connect(s.toggle_regions)
        s.peakMarkerToggle.stateChanged.connect(s.toggle_peaks)

        s.caseChooser = QtGui.QComboBox()
        s.peakChooser = QtGui.QComboBox()
        for peak in s.peak_choices:
            s.peakChooser.addItem(peak)

        s.peakControls = QtGui.QHBoxLayout()
        s.peakControls.addWidget(s.caseChooser)
        s.peakControls.addWidget(s.peakChooser)

        pick_buttons = [('Pick',s.pick_init),('Apply',s.apply_selections)]
        #s.add_buttons(s.peakControls,pick_buttons)
        for label,handler in pick_buttons:
            s.buttons[label] = QtGui.QPushButton(label)
            s.peakControls.addWidget(s.buttons[label])
            if handler:
                s.buttons[label].clicked.connect(handler)

        all_single_label = QtGui.QLabel("repick mode:")
        all_single_label.setAlignment(Qt.AlignRight)
        s.pickModeToggle = QtGui.QPushButton(s.app_data['pick state']['repick mode'])
        s.peakControls.addWidget(all_single_label)
        s.peakControls.addWidget(s.pickModeToggle)

        s.pickModeToggle.clicked.connect(s.mode_toggle)

        s.plotsGrid = pg.GraphicsLayoutWidget()#QtGui.QGridLayout()
        s.zoomDialog = QtGui.QDialog(s)
        s.zoomGW = pg.GraphicsWindow(parent=s.zoomDialog)
        s.zoomPlot = s.zoomGW.addPlot()#pg.PlotWidget(parent=s.zoomDialog)
        s.plotsGrid.ci.layout.setContentsMargins(0, 0, 0, 0)
        s.plotsGrid.ci.layout.setSpacing(0)
        s.pick_electrodes = []
        s.plots = {}
        s.plot_labels = {}
        s.curves = {}
        s.vb_map = {}

        s.peak_data = {}

        s.app_data['picks'] = set()

        s.load_file(initialize=True)
        s.caseChooser.clear()

        prev_plot = None
        s.legend_plot = None
        for rN,prow in enumerate(s.plot_desc):
            if rN == 0:
                for cN,p_desc in enumerate(prow):
                    s.plotsGrid.addLabel('',rN,cN)
            for cN,p_desc in enumerate(prow):
                if p_desc:
                    elec = p_desc['electrode']
                    if elec not in s.ignore + s.show_only:
                        s.pick_electrodes.append(elec)
                    plot = s.plotsGrid.addPlot(rN+1,cN)#,title=elec)
                    s.proxyMouse = pg.SignalProxy(plot.scene().sigMouseClicked, slot=s.show_zoom_plot)
                    #plot.resize(300,250)
                    plot.vb.sigRangeChanged.connect(s.update_ranges)
                    s.vb_map[plot.vb] = elec
                    s.plots[elec] = plot
                    if prev_plot:
                        plot.setXLink(prev_plot)
                        plot.setYLink(prev_plot)
                    prev_plot = plot
                elif not s.legend_plot:
                    s.legend_plot = s.plotsGrid.addPlot(rN+1,cN)
                    s.legend_plot.getAxis('left').hide()
                    s.legend_plot.getAxis('bottom').hide()



        s.pickLayout.addLayout(s.controls_1)
        s.pickLayout.addLayout(s.peakControls)
        s.plotsScroll = QtGui.QScrollArea()
        s.plotsGrid.resize(1150,1800)
        s.plotsScroll.setWidget(s.plotsGrid)
        s.pickLayout.addWidget(s.plotsScroll)

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
            s.eeg = eeg
            experiment = eeg.file_info['experiment']
            s.gather_info(eeg)
            cases = eeg.case_list
            chans = eeg.electrodes

            print('Load  ', experiment,' n paths, ind: ', len(paths), ind, eeg.file_info)
            s.app_data['current experiment'] = experiment
            s.app_data['experiment cases'] = eeg.case_list

            s.caseChooser.clear()
            print('removing case toggles', s.case_toggles)
            for case,toggle in s.case_toggles.items():
                toggle.stateChanged.disconnect(s.toggle_case)
                toggle.setParent(None)#s.casesLayout.removeWidget(toggle)
            s.case_toggles = {}
            for case in cases:
                s.caseChooser.addItem(case)
                case_toggle = QtGui.QCheckBox(case)
                case_toggle.setChecked(True)
                case_toggle.stateChanged.connect(s.toggle_case)
                s.casesLayout.addWidget(case_toggle)
                s.case_toggles[case] = case_toggle
            #expD = s.app_data[experiment]
            # reversing initialize flag for testing
            data_sourceD, peak_sourcesD = eeg.make_data_sources(empty_flag=initialize, 
                                    time_range=s.app_data['display props']['time range'])
            s.current_data = data_sourceD

            s.plot_desc = eeg.selected_cases_by_channel(mode='server',style='layout')

            if not initialize:
                s.legend_plot.clear()
                s.legend_plot.addLegend(size=(60,40),offset=(-60,0) )
                for c_ind,case in enumerate(cases):
                    s.legend_plot.plot(x=[-5,-4],y=[-20,-20],
                        pen=s.plot_props['line colors'][c_ind],
                        name=case)
                    s.legend_plot.vb.setRange(xRange=[0,1],yRange=[0,1])

                for elec in [ ch for ch in chans if ch not in s.ignore ]:
                    plot = s.plots[elec]
                    plot.clear()
                    for c_ind,case in enumerate(cases):
                        s.curves[(elec,case)] = s.plots[elec].plot(x=s.current_data['times'],
                                y=s.current_data[elec+'_'+case], 
                                pen=s.plot_props['line colors'][c_ind],
                                name=case )
                    label = pg.TextItem(text=elec,anchor=(0,0.2))
                    plot.addItem(label)                    
                    s.plot_labels[plot.vb] = label
                    s.adjust_label(plot.vb)
                #print(dir(label))


        s.peak_markers = {}
        s.region_case_peaks = {}

        #check for and load old mt
        picked_file_exists = s.eeg.extract_mt_data()
        if picked_file_exists:
            print('Already picked')
            for cs_pk in s.eeg.case_peaks:
                s.app_data['picks'].add( (s.eeg.num_case_map[int(cs_pk[0])], cs_pk[1]  ) )

            for elec in s.pick_electrodes:
                for cs_pk in s.eeg.case_peaks:
                    #print(elec,cs_pk,  s.eeg.get_peak_data( elec, cs_pk[0], cs_pk[1] ) )
                    s.peak_data[ (elec,cs_pk[0],cs_pk[1]) ] = \
                        tuple([float(v) for v in s.eeg.get_peak_data( elec, cs_pk[0], cs_pk[1] ) ])
            print('picks',s.app_data['picks'])
            s.show_peaks()
            s.show_state()

        s.pick_regions = {}
        s.pick_region_labels = {}

    def save_mt(s):
        print('Save mt')
        picked_cp = s.app_data['picks']
        cases_Ns = list(set([(cp[0],s.eeg.case_num_map[cp[0]]) for cp in picked_cp]))
        peaks = list(set([cp[1] for cp in picked_cp]))

        n_cases = len(cases_Ns)
        n_chans = 61 # only core 61 chans
        n_peaks = len(peaks)
        amps = np.empty( (n_peaks, n_chans, n_cases) )
        lats = np.empty( (n_peaks, n_chans, n_cases) )
        for icase, case_N in enumerate(cases_Ns):
            case_name = case_N[0]
            for ichan, chan in enumerate(s.eeg.electrodes_61): # only core 61 chans
                case_peaks = [ cp[1] for cp in picked_cp if cp[0]==case_name ]
                case_peaks.sort()
                for peak in case_peaks:
                    ipeak = peaks.index(peak)
                    amp_lat =  s.peak_data[(chan,case_name,peak)]
                    amps[ipeak, ichan, icase] = amp_lat[0]
                    lats[ipeak, ichan, icase] = amp_lat[1]

        # reshape into 1d arrays
        amps1d = amps.ravel('F')
        lats1d = lats.ravel('F')

        # build mt text (makes default output location), write to a test location
        s.eeg.build_mt([cn[1] for cn in cases_Ns], peaks, amps1d, lats1d)
        #print(eeg.mt)
        test_dir = os.path.join('/active_projects/test', s.app_data['user']+'Q' )
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        fullpath = os.path.join( test_dir, s.eeg.mt_name )
        of = open( fullpath, 'w' )
        of.write( s.eeg.mt )
        of.close()

        #exp['status display'].text = 'Saved .mt file to' + test_dir
        print('Saved', fullpath)
            



    def adjust_label(s,viewbox):

        if viewbox in s.plot_labels:
            label = s.plot_labels[viewbox]
            region = viewbox.getState()['viewRange']
            label.setPos(region[0][0],region[1][1])

    def update_ranges(s):
        ''' Called when axis ranges change 
        '''
        s.adjust_label(s.sender())
        s.update_region_label_positions()

    def update_pick_regions(s):
        
        sender = s.sender()
        region = sender.getRegion()
        elec, case, peak = s.region_case_peaks[sender]

        for el_cs_pk,reg in s.pick_regions.items():
            if s.app_data['pick state']['repick mode'] == 'all' or el_cs_pk == (elec,case,peak):
                if el_cs_pk[1] == case and el_cs_pk[2] == peak:
                    reg.setRegion( region )

        s.update_region_label_positions()

    def pick_init(s):
        ffs = ['Verdana', 'Arial', 'Helvetica', 'sans-serif', 'Times']
        ffind = 0
        case = s.caseChooser.currentText()
        peak = s.peakChooser.currentText()
        s.app_data['pick state']['case'] = case
        s.app_data['pick state']['peak'] = peak

        print('Pick init for ',case, peak)

        s.app_data['picks'].add( (case,peak) )

        pick_case_peaks = set([(ecp[1],ecp[2]) for ecp in s.pick_regions])

        if (case,peak) not in pick_case_peaks:
            peak_center_ms = 100*int(peak[1])
            start_range = (peak_center_ms-75,peak_center_ms+75)
            for elec in [ p for p in s.plots if p not in s.show_only ]:
                region = pg.LinearRegionItem(values=start_range,movable=True,
                        brush=s.app_data['display props']['pick region'])
                region.sigRegionChangeFinished.connect(s.update_pick_regions)
                #ffind +=1
                #ff = ffs[ ffind%len(ffs) ]
                # region_label = pg.TextItem(
                #     html='<div style="color: #FF0; font-size: 7pt; font-family: '+ff+'">'+peak+'-'+ff+'</div>',
                #     anchor=(-0.025,0.2))
                region_label = pg.TextItem(
                    html='<div style="color: #FF0; font-size: 7pt; font-family: Helvetica">'+peak+'</div>',
                    anchor=(-0.025,0.2))

                s.pick_regions[(elec,case,peak)] = region
                s.pick_region_labels[(elec,case,peak)] = region_label
                s.region_case_peaks[region] = (elec,case,peak)
                s.plots[elec].addItem(region)
                s.plots[elec].addItem(region_label)
        s.update_region_label_positions()
        
        for disp_case in s.eeg.case_list:
            s.set_case_display(disp_case,disp_case == case)



        print('pick_init finish')

    def update_region_label_positions(s):
        for reg_key, region in s.pick_regions.items():
            elec = reg_key[0]
            vb_range = s.plots[elec].vb.getState()['viewRange']
            start_fin = region.getRegion()
            s.pick_region_labels[reg_key].setPos( start_fin[0],vb_range[1][1] )

    def show_state(s):
        picks = s.app_data['picks']
        cases = s.eeg.case_list
        state_string = ''
        for case in cases:
            state_string += case+':['
            state_string += ','.join([cp[1] for cp in picks if cp[0]==case])+'] '

        s.stateInfo.setText(state_string)

    def show_zoom_plot(s,ev):
        print('zoom_plot',ev)
        if ev[0].button() == 1 and ev[0].currentItem in s.vb_map:
            elec = s.vb_map[ ev[0].currentItem ]
            Pstate = s.app_data['pick state']
            #ev[0].accept()

            print(elec)

            if Pstate['case']:
                s.zoomDialog.setGeometry(*s.app_data['display props']['zoom position'])
                s.zoomDialog.setWindowTitle(elec + ' - '+ Pstate['peak']+'     ')
                
                c_ind = s.eeg.case_list.index(Pstate['case'])
                s.zoomPlot.clear()
                s.zoomCurve = s.zoomPlot.plot(x=s.current_data['times'],
                                    y=s.current_data[elec+'_'+Pstate['case']], 
                                    pen=s.plot_props['line colors'][c_ind],
                                    name=Pstate['case'] )
                #Need a custom axis linking mechanism
                #s.zoomPlot.setXLink(s.plots[elec])
                #s.zoomPlot.setYLink(s.plots[elec])
                small_region = s.pick_regions[(elec,Pstate['case'],Pstate['peak'])]
                start_fin = small_region.getRegion()
                region = pg.LinearRegionItem(values=start_fin,movable=True,
                                brush=s.app_data['display props']['pick region'])
                region.sigRegionChangeFinished.connect(s.update_pick_regions)
                s.region_case_peaks[region] = (elec,Pstate['case'],Pstate['peak'])
                s.zoomPlot.addItem(region)
                s.zoomDialog.show()
                # set apply on close and relink axes
            #print(dir(s.zoomDialog))

    def toggle_regions(s):
        checked = s.sender().isChecked()
        for el_cs_pk,reg in s.pick_regions.items():
            reg.setVisible(checked) 
            s.pick_region_labels[el_cs_pk].setVisible(checked)

    def toggle_peaks(s):
        checked = s.sender().isChecked()
        for el_cs_pk,mark in s.peak_markers.items():
            mark.setVisible(checked)

    def toggle_case(s):
        sender = s.sender()
        case = sender.text()
        print(case)
        checked = sender.isChecked()
        s.set_case_display(case,checked)
        #for el_cs in [ec for ec in s.curves if ec[1]==case]:
        #    s.curves[el_cs].setVisible(checked)

    def set_case_display(s,case,state):

        s.case_toggles[case].setChecked(state)
        for el_cs in [ec for ec in s.curves if ec[1] == case]:
            s.curves[el_cs].setVisible(state)
        for el_cs_pk in [ ecp for ecp in s.pick_regions if ecp[1] == case ]:
            s.pick_regions[el_cs_pk].setVisible(state)
            s.pick_region_labels[el_cs_pk].setVisible(state)

        for el_cs_pk in [ ecp for ecp in s.peak_markers if ecp[1] == case ]:
            s.peak_markers[el_cs_pk].setVisible(state)

    def mode_toggle(s):
        current_mode = s.app_data['pick state']['repick mode']
        current_mode_i = s.repick_modes.index(current_mode)
        s.app_data['pick state']['repick mode'] = \
            s.repick_modes[(current_mode_i+1)%len(s.repick_modes)]
        s.pickModeToggle.setText(s.app_data['pick state']['repick mode'])

    
    def show_peaks(s):

        bar_len = s.app_data['display props']['bar length']

        for el_cs_pk, amp_lat in s.peak_data.items():
            if el_cs_pk in s.peak_markers:
                s.plots[el_cs_pk[0]].removeItem(s.peak_markers[el_cs_pk])

            marker = pg.ErrorBarItem(x=[amp_lat[1]],y=[amp_lat[0]],
                top=bar_len,bottom=bar_len,beam=0,pen=(255,255,255))
            s.peak_markers[el_cs_pk] = marker
            s.plots[el_cs_pk[0]].addItem(marker)

    def apply_selections(s):
        case = s.app_data['pick state']['case']
        peak = s.app_data['pick state']['peak']
        polarity = peak[0].lower()
        starts = []
        finishes = []
        elecs = []

        for elec_case_peak in s.pick_regions:
            if elec_case_peak[1] == case and elec_case_peak[2] == peak:
                region = s.pick_regions[elec_case_peak]
                start_finish = region.getRegion()
                elecs.append(elec_case_peak[0])
                starts.append(start_finish[0])
                finishes.append(start_finish[1])
        #print('starts:',starts)
        pval,pms = s.eeg.find_peaks(case,elecs,
            starts_ms=starts,ends_ms=finishes, polarity=polarity)
        for e_ind,elec in enumerate(elecs):
            # if (elec,case,peak) in s.peak_markers:
            #     s.plots[elec].removeItem(s.peak_markers[(elec,case,peak)])

            latency = pms[e_ind]
            amplitude = pval[e_ind]
            s.peak_data[ (elec,case,peak) ] = (amplitude, latency)

            # marker = pg.ErrorBarItem(x=[latency],y=[amplitude],
            #     top=bar_len,bottom=bar_len,beam=0,pen=(255,255,255))
            # s.peak_markers[(elec,case,peak)] = marker
            # s.plots[elec].addItem(marker)

        s.show_peaks()

        s.show_state()


app = QtGui.QApplication(sys.argv)
GUI = Picker()
sys.exit(app.exec_())