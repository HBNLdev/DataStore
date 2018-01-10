''' Qt version of HBNL Peak Picker

    Just run me with python3 in conda 'upgrade' environment.
'''

import os
import sys
import subprocess
import pickle

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from datetime import datetime

from picker.EEGdata import avgh1

Qt = QtCore.Qt

#configuration of case display
# If an experiment is included, only specified cases will be shown with their aliases
default_case_display_aliases = {'ern':[('N50','N50'),('N10','N10'),('P10','P10'),('P50','P50')],
                                'cpt':[('T','Go'),('CN','Cue No-Go'),('N','No-Go')],
                                'gng':[('G','Go'),('NG','No-Go')]
                                }

def break_text(text,chars_per_line):

    lines = []; this_line = ''
    parts = text.split(' ')
    while len(parts):
        next_piece = parts.pop(0)
        if len(this_line) + len(next_piece) +1 > chars_per_line:
            lines.append(this_line)
            this_line = ''
        this_line += next_piece+' '
    lines.append(this_line)
    return '<br>'.join(lines)


class Picker(QtGui.QMainWindow):
    ''' main GUI '''
    module_path = os.path.split(__file__)[0]

    init_files_by_exp = {'ant': 'ant_0_a0_11111111_avg.h1',
                         'vp3': 'vp3_0_a0_11111111_avg.h1',
                         'aod': 'aod_1_a1_11111111_avg.h1'}

    dir_paths_by_exp = {'ant': ['/processed_data/avg-h1-files/ant/l8-h003-t75-b125/suny/ns32-64/',
                                'ant_5_e1_40143015_avg.h1 ant_5_e1_40146034_avg.h1 ant_5_a1_40026180_avg.h1'],
                        'vp3': ['/processed_data/avg-h1-files/vp3/l16-h003-t75-b125/suny/ns32-64/',
                                'vp3_5_a1_40021069_avg.h1 vp3_5_a1_40017006_avg.h1 vp3_5_a1_40026204_avg.h1'],
                        'aod': ['/processed_data/avg-h1-files/aod/l16-h003-t75-b125/suny/ns32-64/',
                                'aod_6_a1_40021070_avg.h1 aod_6_a1_40021017_avg.h1 aod_6_a1_40017007_avg.h1']
                        }
    initD = '/active_projects/test/testQ/avgh1s'#s.dir_paths_by_exp['ant'][0]
    initFs = 'cpt_4_f1_40719005_avg.h1 ern_7_b1_40001008_avg.h1 gng_2_b1_40355069_avg.h1' +\
                ' ans_5_f1_40293005_avg.h1 ant_5_a1_49403009_avg.h1 stp_3_e1_40277003_avg.h1'+\
                ' aod_6_a1_40063010_avg.h1 err_8_a1_40251004_avg.h1 vp3_5_a1_40017006_avg.h1'+\
                ' cas_1_e1_40701003_avg.h1'
    debug_dir = '/active_projects/programs/picker/debug'
    temp_store_dir = '/active_projects/picker/store'
#s.dir_paths_by_exp['ant'][1]

    ignore = ['BLANK']
    show_only = ['X', 'Y']
    repick_modes = ('all', 'single')
    peak_choices = ['P1', 'P2', 'P3', 'P4', 'N1', 'N2', 'N3', 'N4']

    plot_props = {'width': 233, 'height': 102,
                  'extra_bottom_height': 40,  # for bottom row
                  'min_border': 4,
                  'line colors': [(221, 34, 34),  # red
                                  (102, 221, 102),  # green
                                  (55, 160, 255),  # light blue
                                  (255, 200, 20),  # orange
                                  (200, 255, 40),  # yellow-green
                                  (20, 255, 200),  # blue green
                                  (160, 0, 188),  # gray
                                  (221, 34, 221),  # magentagits
				                  (225,225,225), # light gray
                                  ],
                  'XY gridlines': ([0, 200, 400, 600, 800], [0]),
                  'grid color': '#555',
                  'label size': 20}

    app_data = {'display props': {'marker size': 8,
                                  'pick dash': [4, 1],
                                  'pick width': 2,
                                  'current color': '#ffbc00',
                                  'picked color': '#886308',
                                  'time range': [-10, 850],
                                  'bar length': np.float64(1.25),
                                  'pick region': (80, 80, 80, 50),
                                  'background': (0, 0, 0),
                                  'foreground': (135, 135, 135),
                                  'main position': (50, 50, 1200, 942),
                                  'zoom position': [300, 200, 780, 650],
                                  }}

    region_label_html = '<div style="color: #FF0; font-size: 7pt; font-family: Helvetica">__PEAK__</div>'

    user = ''
    if len(sys.argv) > 1:
        user = sys.argv[1]
    # userName for store path
    userName = user
    if '_' in user:
        userName = user.split('_')[1]

    if userName == '':
        userName = 'default'

    app_data['user'] = userName

    if len(sys.argv) > 2:
        try:
            app_data['debug'] = int(sys.argv[2])
        except:
            print('bad value for debug flag: ', sys.argv[2], ' should be int, setting to 0')
            app_data['debug'] = 0

 
    app_data['case display'] = default_case_display_aliases #update for custom here

    app_data['file paths'] = [
        '/processed_data/avg-h1-files/ant/l8-h003-t75-b125/suny/ns32-64/ant_5_e1_40143015_avg.h1']
    # os.path.join(os.path.dirname(__file__),init_files_by_exp['ant'] ) ]
    app_data['paths input'] = []

    app_data['status history'] = []

    app_data['file ind'] = 0
    app_data['pick state'] = {'case': None, 'peak': None,
                              'repick mode': repick_modes[0]}
    app_data['zoom electrode'] = None
    app_data['regions by filepath'] = {}


    def __init__(s):
        ''' the init lays out all of the GUI elements, connects interactive elements to call-backs '''

        DProps = s.app_data['display props']
        super(Picker, s).__init__()
        s.setGeometry(*DProps['main position'])
        s.setWindowTitle("HBNL Peak Picker (" + s.user + ")      ")

        s.app_data['debug path'] = \
            os.path.join(s.debug_dir,s.user+'_'+str(int(datetime.now().timestamp()))+'.log')

        pg.setConfigOption('background', DProps['background'])
        pg.setConfigOption('foreground', DProps['foreground'])

        s.buttons = {}

        s.tabs = QtGui.QTabWidget()

        ### Navigation ###
        # temporary placeholders
        s.navTab = QtGui.QWidget()

        s.navLayout = QtGui.QVBoxLayout()

        s.directoryLayout = QtGui.QHBoxLayout()
        directory_label = QtGui.QLabel('Directory:')
        s.directoryInput = QtGui.QLineEdit(s.initD)#dir_files[0])
        s.directoryLayout.addWidget(directory_label)
        s.directoryLayout.addWidget(s.directoryInput)

        s.filesLayout = QtGui.QHBoxLayout()
        files_label = QtGui.QLabel('Files:')
        s.filesInput = QtGui.QLineEdit(s.initFs)
        s.filesLayout.addWidget(files_label)
        s.filesLayout.addWidget(s.filesInput)
        s.startButton = QtGui.QPushButton("Start")
        s.startButton.clicked.connect(s.start_handler)
        s.startStatus = QtGui.QLabel('waiting for start...')
        s.startStatus.setFixedHeight(150)
        s.startStatus.setWordWrap(True)

        s.navLayout.addLayout(s.directoryLayout)
        s.navLayout.addLayout(s.filesLayout)
        s.navLayout.addWidget(s.startButton)
        s.navLayout.addWidget(s.startStatus)
        s.navTab.setLayout(s.navLayout)

        ### Picking ###
        s.pickTab = QtGui.QWidget()

        s.pickLayout = QtGui.QVBoxLayout()

        s.dispNstatus = QtGui.QHBoxLayout()

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
        s.peakTopToggle = QtGui.QCheckBox("markers")
        s.peakTopToggle.setChecked(False)
        s.textToggle = QtGui.QCheckBox("values")
        s.textToggle.setChecked(True)
        s.dispLayout.addWidget(disp_label)
        s.dispLayout.addWidget(s.pickRegionToggle)
        s.dispLayout.addWidget(s.peakMarkerToggle)
        s.dispLayout.addWidget(s.peakTopToggle)
        s.dispLayout.addWidget(s.textToggle)
        disp_cases_label = QtGui.QLabel("cases:")
        s.casesLayout.addWidget(disp_cases_label)
        s.dispNstatus.addLayout(s.dispLayout)
        s.dispNstatus.addLayout(s.casesLayout)

        s.buttons['Rescale'] = QtGui.QPushButton('Rescale')  # , sizeHint=QtCore.QSize(60,25) )
        s.buttons['Rescale'].clicked.connect(s.rescale_yaxis)
        s.buttons['First31'] = QtGui.QPushButton('1-31')
        s.buttons['First31'].clicked.connect(s.scroll_to_top)
        s.buttons['Last31'] = QtGui.QPushButton('33-61')
        s.buttons['Last31'].clicked.connect(s.scroll_to_bottom)
        s.viewControlLayout = QtGui.QHBoxLayout()
        s.viewControlLayout.addWidget(s.buttons['Rescale'])
        s.viewControlLayout.addWidget(s.buttons['First31'])
        s.viewControlLayout.addWidget(s.buttons['Last31'])
        s.viewControlLayout.setAlignment(Qt.AlignLeft)
        s.dispNstatus.addLayout(s.viewControlLayout)

        # Picks display
        s.stateLayout = QtGui.QHBoxLayout()
        s.stateLayout.setAlignment(Qt.AlignLeft)
        state_label = QtGui.QLabel("Picked:")
        s.stateInfo = QtGui.QLabel()
        #s.stateInfo.setMaximumWidth( int(s.plot_props['width']*0.9) )
        font = QtGui.QFont()
        font.setFamily('Helvetica')
        font.setPointSize(7)
        s.stateInfo.setFont(font)
        s.stateInfo.setWordWrap(True)
        s.stateInfo.setAlignment(Qt.AlignLeft)
        s.stateLayout.addWidget(state_label)
        s.stateLayout.addWidget(s.stateInfo)
        s.dispNstatus.addLayout(s.stateLayout)

        s.caseToggles = {}  # checkboxes populated for each file
        s.zoomCaseToggles = {}

        s.pickRegionToggle.stateChanged.connect(s.toggle_regions)
        s.peakMarkerToggle.stateChanged.connect(s.toggle_peaks)
        s.peakTopToggle.stateChanged.connect(s.toggle_peak_tops)
        s.textToggle.stateChanged.connect(s.toggle_value_texts)

        pick_label = QtGui.QLabel("Pick:")
        s.caseChooser = QtGui.QComboBox()
        s.peakChooser = QtGui.QComboBox()
        for peak in s.peak_choices:
            s.peakChooser.addItem('  ' + peak + '  ')

        s.pickNavControls = QtGui.QHBoxLayout()
        s.pickNavControls.setAlignment(Qt.AlignLeft)
        s.pickNavControls.addWidget(pick_label)
        s.pickNavControls.addWidget(s.caseChooser)
        s.pickNavControls.addWidget(s.peakChooser)

        pick_buttons = [('Pick', s.pick_init), ('Apply', s.apply_selections),
                        ('Back', s.previous_apply), ('Forward', s.next_apply),
                        ('Fix', s.fix_peak),('Settings',s.settings)]
        # s.add_buttons(s.pickNavControls,pick_buttons)
        for label, handler in pick_buttons:
            s.buttons[label] = QtGui.QPushButton(label)
            s.pickNavControls.addWidget(s.buttons[label])
            if handler:
                s.buttons[label].clicked.connect(handler)

        all_single_label = QtGui.QLabel("repick mode:")
        all_single_label.setAlignment(Qt.AlignRight)
        s.pickModeToggle = QtGui.QPushButton(s.app_data['pick state']['repick mode'])
        s.pickNavControls.addWidget(all_single_label)
        s.pickNavControls.addWidget(s.pickModeToggle)
        spacer = QtGui.QSpacerItem(40, 1)
        s.pickNavControls.addItem(spacer)

        s.pickModeToggle.clicked.connect(s.mode_toggle)

        nav_buttons = [('Save', s.save),
                       ('Prev', s.previous_file),
                       ('Next', s.next_file)]

        s.navLayout = QtGui.QHBoxLayout()
        for label, handler in nav_buttons:
            s.buttons[label] = QtGui.QPushButton(label)
            s.navLayout.addWidget(s.buttons[label])
            if handler:
                s.buttons[label].clicked.connect(handler)

        s.pickNavControls.addLayout(s.navLayout)

        s.plotsGrid = pg.GraphicsLayoutWidget()  # QtGui.QGridLayout()
        s.zoomDialog = QtGui.QDialog(s)
        s.zoomDialog.closeEvent = s.zoom_close
        s.zoomDialog._open = False
        s.zoomLayout = QtGui.QVBoxLayout()
        s.zoomDialog.setLayout(s.zoomLayout)
        s.zoomControls = QtGui.QHBoxLayout()
        s.zoomLayout.addLayout(s.zoomControls)
        s.zoomPlotWidget = QtGui.QWidget()
        s.zoomLayout.addWidget(s.zoomPlotWidget)
        s.zoomGW = pg.GraphicsWindow(parent=s.zoomPlotWidget)  # zoomDialogWidget)
        s.zoomPlot = s.zoomGW.addPlot()  # pg.PlotWidget(parent=s.zoomDialog)
        zoom_region = pg.LinearRegionItem(values=[-0.01, -0.009], movable=True,
                                             brush=s.app_data['display props']['pick region'])
        zoom_region.setVisible(False)
        s.zoomPlot.addItem(zoom_region)
        s.pick_regions = { 'zoom': zoom_region }
        s.region_elecs = { zoom_region: 'zoom' }

        s.fixDialog = QtGui.QDialog(s)
        s.fixLayout = QtGui.QVBoxLayout()
        s.fixLabel = QtGui.QLabel('Case and Peak to fix:')
        s.fixDialog.setLayout(s.fixLayout)
        s.fixCase = pg.ComboBox()
        s.fixCase.currentIndexChanged.connect(s.choose_fix_case)
        s.oldPeak = pg.ComboBox()
        s.oldPeak.currentIndexChanged.connect(s.choose_old_peak)
        s.removePeak = QtGui.QPushButton('Remove Peak')
        s.removePeak.clicked.connect(s.remove_peak)
        s.renameLabel = QtGui.QLabel('or Change to:')
        s.newPeak = pg.ComboBox()
        s.applyChange = QtGui.QPushButton('Apply Change')
        s.applyChange.clicked.connect(s.apply_peak_change)
        [s.fixLayout.addWidget(w) for w in
         [s.fixLabel, s.fixCase, s.oldPeak, s.removePeak, 
                        s.renameLabel, s.newPeak, s.applyChange]]

        s.settingsDialog = QtGui.QDialog(s)
        s.settingsLayout = QtGui.QVBoxLayout()
        s.settingsDialog.setLayout(s.settingsLayout)
        s.useMainCasesToggle = QtGui.QCheckBox("zoom follows main cases display")
        s.useMainCasesToggle.setChecked(False)
        [s.settingsLayout.addWidget(w) for w in 
         [ s.useMainCasesToggle ] ]

        s.plotsGrid.ci.layout.setContentsMargins(0, 0, 0, 0)
        s.plotsGrid.ci.layout.setSpacing(0)

        s.pick_electrodes = []
        s.plots = {}
        s.plot_labels = {}
        s.plot_texts = {}
        s.curves = {}
        s.zoom_curves = {}
        s.vb_map = {}
        s.edge_notified_plots = set()

        s.load_file(initialize=True)
        s.caseChooser.clear()

        prev_plot = None
        s.legend_plot = None
        s.status_plot = None
        n_rows = len(s.plot_desc)
        n_columns = len(s.plot_desc[0])
        for rN, prow in enumerate(s.plot_desc):
            if rN == 0:
                for cN, p_desc in enumerate(prow):
                    s.plotsGrid.addLabel('', rN, cN)
            for cN, p_desc in enumerate(prow):
                if p_desc:
                    elec = p_desc['electrode']
                    pick_elec = False
                    if elec not in s.ignore + s.show_only:
                        pick_elec = True
                        s.pick_electrodes.append(elec)
                    plot = s.plotsGrid.addPlot(rN + 1, cN)  # ,title=elec)
                    s.proxyMouse = pg.SignalProxy(plot.scene().sigMouseClicked, slot=s.show_zoom_plot)
                    # s.proxyScroll = pg.SignalProxy(plot.scene().sigMouseMoved, slot=s.scroll_handler)
                    # plot.resize(300,250)

                    s.vb_map[plot.vb] = elec
                    s.plots[elec] = plot
                    if prev_plot:
                        plot.setXLink(prev_plot)
                        plot.setYLink(prev_plot)
                    prev_plot = plot

                    if pick_elec: #elec in s.app_data['active channels']:
                        region = pg.LinearRegionItem(values=[-0.01,-0.009], movable=True,
                                             brush=s.app_data['display props']['pick region'])
                        region.setVisible(False)
                        region.sigRegionChanged.connect(s.update_region_label_position)
                        region.sigRegionChangeFinished.connect(s.update_pick_regions)

                        #plot.vb.sigRangeChanged.connect(s.update_ranges)

                        s.pick_regions[ elec ] = region
                        s.plots[elec].addItem(region)                    

                        s.region_elecs[region] = elec

                elif not s.legend_plot:
                    s.legend_plot = s.plotsGrid.addPlot(rN + 1, cN)
                    s.legend_plot.getAxis('left').hide()
                    s.legend_plot.getAxis('bottom').hide()
                elif not s.status_plot:
                    s.status_plot = s.plotsGrid.addPlot(rN + 1, cN)
                    s.status_plot.getAxis('left').hide()
                    s.status_plot.getAxis('bottom').hide()

        s.pickLayout.addLayout(s.pickNavControls)
        s.pickLayout.addLayout(s.dispNstatus)
        s.plotsScroll = QtGui.QScrollArea()
        s.plotsGrid.resize(n_columns * s.plot_props['width'], n_rows * s.plot_props['height'])
        s.plotsScroll.setWidget(s.plotsGrid)
        s.pickLayout.addWidget(s.plotsScroll)

        s.pickTab.setLayout(s.pickLayout)

        s.tabs.addTab(s.navTab, "Navigate")
        s.tabs.addTab(s.pickTab, "Pick")

        s.setCentralWidget(s.tabs)

        s.show()

    def scroll_to_top(s):
        vert_scroll = s.plotsScroll.verticalScrollBar()
        vert_scroll.setValue(1)

    def scroll_to_bottom(s):
        vert_scroll = s.plotsScroll.verticalScrollBar()
        vert_scroll.setValue(vert_scroll.maximum())

    def start_handler(s, signal):
        ''' Start button inside Navigate tab '''
        s.debug(['Start:', signal],2)
        directory = s.directoryInput.text().strip()
        s.app_data['working directory'] = directory

        files_raw = s.filesInput.text().split('.h1')
        s.debug(['files_raw: ', files_raw ],2 )
        files_h1 = [f.strip()+'.h1' for f in files_raw[:-1] ]
        files_ck = [ f for f in files_h1 if os.path.exists( os.path.join(directory, f) ) ]
        if len(files_ck) != len(files_h1):
            bad = set(files_h1).difference(files_ck)
            bfst = ''
            for f in bad:
                bfst = bfst+f+', '
            s.startStatus.setText('Bad files: '+bfst[:-1] + ' You can proceed with the others.')
        else:
            s.startStatus.setText('All file paths check out, ready to pick!')

        paths = [os.path.join(directory, f) for f in files_ck]

        # if have already Started
        if len(s.app_data['paths input']) > 0 and paths == s.app_data['paths input'][-1]:
            return
        s.app_data['paths input'].append(paths)

        s.app_data['file paths'] = paths
        s.app_data['file ind'] = -1

        s.next_file()

    def next_file(s):
        ''' Next button inside Pick tab '''
        s.debug(['Next'],1)
        before = datetime.now()
        s.load_file(next_file=True)
        after = datetime.now()
        #s.app_data['applied region limit histories'][s.app_data['current path']] = s.applied_region_limits
        s.debug(['Load took: ', after-before],2)

    def previous_file(s):
        ''' Previous button inside Pick tab '''
        s.debug(['Previous'],1)
        if s.app_data['file ind'] > 0:
            s.app_data['file ind'] -= 1
            s.load_file()
        #s.app_data['applied region limit histories'][s.app_data['current path']] = s.applied_region_limits

    def gather_info(s, eeg):
        ''' given EEGdata.avgh1 object, gather info about transforms and cases '''

        eeg.extract_transforms_data()
        eeg.extract_case_data()
        filter_info = 'Filter band: ' + '{:4.3f}'.format(eeg.transforms['hi_pass_filter']) \
                      + ' Hz to ' + '{:4.1f}'.format(eeg.transforms['lo_pass_filter']) + ' Hz'
        case_info = ['cases: trials accepted/total']
        trials_str = ''
        for caseN, caseD in eeg.cases.items():
            trials_str += caseD['case_type'] + ': ' + str(caseD['n_trials_accepted']) + '/' \
                          + str(caseD['n_trials']) + ',   '
        trials_str = trials_str[:-4]
        case_info.append(trials_str)

        s.app_data['info'] = [filter_info] + case_info

    def load_file(s, next_file=False, initialize=False):
        ''' load an avgh1 object as the current file to be picked and plot its data in the plotgrid '''
        s.debug(['load_file','next_file =',next_file,'initialize =',initialize],3)
        T_load = datetime.now()
        paths = s.app_data['file paths']

        if 'applied_region_limits' in dir(s) and len(s.applied_region_limits) > 0:
            s.app_data['regions by filepath'][paths[s.app_data['file ind']]] = s.applied_region_limits

        if next_file:
            if s.app_data['file ind'] < len(paths) - 1:
                s.app_data['file ind'] += 1
            else:
                s.debug(['already on last file'],1)
                s.status_message('already on last file')
                return

        s.buttons['Back'].setEnabled(False)
        s.buttons['Forward'].setEnabled(False)

        # main plot / drawing portion
        ind = s.app_data['file ind']
        s.debug(['load file: #', ind, paths[ind]],2)
        if ind < len(paths):
            # path = paths[ind]
            # s.app_data['current path'] = path
            # if path in s.app_data['applied region limit histories']:
            #     s.applied_region_limits = s.app_data['applied region limit histories'][path]
            # else:
            #     s.applied_region_limits = {}
            eeg = avgh1(paths[ind])
            s.eeg = eeg
            experiment = eeg.file_info['experiment']
            s.gather_info(eeg)
            cases = eeg.case_list
            if experiment in s.app_data['case display']:
                exp_cases_aliases = s.app_data['case display'][experiment]
                s.app_data['working cases'] = [ca[0] for ca in exp_cases_aliases]
                s.app_data['case aliases'] = dict(exp_cases_aliases)
            else:
                s.app_data['working cases'] = cases
                s.app_data['case aliases'] = {c:c for c in cases}
            s.app_data['case inds'] = {c:cases.index(c) for c in s.app_data['working cases']}
            s.app_data['case alias lookup'] = {a:c for c,a in s.app_data['case aliases'].items()}

            chans = eeg.electrodes

            #s.previous_peak_limits = {}
            if paths[ind] in s.app_data['regions by filepath']:
                s.applied_region_limits = s.app_data['regions by filepath'][paths[ind]]
                s.debug(['loaded previous applied region limit',
                    [(k,len(rs)) for k,rs in s.applied_region_limits.items()]],3)
            else:
                s.applied_region_limits = {}

            s.debug(['Loaded', experiment, ',', len(paths), 'paths, ind:', ind, ', info:', eeg.file_info],2)
            s.app_data['current experiment'] = experiment
            s.app_data['experiment cases'] = eeg.case_list

            s.peak_data = {}
            s.peak_edges = {}
            s.app_data['picks'] = set()
            s.show_state()

            # initialize (disconnect) current state of case toggles
            s.caseChooser.clear()
            s.debug(['removing case toggles', s.caseToggles],3)
            for case, toggle in s.caseToggles.items():
                toggle.stateChanged.disconnect(s.toggle_case)
                toggle.setParent(None)  # s.casesLayout.removeWidget(toggle)
            for case, toggle in s.zoomCaseToggles.items():
                toggle.stateChanged.disconnect(s.toggle_zoom_case)
                toggle.setParent(None)  # s.casesLayout.removeWidget(toggle)

            # connect case toggles
            s.caseToggles = {}
            s.zoomCaseToggles = {}
            for ci, case in enumerate(s.app_data['working cases'] ):
                alias = s.app_data['case aliases'][case]
                s.caseChooser.addItem('  ' + alias + '  ')
                case_toggle = QtGui.QCheckBox(alias)
                color_str = 'rgb' + str(s.plot_props['line colors'][ci])
                style_string = "background:" + color_str + ";"
                case_toggle.setStyleSheet(style_string)
                case_toggle.setChecked(True)
                case_toggle.stateChanged.connect(s.toggle_case)
                s.casesLayout.addWidget(case_toggle)
                s.caseToggles[case] = case_toggle
                zoom_case_toggle = QtGui.QCheckBox(alias)
                zoom_case_toggle.setChecked(True)
                zoom_case_toggle.stateChanged.connect(s.toggle_zoom_case)
                s.zoomCaseToggles[case] = zoom_case_toggle
                s.zoomControls.addWidget(zoom_case_toggle)

            # reversing initialize flag for testing
            data_sourceD, peak_sourcesD = eeg.make_data_sources(empty_flag=initialize,
                                                                time_range=s.app_data['display props']['time range'])
            s.current_data = data_sourceD

            # channel layout determined by this
            s.app_data['displayed channels'] = [ch for ch in chans if (ch not in s.ignore)]
            s.app_data['active channels'] = [ch for ch in chans if (ch not in s.show_only + s.ignore)]
            s.plot_desc = eeg.selected_cases_by_channel( cases=s.app_data['working cases'],
                                        time_range=s.app_data['display props']['time range'],
                                        channels=s.app_data['active channels'], mode='server', style='layout')
            s.ylims = s.plot_desc[0][1]['props']['yrange']
            s.times = s.plot_desc[1][1]['props']['times']

            T_prelims = datetime.now()
            if not initialize:
                s.legend_plot.clear()
                if 'items' in dir(s.legend_plot.legend):
                    s.legend_plot.legend.items = []
                for mk in [k for k in s.peak_markers.keys() if '_zoom' in k[0]]:
                    s.zoomPlot.removeItem( s.peak_markers[mk] )

                file_info = s.app_data['info']
                case_info = file_info[-1]
                sep_cases = case_info.split(',')
                s.debug(['sep cases', sep_cases],1)
                nC = len(sep_cases)
                if nC % 3 != 0:
                    sep_cases.append('')
                case_lines = [' '.join(sep_cases[si * 3:si * 3 + 3]) for si in range(int(len(sep_cases) / 3))]

                # create HTML to display the current file's info in top-left
                html = '<div>'
                for line in [os.path.split(paths[ind])[1]] + file_info[:-1] + case_lines:
                    html += '<span style="text-align: center; color: #EEE;' \
                            'font-size: 8pt; font-family: Helvetica;">' + line + '</span><br>'
                html += '</div>'
                s.debug(['info html', html],3)
                info_text = pg.TextItem(html=html, anchor=(0, 0))
                info_text.setPos(0.1, 1.18)
                s.legend_plot.addItem(info_text)

                s.legend_plot.vb.setRange(xRange=[0, 1], yRange=[0, 1])

                T_1 = datetime.now(); T_plots = []
                # main gridplot loop
                x_gridlines, y_gridlines = s.plot_props['XY gridlines']
                grid_pen = s.plot_props['grid color']
                #keep_items = [k for k in s.region_elecs.keys]+[v for k,v in s.plots.items]
                s.curves = {}
                value_texts = [ v for k,v in s.plot_texts.items() ]
                text_visible = s.textToggle.isChecked()
                for elec in s.app_data['displayed channels']:
                    plot = s.plots[elec]
                    # clear plot
                    for item in plot.listDataItems():
                        if item not in s.region_elecs and item not in value_texts:
                            plot.removeItem( item )
                    for ecp, lab in s.pick_region_labels.items():
                        if ecp[0] == elec:
                            plot.removeItem( lab )
                    for ecp, marker in s.peak_markers.items():
                        if ecp[0] == elec:
                            plot.removeItem( marker )
                    for text in value_texts:
                        text.setText('')
                        text.setVisible(text_visible)

                    for reg in s.region_elecs:
                        reg.setVisible(False)

                    if elec == s.app_data['displayed channels'][0]:
                        s.debug(['clear first:',datetime.now()-T_1],3)

                    # # grid lines
                    for xval in x_gridlines:
                        plot.addLine(x=xval, pen=grid_pen)
                    for yval in y_gridlines:
                        plot.addLine(y=yval, pen=grid_pen)

                    # ERP amplitude curves for each case
                    for case in s.app_data['working cases']:
                        s.curves[(elec, case)] = s.plot_curve(s.plots[elec], elec, case)
                    if elec == s.app_data['displayed channels'][0]:
                        s.debug(['plot first:',datetime.now()-T_1],3)
                    # set y limits
                    plot.setYRange(s.ylims[0], s.ylims[1])

                    # peak text
                    if plot.vb not in s.plot_texts:
                        peak_text = pg.TextItem(text='', anchor=(-0.1, 0.3))
                        plot.addItem(peak_text)
                        s.plot_texts[plot.vb] = peak_text

                    s.adjust_text(plot.vb)
                    if elec == s.app_data['displayed channels'][0]:
                        s.debug(['peak texts:',datetime.now()-T_1],3)

                    if plot.vb not in s.plot_labels:
                        bLabel = pg.ButtonItem(
                            imageFile=os.path.join(s.module_path, os.path.join('chanlogos', elec + '.png')),
                            width=s.plot_props['label size'], parentItem=plot)
                        bLabel.setPos(12, -8)
                        if elec == s.app_data['displayed channels'][0]:
                            s.debug(['label plots:',datetime.now()-T_1],3)
                        s.plot_labels[plot.vb] = bLabel

                    plot.vb.setMouseEnabled(x=False, y=False)
                    T_plots.append(datetime.now())
                
                # Adjust x range - requires extra call to setXRange at end of function below
                adj_plot = s.plots[s.app_data['displayed channels'][0]]
                adj_plot.setXRange(min(s.times),max(s.times))

                s.debug(['Plot setup time: ', T_1-T_prelims],3)
                Tp = T_1
                for T in T_plots:
                    #s.debug(['Plot time:', T-Tp],3)
                    Tp = T
                # update zoom plot
                s.zoom_curves = {}

                if s.app_data['zoom electrode'] is not None:
                    s.show_zoom_plot(s.app_data['zoom electrode'])
        T_end = datetime.now()
        s.debug(['Prelim load time:',T_prelims - T_load],3)
        s.debug(['Display load time:',T_end - T_prelims],3)

        s.peak_markers = {}
        s.peak_tops = {}

        # check for and load old mt
        picked_file_exists = s.eeg.extract_mt_data()
        if picked_file_exists:
            s.debug(['Already picked'],1)
            for cs_pk in s.eeg.case_peaks:
                s.app_data['picks'].add((s.eeg.num_case_map[int(cs_pk[0])], cs_pk[1]))

            for elec in s.pick_electrodes:
                for cs_pk in s.eeg.case_peaks:
                    s.peak_data[(elec, s.eeg.case_letter_from_number(cs_pk[0]), cs_pk[1])] = \
                        tuple([float(v) for v in s.eeg.get_peak_data(elec, cs_pk[0], cs_pk[1])])
            s.debug(['picks', s.app_data['picks']],1)
            s.show_peaks()
            s.show_state()

        s.pick_region_labels = {}

        if not initialize:
            s.reset_edge_notification_backgrounds()
            s.status_message('File loaded')
            # Extra setXRange call - to handle first file 
            adj_plot.setXRange(min(s.times),max(s.times))

    def rescale_yaxis(s):
        ''' set y limits of all plots in plotgrid to the recommended vals '''
        plot = s.plots['PZ']
        plot.setYRange(s.ylims[0], s.ylims[1])
        s.zoomPlot.autoRange()

    def plot_curve(s, plot, electrode, case):
        ''' given a plot handle, electrode, and case, return the line plot of its amplitude data '''
        c_ind = s.app_data['working cases'].index(case)
        curve = plot.plot(x=s.current_data['times'],
                          y=s.current_data[electrode + '_' + case],
                          pen=s.plot_props['line colors'][c_ind],
                          name=case)

        return curve

    def save(s):
        # save picks in pickle file
        s.debug(['Save'],2)
        pickD = {'picks':s.app_data['picks'],
                'peak data':s.peak_data,
                'avgh1 path':s.eeg.filepath,
                'save dir':s.app_data['working directory'],
                'plot props':s.plot_props,
                'experiment cases':s.app_data['experiment cases'],
                'working cases':s.app_data['working cases'],
                'case aliases':s.app_data['case aliases'],
                'current data':s.current_data,
                'info':s.app_data['info'],
                'plot desc':s.plot_desc,
                }
        store_name = s.app_data['user']+'_'+os.path.split(s.eeg.filepath)[1]+'.p'
        store_path = os.path.join(s.temp_store_dir,store_name)
        with open( store_path,'wb') as of:
            pickle.dump(pickD,of)
        s.debug(['stored pickled picks to ',store_path],2)
        subprocess.Popen(['/usr/local/PeakPicker/storePicks.sh', store_path])#+\
               #store_path, shell=True ) 
        s.debug(['saving in background'],2)

    def output_page(s, layout_desc):
        # setup
        xlim = [-100, 850]
        xticks = [0, 250, 500, 750]
        tick_col = 0
        ylim = [-4, 12]
        yticks = [0, 10]
        arrow_size = 4.5
        linewidth = 0.5

        ccns = [[v / 255 for v in cc] for cc in s.plot_props['line colors']]

        fig = plt.figure(figsize=(11, 8.5))
        nrows = len(layout_desc)
        ncols = len(layout_desc[0])
        spNum = 0
        for rN, prow in enumerate(layout_desc):

            for cN, p_desc in enumerate(prow):
                spNum += 1
                if p_desc:
                    elec = p_desc['electrode']

                    #    plot = s.plotsGrid.addPlot(rN + 1, cN)  # ,title=elec)
                    ax = plt.subplot(nrows + 1, ncols, spNum)
                    plt.subplots_adjust(hspace=0.001, wspace=0.001)
                    ax.margins(0)
                    ax.set_xlim(xlim)
                    ax.set_xticks(xticks)
                    ax.set_ylim(ylim)
                    ax.set_yticks(yticks)
                    ax.tick_params(direction='out', pad=5, labelsize=7)

                    ax.add_artist(plt.Line2D((xticks[0], xticks[-1]), (yticks[0], yticks[0]),
                                             color='black', linewidth=0.5))
                    ax.add_artist(plt.Line2D((0, 0), yticks, color='black', linewidth=0.5))
                    if rN == nrows - 1:
                        ax.set_xticklabels(xticks, fontsize=6)
                    else:
                        ax.set_xticklabels([])
                    if cN == tick_col:
                        ax.set_yticklabels(yticks, fontsize=6)
                    else:
                        ax.set_yticklabels([])

                    ax.set_ylabel(elec, rotation=0, fontsize=10, labelpad=5)
                    ax.set_frame_on(False)
                    ax.get_xaxis().tick_bottom()
                    ax.get_yaxis().tick_left()
                    ax.spines['left'].set_position('zero')
                    ax.spines['bottom'].set_position('zero')
                    ax.xaxis.set_ticks_position('bottom')
                    ax.yaxis.set_ticks_position('left')
                    for caseN, case in enumerate(s.app_data['experiment cases']):
                        if case in s.app_data['working cases']:
                            workingN = s.app_data['working cases'].index(case)
                            # case_color = s.plot_props['line colors'][caseN]
                            ccn = ccns[workingN]  # [v / 255 for v in case_color]
                            ax.plot(s.current_data['times'],
                                    s.current_data[elec + '_' + case],
                                    color=ccn, clip_on=False,
                                    linewidth=linewidth)

                            peak_keys = [k for k in s.peak_data.keys() if k[0] == elec and k[1] == case]
                            for pk in peak_keys:
                                if pk[2][0] == 'P':
                                    arrow_len = arrow_size
                                else:
                                    arrow_len = -arrow_size
                                amp, lat = s.peak_data[pk]
                                ax.annotate('', (lat, amp), (lat, amp + arrow_len),
                                            size=7, clip_on=False, annotation_clip=False,
                                            arrowprops=dict(arrowstyle='-|>',
                                                            fc=ccn, ec=ccn))
        # info row on bottom
        file_info = s.app_data['info']
        subD = s.eeg.subject_data()
        expD = s.eeg.exp_data()
        tformD = s.eeg.transforms_data()
        casesD = s.eeg.case_data()
        runD = s.eeg.run_data()
        # print('file_info',file_info)
        # print('subject data',subD)
        # print('exp data', expD)
        # print('transform data', tformD)
        # print('run data',runD)
        # print( 'eeg cases',casesD )


        filename = os.path.split(s.app_data['file paths'][s.app_data['file ind']])[1]
        desc_ax = plt.subplot(nrows + 1, ncols, spNum + 1)
        desc_ax.set_frame_on(False)
        desc_ax.set_xticks([])
        desc_ax.set_yticks([])
        desc_ax.set_xlim([0, 10])
        desc_ax.set_ylim([0, 10])
        desc_ax.text(0, 0, filename + '\n' + \
                     ' '.join([str(round(subD['age'] * 100) / 100)[:5], ' ', subD['gender'],
                               ' ', subD['handedness'], '  ', 'artf thresh',
                               str(expD['threshold_value'])]) + ' uV \n' + \
                     runD['run_date_time'],
                     fontsize=9)

        trials_table_rows_ax = plt.subplot(nrows + 1, ncols, spNum + 3)
        trials_table_rows_ax.set_frame_on(False)
        trials_table_rows_ax.set_xticks([])
        trials_table_rows_ax.set_yticks([])
        trials_table_rows_ax.set_ylim([0, 10])
        trials_table_rows_ax.set_xlim([0, 10])

        trials_table_ax = plt.subplot(nrows + 1, ncols, spNum + 4)
        trials_table_ax.set_frame_on(False)
        trials_table_ax.set_xticks([])
        trials_table_ax.set_yticks([])
        trials_table_ax.set_ylim([0, 10])
        trials_table_ax.text(0, 8, 'trials     resps', fontsize=9)

        rx_time_ax = plt.subplot(nrows + 1, ncols, spNum + 5)
        rx_time_ax.set_ylim([0, 1])
        rx_time_ax.text(0, 0.8, 'Response Times', fontsize=8)
        rx_time_ax.set_frame_on(False)
        rx_time_ax.set_xticks(xticks)
        rx_time_ax.set_xticklabels(xticks, fontsize=6)
        rx_time_ax.xaxis.set_ticks_position('bottom')
        rx_time_ax.set_yticks([])
        rx_time_ax.set_xlim(xlim)
        rx_time_ax.add_artist(plt.Line2D((xticks[0], xticks[-1]), (yticks[0], yticks[0]),
                                         color='black', linewidth=0.5))
        for caseN, case in enumerate(s.app_data['working cases']):
            case_alias = s.app_data['case aliases'][case]
            ccn = ccns[caseN]
            cD = casesD[case]
            trials_table_rows_ax.text(9, 6 - 2 * caseN, cD['descriptor'],
                                      horizontalalignment='right',
                                      fontsize=9, color=ccn)
            n_trials = str(cD['n_trials_accepted'])
            n_resp = str(min([cD['n_responses'], cD['n_trials_accepted']]))
            trials_table_ax.text(0, 6 - 2 * caseN, ' ' + n_trials + \
                                 ' ' * (3 - len(n_trials)) + '       ' \
                                 + n_resp,
                                 fontsize=9, color=ccn)

            rx_tm = cD['mean_resp_time']
            rx_time_ax.plot(2 * [rx_tm], [0.15, 0.5], color=ccn)
            rx_time_ax.text(rx_tm, 0.55, str(round(rx_tm * 100) / 100), fontsize=7, color=ccn)

        fig.tight_layout()

    def save_pdf(s, directory):
        filename = os.path.split(s.app_data['file paths'][s.app_data['file ind']])[1]
        with PdfPages(os.path.join(directory, filename + '.pdf')) as pdf:
            s.output_page(s.plot_desc[:8])
            pdf.savefig()
            plt.close()
            s.output_page(s.plot_desc[8:])
            pdf.savefig()
            plt.close()

    def update_curve_weights(s):
        ps = s.app_data['pick state']
        all_curves = [ (k,v) for k,v in s.curves.items() ]
        all_curves.extend([ (('zoom',k),v) for k,v in s.zoom_curves.items()])
        #s.debug(['update_curve_weights','all_curves keys:',[t[0] for t in all_curves]],3)
        for elec_case, curve in all_curves:
            weight = 1
            if elec_case[1] == ps['case']:
                weight = 2
            c_ind = s.app_data['working cases'].index(elec_case[1])
            pen = pg.mkPen(color=s.plot_props['line colors'][c_ind],
                           width=weight)
            curve.setPen(pen)

    def save_mt(s, save_dir):
        ''' save the current picks as an HBNL-formatted *.mt text file '''

        s.debug('save_mt',1)
        picked_cp = s.app_data['picks']
        cases_Ns = list(set([(cp[0], s.eeg.case_num_map[cp[0]]) for cp in picked_cp]))
        peaks = list(set([cp[1] for cp in picked_cp]))

        n_cases = len(cases_Ns)
        n_chans = 61  # only core 61 chans
        n_peaks = len(peaks)
        amps = np.empty((n_peaks, n_chans, n_cases))
        lats = np.empty((n_peaks, n_chans, n_cases))
        amps.fill(np.NAN)
        lats.fill(np.NAN)
        for icase, case_N in enumerate(cases_Ns):
            case_name = case_N[0]
            for ichan, chan in enumerate(s.eeg.electrodes_61):  # only core 61 chans
                case_peaks = [cp[1] for cp in picked_cp if cp[0] == case_name]
                case_peaks.sort()
                for peak in case_peaks:
                    ipeak = peaks.index(peak)
                    pd_key = (chan, case_name, peak)
                    if pd_key in s.peak_data:
                        amp_lat = s.peak_data[(chan, case_name, peak)]
                        amps[ipeak, ichan, icase] = amp_lat[0]
                        lats[ipeak, ichan, icase] = amp_lat[1]
                    else:
                        s.debug(['Missing peak data for:',pd_key],3)

        # reshape into 1d arrays
        amps1d = amps.ravel('F')
        lats1d = lats.ravel('F')

        # build mt text (makes default output location), write to a test location
        s.eeg.build_mt([cn[1] for cn in cases_Ns], peaks, amps1d, lats1d)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        fullpath = os.path.join(save_dir, s.eeg.mt_name)
        of = open(fullpath, 'w')
        of.write(s.eeg.mt)
        of.close()

        s.status_message(text='Saved to ' + os.path.split(fullpath)[0])
        s.debug(['Saved', fullpath],1)

    def adjust_text(s, viewbox):
        ''' adjusts position of electrode text label objects to be visible if axis limits change '''

        if viewbox in s.plot_texts:
            text = s.plot_texts[viewbox]
            region = viewbox.getState()['viewRange']
            # reg_height = region[1][1]-region[1][0]
            text.setPos(region[0][0], region[1][1])  # +reg_height/10)

    def update_ranges(s):
        ''' called when axis limits change (e.g. on pan/zoom) '''
        Pstate = s.app_data['pick state']
        # s.adjust_label(s.sender())
        for el in s.pick_regions:
            #if el_cs_pk[1] == Pstate['case'] and el_cs_pk[2] == Pstate['peak']:
            s.update_region_label_position(el)

    def update_pick_regions(s,call,regions_in=None):
        ''' called when pick regions are adjusted '''
        #s.debug(['update_pick_regions caller: ',call,'regions_in: ',regions_in],2)
        if not regions_in:
            sender = s.sender()
            region = sender.getRegion()
            elec = s.region_elecs[sender]

            if elec == 'zoom':
                elec = s.app_data['zoom electrode']
        else:
            elec = None
         
        case = s.app_data['pick state']['case']
        peak = s.app_data['pick state']['peak']

        for el, reg in s.pick_regions.items():
            if s.app_data['pick state']['repick mode'] == 'all' or regions_in is not None\
              or el == elec:
                if not regions_in:
                    reg.setRegion(region)
                else:
                    if el in regions_in:
                        reg.setRegion( regions_in[ el ] )
            elif el == 'zoom' and elec == s.app_data['zoom electrode']:
                reg.setRegion( region ) 

        if not regions_in:
            if 'zoomRegion' in dir(s) and sender != s.pick_regions['zoom']\
                                and elec == s.app_data['zoom electrode']:
                s.pick_regions['zoom'].setRegion(region)
        elif s.app_data['zoom electrode']:
            s.pick_regions['zoom'].setRegion( regions_in[ s.app_data['zoom electrode'] ] )


    def pick_init(s):
        ''' Pick inside the Pick tab (start picking a certain peak) '''

        case_alias = s.caseChooser.currentText().strip()
        case = s.app_data['case alias lookup'][case_alias] 
        peak = s.peakChooser.currentText().strip()
        s.app_data['pick state']['case'] = case
        s.app_data['pick state']['peak'] = peak

        s.debug(['Pick init for ', case_alias+'('+case+')', peak,],1)

        s.app_data['picks'].add((case, peak))

        #pick_case_peaks = set([(ecp[1], ecp[2]) for ecp in s.pick_region_labels])
        previous_case_peaks = set( [ (ecp[1],ecp[2]) for ecp in s.applied_region_limits ] )
        #previous_peaks = set([el_pk[1] for el_pk in s.previous_peak_limits])

        for ztcase, checkbox in s.zoomCaseToggles.items():
            checkbox.setChecked(True)  # ztcase == case )

        s.buttons['Back'].setEnabled(False)
        s.buttons['Forward'].setEnabled(False)
        
        s.pick_regions['zoom'].setVisible(True)
        s.pick_regions['zoom'].sigRegionChangeFinished.connect(s.update_pick_regions)

        if (case, peak) not in previous_case_peaks:
            s.app_data['apply count'] = 0
            s.app_data['apply scan ind'] = -1

            peak_center_ms = 100 * int(peak[1])

            existing_lim_CPs = set([(ecp[1], ecp[2]) for ecp in s.applied_region_limits])
            if (case, peak) in existing_lim_CPs:
                new = False
                start_ranges = {el: s.applied_region_limits[(el, case, peak)] for el in s.app_data['active channels']}
            # elif peak in previous_peaks:
            #     s.debug(['using previous peak range'],1)
            #     start_ranges = {el: s.previous_peak_limits[(el, peak)] for el in s.app_data['active channels']}
            else:
                new = True
                start_ranges = {el: (peak_center_ms - 75, peak_center_ms + 75) for el in s.app_data['active channels']}

            for elec in s.app_data['active channels']:  # [ p for p in s.plots if p not in s.show_only ]:

                s.pick_regions[ elec ].setRegion( start_ranges[elec] )
                s.pick_regions[ elec ].setVisible( True )

                s.plot_texts[s.plots[elec].vb].setHtml('')
                #s.debug(['about to update region label pos',(elec, case, peak)],3)
                s.update_region_label_position((elec, case, peak))
                if new:
                    s.applied_region_limits[ (elec,case,peak) ] = []
        else:
            check_key = ('CZ',case,peak)
            if check_key in s.applied_region_limits:
                s.app_data['apply count'] = len(s.applied_region_limits[check_key])
                s.debug(['pick init for already existing, CZ history: ',
                    s.applied_region_limits[('CZ',case,peak)]],3)
            else:
                s.app_data['apply_count'] = 0
            s.app_data['apply scan ind'] = s.app_data['apply count'] - 1
            if s.app_data['apply scan ind'] > 0:
                s.buttons['Back'].setEnabled(True)

        # for disp_case in s.app_data['experiment cases']:
        #     s.set_case_display(disp_case, disp_case == case)
        s.add_region_labels(case, peak)
        s.update_curve_weights()
        s.caseToggles[case].setChecked(True)
        s.toggle_regions(True)

        s.update_zoom_region()

        s.reset_edge_notification_backgrounds()

        s.status_message(text="Picking " + case_alias+'('+case+')' + ',' + peak)
        s.debug(['pick_init finish'],1)

    def add_region_labels(s,case,peak):

        for elec in s.app_data['active channels']:
            ecp = (elec, case, peak)
            if ecp not in s.pick_region_labels:
                region_label = pg.TextItem(
                    html=s.region_label_html.replace('__PEAK__', peak),
                    anchor=(-0.025, -0.2) )
                s.pick_region_labels[ecp] = region_label
                s.plots[elec].addItem(region_label)
                s.update_region_label_position(ecp)

    def update_region_label_position(s, reg_key=None):
        ''' given region-identifying 3-tuple of (electrode, case, peak), update a region's label position '''

        region_label = None
        if s.app_data['pick state']['case']:
            if not isinstance(reg_key, str):
                region = s.sender()
                if region in s.region_elecs:
                    elec = s.region_elecs[ region ]
                    case = s.app_data['pick state']['case']
                    peak = s.app_data['pick state']['peak']
                    #s.debug(['update_region_label_position signal',elec,case,peak],3)
                elif isinstance(reg_key,tuple):
                    region = s.pick_regions[ reg_key[0] ]
                    elec, case, peak = reg_key
                else:
                    elec, case, peak = None, None, None
                    s.debug(['update_region_label_position, bad input, reg_key: ',reg_key],1)

                if (elec, case, peak) in s.pick_region_labels:
                    region_label = s.pick_region_labels[ (elec, case, peak) ]

            else:
                region = s.pick_regions[reg_key]
                if reg_key in s.pick_region_labels:
                    region_label = s.pick_region_labels[reg_key]

            # check region for one electrode
            elec = 'FZ'  # s.eeg.electrodes[0]#reg_key[0]
            vb_range = s.plots[elec].vb.getState()['viewRange']
            
            if region_label:
                start_fin = region.getRegion()
                region_label.setPos(start_fin[0], vb_range[1][1])

    def show_state(s, cases='all'):
        ''' display the current pick state as a list of cases, with corresponding picked peaks following in brackets '''

        picks = s.app_data['picks']
        if cases == 'all':
            cases = s.app_data['working cases']
        state_string = ''
        for case in cases:
            case_alias = s.app_data['case aliases'][case]
            state_string += case_alias + ':['
            for cp in picks:
                if cp[0] == case:
                    state_string += cp[1]
                    if s.get_edge_elecs_casepeak(case, cp[1]):
                        state_string += '*'
                    state_string += ','
            state_string = state_string[:-1]  # drop trailing comma
            state_string += '] '

        s.stateInfo.setText(state_string)

    def get_edge_elecs_casepeak(s, case, peak):
        ''' given a case and peak, check if any of its channels currently picked are at an edge '''

        el_lst = []
        for elec_case_peak, is_atedge in s.peak_edges.items():
            if elec_case_peak[1] == case and elec_case_peak[2] == peak:
                if is_atedge:
                    el_lst.append(elec_case_peak[0])

        return el_lst

    # def scroll_handler(s,ev):
    #     print('scroll handler', ev[0] )
    #     print(dir(ev[0]))

    def show_zoom_plot(s, ev_or_elec):
        ''' if a single plot is clicked, show a larger version of the plot on a detached window '''

        if s.zoomDialog._open:
            s.get_zoompos()

        s.debug(['zoom_plot', ev_or_elec],1)

        Pstate = s.app_data['pick state']

        proceed = False
        if type(ev_or_elec) == str:
            elec = ev_or_elec
            proceed = True
        elif ev_or_elec[0].button() == 1 and ev_or_elec[0].currentItem in s.vb_map:
            elec = s.vb_map[ev_or_elec[0].currentItem]
            s.app_data['zoom electrode'] = elec
            proceed = True

        if s.useMainCasesToggle.isChecked():
            for case in s.app_data['working cases']:
                s.zoomCaseToggles[case].setChecked( s.caseToggles[case].isChecked() )

        if proceed:
            s.debug([elec],1)
            for el_cs_pk, mark in s.peak_markers.items():
                if '_zoom' in el_cs_pk[0]:
                    s.zoomPlot.removeItem(mark)
            for item in s.zoomPlot.listDataItems():
                if item != s.pick_regions['zoom']:
                    s.zoomPlot.removeItem( item )

            if Pstate['case']:
                s.zoomDialog.show()
                s.zoomDialog._open = True

                x_lines, y_lines = s.plot_props['XY gridlines']
                grid_pen = s.plot_props['grid color']
                for xval in x_lines:
                    s.zoomPlot.addLine(x=xval, pen=grid_pen)
                for yval in y_lines:
                    s.zoomPlot.addLine(y=yval, pen=grid_pen)

                for case in s.app_data['working cases']:
                    s.zoom_curves[case] = s.plot_curve(s.zoomPlot, elec, case)
                    # s.set_case_display(case, s.zoomCaseToggles[case].isChecked(), zoom=True)
                    peak_keys = [ecp for ecp in s.peak_data.keys() if ecp[0] == elec and ecp[1] == case]
                    for pK in peak_keys:
                        zkey = (pK[0] + '_zoom', pK[1], pK[2])
                        marker = s.show_zoom_marker( s.peak_data[pK] )
                        s.peak_markers[zkey] = marker

                s.zoomDialog.setGeometry(*s.app_data['display props']['zoom position'])
                s.zoomDialog.setWindowTitle(elec + ' - ' + \
                                            Pstate['case'] + ' - ' + Pstate['peak'] + '     ')

                s.update_zoom_region()

                for case in s.app_data['working cases']:  # unsure why this doesn't work in above loop, maybe timing
                    s.set_case_display(case, s.zoomCaseToggles[case].isChecked(), zoom=True)

            s.update_curve_weights()

    def get_zoompos(s):
        ''' store the current position of the zoom window '''
        s.app_data['display props']['zoom position'][0] = s.zoomDialog.pos().x()
        s.app_data['display props']['zoom position'][1] = s.zoomDialog.pos().y()

    def zoom_close(s, event):
        ''' custom handler for the closing of the zoom window that remembers its position '''
        s.get_zoompos()
        QtGui.QDialog.closeEvent(s.zoomDialog, event)
        s.zoomDialog._open = False

    def update_zoom_region(s):

        elec = s.app_data['zoom electrode']
        if elec is not None:
            # if 'zoomRegion' in dir(s) and s.zoomRegion in s.zoomPlot.items:
            #     s.zoomPlot.removeItem(s.zoomRegion)
            Pstate = s.app_data['pick state']
            if elec in s.pick_regions:
                small_region = s.pick_regions[elec]
                start_fin = small_region.getRegion()
                s.pick_regions['zoom'].setRegion(start_fin)

    def toggle_regions(s, state=None):
        ''' toggle display of regions (if peak being picked is changed or the display checkbox is toggled) '''

        if state is None:
            state = s.sender().isChecked()

        Pstate = s.app_data['pick state']
        s.debug(['toggle_regions', state, Pstate],3)

        for prlK,prl in s.pick_region_labels.items():
            if prlK[1] != Pstate['case'] or prlK[2] != Pstate['peak']:
                prl.setVisible(False)

        for el, reg in s.pick_regions.items():
            reg.setVisible(state)
            if el != 'zoom':
                prl_key = (el,Pstate['case'],Pstate['peak'])
                if prl_key in s.pick_region_labels:
                    s.pick_region_labels[ prl_key ].setVisible(state)

    def toggle_peaks(s):
        ''' toggle display of peak-marking glyphs (if checkbox is toggled) '''

        checked = s.sender().isChecked()
        for el_cs_pk, mark in s.peak_markers.items():
            if '_zoom' in el_cs_pk[0]:
                if s.zoomCaseToggles[ el_cs_pk[1] ].isChecked():
                    mark.setVisible( checked )
            elif s.caseToggles[ el_cs_pk[1] ].isChecked():
                mark.setVisible(checked)

    def toggle_peak_tops(s):
        ''' toggle display of marking top glyphs above peak-marking glyphs (if checkbox is toggled) '''

        checked = s.sender().isChecked()
        for el_cs_pk, top in s.peak_tops.items():
            if s.caseToggles[ el_cs_pk[1] ].isChecked():
                top.setVisible(checked)

    def toggle_value_texts(s):
        ''' toggle display of amplitude latency text (if checkbox is toggled) '''

        checked = s.sender().isChecked()
        for vb, text in s.plot_texts.items():
            text.setVisible(checked)

    def toggle_case(s):
        ''' toggle display of case ERP curves (if checkbox is toggled) '''

        sender = s.sender()
        case_alias = sender.text()
        # print('toggle_case',case)
        checked = sender.isChecked()
        s.set_case_display( s.app_data['case alias lookup'][case_alias], checked)

    def toggle_zoom_case(s):
        ''' toggle display of case ERP curves inside zoomplot (if checkbox is toggled) '''

        sender = s.sender()
        case_alias = sender.text()
        checked = sender.isChecked()
        s.set_case_display( s.app_data['case alias lookup'][case_alias], checked, zoom=True)

    def sync_case_display(s):
        for case in s.app_data['working cases']:
            state = s.caseToggles[case].isChecked()
            s.set_case_display( case, state)

    def set_case_display(s, case, state, zoom=False):
        ''' given case string and boolean state, sets display settings  '''

        # print('set_case_display',case,state,'zoom',zoom)
        curves = {}
        curve_keys = []
        if zoom or s.useMainCasesToggle.isChecked():
            toggles = s.zoomCaseToggles
            curves.update(s.zoom_curves)
            curve_keys.extend([case])
            for zel_cs_pk in [ k for k in s.peak_markers if '_zoom' in k[0] and k[1]==case ]:
                s.peak_markers[ zel_cs_pk ].setVisible(state)

        if not zoom:
            toggles = s.caseToggles
            curves.update(s.curves)
            curve_keys.extend([e_c for e_c in curves.keys() if isinstance(e_c,tuple) and e_c[1] == case])

        if not zoom:
            try:
                toggles[case].stateChanged.disconnect(s.toggle_case)
            except: s.debug(['set_case_display, case',case,' state',state, ' toggle not connected'],3)
        toggles[case].setChecked(state)
        if not zoom:
            toggles[case].stateChanged.connect(s.toggle_case)
        for ck in curve_keys:
            if ck in curves:
                curves[ck].setVisible(state)

        if not zoom:
            curves[ck].setVisible(state)

            marker_ck_state = s.peakMarkerToggle.isChecked()
            marker_state = state and marker_ck_state
            for el_cs_pk in [ecp for ecp in s.peak_markers if isinstance(ecp,tuple) and\
                                ecp[1] == case and '_zoom' not in ecp[0]]:
                s.peak_markers[el_cs_pk].setVisible(marker_state)

    def mode_toggle(s):
        ''' toggles between 'all' and 'single' peak picking modes which refer to the scope of actions
            including applying a pick and adjusting region limits '''

        current_mode = s.app_data['pick state']['repick mode']
        current_mode_i = s.repick_modes.index(current_mode)
        s.app_data['pick state']['repick mode'] = \
            s.repick_modes[(current_mode_i + 1) % len(s.repick_modes)]
        s.pickModeToggle.setText(s.app_data['pick state']['repick mode'])

    def status_message(s, text='', color='#EEE'):
        ''' Clears message by default'''

        s.app_data['status history'].append(text)

        html = '''<div style="width:__width__; word-wrap:break-word;">
                    <span style="text-align: center; color: __color__; font-size: 8pt; font-family: Helvetica;">
                '''
        html += break_text(text,s.plot_props['width']/8)
        html += '</span><br></div>'
        html = html.replace('__width__', str(s.plot_props['width'] - 10))

        html = html.replace('__color__', color)

        if 'info_text' not in dir(s):
            s.info_text = pg.TextItem(html=html, anchor=(-0.05, 0))
            s.info_text.setPos(-0.1, 0.8)
            s.status_plot.addItem(s.info_text)
        else:
            s.info_text.setHtml(html)



    def notify_applied_ckEdges(s, case, peak):
        ''' for a given case / peak combination, check if any peaks are at an edge, and provide a notification'''
        case_alias = s.app_data['case aliases'][case]
        edge_elecs = s.get_edge_elecs_casepeak(case,peak)
        if edge_elecs:
            plural = [' is',' an edge:']
            if len(edge_elecs) > 1:
                plural = ['s are', ' edges:']
            text = 'Peak'+plural[0]+ ' at '+ plural[1]+'\n'
            for el in edge_elecs:
                text += ' '+el
                s.plots[el].getViewBox().setBackgroundColor('#a37437')
                s.edge_notified_plots.add(el)
            
            s.reset_edge_notification_backgrounds( s.edge_notified_plots.difference(edge_elecs) )
                
            s.status_message(text=text, color='#E00')
        else:
            s.status_message(text=case_alias+'('+case+')' + ' , ' + peak + ' applied. All peaks within range.')
            s.reset_edge_notification_backgrounds()

    def reset_edge_notification_backgrounds(s,elecs='all'):

        if elecs == 'all':
            elecs = s.edge_notified_plots
        for el in elecs:
            s.plots[el].getViewBox().setBackgroundColor('#000000')

    def show_zoom_marker(s, amp_lat):
        bar_len = s.app_data['display props']['bar length']
        marker = pg.ErrorBarItem(x=[amp_lat[1]], y=[amp_lat[0]],
                                 top=bar_len / 5, bottom=bar_len / 5, beam=0, pen=(255, 255, 255))
        s.zoomPlot.addItem(marker)
        return marker

    def show_peaks(s, cases='all'):
        ''' display chosen extrema as glyphs '''
        s.debug(['show_peaks',' cases:',cases],2)
        bar_len = s.app_data['display props']['bar length']

        if cases == 'all':
            cases = s.app_data['working cases']

        for el_cs_pk, amp_lat in s.peak_data.items():
            if el_cs_pk[1] in cases:
                if el_cs_pk in s.peak_markers:
                    s.peak_markers[el_cs_pk].setData(x=[amp_lat[1]], y=[amp_lat[0]])
                    if s.app_data['zoom electrode'] == el_cs_pk[0]:
                        zkey = (el_cs_pk[0] + '_zoom', el_cs_pk[1], el_cs_pk[2])
                        if zkey in s.peak_markers:
                            s.peak_markers[zkey].setData(x=[amp_lat[1]], y=[amp_lat[0]])
                        else:
                            zoom_marker = s.show_zoom_marker(amp_lat)
                            s.peak_markers[zkey] = zoom_marker
                else:
                    marker = pg.ErrorBarItem(x=[amp_lat[1]], y=[amp_lat[0]],
                                             top=bar_len, bottom=bar_len, beam=0, pen=(255, 255, 255))
                    s.peak_markers[el_cs_pk] = marker
                    s.plots[el_cs_pk[0]].addItem(marker)
                    if s.app_data['zoom electrode'] == el_cs_pk[0]:
                        zoom_marker = s.show_zoom_marker(amp_lat)
                        zkey = (el_cs_pk[0] + '_zoom', el_cs_pk[1], el_cs_pk[2])
                        s.peak_markers[zkey] = zoom_marker

                c_ind = s.app_data['experiment cases'].index(el_cs_pk[1])
                if el_cs_pk in s.peak_edges and s.peak_edges[el_cs_pk]:
                    sym = 'x'
                    sz = 16
                else:
                    sym = 'o'
                    sz = 4

                if el_cs_pk in s.peak_tops:
                    s.peak_tops[el_cs_pk].setData(x=[amp_lat[1]], y=[amp_lat[0] + bar_len])
                    s.peak_tops[el_cs_pk].setSymbol(sym)
                    s.peak_tops[el_cs_pk].setSize(sz)
                else:
                    color_ind = s.app_data['working cases'].index(el_cs_pk[1])
                    top = pg.ScatterPlotItem(x=[amp_lat[1]], y=[amp_lat[0] + bar_len],
                                             symbol=sym, size=sz, pen=None, brush=s.plot_props['line colors'][color_ind])
                    s.peak_tops[el_cs_pk] = top
                    s.plots[el_cs_pk[0]].addItem(top)
                s.peak_tops[el_cs_pk].setVisible(s.peakTopToggle.isChecked())

        s.sync_case_display()

    def relabel_peak(s, case, old_peak, new_peak):
        ''' re-labels a picked peak to have a new pick identity '''

        old_keys = [k for k in s.peak_data.keys() if k[1] == case and k[2] == old_peak]

        for el, cs, opk in old_keys:
            oK, nK = (el, cs, opk), (el, cs, new_peak)
            s.peak_data[nK] = s.peak_data[oK]
            s.peak_data.pop(oK)

            s.peak_markers[nK] = s.peak_markers[oK]
            s.peak_markers.pop(oK)

            s.peak_tops[nK] = s.peak_tops[oK]
            s.peak_tops.pop(oK)

            s.pick_region_labels[nK] = s.pick_region_labels[oK]
            reg = s.pick_region_labels.pop(oK)
            reg.setHtml(s.region_label_html.replace('__PEAK__', new_peak))

        s.app_data['picks'].remove((case, old_peak))
        s.app_data['picks'].add((case, new_peak))

        s.show_state()
        case_alias = s.app_data['case aliases'][case]
        message = 'Changed ' + case_alias+'('+case+')' + ' , ' + old_peak + '  to  ' + new_peak
        s.debug([message],2)
        s.status_message(message)

    def remove_peak(s):
        ''' callback for Remove button inside Fix button dialog box '''
        case_alias = s.fixCase.currentText()
        case = s.app_data['case alias lookup'][ case_alias ]
        peak = s.oldPeak.currentText()
        s.debug(['remove_peak','case:',case,'peak:',peak,'keys:',[k for k in s.peak_data.keys()]],3)
        for el_cs_pk in list(s.peak_data.keys()):

            if el_cs_pk[1] == case and el_cs_pk[2] == peak:
                s.peak_data.pop(el_cs_pk)

                zk_check = (el_cs_pk[0]+'_zoom',case,peak) 
                if zk_check in s.peak_markers:
                    zoom_marker = s.peak_markers[ zk_check ]
                    zoom_marker.setVisible(False)    
                    s.zoomPlot.removeItem(zoom_marker)

                plot = s.plots[el_cs_pk[0]]
                label = s.pick_region_labels.pop(el_cs_pk)
                plot.removeItem(label)
                marker = s.peak_markers.pop(el_cs_pk)
                marker.setVisible(False)  # to improve display responsiveness
                plot.removeItem(marker)
                top = s.peak_tops.pop(el_cs_pk)
                top.setVisible(False)
                plot.removeItem(top)

        s.app_data['picks'].remove((case, peak))
        s.show_state()
        s.fixDialog.setVisible(False)

        s.status_message('Removed ' + case_alias+'('+case+')' + ' , ' + peak)

    def settings(s):

        s.settingsDialog.show()

    def fix_peak(s):
        ''' callback for Fix button inside Pick tab '''
        s.debug(['fix_peak'],1)
        s.fixCase.setItems([])
        s.oldPeak.setItems([])
        s.newPeak.setItems([])
        picked_cases = set([c_p[0] for c_p in s.app_data['picks']])
        for case in picked_cases:
            s.fixCase.addItem(s.app_data['case aliases'][case])
        s.fixDialog.show()

    def choose_fix_case(s):
        ''' populates drop-down box of cases to potentially fix '''

        case_alias = s.sender().currentText()
        if case_alias:
            case = s.app_data['case alias lookup'][ case_alias ]
            s.debug([ 'choose_fix_case', case_alias+'('+case+')', s.app_data['picks'] ],3)
            s.oldPeak.setItems([])
            s.newPeak.setItems([])
            available_peaks = set([c_p[1] for c_p in s.app_data['picks'] if c_p[0] == case])
            for peak in available_peaks:
                s.oldPeak.addItem(peak)
        else:
            s.debug(['choose_fix_case','case_alias: ',case_alias, 'sender:',
                    {d:getattr(s.sender,d) for d in dir(s.sender) if d[:2]!='__'}],3)

    def choose_old_peak(s):
        ''' populates drop-down box of peaks to potentially fix '''
        s.debug(['choose_old_peak','sender:',s.sender()],3)
        case_alias = s.fixCase.currentText()
        if case_alias:
            case = s.app_data['case alias lookup'][ case_alias ]
            old_peak = s.sender().currentText()
            s.debug(['choose_old_peak', case_alias+'('+case+')', old_peak],3)
            s.newPeak.setItems([])
            if old_peak: # handles calls generated by internal updates to the widgets
                possible_peaks = [p for p in s.peak_choices if p[0] == old_peak[0]]
                for peak in possible_peaks:
                    s.newPeak.addItem(peak)

    def apply_peak_change(s):
        ''' callback for Apply Change button inside of Fix dialog box '''

        case_alias = s.fixCase.currentText()
        if case_alias:
            case = s.app_data['case alias lookup'][ case_alias ]
            old_peak = s.oldPeak.currentText()
            new_peak = s.newPeak.currentText()
            s.relabel_peak(case, old_peak, new_peak)
            s.fixDialog.setVisible(False)

    def apply_selections(s,state,mode='new',regions=None):
        ''' find the extremum in the given region '''
        s.debug(['apply selections, mode:',mode],3)
        case = s.app_data['pick state']['case']
        peak = s.app_data['pick state']['peak']
        polarity = peak[0].lower()
        starts = []
        finishes = []
        elecs = []

        if mode == 'new':
            regions = { e:region.getRegion() for e,region in s.pick_regions.items() }
            s.app_data['apply count'] += 1
            s.app_data['apply scan ind'] = s.app_data['apply count']-1
            s.buttons['Forward'].setEnabled(False)
            if s.app_data['apply count'] > 1:
                s.buttons['Back'].setEnabled(True)

        for elec,start_finish in regions.items():
            if elec != 'zoom':
                elec_case_peak = (elec, case, peak)
                elecs.append( elec )
                starts.append(start_finish[0])
                finishes.append(start_finish[1])
                if mode =='new':
                    s.applied_region_limits[elec_case_peak].append(start_finish)
                # s.applied_region_limits[elec_case_peak].append(start_finish)
                # s.previous_peak_limits[(elec_case_peak[0], peak)] = start_finish
        #print(s.app_data['applied region limit histories'])
        # print('starts:',starts)
        pval, pms = s.eeg.find_peaks(case, elecs,
                                     starts_ms=starts, ends_ms=finishes, polarity=polarity)
        for e_ind, elec in enumerate(elecs):

            latency = pms[e_ind]
            amplitude = pval[e_ind]
            s.plot_texts[s.plots[elec].vb].setHtml(
                '<div style="font-size: 8pt; font-family: Helvetica; font-weight: bolder">' + '%.3f' % amplitude + ', ' + '%.1f' % latency + '</div>')
            s.peak_data[(elec, case, peak)] = (amplitude, latency)
            if (np.fabs(latency - starts[e_ind]) < 3) or (np.fabs(latency - finishes[e_ind]) < 3):
                s.peak_edges[(elec, case, peak)] = True
            else:
                s.peak_edges[(elec, case, peak)] = False

                # marker = pg.ErrorBarItem(x=[latency],y=[amplitude],
                #     top=bar_len,bottom=bar_len,beam=0,pen=(255,255,255))
                # s.peak_markers[(elec,case,peak)] = marker
                # s.plots[elec].addItem(marker)

        s.show_peaks(cases=[case])
        s.notify_applied_ckEdges(case, peak)

        s.show_state()

    def previous_apply(s):
        if s.app_data['apply scan ind'] > 0:
            s.app_data['apply scan ind'] -= 1
            if s.app_data['apply scan ind'] == 0:
                s.buttons['Back'].setEnabled(False)
        s.buttons['Forward'].setEnabled(True)
        s.debug(['previous_apply','apply scan ind:',s.app_data['apply scan ind'],
                ' apply count:',s.app_data['apply count']],3)

        regions = { ecp[0]:hist[s.app_data['apply scan ind']]\
            for ecp,hist in s.applied_region_limits.items()\
            if ecp[1] == s.app_data['pick state']['case'] and ecp[2] == s.app_data['pick state']['peak'] }
        if s.app_data['zoom electrode']:
            regions['zoom'] =  regions[ s.app_data['zoom electrode'] ]
        s.update_pick_regions('',regions_in=regions)
        s.apply_selections('',mode='prev',regions=regions)

    def next_apply(s):
        if s.app_data['apply scan ind'] < s.app_data['apply count']:
            s.app_data['apply scan ind'] += 1
            if s.app_data['apply scan ind'] == s.app_data['apply count'] -1:
                s.buttons['Forward'].setEnabled(False)
        s.buttons['Back'].setEnabled(True)
        s.debug(['next_apply','apply scan ind:',s.app_data['apply scan ind'],
                ' apply count:',s.app_data['apply count']],3)
        regions = { ecp[0]:hist[s.app_data['apply scan ind']]\
            for ecp,hist in s.applied_region_limits.items()\
            if ecp[1] == s.app_data['pick state']['case'] and ecp[2] == s.app_data['pick state']['peak'] }
        if s.app_data['zoom electrode']:
            regions['zoom'] =  regions[ s.app_data['zoom electrode'] ]
        s.update_pick_regions('',regions_in=regions)
        s.apply_selections('',mode='next',regions=regions)

    def debug(s,message,priority):
        '''message should be a list

        prints if debug_level (second input) meets or exceeds priority
        '''
        if s.app_data['debug'] >= priority:
            print(*message)
        with open(s.app_data['debug path'],'a') as odf:
            odf.write(str(priority)+'\t'+' '.join([str(m) for m in message])+'\n')

        # for i in s.plots['FP1'].items:
        #     if 'data' in dir(i):
        #         print( type(i), i.data[:2]  )  


app = QtGui.QApplication(sys.argv)
GUI = Picker()
sys.exit(app.exec_())
