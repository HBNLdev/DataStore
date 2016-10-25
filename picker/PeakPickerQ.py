''' Qt version of HBNL Peak Picker

    Just run me with python3 in conda 'upgrade' environment.
'''

import os
import sys
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])
import numpy as np
from picker.EEGdata import avgh1
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore

Qt = QtCore.Qt


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

    ignore = ['BLANK']
    show_only = ['X', 'Y']
    repick_modes = ('all', 'single')
    peak_choices = ['P1', 'P2', 'P3', 'P4', 'N1', 'N2', 'N3', 'N4']

    plot_props = {'width': 233, 'height': 102,
                  'extra_bottom_height': 40,  # for bottom row
                  'min_border': 4,
                  'line colors': [  (221, 34, 34), # red
                                    (102, 221, 102), # green
                                    (55, 160, 255), # light blue
                                    (221, 34, 221), # magenta
                                    (255, 200, 20), # orange
                                    (200, 255, 40), # yellow-green
                                    (20, 255, 200), # blue green
                                    (160, 0, 188), # gray
                                ],
                  'XY gridlines': ([0, 200, 400, 600, 800], [0]),
                  'grid color': '#555',
                  'label size':20}

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
    app_data['file paths'] = [
        '/processed_data/avg-h1-files/ant/l8-h003-t75-b125/suny/ns32-64/ant_5_e1_40143015_avg.h1']
    # os.path.join(os.path.dirname(__file__),init_files_by_exp['ant'] ) ]
    app_data['paths input'] = []

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

        pg.setConfigOption('background', DProps['background'])
        pg.setConfigOption('foreground', DProps['foreground'])

        s.buttons = {}

        s.tabs = QtGui.QTabWidget()

        ### Navigation ###
        # temporary placeholders
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

        s.buttons['Rescale'] = QtGui.QPushButton('Rescale')#, sizeHint=QtCore.QSize(60,25) )
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
            s.peakChooser.addItem('  '+peak+'  ')

        s.pickNavControls = QtGui.QHBoxLayout()
        s.pickNavControls.setAlignment(Qt.AlignLeft)
        s.pickNavControls.addWidget(pick_label)
        s.pickNavControls.addWidget(s.caseChooser)
        s.pickNavControls.addWidget(s.peakChooser)

        pick_buttons = [('Pick', s.pick_init), ('Apply', s.apply_selections),
                        ('Fix', s.fix_peak)]
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
        spacer = QtGui.QSpacerItem(40,1)
        s.pickNavControls.addItem(spacer)

        s.pickModeToggle.clicked.connect(s.mode_toggle)

        nav_buttons = [('Save', s.save_mt),
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

        s.fixDialog = QtGui.QDialog(s)
        s.fixLayout = QtGui.QVBoxLayout()
        s.fixDialog.setLayout(s.fixLayout)
        s.fixCase = QtGui.QComboBox()
        s.fixCase.currentIndexChanged.connect(s.choose_fix_case)
        s.oldPeak = QtGui.QComboBox()
        s.oldPeak.currentIndexChanged.connect(s.choose_old_peak)
        s.removePeak = QtGui.QPushButton('Remove Peak')
        s.removePeak.clicked.connect(s.remove_peak)
        s.newPeak = QtGui.QComboBox()
        s.applyChange = QtGui.QPushButton('Apply Change')
        s.applyChange.clicked.connect(s.apply_peak_change)
        [s.fixLayout.addWidget(w) for w in
         [s.fixCase, s.oldPeak, s.removePeak, s.newPeak, s.applyChange]]

        s.plotsGrid.ci.layout.setContentsMargins(0, 0, 0, 0)
        s.plotsGrid.ci.layout.setSpacing(0)

        s.pick_electrodes = []
        s.plots = {}
        s.plot_labels = {}
        s.plot_texts = {}
        s.curves = {}
        s.zoom_curves = {}
        s.vb_map = {}

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
                    if elec not in s.ignore + s.show_only:
                        s.pick_electrodes.append(elec)
                    plot = s.plotsGrid.addPlot(rN + 1, cN)  # ,title=elec)
                    s.proxyMouse = pg.SignalProxy(plot.scene().sigMouseClicked, slot=s.show_zoom_plot)
                    # s.proxyScroll = pg.SignalProxy(plot.scene().sigMouseMoved, slot=s.scroll_handler)
                    # plot.resize(300,250)
                    plot.vb.sigRangeChanged.connect(s.update_ranges)
                    s.vb_map[plot.vb] = elec
                    s.plots[elec] = plot
                    if prev_plot:
                        plot.setXLink(prev_plot)
                        plot.setYLink(prev_plot)
                    prev_plot = plot
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

    def get_zoompos(s):
        ''' store the current position of the zoom window '''
        s.app_data['display props']['zoom position'][0] = s.zoomDialog.pos().x()
        s.app_data['display props']['zoom position'][1] = s.zoomDialog.pos().y()

    def zoom_close(s, event):
        ''' custom handler for the closing of the zoom window that remembers its position '''
        s.get_zoompos()
        QtGui.QDialog.closeEvent(s.zoomDialog, event)
        s.zoomDialog._open = False

    def scroll_to_top(s):
        vert_scroll = s.plotsScroll.verticalScrollBar()
        vert_scroll.setValue( 1 )

    def scroll_to_bottom(s):
        vert_scroll = s.plotsScroll.verticalScrollBar()
        vert_scroll.setValue( vert_scroll.maximum() )

    def start_handler(s, signal):
        ''' Start button inside Navigate tab '''
        print('Start:', signal)
        directory = s.directoryInput.text().strip()

        files = s.filesInput.text().split(' ')
        files = [f for f in files if '.h1' in f]
        paths = [os.path.join(directory, f) for f in files]

        # if have already Started
        if len(s.app_data['paths input']) > 0 and paths == s.app_data['paths input'][-1]:
            return
        s.app_data['paths input'].append(paths)

        s.app_data['file paths'] = paths
        s.app_data['file ind'] = -1

        s.next_file()

    def next_file(s):
        ''' Next button inside Pick tab '''
        print('Next')
        s.load_file(next_file=True)

    def previous_file(s):
        ''' Previous button inside Pick tab '''
        print('Previous')
        if s.app_data['file ind'] > 0:
            s.app_data['file ind'] -= 1
            s.load_file()

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

        paths = s.app_data['file paths']

        if 'applied_regions' in dir(s) and s.applied_regions is not None:
            s.app_data['regions by filepath'][ paths[s.app_data['file ind']] ] = s.applied_regions

        if next_file:
            if s.app_data['file ind'] < len(paths) - 1:
                s.app_data['file ind'] += 1
            else:
                print('already on last file')
                s.status_message('already on last file')
                return

        # main plot / drawing portion
        ind = s.app_data['file ind']
        print('load file: #', ind, paths[ind])
        if ind < len(paths):
            eeg = avgh1(paths[ind])
            s.eeg = eeg
            experiment = eeg.file_info['experiment']
            s.gather_info(eeg)
            cases = eeg.case_list
            chans = eeg.electrodes

            s.applied_regions = None
            s.previous_peak_limits = {}
            if paths[ind] in s.app_data['regions by filepath']:
                s.applied_regions = s.app_data['regions by filepath'][paths[ind]]

            print('Load', experiment, ',', len(paths), 'paths, ind:', ind, ', info:', eeg.file_info)
            s.app_data['current experiment'] = experiment
            s.app_data['current cases'] = eeg.case_list

            s.peak_data = {}
            s.peak_edges = {}
            s.app_data['picks'] = set()
            s.show_state()

            # initialize (disconnect) current state of case toggles
            s.caseChooser.clear()
            print('removing case toggles', s.caseToggles)
            for case, toggle in s.caseToggles.items():
                toggle.stateChanged.disconnect(s.toggle_case)
                toggle.setParent(None)  # s.casesLayout.removeWidget(toggle)
            for case, toggle in s.zoomCaseToggles.items():
                toggle.stateChanged.disconnect(s.toggle_zoom_case)
                toggle.setParent(None)  # s.casesLayout.removeWidget(toggle)

            # connect case toggles
            s.caseToggles = {}
            s.zoomCaseToggles = {}
            for ci, case in enumerate(cases):
                s.caseChooser.addItem('  '+case+'  ')
                case_toggle = QtGui.QCheckBox(case)
                color_str = 'rgb'+str(s.plot_props['line colors'][ci])
                style_string = "background:"+color_str+";"
                case_toggle.setStyleSheet(style_string)
                case_toggle.setChecked(True)
                case_toggle.stateChanged.connect(s.toggle_case)
                s.casesLayout.addWidget(case_toggle)
                s.caseToggles[case] = case_toggle
                zoom_case_toggle = QtGui.QCheckBox(case)
                zoom_case_toggle.setChecked(True)
                zoom_case_toggle.stateChanged.connect(s.toggle_zoom_case)
                s.zoomCaseToggles[case] = zoom_case_toggle
                s.zoomControls.addWidget(zoom_case_toggle)

            # reversing initialize flag for testing
            data_sourceD, peak_sourcesD = eeg.make_data_sources(empty_flag=initialize,
                                                                time_range=s.app_data['display props']['time range'])
            s.current_data = data_sourceD

            # channel layout determined by this
            s.app_data['displayed channels'] = [ ch for ch in chans if ( ch not in s.ignore  ) ]
            s.app_data['active channels'] = [ ch for ch in chans if (ch not in s.show_only+s.ignore) ]
            s.plot_desc = eeg.selected_cases_by_channel(mode='server', style='layout',
                                                        time_range=s.app_data['display props']['time range'],
                                                        channels=s.app_data['active channels'])
            s.ylims = s.plot_desc[0][1]['props']['yrange']

            if not initialize:
                s.legend_plot.clear()
                if 'items' in dir(s.legend_plot.legend):
                    s.legend_plot.legend.items = []

                file_info = s.app_data['info']
                case_info = file_info[-1]
                sep_cases = case_info.split(',')
                print('sep cases', sep_cases)
                nC = len(sep_cases)
                if nC % 2 != 0:
                    sep_cases.append('')
                case_lines = [ ' '.join(sep_cases[si*2:si*2+2]) for si in range( int(len(sep_cases)/2) ) ]


                # create HTML to display the current file's info in top-left
                html = '<div>'
                for line in [os.path.split(paths[ind])[1]] + file_info[:-1] + case_lines:
                    html += '<span style="text-align: center; color: #EEE;' \
                            'font-size: 8pt; font-family: Helvetica;">' + line + '</span><br>'
                html += '</div>'
                print('html', html)
                info_text = pg.TextItem(html=html, anchor=(-0.05, 0))
                info_text.setPos(-0.1, 1.18)
                s.legend_plot.addItem(info_text)

                # create color-coded case legend
                s.legend_plot.addLegend(size=(55, 0), offset=(-4, 0.001))
                for c_ind, case in enumerate(cases):
                    s.legend_plot.plot(x=[-5, -4], y=[-20, -20],
                                       pen=s.plot_props['line colors'][c_ind],
                                       name=case)
                    s.legend_plot.vb.setRange(xRange=[0, 1], yRange=[0, 1])

                # main gridplot loop
                x_gridlines, y_gridlines = s.plot_props['XY gridlines']
                grid_pen = s.plot_props['grid color']
                s.curves= {}
                for elec in s.app_data['displayed channels']:
                    plot = s.plots[elec]
                    plot.clear()
                    if plot.vb in s.plot_labels:
                        plot.vb.removeItem( s.plot_labels[plot.vb] )

                    # # grid lines
                    for xval in x_gridlines:
                        plot.addLine(x=xval, pen=grid_pen)
                    for yval in y_gridlines:
                        plot.addLine(y=yval, pen=grid_pen)

                    # ERP amplitude curves for each case
                    for case in cases:
                        s.curves[(elec, case)] = s.plot_curve(s.plots[elec], elec, case)

                    # set y limits
                    plot.setYRange(s.ylims[0], s.ylims[1])

                    # peak text
                    peak_text = pg.TextItem(text='', anchor=(-0.1, 0.3))
                    plot.addItem(peak_text)
                    s.plot_texts[plot.vb] = peak_text
                    s.adjust_text(plot.vb)

                    bLabel = pg.ButtonItem(imageFile=os.path.join( s.module_path, os.path.join('chanlogos',elec+'.png') ),
                                    width=s.plot_props['label size'], parentItem=plot )
                    bLabel.setPos(12,-8)

                    s.plot_labels[plot.vb] = bLabel
                    

                    plot.vb.setMouseEnabled(x=False, y=False)

                # update zoom plot
                s.zoom_curves = {}
                s.pick_regions = {}
                if s.app_data['zoom electrode'] is not None:
                    s.show_zoom_plot( s.app_data['zoom electrode'] )


        s.peak_markers = {}
        s.peak_tops = {}
        s.region_case_peaks = {}

        # check for and load old mt
        picked_file_exists = s.eeg.extract_mt_data()
        if picked_file_exists:
            print('Already picked')
            for cs_pk in s.eeg.case_peaks:
                s.app_data['picks'].add((s.eeg.num_case_map[int(cs_pk[0])], cs_pk[1]))

            for elec in s.pick_electrodes:
                for cs_pk in s.eeg.case_peaks:
                    # print(elec,cs_pk,  s.eeg.get_peak_data( elec, cs_pk[0], cs_pk[1] ) )
                    s.peak_data[(elec, s.eeg.case_letter_from_number(cs_pk[0]), cs_pk[1])] = \
                        tuple([float(v) for v in s.eeg.get_peak_data(elec, cs_pk[0], cs_pk[1])])
            print('picks', s.app_data['picks'])
            s.show_peaks()
            s.show_state()

        s.pick_regions = {}
        s.applied_region_limits = {}
        s.pick_region_labels = {}
        s.pick_region_labels_byRegion = {}

        if not initialize:
            s.status_message('File loaded')

    def rescale_yaxis(s):
        ''' set y limits of all plots in plotgrid to the recommended vals '''
        plot = s.plots['PZ']
        plot.setYRange(s.ylims[0], s.ylims[1])

    def plot_curve(s, plot, electrode, case):
        ''' given a plot handle, electrode, and case, return the line plot of its amplitude data '''
        c_ind = s.app_data['current cases'].index(case)
        curve = plot.plot(x=s.current_data['times'],
                          y=s.current_data[electrode + '_' + case],
                          pen=s.plot_props['line colors'][c_ind],
                          name=case)

        return curve

    def update_curve_weights(s):
        ps = s.app_data['pick state']
        for elec_case, curve in s.curves.items():
            weight = 1
            if elec_case[1] == ps['case']:
                weight = 2
            c_ind = s.app_data['current cases'].index(elec_case[1])
            pen = pg.mkPen( color=s.plot_props['line colors'][c_ind],
                            width=weight ) 
            curve.setPen( pen )

    def save_mt(s):
        ''' save the current picks as an HBNL-formatted *.mt text file '''

        print('Save mt')
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
                    amp_lat = s.peak_data[(chan, case_name, peak)]
                    amps[ipeak, ichan, icase] = amp_lat[0]
                    lats[ipeak, ichan, icase] = amp_lat[1]

        # reshape into 1d arrays
        amps1d = amps.ravel('F')
        lats1d = lats.ravel('F')

        # build mt text (makes default output location), write to a test location
        s.eeg.build_mt([cn[1] for cn in cases_Ns], peaks, amps1d, lats1d)

        test_dir = os.path.join('/active_projects/test', s.app_data['user'] + 'Q')
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        fullpath = os.path.join(test_dir, s.eeg.mt_name)
        of = open(fullpath, 'w')
        of.write(s.eeg.mt)
        of.close()

        s.status_message( text='Saved to '+os.path.split(fullpath)[0] )
        print('Saved', fullpath)

    def adjust_text(s, viewbox):
        ''' adjusts position of electrode text label objects to be visible if axis limits change '''

        if viewbox in s.plot_texts:
            text = s.plot_texts[viewbox]
            region = viewbox.getState()['viewRange']
            #reg_height = region[1][1]-region[1][0]
            text.setPos(region[0][0], region[1][1])#+reg_height/10)

    def update_ranges(s):
        ''' called when axis limits change (e.g. on pan/zoom) '''
        Pstate = s.app_data['pick state']
        #s.adjust_label(s.sender())
        for el_cs_pk in s.pick_regions:
            if el_cs_pk[1] == Pstate['case'] and el_cs_pk[2] == Pstate['peak']:
                s.update_region_label_position(el_cs_pk)

    def update_pick_regions(s):
        ''' called when pick regions are adjusted '''

        sender = s.sender()
        region = sender.getRegion()
        elec, case, peak = s.region_case_peaks[sender]

        for el_cs_pk, reg in s.pick_regions.items():
            if s.app_data['pick state']['repick mode'] == 'all' or el_cs_pk == (elec, case, peak):
                if el_cs_pk[1] == case and el_cs_pk[2] == peak:
                    reg.setRegion(region)

        if 'zoomRegion' in dir(s) and sender != s.zoomRegion and elec == s.app_data['zoom electrode']:
            s.zoomRegion.setRegion(region)


    def pick_init(s):
        ''' Pick inside the Pick tab (start picking a certain peak) '''

        ffs = ['Verdana', 'Arial', 'Helvetica', 'sans-serif', 'Times', 'Times New Roman', 'Georgia', 'serif',
               'Lucida Console', 'Courier', 'monospace']
        ffind = 0
        case = s.caseChooser.currentText().strip()
        peak = s.peakChooser.currentText().strip()
        s.app_data['pick state']['case'] = case
        s.app_data['pick state']['peak'] = peak

        print('Pick init for ', case, peak)

        s.app_data['picks'].add((case, peak))

        pick_case_peaks = set([(ecp[1], ecp[2]) for ecp in s.pick_regions])
        previous_peaks = set([ el_pk[1] for el_pk in s.previous_peak_limits ])

        for ztcase, checkbox in s.zoomCaseToggles.items():
                checkbox.setChecked( True ) #ztcase == case )

        if (case, peak) not in pick_case_peaks:
            peak_center_ms = 100 * int(peak[1])
            
            existing_lim_CPs = set([ (ecp[1],ecp[2]) for ecp in s.applied_region_limits ])
            if (case, peak) in existing_lim_CPs:
                start_ranges = { el:s.applied_region_limits[(el,case,peak)] for el in s.app_data['active channels'] }
            elif peak in previous_peaks:
                print( 'using previous peak range' )
                start_ranges = { el:s.previous_peak_limits[(el,peak)] for el in s.app_data['active channels'] } 
            else:
                start_ranges = { el:(peak_center_ms-75,peak_center_ms+75) for el in s.app_data['active channels']}
            
            for elec in s.app_data['active channels']:#[ p for p in s.plots if p not in s.show_only ]:
                region = pg.LinearRegionItem(values=start_ranges[elec],movable=True,
                        brush=s.app_data['display props']['pick region'])

                region.sigRegionChanged.connect(s.update_region_label_position)
                region.sigRegionChangeFinished.connect(s.update_pick_regions)

                region_label = pg.TextItem(
                    html=s.region_label_html.replace('__PEAK__', peak),
                    anchor=(-0.025, -0.2))

                s.pick_regions[(elec, case, peak)] = region
                s.pick_region_labels[(elec, case, peak)] = region_label
                s.pick_region_labels_byRegion[region] = region_label
                s.region_case_peaks[region] = (elec, case, peak)
                s.plots[elec].addItem(region)
                s.plots[elec].addItem(region_label)

                s.plot_texts[ s.plots[elec].vb ].setHtml('')

                s.update_region_label_position((elec, case, peak))

        # for disp_case in s.app_data['current cases']:
        #     s.set_case_display(disp_case, disp_case == case)
        s.update_curve_weights()

        s.toggle_regions(True)

        s.update_zoom_region()

        s.status_message(text="Picking "+case+','+peak)
        print('pick_init finish')

    def update_region_label_position(s, reg_key=None):
        ''' given region-identifying 3-tuple of (electrode, case, peak), update a region's label position '''

        if type(reg_key) != tuple:
            region = s.sender()
            region_label = s.pick_region_labels_byRegion[region]
        else:
            region = s.pick_regions[reg_key]
            region_label = s.pick_region_labels[reg_key]

        # check region for one electrode
        elec = 'FZ'  # s.eeg.electrodes[0]#reg_key[0]
        vb_range = s.plots[elec].vb.getState()['viewRange']
        start_fin = region.getRegion()

        region_label.setPos(start_fin[0], vb_range[1][1])

    def show_state(s, cases='all'):
        ''' display the current pick state as a list of cases, with corresponding picked peaks following in brackets '''

        picks = s.app_data['picks']
        if cases == 'all':
            cases = s.app_data['current cases']
        state_string = ''
        for case in cases:
            state_string += case + ':['
            for cp in picks:
                if cp[0] == case:
                    state_string += cp[1]
                    if s.any_casepeak_edges(case, cp[1]):
                        state_string += '*'
                    state_string += ','
            state_string += '] '

        s.stateInfo.setText(state_string)

    def any_casepeak_edges(s, case, peak):
        ''' given a case and peak, check if any of its channels currently picked are at an edge '''

        bool_lst = []
        for elec_case_peak, is_atedge in s.peak_edges.items():
            if elec_case_peak[1] == case and elec_case_peak[2] == peak:
                bool_lst.append(is_atedge)

        if any(bool_lst):
            return True
        else:
            return False

    # def scroll_handler(s,ev):
    #     print('scroll handler', ev[0] )
    #     print(dir(ev[0]))

    def show_zoom_plot(s, ev_or_elec):
        ''' if a single plot is clicked, show a larger version of the plot on a detached window '''

        if s.zoomDialog._open:
            s.get_zoompos()

        print('zoom_plot', ev_or_elec)

        Pstate = s.app_data['pick state']

        proceed = False
        if type(ev_or_elec) == str:
            elec = ev_or_elec
            proceed = True
        elif ev_or_elec[0].button() == 1 and ev_or_elec[0].currentItem in s.vb_map:
            elec = s.vb_map[ev_or_elec[0].currentItem]
            s.app_data['zoom electrode'] = elec
            proceed = True 

        if proceed:
            print(elec)
            s.zoomPlot.clear()

            if Pstate['case']:
                s.zoomDialog.show()
                s.zoomDialog._open = True

                x_lines, y_lines = s.plot_props['XY gridlines']
                grid_pen = s.plot_props['grid color']
                for xval in x_lines:
                    s.zoomPlot.addLine(x=xval, pen=grid_pen)
                for yval in y_lines:
                    s.zoomPlot.addLine(y=yval, pen=grid_pen)

                for case in s.app_data['current cases']:
                    s.zoom_curves[case] = s.plot_curve(s.zoomPlot, elec, case)
                    # s.set_case_display(case, s.zoomCaseToggles[case].isChecked(), zoom=True)
                    peak_keys = [ ecp for ecp in s.peak_data.keys() if ecp[0]==elec and ecp[1]==case ]
                    for pK in peak_keys:
                        zkey = ( pK[0]+'_zoom', pK[1], pK[2] )
                        marker = s.show_zoom_marker( s.peak_data[pK] )
                        s.peak_markers[zkey] = marker

                s.zoomDialog.setGeometry(*s.app_data['display props']['zoom position'])
                s.zoomDialog.setWindowTitle(elec + ' - ' + \
                                            Pstate['case'] + ' - ' + Pstate['peak'] + '     ')

                s.update_zoom_region()


                for case in s.app_data['current cases']:  # unsure why this doesn't work in above loop, maybe timing
                    s.set_case_display(case, s.zoomCaseToggles[case].isChecked(), zoom=True)

    def update_zoom_region(s):

        elec = s.app_data['zoom electrode']
        if elec is not None:
            if 'zoomRegion' in dir(s) and s.zoomRegion in s.zoomPlot.items:
                s.zoomPlot.removeItem( s.zoomRegion )
            Pstate = s.app_data['pick state']
            reg_key = (elec, Pstate['case'], Pstate['peak'])
            if reg_key in s.pick_regions:
                small_region = s.pick_regions[reg_key]
                start_fin = small_region.getRegion()
                region = pg.LinearRegionItem(values=start_fin, movable=True,
                                             brush=s.app_data['display props']['pick region'])
                region.sigRegionChangeFinished.connect(s.update_pick_regions)
                s.zoomRegion = region
                s.region_case_peaks[region] = (elec, Pstate['case'], Pstate['peak'])
                s.zoomPlot.addItem(region)
                

    def toggle_regions(s, state=None):
        ''' toggle display of regions (if peak being picked is changed or the display checkbox is toggled) '''

        if state is None:
            state = s.sender().isChecked()

        Pstate = s.app_data['pick state']
        print('toggle_regions',state, Pstate)

        for el_cs_pk,reg in s.pick_regions.items():
            show = False
            if state and el_cs_pk[1] == Pstate['case'] and el_cs_pk[2] == Pstate['peak']:
                show = True
            reg.setVisible(show) 
            s.pick_region_labels[el_cs_pk].setVisible(show)

    def toggle_peaks(s):
        ''' toggle display of peak-marking glyphs (if checkbox is toggled) '''

        checked = s.sender().isChecked()
        for el_cs_pk, mark in s.peak_markers.items():
            if s.caseToggles[el_cs_pk[1]].isChecked():
                mark.setVisible(checked)

    def toggle_peak_tops(s):
        ''' toggle display of marking top glyphs above peak-marking glyphs (if checkbox is toggled) '''

        checked = s.sender().isChecked()
        for el_cs_pk, top in s.peak_tops.items():
            if s.caseToggles[el_cs_pk[1]].isChecked():
                top.setVisible(checked)

    def toggle_value_texts(s):
        ''' toggle display of amplitude latency text (if checkbox is toggled) '''

        checked = s.sender().isChecked()
        for vb,text in s.plot_texts.items():
            text.setVisible(checked)

    def toggle_case(s):
        ''' toggle display of case ERP curves (if checkbox is toggled) '''

        sender = s.sender()
        case = sender.text()
        # print('toggle_case',case)
        checked = sender.isChecked()
        s.set_case_display(case, checked)
        # for el_cs in [ec for ec in s.curves if ec[1]==case]:
        #    s.curves[el_cs].setVisible(checked)

    def toggle_zoom_case(s):
        ''' toggle display of case ERP curves inside zoomplot (if checkbox is toggled) '''

        sender = s.sender()
        case = sender.text()
        checked = sender.isChecked()
        s.set_case_display(case, checked, zoom=True)

    def sync_case_display(s):
        for case in s.app_data['current cases']:
            state = s.caseToggles[case].isChecked()
            s.set_case_display(case,state)

    def set_case_display(s,case,state,zoom=False):
        ''' given case string and boolean state, sets display settings  '''

        #print('set_case_display',case,state,'zoom',zoom)

        if zoom:
            toggles = s.zoomCaseToggles
            curves = s.zoom_curves
            curve_keys = [case]
        else:
            toggles = s.caseToggles
            curves = s.curves
            curve_keys = [e_c for e_c in curves.keys() if e_c[1] == case]

        if not zoom: 
            toggles[case].stateChanged.disconnect(s.toggle_case)
        toggles[case].setChecked(state)
        if not zoom: 
            toggles[case].stateChanged.connect(s.toggle_case)
        for ck in curve_keys:
            if ck in curves:
                curves[ck].setVisible(state)

        if not zoom:
            curves[ck].setVisible(state)

            marker_ck_state =  s.peakMarkerToggle.isChecked()
            marker_state = state and marker_ck_state
            for el_cs_pk in [ecp for ecp in s.peak_markers if ecp[1] == case]:
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

        html = '''<div style="width:__width__; word-wrap:break-word;">
                    <span style="text-align: center; color: __color__; font-size: 8pt; font-family: Helvetica;">
                '''
        html += text
        html += '</span><br></div>'
        html = html.replace('__width__', str(s.plot_props['width']-10) )
 
        html = html.replace('__color__',color)

        if 'info_text' not in dir(s):
            s.info_text = pg.TextItem(html=html, anchor=(-0.05, 0) )            
            s.info_text.setPos(0.15, 0.8)
            s.status_plot.addItem(s.info_text)
        else:
            s.info_text.setHtml(html)


    def notify_applied_ckEdges(s, case, peak):
        ''' for a given case / peak combination, check if any peaks are at an edge, and provide a notification'''

        if s.any_casepeak_edges(case, peak):

            text = 'At least one peak is at an edge'
            s.status_message(text=text, color='#E00')
        else: 
            s.status_message(text=case+' , '+peak+' applied. All peaks within range.')

    def show_zoom_marker(s,amp_lat):
        bar_len = s.app_data['display props']['bar length']
        marker = pg.ErrorBarItem(x=[amp_lat[1]], y=[amp_lat[0]],
                            top=bar_len/5, bottom=bar_len/5, beam=0, pen=(255, 255, 255))
        s.zoomPlot.addItem(marker)
        return marker

    def show_peaks(s, cases='all'):
        ''' display chosen extrema as glyphs '''

        bar_len = s.app_data['display props']['bar length']

        if cases == 'all':
            cases = s.app_data['current cases']
            
        for el_cs_pk, amp_lat in s.peak_data.items():
            if el_cs_pk[1] in cases:
                if el_cs_pk in s.peak_markers:
                    s.peak_markers[el_cs_pk].setData(x=[amp_lat[1]],y=[amp_lat[0]])
                    if s.app_data['zoom electrode'] == el_cs_pk[0]:
                        zkey = (el_cs_pk[0]+'_zoom',el_cs_pk[1],el_cs_pk[2])
                        if zkey in s.peak_markers:
                            s.peak_markers[zkey].setData(x=[amp_lat[1]],y=[amp_lat[0]])
                        else:
                            zoom_marker = s.show_zoom_marker( amp_lat )
                            s.peak_markers[ zkey ] = zoom_marker
                else:
                    marker = pg.ErrorBarItem(x=[amp_lat[1]], y=[amp_lat[0]],
                                             top=bar_len, bottom=bar_len, beam=0, pen=(255, 255, 255))
                    s.peak_markers[el_cs_pk] = marker
                    s.plots[el_cs_pk[0]].addItem(marker)
                    if s.app_data['zoom electrode'] == el_cs_pk[0]:
                        zoom_marker = s.show_zoom_marker( amp_lat )
                        zkey = (el_cs_pk[0]+'_zoom',el_cs_pk[1],el_cs_pk[2])
                        s.peak_markers[ zkey ] = zoom_marker

                c_ind = s.app_data['current cases'].index(el_cs_pk[1])
                if el_cs_pk in s.peak_edges and s.peak_edges[el_cs_pk]:
                    sym = 'x'
                    sz = 12
                else:
                    sym = 'o'
                    sz = 4

                if el_cs_pk in s.peak_tops:
                    s.peak_tops[el_cs_pk].setData(x=[amp_lat[1]],y=[amp_lat[0] + bar_len])
                    s.peak_tops[el_cs_pk].setSymbol(sym)
                    s.peak_tops[el_cs_pk].setSize(sz)
                else:
                    top = pg.ScatterPlotItem(x=[amp_lat[1]], y=[amp_lat[0] + bar_len],
                                         symbol=sym, size=sz, pen=None, brush=s.plot_props['line colors'][c_ind])
                    s.peak_tops[el_cs_pk] = top
                    s.plots[el_cs_pk[0]].addItem(top)
                s.peak_tops[el_cs_pk].setVisible( s.peakTopToggle.isChecked() )

        s.sync_case_display()

    def relabel_peak(s, case, old_peak, new_peak):
        ''' re-labels a picked peak to have a new pick identity '''

        old_keys = [k for k in s.peak_data.keys() if k[1] == case and k[2] == old_peak]

        for el, cs, opk in old_keys:
            oK, nK = (el, cs, opk), (el, cs, new_peak)
            s.peak_data[nK] = s.peak_data[oK]
            s.peak_data.pop(oK)

            s.pick_regions[nK] = s.pick_regions[oK]
            region = s.pick_regions.pop(oK)
            s.region_case_peaks[region] = nK

            s.peak_markers[nK] = s.peak_markers[oK]
            s.peak_markers.pop(oK)

            s.pick_region_labels[nK] = s.pick_region_labels[oK]
            reg = s.pick_region_labels.pop(oK)
            reg.setHtml(s.region_label_html.replace('__PEAK__', new_peak))

        s.app_data['picks'].remove((case, old_peak))
        s.app_data['picks'].add((case, new_peak))

        s.show_state()

        s.status_message('Changed '+case+' , '+old_peak+'  to  '+new_peak)

    def remove_peak(s):
        ''' callback for Remove button inside Fix button dialog bix '''

        case = s.fixCase.currentText()
        peak = s.oldPeak.currentText()

        for el_cs_pk in list(s.peak_data.keys()):

            if el_cs_pk[1] == case and el_cs_pk[2] == peak:
                s.peak_data.pop(el_cs_pk)

                plot = s.plots[el_cs_pk[0]]
                region = s.pick_regions[el_cs_pk]
                s.pick_regions.pop(el_cs_pk)
                s.region_case_peaks.pop(region)
                s.pick_region_labels_byRegion.pop(region)
                plot.removeItem(region)
                label = s.pick_region_labels.pop(el_cs_pk)
                plot.removeItem(label)
                marker = s.peak_markers.pop(el_cs_pk)
                marker.setVisible(False) # to improve display responsiveness
                plot.removeItem(marker)
                top = s.peak_tops.pop(el_cs_pk)
                top.setVisible(False)
                #plot.removeItem(top)

        s.app_data['picks'].remove((case, peak))
        s.show_state()
        s.fixDialog.setVisible(False)

        s.status_message('Removed '+case+' , '+peak)

    def fix_peak(s):
        ''' callback for Fix button inside Pick tab '''

        s.fixCase.clear()
        s.oldPeak.clear()
        s.newPeak.clear()
        picked_cases = set([c_p[0] for c_p in s.app_data['picks']])
        for case in picked_cases:
            s.fixCase.addItem(case)
        s.fixDialog.show()

    def choose_fix_case(s):
        ''' populates drop-down box of cases to potentially fix '''

        case = s.sender().currentText()
        print('choose_fix_case', case, s.app_data['picks'])
        available_peaks = [c_p[1] for c_p in s.app_data['picks'] if c_p[0 == case]]
        for peak in available_peaks:
            s.oldPeak.addItem(peak)

    def choose_old_peak(s):
        ''' populates drop-down box of peaks to potentially fix '''

        case = s.fixCase.currentText()
        old_peak = s.sender().currentText()
        print('choose_old_peak', case, old_peak)
        possible_peaks = [p for p in s.peak_choices if p[0] == old_peak[0]]
        for peak in possible_peaks:
            s.newPeak.addItem(peak)

    def apply_peak_change(s):
        ''' callback for Apply Change button inside of Fix dialog box '''

        case = s.fixCase.currentText()
        old_peak = s.oldPeak.currentText()
        new_peak = s.newPeak.currentText()
        s.relabel_peak(case, old_peak, new_peak)
        s.fixDialog.setVisible(False)

    def apply_selections(s):
        ''' find the extremum in the given region '''

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
                s.applied_region_limits[elec_case_peak] = start_finish
                s.previous_peak_limits[ (elec_case_peak[0], peak) ] = start_finish

        # print('starts:',starts)
        pval, pms = s.eeg.find_peaks(case, elecs,
                                     starts_ms=starts, ends_ms=finishes, polarity=polarity)
        for e_ind, elec in enumerate(elecs):

            latency = pms[e_ind]
            amplitude = pval[e_ind]
            s.plot_texts[ s.plots[elec].vb ].setHtml('<div style="font-size: 8pt; font-family: Helvetica; font-weigth: bolder">'+'%.3f'%amplitude+', '+'%.1f'%latency+'</div>')
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

        ## printing for debug
        # for i in s.plots['FP1'].items:
        #     if 'data' in dir(i):
        #         print( type(i), i.data[:2]  )  

app = QtGui.QApplication(sys.argv)
GUI = Picker()
sys.exit(app.exec_())
