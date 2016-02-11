'''HBNL Peak Picker

to start:
	/usr/local/bin/bokeh serve PeakPicker.py
	*to make the page accessible across out LAN
		add --address 138.5.49.214 --host 138.5.49.214:5006
point browser to:
	http://localhost:5006/PeakPicker/
	*replace 'localhost' with url if on another computer

'''

# import logging
# logging.basic.Config(level=logging.DEBUG)
import os
import sys
#repo_path = '/export/home/mike/python/mort-collab'
repo_path = '/export/home/mort/programs/dev'
if repo_path not in sys.path:
	sys.path.append(repo_path)
import numpy as np
import pandas as pd

import organization as O
import EEGdata

from bokeh.plotting import Figure, gridplot, hplot, vplot, output_server
from bokeh.models import ( Plot, Panel, Tabs, ColumnDataSource, CustomJS,
					BoxSelectTool, TapTool, GridPlot,
				BoxZoomTool, ResetTool, PanTool, WheelZoomTool, ResizeTool,
				Asterisk, Segment, Line, Grid, 
				LinearAxis, Range1d, AdaptiveTicker, CompositeTicker )

from bokeh.models.widgets import VBox, Slider, TextInput, VBoxForm, Select, CheckboxGroup, \
				RadioButtonGroup, Button
from bokeh.client import push_session
from bokeh.io import curdoc, curstate, set_curdoc

experiments = ['ant','vp3']#,'aod']
app_data = { expr:{} for expr in experiments }
init_files = ['ant_0_a0_11111111_avg.h1', 'vp3_0_a0_11111111_avg.h1']
app_data['file paths'] = [ os.path.join(os.path.dirname(__file__),f) for f in init_files ]
app_data['paths input'] = []
#fields: file paths, file ind

file_chooser = TextInput( title="files", name='file_chooser',
				 value='/processed_data/avg-h1-files/ant/l8-h003-t75-b125/suny/ns32-64/ant_5_e1_40143015_avg.h1 /processed_data/avg-h1-files/ant/l8-h003-t75-b125/suny/ns32-64/ant_5_e1_40146034_avg.h1')
start_button = Button( label="Start" )

text = TextInput( title="file", name='file', value='')


def load_file(next=False, initialize=False, reload_flag=False):
	
	paths = app_data['file paths']

	if not initialize and not reload_flag:
		if next and app_data['file ind'] < len(paths)-1:
			app_data['file ind'] += 1
		else:
			print('already on last file')
			return

	ind = app_data['file ind']
	
	if ind < len(paths) or not next:
		eeg = EEGdata.avgh1( paths[ind] )
		experiment = eeg.file_info['experiment']
		print('Load  ', experiment,' n paths, ind: ', len(paths), ind, eeg.file_info)
		app_data['current experiment'] = experiment
		expD = app_data[experiment]
		expD['eeg'] =  eeg
		data_sourceD, peak_sourcesD = expD['eeg'].make_data_sources()
		if initialize: # initialize
			expD['peak sources'] = { case:ColumnDataSource( data = D ) for case,D in peak_sourcesD.items() }				
			expD['data source'] = ColumnDataSource( data = data_sourceD )
			expD['cases'] = eeg.case_list
			print( 'initial expD: ', expD, app_data[experiment])
		else:
			expD['data source'].data = data_sourceD
			for case,D in peak_sourcesD.items():
				expD['peak sources'][case].data = D
				expD['peak sources'][case].set()


		text.value = paths[ind]

		expD['data source'].trigger('data',expD['data source'].data,
										expD['data source'].data)
		


def next_file():
	load_file(next=True)

def start_handler():
	#exp_path = '/processed_data/mt-files/vp3/suny/ns/a-session/vp3_3_a1_40025009_avg.h1'
	#'/processed_data/avg-h1-files/ant/l8-h003-t75-b125/suny/ns32-64/ant_5_a1_40026180_avg.h1'
	print('Start:  ', app_data['current experiment'])

	paths = file_chooser.value.split(' ')
	if len(app_data['paths input']) > 0 and paths == app_data['paths input'][-1]:
		return
	app_data['paths input'].append(paths)

	for expch in experiments:
		if expch in paths[0]:
			app_data['current experiment'] = expch
			continue
	######################  could validate other paths here

	app_data['file paths'] = paths
	app_data['file ind'] = -1
	print('Start done:  ', app_data['current experiment'])
	next_file()

# Initialize tabs
app_data['file ind'] = 0
load_file( initialize=True )
app_data['file ind'] = 1
load_file( initialize=True )


# ***************************** Temporary setup **********************
#start_handler()


def make_box_callback( experiment ):
	pick_source = app_data[experiment]['pick source']
	box_callback = CustomJS(args=dict(source=pick_source), code="""
			        // get data source from Callback args
			        console.dir(cb_data)
			        var data = source.get('data');

			        /// get BoxSelectTool dimensions from cb_data parameter of Callback
			        var geometry = cb_data['geometry'];

			        /// calculate Rect attributes
			        var width = geometry['x1'] - geometry['x0'];
			        var height = geometry['y1'] - geometry['y0'];
			        var x = geometry['x0'] + width/2;
			        var y = geometry['y0'] + height/2;

			        /// update data source with new Rect attributes
			        data['x'].push(x);
			        data['y'].push(y);
			        data['width'].push(width);
			        data['height'].push(height);
			        data['start'].push(geometry['x0'])
			        data['finish'].push(geometry['x1'])
			        data['bots'].push(-5)
			        data['tops'].push(25)
	 		        console.dir(data)
			        // trigger update of data source
			        source.trigger('change');
			        source.set('data',data)
			        console.dir(source)
			    """)
	print('box_callback: ', type(box_callback))
	return box_callback

def box_gen_gen( experiment ):
	box_callback = make_box_callback(experiment)
	def box_generator():
		return BoxSelectTool( callback=box_callback )
	return box_generator


plot_props = {'width':180, 'height':110,
				 'extra_bottom_height':40, # for bottom row
				'min_border':4}

#chans = ['FZ','CZ','PZ','F3','C3','P3']
#chans = eeg.electrodes[:31]
#chans.append(eeg.electrodes[63])
chans = ['FP1', 'Y',  'FP2', 'X', 'F7', 'AF1', 'AF2', 'F8', 'F3', 'FZ',  'F4',
		 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ',  'C4',  'T8', 'CP5',
		 'CP1', 'CP2', 'CP6', 'P3', 'PZ',  'P4', 'P7', 'PO1', 'PO2', 'P8',
		 'O1',  'O2']

def make_plot(plot_setup, experiment):
	if plot_setup == None:
		return None
	PS = plot_setup
	props = PS['props']

	plot = Plot( title=PS['electrode'], tools=PS['tools'])
	plot.title_standoff = 0
	plot.title_text_align='left'
	plot.title_text_baseline='top'
	plot.min_border_left = props['min_border']
	plot.min_border_right = props['min_border']
	plot.min_border_top = props['min_border']
	plot.min_border_bottom = props['min_border']
	plot.plot_width = props['width']
	plot.plot_height = PS['adjusted height']
	plot.y_range = Range1d(*props['yrange'])
	plot.x_range = Range1d(*props['xrange'])#, name='sharedX')
	plot.title_text_font_size = str(props['font size'])+'pt'
	plot.outline_line_alpha = props['outline alpha']
	plot.outline_line_width = None
	plot.outline_line_color = None

	# Axes
	xAxis = LinearAxis()#x_range_name='sharedX')
	#xTicker = AdaptiveTicker(base=10,mantissas=[0,4],min_interval=50)
	xTicker_0 = AdaptiveTicker(base=100,mantissas=[0,4],min_interval=400)#SingleIntervalTicker(interval=400)
	xTicker_1 = AdaptiveTicker(base=10,mantissas=[2,5],min_interval=20,max_interval=400)
	xTicker = CompositeTicker(tickers=[xTicker_0,xTicker_1])
	xAxis.ticker = xTicker
	xGrid = Grid(dimension=0, ticker=xTicker)
	
	yAxis = LinearAxis()
	yTicker_0 = AdaptiveTicker(base=10,mantissas=[1],min_interval=10)#SingleIntervalTicker(interval=10)#desired_num_ticks=2,num_minor_ticks=1)
	yTicker_1 = AdaptiveTicker(base=2,mantissas=[2],max_interval=10,min_interval=2)#SingleIntervalTicker(interval=1, max_interval=10)
	yTicker_2 = AdaptiveTicker(base=0.1,mantissas=[4],max_interval=2)
	yTicker = CompositeTicker(tickers=[yTicker_0, yTicker_1, yTicker_2])
	yAxis.ticker = yTicker
	
	xAxis.axis_label_text_font_size = str(props['font size'])+'pt'
	xAxis.major_label_text_font_size = str(props['font size']-2)+'pt'
	xAxis.major_label_text_align = 'right'
	xAxis.major_label_standoff = 2
	xAxis.minor_tick_line_color = None
	xAxis.major_tick_out = 0
	xAxis.major_tick_in = 2
	plot.add_layout(xAxis,'below')
	xGrid.grid_line_alpha = props['grid alpha']
	plot.add_layout(xGrid)

	yAxis.axis_label_text_font_size = str(props['font size'])+'pt'
	yAxis.major_label_text_font_size = str(props['font size']-2)+'pt'
	yAxis.major_label_standoff = 2
	yAxis.minor_tick_line_color = None
	yAxis.major_tick_out = 0
	yAxis.major_tick_in = 4
	plot.add_layout(yAxis,'left')

	if PS['tool generators']:
		plot.add_tools(*[ g() for g in PS['tool generators'] ])

	for cs_ind,case in enumerate(PS['case list']):
		line= Line( x='times', y=PS['electrode']+'_'+case, line_color=props['colors'][cs_ind],
				line_width=1.5, line_alpha=0.85, name=case+'_line')
		plot.add_glyph(app_data[experiment]['data source'],line)

	return plot

#########################
##		Callbacks
#########################
def update_data( peak_data ):
	app_data[app_data['current experiment']]['peak source'].data = peak_data

# def input_change(attr,old,new):
# 	pass

def apply_handler():
	print('Apply')
	exp = app_data[app_data['current experiment']]
	eeg = exp['eeg']
	#print( peak_source.data )
	limitsDF = exp['pick source'].to_df()
	start = limitsDF[ 'start' ].values[-1]
	finish = limitsDF[ 'finish' ].values[-1]
	
	case = exp['pick state']['case']
	peak = exp['pick state']['peak']	
	pval,pms = eeg.find_peak(case,start_ms=start,end_ms=finish)
	eeg.update_peak_source( exp['peak sources'][case].data, 
				case,peak,pval, pms)
	exp['peak sources'][case].set()

	print( 'pick_state: ', exp['pick state'])
	print( 'Values:',pval, 'Times:',pms)
	print( 'Values:',len(pval), 'Times:',len(pms) )
	#print( dir(peak_source) )
	#push_session(curdoc())
	exp['peak sources'][case].trigger('data', exp['peak sources'][case].data, 
									exp['peak sources'][case].data)
	print(exp['peak sources'])
	for c in exp['peak sources'].keys():
		print(exp['peak sources'][c].data)


def save_handler():
	print('Save')
	exp = app_data[app_data['current experiment']]
	eeg = exp['eeg']
	# get list of cases which have picks and unique peaks
	case_lst = []
	peak_lst = []
	for case in eeg.case_list:
		if exp['peak sources'][case].data['peaks']: # if case contains picks
			case_lst.append( eeg.case_num_map[case] ) #use numeric reference
			pks = exp['peak sources'][case].data['peaks']
			for pk in pks:
				peak_lst.append(pk)
		else:
			print('A case is missing picks.')
			# return
	cases = list(set(case_lst))
	peaks = list(set(peak_lst))

	# get amps and lats as ( peak, chan, case ) shaped arrays
	n_cases = len(cases)
	n_chans = 61 # only core 61 chans
	n_peaks = len(peaks)
	amps = np.empty( (n_peaks, n_chans, n_cases) )
	lats = np.empty( (n_peaks, n_chans, n_cases) )
	for icase, case in enumerate(cases):
		case_name = eeg.case_list[icase]
		for ichan, chan in enumerate(eeg.electrodes_61): # only core 61 chans
			for ipeak, peak in enumerate(peaks):
				amps[ipeak, ichan, icase] = \
						exp['peak sources'][case_name].data[chan+'_pot'][peaks.index( peak )]
				lats[ipeak, ichan, icase] =	\
						exp['peak sources'][case_name].data[chan+'_time'][peaks.index( peak )]

	# reshape into 1d arrays
	amps1d = amps.ravel('F')
	lats1d = lats.ravel('F')

	# build mt text (makes default output location)
	eeg.build_mt(cases, peaks, amps1d, lats1d)

	# write to a test location
	print(eeg.mt)
	test_dir = '/processed_data/mt-files/test'
	fullpath = os.path.join( test_dir, eeg.mt_name )
	of = open( fullpath, 'w' )
	of.write( eeg.mt )
	of.close()

def next_handler():
	print('Next')
	next_file()	

def reload_handler():
	print('Reload')
	load_file(reload_flag=True)

def case_toggle_handler(active):
	exp = app_data[app_data['current experiment']]
	chosen_case = exp['cases'][active]
	exp['pick state']['case'] = chosen_case
	for case in exp['cases']:
		width = 2.5 if case == chosen_case else 1.5
		selections = exp['grid'].select(dict(name=case+'_line'))
		#print( dir(selections[0]))
		for sel in selections:
			sel.line_width = width

def peak_toggle_handler(active):
	exp = app_data[app_data['current experiment']]
	exp['pick state']['peak'] = exp['cases'][active]

def checkbox_handler(active):
	exp = app_data[app_data['current experiment']]
	for n,nm in enumerate(exp['cases']):
		alpha = 1 if n in active else 0
		label = nm+'_line'
		selections = exp['grid'].select(dict(name=label))
		for sel in selections:
			sel.line_alpha = alpha
		marker_label = nm+'_peak'
		selections = exp['grid'].select(dict(name=marker_label))
		for sel in selections:
			#sel.fill_alpha = alpha
			sel.line_alpha = alpha

def input_change(attr, old, new):
	update_data()


peak_choices = ['P1','P3','P4','N1','N2','N3','N4']
def build_experiment_tab(experiment):
	print('Building tab for', experiment)
	components = {}
	expD = app_data[ experiment ]
	print([k for k in app_data.keys()], app_data[experiment])
	case_choices = expD['cases']

	expD['pick source'] = ColumnDataSource( data= dict( x=[], y=[], width=[], height=[],
								 start=[], finish=[], bots=[], tops=[] ))
	expD['pick state'] =  {'case':case_choices[0], 'peak':peak_choices[0]}

	case_pick_chooser = RadioButtonGroup( labels=case_choices, active=0 )

	case_display_toggle = CheckboxGroup( labels=case_choices, inline=True,
				active=[n for n in range(len(case_choices))] )

	peak_chooser = RadioButtonGroup( labels=peak_choices, active=0)

	apply_button = Button( label="Apply", type='default' )
	save_button = Button( label="Save" )
	next_button = Button( label="Next" )
	reload_button = Button( label="Reload")

	expD['controls'] = { 'case' : case_pick_chooser,
						 'peak' : peak_chooser,
						 'apply' : apply_button,
						 'save' : save_button,
						 'reload' : reload_button,
						 'case toggle' : case_display_toggle
						}

	components['inputs'] = [ case_pick_chooser, peak_chooser, apply_button, save_button,
							 next_button, reload_button, case_display_toggle ]

	start_button.on_click(start_handler)

	case_pick_chooser.on_click(case_toggle_handler)
	peak_chooser.on_click(peak_toggle_handler)

	case_display_toggle.on_click(checkbox_handler)
	apply_button.on_click(apply_handler)
	save_button.on_click(save_handler)
	next_button.on_click(next_handler)
	reload_button.on_click(reload_handler)

	box_gen=box_gen_gen(experiment)

	gridplots_setup = expD['eeg'].selected_cases_by_channel(cases='all',
				channels=chans,
				props=plot_props,  mode='server',
				source=expD['data source'],
				tool_gen=[box_gen,BoxZoomTool, WheelZoomTool, 
						ResetTool, PanTool, ResizeTool],
				style='layout'
				)

	gridplots = []
	for growS in gridplots_setup:
		gridplots.append([])
		for plotS in growS:
			gridplots[-1].append( make_plot( plotS, experiment ) )

	components['plots'] = gridplots

	pick_starts = Segment(x0='start',x1='start',y0='bots',y1='tops',
					line_width=1.5,line_alpha=0.95,line_color='darkgoldenrod',
					line_dash='dashed')
	pick_finishes = Segment(x0='finish',x1='finish',y0='bots',y1='tops',
					line_width=1.5,line_alpha=0.95,line_color='darkgoldenrod',
					line_dash='dashdot')
	expD['pick starts'] = pick_starts
	expD['pick finishes'] = pick_finishes

	gcount = -1
	for g_row in gridplots:
		for gp in g_row:
			if gp != None:
				gcount +=1
				chan = chans[gcount]
				for case in case_choices:
					marker = Asterisk( x=chan+'_time',y=chan+'_pot',
							size=4, line_alpha=1,line_color='black',
							name=case+'_peak')
					gp.add_glyph( expD['peak sources'][case], marker)
				gp.add_glyph(expD['pick source'],pick_starts)
				gp.add_glyph(expD['pick source'],pick_finishes)

	return components

files_setup = VBox(children=[ file_chooser, start_button ])
# LAYOUT
navigation = Panel( child=files_setup, title='Navigate' )

tab_setup = [ navigation ]

for expr in experiments:

	components = build_experiment_tab(expr)
	inputs = VBox( children=components['inputs'])
	grid = GridPlot( children=components['plots'] )
	app_data[expr]['grid'] = grid
	page = VBox( children=[inputs, grid])

	tab_setup.append( Panel( child=page, title='pick '+expr ) )


tabs = Tabs( tabs=tab_setup )

curdoc().add_root(tabs)

# case_toggle_handler(0)
# peak_toggle_handler(0)
# checkbox_handler([ n for n in range(len(case_choices))])