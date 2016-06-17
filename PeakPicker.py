'''HBNL Peak Picker

to start:
	1) Start the bokeh server:
		/usr/local/bin/bokeh serve --address 138.5.49.214 --host 138.5.49.214:5006 --port 5006 --allow-websocket-origin 138.5.49.214:8000
		* make sure that the first port (here, 5006) matches those found at the end of this script in push_session and autoload_server
	2) Add the app:
		python3 PeakPicker.py [username] [experiments (space delimited)]
	3) Start the python webserver to receive requests:
		python3 -m http.server 8000 --bind 138.5.49.214

	** on updating code, only step 2 needs to be repeated
	** NOTE: for steps 2 and 3, must be in same directory as app

point browser to:
	http://138.5.49.214:8000/PeakPicker.html
	*replace 'localhost' with url if on another computer
	** if a username was provided in step 2, add the suffix: '_username' between 'PeakPicker' and '.html'
	*** if experiments were specified add the additional suffix _exp1-exp2.html

'''

# import logging
# logging.basic.Config(level=logging.DEBUG)
import os
import sys
repo_path = '/export/home/mike/python/mort-collab'
#repo_path = '/export/home/mort/programs/dev'
if repo_path not in sys.path:
	sys.path.append(repo_path)
import numpy as np
import pandas as pd

import organization as O
import EEGdata

from bokeh.embed import autoload_server
from bokeh.document import Document
from bokeh.plotting import Figure, gridplot, hplot, vplot, output_server
from bokeh.models import ( Panel, Tabs, ColumnDataSource, CustomJS,
						   Plot, GridPlot, Grid, Renderer,
						   BoxSelectTool, TapTool, BoxZoomTool, ResetTool,
						   LinearAxis, Range1d, AdaptiveTicker, CompositeTicker, SingleIntervalTicker,
				 		   PanTool, WheelZoomTool, ResizeTool,
						   Asterisk, Segment, Line,
						   VBox, HBox )
from bokeh.models.widgets import ( Slider, TextInput, Select, CheckboxGroup,
				RadioButtonGroup, Button, Paragraph, Toggle )
from bokeh.client import push_session
from bokeh.io import curdoc, curstate, set_curdoc

init_files_by_exp = {'ant':'ant_0_a0_11111111_avg.h1', 
					'vp3':'vp3_0_a0_11111111_avg.h1',
					'aod':'aod_1_a1_11111111_avg.h1'}
				
# Process inputs
user = ''
if len(sys.argv) > 1:
	user = '_'+sys.argv[1]
experiments = ['ant','vp3','aod']
if len(sys.argv) > 2:
	experiments = []
	for exp in sys.argv[2:]:
		experiments.append(exp)
	user+= '_'+'-'.join(experiments)
init_files = [ init_files_by_exp[ ex ] for ex in experiments ]


app_data = { expr:{} for expr in experiments }

#userName for store path
if '_' in user:
	userName = user.split('_')[1]
else: userName = 'default'
app_data['user'] = userName
app_data['file paths'] = [ os.path.join(os.path.dirname(__file__),f) for f in init_files ]
app_data['paths input'] = []
	
#fields: file paths, file ind
dir_paths_by_exp = { 'ant':['/processed_data/avg-h1-files/ant/l8-h003-t75-b125/suny/ns32-64/',
							'ant_5_e1_40143015_avg.h1 ant_5_e1_40146034_avg.h1 ant_5_a1_40026180_avg.h1'],
					'vp3':['/processed_data/avg-h1-files/vp3/l16-h003-t75-b125/suny/ns32-64/',
							'vp3_5_a1_40021069_avg.h1 vp3_5_a1_40017006_avg.h1 vp3_5_a1_40026204_avg.h1'],
					'aod':['/processed_data/avg-h1-files/aod/l16-h003-t75-b125/suny/ns32-64/',
							'aod_6_a1_40021070_avg.h1 aod_6_a1_40021017_avg.h1 aod_6_a1_40017007_avg.h1']
					}

directory_chooser = TextInput( title="directory", name='directory_chooser',
						value=dir_paths_by_exp[experiments[0]][0] )
file_chooser = TextInput( title="files", name='file_chooser',
				 value=dir_paths_by_exp[experiments[0]][1])
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
	print('load file: #',ind, paths)
	if ind < len(paths) or not next:
		eeg = EEGdata.avgh1( paths[ind] )
		experiment = eeg.file_info['experiment']
		print('Load  ', experiment,' n paths, ind: ', len(paths), ind, eeg.file_info)
		app_data['current experiment'] = experiment
		expD = app_data[experiment]
		expD['eeg'] =  eeg
		data_sourceD, peak_sourcesD = expD['eeg'].make_data_sources(empty_flag=initialize)

		if initialize: # initialize
			expD['peak sources'] = { case:ColumnDataSource( data = D ) for case,D in peak_sourcesD.items() }				
			expD['data source'] = ColumnDataSource( data = data_sourceD )
			expD['cases'] = eeg.case_list
			print( 'initial expD: ', expD, app_data[experiment])
		else:
			expD['data source'].data = data_sourceD
			for case,D in peak_sourcesD.items():
				expD['peak sources'][case].data = D
				#expD['peak sources'][case].set()
				expD['peak sources'][case].trigger('data',expD['peak sources'][case].data,
														expD['peak sources'][case].data)

				expD['applied'][case] = []

				pickedD = expD['picked sources'][case].data
				expD['picked sources'][case].data = { k:[] for k in pickedD}
				for fd in pickedD.keys():
					pickedD[fd] = []
				#expD['picked sources'][case].set()
				expD['picked sources'][case].trigger('data',expD['picked sources'][case].data,
														expD['picked sources'][case].data)

			expD['applied picks display'].text = picked_state_text( app_data['current experiment'] )

			reset_current_pick_source(expD)

			print('info plot', dir(expD['info']))

			expD['info'][0].text = eeg.filename #+ '<br>'.join([ str(k)+':'+str(v) for k,v in eeg.exp.items() ])
			details = gather_info(eeg)
			expD['info'][1].text = details[0]
			expD['info'][2].text = details[1]
			expD['info'][3].text = details[2]

			scale_chans = [ch for ch in chans if ch not in ['X','Y']]
			yscale = eeg.get_yscale(channels=scale_chans)
			#print('updating yscale: ',yscale)
			#print('plot dir', dir(expD['components']['plots'][0][1].y_range))
			expD['y range'].start = yscale[0]
			expD['y range'].trigger('start',expD['y range'].start,expD['y range'].start)
			expD['y range'].end = yscale[1]
			expD['y range'].trigger('end',expD['y range'].end,expD['y range'].end)

			for plt_row in expD['components']['plots']:
				for plt in plt_row:
					if plt and 'y_range' in dir(plt):

						if reload_flag or next:

							chan = plt.title.split(' ')[0]
							plt.title = chan
							plt.trigger('title',plt.title,plt.title)

		text.value = paths[ind]

		expD['data source'].trigger('data',expD['data source'].data,
										expD['data source'].data)
		


def next_file():
	load_file(next=True)
	print( 'next file loaded' )

def start_handler():
	#exp_path = '/processed_data/mt-files/vp3/suny/ns/a-session/vp3_3_a1_40025009_avg.h1'
	#'/processed_data/avg-h1-files/ant/l8-h003-t75-b125/suny/ns32-64/ant_5_a1_40026180_avg.h1'
	print('Start:  ', app_data['current experiment'])

	directory = directory_chooser.value.strip()
	files = file_chooser.value.split(' ')
	files = [f for f in files if '.h1' in f]
	paths = [ os.path.join(directory,f) for f in files ]
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

	case_toggle_handler(0)
	peak_toggle_handler(0)
	checkbox_handler([ n for n in range(len(app_data[app_data['current experiment']]['peak sources']))])

# Initialize tabs
for f_ind, exp in enumerate(experiments):
	app_data['file ind'] = f_ind
	load_file( initialize=True )


# ***************************** Temporary setup **********************
#start_handler()


def make_box_callback( experiment ):
	pick_source = app_data[experiment]['current pick source']
	box_callback = CustomJS(args=dict(source=pick_source), code="""
			        // get data source from Callback args
			        console.dir(cb_data)
			        console.dir(cb_obj)
			        // console.dir(cb_obj.get('selected')['1d'].indices)

			        chans = ['FP1', 'Y',  'FP2', 'X', 'F7', 'AF1', 'AF2', 'F8', 'F3', 'FZ',  'F4',
		 					'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ',  'C4',  'T8', 'CP5',
		 					'CP1', 'CP2', 'CP6', 'P3', 'PZ',  'P4', 'P7', 'PO1', 'PO2', 'P8',
		 					'O1',  'O2',
		 					'AF7','FPZ','AFZ','AF8','F5','F1','F2','F6','FT7','FC3','FCZ','FC4','FT8',
		 					'C5','C1','C2','C6','TP7','CP3','CPZ','CP4','TP8','P5','P1','POZ',
		 					'P2','P6','PO7','OZ','PO8']
			        var data = source.get('data');

			        /// get BoxSelectTool dimensions from cb_data parameter of Callback
			        var geometry = cb_data['geometry'];

			        /// calculate Rect attributes
			        //var width = geometry['x1'] - geometry['x0'];
			        //var height = geometry['y1'] - geometry['y0'];
			        //var x = geometry['x0'] + width/2;
			        //var y = geometry['y0'] + height/2;

			        /// update data source with new Rect attributes
			        //data['x'].push(x);
			        //data['y'].push(y);
			        //data['width'].push(width);
			        //data['height'].push(height);
			        console.log('Single mode: '+data['single'])
			        if( data['single'][0] ){
			        	console.log('running in single channel mode')
			        	title = cb_obj['attributes']['plot']['attributes']['title']
			        	channel = title.slice(0,3)
			        	if( channel.slice(2,3) == ' ' ){ channel=channel.slice(0,2) }
			        	chans = [ channel ]
			        } 
			        for( i=0, clen=chans.length; i<clen; i++ ){
			        	ch = chans[i];
			        	data['start_'+ch][0]=geometry['x0'];
			        	data['finish_'+ch][0]=geometry['x1'];
			        	data['bots_'+ch][0]=geometry['y0'];
			        	data['tops_'+ch][0]=geometry['y1'];
			        }
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
		return BoxSelectTool( callback=box_callback, renderers=[ app_data['dummy_plotR'] ] )
	return box_generator

plot_props = {'width':180, 'height':110,
				 'extra_bottom_height':40, # for bottom row
				'min_border':4,
				'line colors':['#DD2222','#66DD66','#2222DD','#DD22DD']}

chans = ['FP1', 'Y',  'FP2', 'X', 'F7', 'AF1', 'AF2', 'F8', 'F3', 'FZ',  'F4',
		 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ',  'C4',  'T8', 'CP5',
		 'CP1', 'CP2', 'CP6', 'P3', 'PZ',  'P4', 'P7', 'PO1', 'PO2', 'P8',
		 'O1',  'O2',
		 'AF7','FPZ','AFZ','AF8','F5','F1','F2','F6','FT7','FC3','FCZ','FC4','FT8',
		 'C5','C1','C2','C6','TP7','CP3','CPZ','CP4','TP8','P5','P1','POZ',
		 'P2','P6','PO7','OZ','PO8'
		 ]

def make_plot(plot_setup, ranges, experiment, tool_generators):
	PS = plot_setup
	props = PS['props']
	if plot_setup['show'] == False:
		dummy = True#return None#dummy_plot(tool_generators)
		title = ''
	else: 
		dummy = False
		title = PS['electrode']
		# alpha = props['outline alpha']

	plot = Plot( title=title, tools=PS['tools'])
	plot.title_standoff = 0
	plot.title_text_align='center'
	plot.title_text_baseline='top'
	plot.min_border_left = props['min_border']
	plot.min_border_right = props['min_border']
	plot.min_border_top = props['min_border']
	plot.min_border_bottom = props['min_border']
	plot.plot_width = props['width']
	plot.plot_height = PS['adjusted height']
	plot.y_range = ranges[1]
	plot.x_range = ranges[0]
	plot.title_text_font_size = str(props['font size'])+'pt'
	plot.outline_line_alpha = props['outline alpha']
	plot.outline_line_width = None
	plot.outline_line_color = None

	app_data['dummy_plot'] = Asterisk( x='dummy', y='dummy', name='dummy')
	app_data['dummy_plotR'] = plot.add_glyph(app_data[experiment]['data source'], app_data['dummy_plot'])

	if not dummy:
		# Axes
		xAxis = LinearAxis()#x_range_name='sharedX')
		#xTicker = AdaptiveTicker(base=10,mantissas=[0,4],min_interval=50)
		xTicker_0 = AdaptiveTicker(base=100,mantissas=[0,4],min_interval=400)#SingleIntervalTicker(interval=400)
		xTicker_1 = AdaptiveTicker(base=10,mantissas=[2,5],min_interval=20,max_interval=400)
		xTicker = CompositeTicker(tickers=[xTicker_0,xTicker_1])
		xAxis.ticker = xTicker
		xGrid = Grid(dimension=0, ticker=xTicker)
		
		yAxis = LinearAxis()
		# yTicker_0 = AdaptiveTicker(base=10,mantissas=[1],min_interval=10)#SingleIntervalTicker(interval=10)#desired_num_ticks=2,num_minor_ticks=1)
		# yTicker_1 = AdaptiveTicker(base=2,mantissas=[2],max_interval=10,min_interval=2)#SingleIntervalTicker(interval=1, max_interval=10)
		# yTicker_2 = AdaptiveTicker(base=0.1,mantissas=[4],max_interval=4, min_interval=2)
		yTicker_0 = AdaptiveTicker(base=5,mantissas=[1],min_interval=10)#SingleIntervalTicker(interval=10)#desired_num_ticks=2,num_minor_ticks=1)
		yTicker_1 = AdaptiveTicker(base=40,mantissas=[1],max_interval=50,min_interval=10)#SingleIntervalTicker(interval=1, max_interval=10)
		yTicker_2 = AdaptiveTicker(base=100,mantissas=[1],max_interval=200, min_interval=50)
		# yTicker_0 = SingleIntervalTicker(desired_num_ticks=1, interval=5, num_minor_ticks=0)
		# yTicker_1 = SingleIntervalTicker(desired_num_ticks=1, interval=20, num_minor_ticks=0)
		# yTicker_2 = SingleIntervalTicker(desired_num_ticks=1, interval=100, num_minor_ticks=0)		
		
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

		for cs_ind,case in enumerate(PS['case list']):

			line= Line( x='times', y=PS['electrode']+'_'+case, line_color=props['line colors'][cs_ind],
					line_width=1.5, line_alpha=0.85, name=case+'_line')
			plot.add_glyph(app_data[experiment]['data source'],line)


	if PS['tool generators']:
		plot.add_tools(*[ g() for g in PS['tool generators'] ])


	# if dummy:
	# 	return None
	return plot

#########################
##		Utilities
#########################

def reset_current_pick_source(exp):
	limits_data = exp['current pick source'].data
	exp['current pick source'].data = { k:[] for k in limits_data.keys() }
	exp['current pick source'].set()
	exp['current pick source'].trigger('data',exp['current pick source'].data,exp['current pick source'].data)

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
	
	limits_data = exp['current pick source'].data
	starts = [ limits_data['start_' + ch ] for ch in chans ]
	fins = [ limits_data['finish_'+ch ] for ch in chans ]

	reset_current_pick_source(exp)

	case = exp['pick state']['case']
	peak = exp['pick state']['peak']

	repick = False
	if peak in exp['applied'][case]:
		repick = True
		pk_ind = exp['applied'][case].index( peak )
	else:
		exp['applied'][case].append(peak)

	exp['applied picks display'].text = picked_state_text( app_data['current experiment'] )
	picked_data = limits_data.copy()

	picked_data['peaks'] = [ peak ]

	case_picks = exp['picked sources'][case].data
	if not repick: 
		for key in case_picks.keys():
			case_picks[key].append(picked_data[key][0])
	else:
		for key in case_picks.keys():
			case_picks[key][pk_ind] = picked_data[key][0]

	exp['picked sources'][case].data = case_picks
	exp['picked sources'][case].set() 
	exp['picked sources'][case].trigger('data', exp['picked sources'][case].data, exp['picked sources'][case].data)
	print('picked source data: ',exp['picked sources'][case].data)
	
	#if not exp['pick state']['single']:
	pval,pms = eeg.find_peaks(case,chans,starts_ms=starts,ends_ms=fins,polarity=peak[0].lower())
	#else:
	#	print('Need to implement single repick')

	# need to fill unused channels until all are implemented
	extra_chans = set(eeg.electrodes).difference(chans)
	print(extra_chans)
	psData = exp['peak sources'][case].data
	if not repick:
		psData[ 'peaks' ].append( peak )
		for chan, val, tm in zip(chans, pval, pms):
			psData[ chan+'_pot' ].append( val )
			psData[ chan+'_time' ].append( tm )
		for chan in extra_chans:
			psData[ chan+'_pot' ].append( 0 )
			psData[ chan+'_time'].append( 0 )
	else:
		for chan, val, tm in zip(chans, pval, pms):
			psData[ chan+'_pot' ][pk_ind] = val 
			psData[ chan+'_time' ][pk_ind] = tm		

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

	for plt_row in exp['components']['plots']:
		for plt in plt_row:
			if plt:
				chan = plt.title.split(' ')[0]
				latency = exp['peak sources'][case].data[chan+'_time'][-1]
				potential = exp['peak sources'][case].data[chan+'_pot'][-1]
				plt.title = chan + ' - lat: '+'{:3.1f}'.format(latency)+' amp: '+'{:4.3f}'.format(potential)
				plt.trigger('title',plt.title,plt.title)

	sync_current_selection()

def save_handler():
	print('Save')
	exp = app_data[app_data['current experiment']]
	eeg = exp['eeg']
	# get list of cases which have picks and unique peaks
	case_lst = []
	peak_lst = []
	peaks_by_case = {}
	for case in eeg.case_list:
		peaks_by_case[case] = []
		pks = exp['peak sources'][case].data['peaks']
		print(case, 'peaks: ', pks )
		if pks:#exp['peak sources'][case].data[chans[0]+'_pot']: # if case contains picks
			case_lst.append( eeg.case_num_map[case] ) #use numeric reference
			for pk in pks:
				peak_lst.append(pk)
				peaks_by_case[case] .append(pk)
		else:
			print('A case is missing picks.') # should handle this
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
			for peak in peaks_by_case[case_name]:
				ipeak = peaks.index(peak)
				amps[ipeak, ichan, icase] = \
						exp['peak sources'][case_name]\
						.data[chan+'_pot'][peaks_by_case[case_name].index( peak )]
				lats[ipeak, ichan, icase] =	\
						exp['peak sources'][case_name]\
						.data[chan+'_time'][peaks_by_case[case_name].index( peak )]

	# reshape into 1d arrays
	amps1d = amps.ravel('F')
	lats1d = lats.ravel('F')

	# build mt text (makes default output location), write to a test location
	eeg.build_mt(cases, peaks, amps1d, lats1d)
	print(eeg.mt)
	test_dir = os.path.join('/active_projects/test', app_data['user'] )
	if not os.path.exists(test_dir):
		os.mkdir(test_dir)
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

	for case in exp['cases']:
		selections = exp['grid'].select(dict(name=case+'_limit'))
		alpha= 1 if case == chosen_case else 0
		for sel in selections:
			sel.line_alpha = alpha

	if exp['pick state']['peak'] not in exp['applied'][ exp['pick state']['case'] ]:
		exp['controls']['multi-single toggle'].active = False

	sync_current_selection()

def peak_toggle_handler(active):
	exp = app_data[app_data['current experiment']]
	exp['pick state']['peak'] = peak_choices[active]

	if exp['pick state']['peak'] not in exp['applied'][ exp['pick state']['case'] ]:
		exp['controls']['multi-single toggle'].active = False
	
	sync_current_selection()

def sync_current_selection():
	exp = app_data[app_data['current experiment']]
	case, peak = exp['pick state']['case'], exp['pick state']['peak']
	case_picks = exp['picked sources'][case]
	cp_data = case_picks.data
	print( 'Sync for: ', case, peak, cp_data['peaks'])
	if peak in cp_data['peaks']:
		pk_ind = cp_data['peaks'].index(peak)
		update_data = {}
		for fd in cp_data.keys():
			if 'peak' not in fd:
				update_data[fd] = [ cp_data[fd][pk_ind] ]
		update_data['single'] = [ exp['pick state']['single'] ]

		exp['current pick source'].data = update_data
		exp['current pick source'].set()
		exp['current pick source'].trigger('data',update_data,update_data)

def update_case_peak_selection_display():
	for case_peak in app_data[app_data['current experiment']]['picked sources'].keys():
		pass

def checkbox_handler(active):
	exp = app_data[app_data['current experiment']]
	for n,cs in enumerate(exp['cases']):
		alpha = 1 if n in active else 0
#		visible = True if n in active else False
		label = cs+'_line'
		selections = exp['grid'].select(dict(name=label))
		for sel in selections:
			sel.line_alpha = alpha
		for pk in peak_choices:
			marker_label = cs+'_marker'
			selections = exp['grid'].select(dict(name=marker_label))
			for sel in selections:
				#sel.fill_alpha = alpha
				sel.line_alpha = alpha
#				sel.visible = visible 
def multi_single_toggle_handler(state):
	exp = app_data[app_data['current experiment']]
	print('multi-single toggle state:', state)
	exp['controls']['multi-single toggle'].label='one' if state else 'all'
	exp['pick state']['single'] = state
	exp['current pick source'].data['single'] = [state]
	exp['current pick source'].set()

	sync_current_selection()

def input_change(attr, old, new):
	update_data()

def picked_state_text(experiment):
	txt = 'Picked: '
	for case in app_data[experiment]['cases']:
		peak_set = app_data[experiment]['applied'][case]
		txt += case.upper() + ':[' + ','.join(list(peak_set))+']  '
	return txt

peak_choices = ['P1','P2','P3','P4','N1','N2','N3','N4']
def build_experiment_tab(experiment):
	print('Building tab for', experiment)
	components = {}
	expD = app_data[ experiment ]
	print([k for k in app_data.keys()], app_data[experiment])
	case_choices = expD['cases']

	#expD['current pick source'] = ColumnDataSource( data= dict( start=[], finish=[], bots=[], tops=[] ) )

	expD['applied'] = {}; expD['picked sources'] = {}
	for case in case_choices:
		expD['applied'][case] = []
		for peak in peak_choices:
			peak_sourceD = {'peaks':[]}
			for chan in chans:
				peak_sourceD.update( { fd+'_'+chan:[] for fd in ['start', 'finish', 'bots', 'tops'] } ) #, 'peak'
						#dict( start=[], finish=[], bots=[], tops=[], peak=[] ) )
		expD['picked sources'][case] = ColumnDataSource( data=peak_sourceD )

	expD['pick state'] =  {'case':case_choices[0], 'peak':peak_choices[0], 
							'single':False, 'picked':{} }

	case_pick_chooser = RadioButtonGroup( labels=case_choices, active=0 )

	case_display_toggle = CheckboxGroup( labels=case_choices, inline=True,
				active=[n for n in range(len(case_choices))] )

	peak_chooser = RadioButtonGroup( labels=peak_choices, active=0)

	apply_button = Button( label="Apply", type='default' )
	save_button = Button( label="Save" )
	next_button = Button( label="Next" )
	reload_button = Button( label="Reload")
	# toggle to be placed on display line
	multi_single_pick_toggle = Toggle( label= 'all', type="success" )

	expD['controls'] = { 'case' : case_pick_chooser,
						 'peak' : peak_chooser,
						 'apply' : apply_button,
						 'save' : save_button,
						 'reload' : reload_button,
						 'case toggle' : case_display_toggle,
						 'multi-single toggle': multi_single_pick_toggle
						}

	pick_title = Paragraph(height=12, width=65, text='pick: case')
	peak_title = Paragraph(height=12, width=28, text='peak')
	components['pick controls'] = [ pick_title, case_pick_chooser, peak_title, peak_chooser, 
							 apply_button, save_button, next_button, reload_button]
	display_title = Paragraph(height=12, width=54, text='display:')
	# Legend
	legend_title = Paragraph(height=12, width=44, text='legend:')
	display_elements = [ display_title, case_display_toggle, legend_title ]
	for cc in case_choices:
		display_elements.append( Paragraph(height=18, width=25, text=cc ) )

	spacer = Paragraph(height=12, width=15)
	picked_status = Paragraph(height=12, width=230, text='No picks yet')
	display_elements.append( spacer )
	display_elements.append( picked_status )
	expD['applied picks display'] = picked_status

	repick_title = Paragraph(height=15, width=35, text='repick')

	display_elements.extend([repick_title, multi_single_pick_toggle])

	components['display elements'] = display_elements

	start_button.on_click(start_handler)

	case_pick_chooser.on_click(case_toggle_handler)
	peak_chooser.on_click(peak_toggle_handler)

	case_display_toggle.on_click(checkbox_handler)
	apply_button.on_click(apply_handler)
	save_button.on_click(save_handler)
	next_button.on_click(next_handler)
	reload_button.on_click(reload_handler)
	multi_single_pick_toggle.on_click(multi_single_toggle_handler)

	current_pickD = {}
	for chan in chans:
		current_pickD.update( { fd+'_'+chan:[] for fd in ['start','finish','bots','tops'] } )
	current_pickD['single'] = [ False ]
	expD['current pick source'] = ColumnDataSource( data=current_pickD )

	box_gen = box_gen_gen(experiment)

	plot_tool_generators = [box_gen,BoxZoomTool, WheelZoomTool, 
						ResetTool, PanTool, ResizeTool]

	gridplots_setup = expD['eeg'].selected_cases_by_channel(cases='all',
				channels=chans,
				props=plot_props,  mode='server',
				source=expD['data source'],
				tool_gen=plot_tool_generators,
				style='layout'
				)

	# Ranges
	expD['y range'] = Range1d(*gridplots_setup[0][1]['props']['yrange'])
	expD['x range'] = Range1d(*gridplots_setup[0][1]['props']['xrange'])#, name='sharedX')

	gridplots = []
	components['plots'] = []
	for growS in gridplots_setup:
		gridplots.append([])
		components['plots'].append([])
		for plotS in growS:
			if plotS == None:
				plotS = gridplots_setup[0][1].copy()
				plotS['show'] = False
			else:
				plotS['show'] = True

			this_plot = make_plot( plotS, (expD['x range'], expD['y range']),
								experiment, plot_tool_generators )
			gridplots[-1].append( this_plot )
			if plotS['show']:
				components['plots'][-1].append( this_plot )
			else: components['plots'][-1].append( None )

	expD['pick starts'] = {}
	expD['pick finishes'] = {}
	#expD['case pick sources'] = {}
	gcount = -1
	for gr_ind,g_row in enumerate(gridplots):
		for gc_ind,gp in enumerate(g_row):
			if gridplots_setup[gr_ind][gc_ind] != None:
				gcount +=1
				chan = chans[gcount]
				current_pick_start = Segment(x0='start_'+chan,x1='start_'+chan,y0='bots_'+chan,y1='tops_'+chan,
								line_width=1.5,line_alpha=0.95,line_color='#f2b41e',
								line_dash='dashed')
				current_pick_finish = Segment(x0='finish_'+chan,x1='finish_'+chan,y0='bots_'+chan,y1='tops_'+chan,
								line_width=1.5,line_alpha=0.95,line_color='#f2b41e',
								line_dash='dashdot')
				expD['pick starts'][chan] = current_pick_start
				expD['pick finishes'][chan] = current_pick_finish

				gp.add_glyph( expD['current pick source'],current_pick_start)
				gp.add_glyph( expD['current pick source'],current_pick_finish)
				for case in case_choices:

					#for peak in peak_choices:
						#cspk = case+'_'+peak
					marker = Asterisk( x=chan+'_time',y=chan+'_pot',
						size=4, line_alpha=1,line_color='black',
						name=case+'_marker')
					gp.add_glyph( expD['peak sources'][case], marker)
						# gp.add_glyph( expD['picked sources'][cspk],picked_starts)
						# gp.add_glyph( expD['picked sources'][cspk],picked_finishes)

					#expD['case pick sources'][case] = ColumnDataSource( data= dict( start=[], finish=[], bots=[], tops=[] ) )
					case_pick_starts = Segment(x0='start_'+chan,x1='start_'+chan,y0='bots_'+chan,y1='tops_'+chan,
					line_width=1.5,line_alpha=0.95,line_color='#886308',
					line_dash='dashed', name=case+'_limit' )
					case_pick_finishes = Segment(x0='finish_'+chan,x1='finish_'+chan,y0='bots_'+chan,y1='tops_'+chan,
					line_width=1.5,line_alpha=0.95,line_color='#886308',
					line_dash='dashdot', name=case+'_limit')
					gp.add_glyph( expD['picked sources'][case], case_pick_starts )
					gp.add_glyph( expD['picked sources'][case], case_pick_finishes )
	#print( 'Built sources:', expD )

	return components, gridplots

files_setup = VBox(children=[ directory_chooser, file_chooser, start_button ])
# LAYOUT
navigation = Panel( child=files_setup, title='Navigate' )

tab_setup = [ navigation ]


def gather_info(exp):
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

	return [ filter_info ] + case_info

for expr in experiments:
	expD = app_data[expr]
	components, grid_display = build_experiment_tab(expr)
	expD['components'] = components 
	pick_controls = HBox( children=components['pick controls'])
	display = HBox( children=components['display elements'] )
	inputs = VBox( children=[pick_controls, display])

	info_el = Paragraph(height=12, width=300, text='Info')#make_info_plot()
	#info2 = PreText(text='<tr><td><font color="red">Case1</font></td><td><font color="blue">Case2</font></td></tr>')
	info_ch = [info_el]
	proc_info = gather_info(expD['eeg'])
	for text_line in proc_info:
		info_ch.append( Paragraph( height=11, width=300, text='' ) )
	expD['info'] = info_ch
	info = VBox(children=info_ch)
	inputsNinfo = HBox(children=[inputs, info])#GridPlot(children=[[info]])])
	# need to add css: bk-hbox-spacer{ margin-right:0 }

	grid = GridPlot( children=grid_display )
	expD['grid'] = grid
	page = VBox( children=[inputsNinfo, grid])

	tab_setup.append( Panel( child=page, title='pick '+expr ) )


tabs = Tabs( tabs=tab_setup )
#print('custate: ',dir(curstate()))

document = Document()
session = push_session(document,url='http://138.5.49.214:5006')


html = """
<html>
    <head></head>
    <body>
    	<h3> HBNL Peak Picker </h3>
        %s
    </body>
    <style>
    	p{ margin: 3px; }
		.bk-hbox-spacer{ margin-right:5 !important }
		.bk-vbox > p{ margin:1 !important;
					  font-size: 12px !important;
					}
		.bk-vbox > p:first-child{ font-size: 14px !important;
							  font-weight: bold !important;
							  margin-bottom: 3px !important;
							}
		.bk-bs-checkbox-inline{ margin-top: 4px; }
		.bk-bs-button{ padding-left: 6px !important;
					   padding-right: 6px !important; 
					  }
		.bk-bs-btn-group:first-child{ background-color: blue !important;
							  			font-color: white !important; 
							  		}
		.bk-bs-btn-success{ background-color: #fff !important;
							color: black !important;
							border-color: #ccc !important;
							height: 20 !important;
							margin-top: 3 !important;
							padding: 0 12 0 12 !important;
						}
		.bk-sidebar{ vertical-align: top !important; }
	</style>
	<script src="//ajax.googleapis.com/ajax/libs/jquery/1.11.0/jquery.min.js"></script>
	<script>
		setTimeout( function(){
			$("p").filter( function(i){ return this.textContent=='T' } ).css('background-color','#DD2222').css('color','white')
									.css('text-align','center').css('padding','2px')
			$("p").filter( function(i){ return this.textContent=='NT'} ).css('background-color','#66DD66').css('color','white')
									.css('text-align','center').css('padding','2px')
			$("p").filter( function(i){ return this.textContent=='NV'} ).css('background-color','#2222DD').css('color','white')
									.css('text-align','center').css('padding','2px')
			$("p").filter( function(i){ return this.textContent=='A' } ).css('background-color','#DD2222').css('color','white')
									.css('text-align','center').css('padding','2px')
			$("p").filter( function(i){ return this.textContent=='J' } ).css('background-color','#66DD66').css('color','white')
									.css('text-align','center').css('padding','2px')
			$("p").filter( function(i){ return this.textContent=='W' } ).css('background-color','#2222DD').css('color','white')
									.css('text-align','center').css('padding','2px')
			$("p").filter( function(i){ return this.textContent=='P' } ).css('background-color','#DD22DD').css('color','white')
									.css('text-align','center').css('padding','2px')
	
		}, 2000 )
	</script>
</html>


""" % autoload_server(tabs, session_id=session.id, url='http://138.5.49.214:5006')
#curdoc().add_root(tabs)
document.add_root(tabs)


with open("PeakPicker"+user+".html", "w+") as f:
    f.write(html)

if __name__ == "__main__":
    print("\npress ctrl-C to exit")
    session.loop_until_closed()
