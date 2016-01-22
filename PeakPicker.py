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
from bokeh.models import Plot, Segment, ColumnDataSource, CustomJS, \
					BoxSelectTool, TapTool, GridPlot, \
				BoxZoomTool, ResetTool, PanTool, WheelZoomTool, ResizeTool, \
				Asterisk

from bokeh.models.widgets import VBox, Slider, TextInput, VBoxForm, Select, CheckboxGroup, \
				RadioButtonGroup, Button
from bokeh.client import push_session
from bokeh.io import curdoc, curstate, set_curdoc


exp_path = '/processed_data/mt-files/vp3/suny/ns/a-session/vp3_3_a1_40025009_avg.h1'
#exp_path = '/processed_data/avg-h1-files/ant/l8-h003-t75-b125/suny/ns32-64/ant_5_a1_40026180_avg.h1'
eeg_exp = EEGdata.avgh1( exp_path )
eeg = eeg_exp

data_source, peak_sources = eeg.make_data_sources()

text = TextInput( title="file", name='file', value=exp_path)

case_choices = eeg.case_list
case_toggle = CheckboxGroup( labels=case_choices, inline=True,
				active=[n for n in range(len(case_choices))] )

case_chooser = RadioButtonGroup( labels=case_choices, active=0 )

pick_source = ColumnDataSource( data= dict( x=[], y=[], width=[], height=[],
								 start=[], finish=[], bots=[], tops=[] ))
peak_choices = ['P1','P3','P4','N1','N2','N3','N4']
peak_chooser = RadioButtonGroup( labels=peak_choices, active=0)
pick_state = {'case':case_choices[0], 'peak':peak_choices[0]}

apply_button = Button( label="Apply", type='default' )
save_button = Button( label="Save" )

		#toolset = ['crosshair','pan','reset','resize','save','wheel_zoom']

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
def box_generator():
	box = BoxSelectTool( callback=box_callback )
	return box

tap_callback = CustomJS( args=dict(source=data_source), code="""
 	alert('clicked');
 	""" )
tap = TapTool( callback=tap_callback )

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

gridplots = eeg.selected_cases_by_channel(cases='all',
			channels=chans,
			props=plot_props,  mode='server',
			source=data_source,
			tool_gen=[box_generator,BoxZoomTool, WheelZoomTool, 
					ResetTool, PanTool, ResizeTool],
			style='layout'
			)

#print(out_inds)
#print(gridplots)
#print(rangecheck)

pick_starts = Segment(x0='start',x1='start',y0='bots',y1='tops',
				line_width=1.5,line_alpha=0.95,line_color='darkgoldenrod',
				line_dash='dashed')
pick_finishes = Segment(x0='finish',x1='finish',y0='bots',y1='tops',
				line_width=1.5,line_alpha=0.95,line_color='darkgoldenrod',
				line_dash='dashdot')
#gridplots[0][1].add_glyph(pick_source,pick_starts)
#gridplots[0][1].add_glyph(pick_source,pick_finishes)

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
				gp.add_glyph( peak_sources[case], marker)
			gp.add_glyph(pick_source,pick_starts)
			gp.add_glyph(pick_source,pick_finishes)

def update_data( peak_data ):
	peak_source.data = peak_data

def input_change(attr,old,new):
	pass

def apply_handler():
	print('Apply')
	#print( peak_source.data )
	limitsDF = pick_source.to_df()
	start = limitsDF[ 'start' ].values[-1]
	finish = limitsDF[ 'finish' ].values[-1]
	
	case = pick_state['case']	
	pval,pms = eeg.find_peak(case,start_ms=start,end_ms=finish)
	eeg.update_peak_source( peak_sources[case].data, 
				case,pick_state['peak'],pval, pms)
	peak_sources[case].set()

	print( 'pick_state: ', pick_state)
	print( 'Values:',pval, 'Times:',pms)
	print( 'Values:',len(pval), 'Times:',len(pms) )
	#print( dir(peak_source) )
	#push_session(curdoc())
	peak_sources[case].trigger('data', peak_sources[case].data, 
									peak_sources[case].data)

	# cdoc = curdoc()
	# print( cdoc, dir(cdoc) )
	# cstate = curstate()
	# print( cstate, dir(cstate) )
	# set_curdoc( cdoc )
	#update_data(peak_data = peak_source.data )

def save_handler():
	print('Save')
	
	# get list of unique peaks
	peak_lst = []
	for case in eeg.case_list:
		pks = peak_sources[case].data['peaks']
	
		for pk in pks:
			peak_lst.append(pk)
	peaks = list(set(peak_lst))

	# get amps and lats as ( peak, chan, case ) shaped arrays
	n_cases = len(eeg.cases)
	n_chans = 61 # only core 61 chans
	n_peaks = len(peaks)
	amps = np.empty( (n_peaks, n_chans, n_cases) )
	lats = np.empty( (n_peaks, n_chans, n_cases) )
	for icase, case in enumerate(eeg.cases.keys()):
		case_name = eeg.case_list[icase]
		for ichan, chan in enumerate(eeg.electrodes_61): # only core 61 chans
			for ipeak, peak in enumerate(peaks):
				amps[ipeak, ichan, icase] = \
						peak_sources[case_name].data[chan+'_pot'][peaks.index( peak )]
				lats[ipeak, ichan, icase] =	\
						peak_sources[case_name].data[chan+'_time'][peaks.index( peak )]

	# reshape into 1d arrays
	amps1d = amps.ravel('F')
	lats1d = lats.ravel('F')

	eeg.build_mt(peaks, amps1d, lats1d)
	print(eeg.mt)

def case_toggle_handler(active):
	chosen_case = case_choices[active]
	pick_state['case'] = chosen_case
	for case in case_choices:
		width = 2.5 if case == chosen_case else 1.5
		selections = grid.select(dict(name=case+'_line'))
		for sel in selections:
			sel.glyph.line_width = width

def peak_toggle_handler(active):
	pick_state['peak'] = peak_choices[active]

def checkbox_handler(active):
    for n,nm in enumerate(case_choices):
    	alpha = 1 if n in active else 0
    	label = nm+'_line'
    	selections = grid.select(dict(name=label))
    	for sel in selections:
    		sel.glyph.line_alpha = alpha
    	marker_label = nm+'_peak'
    	selections = grid.select(dict(name=marker_label))
    	for sel in selections:
    		#sel.fill_alpha = alpha
    		sel.line_alpha = alpha


def input_change(attr, old, new):
	s.update_data()
	s.plot.title = s.text.value



case_chooser.on_click(case_toggle_handler)
peak_chooser.on_click(peak_toggle_handler)


case_toggle.on_click(checkbox_handler)
apply_button.on_click(apply_handler)
save_button.on_click(save_handler)

text.on_change('value', input_change)



inputs= VBox( children=[ text, case_chooser, peak_chooser, 
 					apply_button, save_button, case_toggle ])

page = VBox( children=[inputs])
curdoc().add_root(inputs)
grid = gridplot( gridplots ) # gridplot works properly outside of curdoc

case_toggle_handler(0)
peak_toggle_handler(0)
#output_server("picker")
#session = push_session( curdoc() )
#session.loop_until_closed()