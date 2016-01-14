'''HBNL dashboard

to start:
	bokeh serve --show PeakPicker.py
	*add --ip 138.5.49.214 to make the page accessible across out LAN
point browser to:
	http://localhost:5006/PeakPicker/
	*replace 'localhost' with url if on another computer

'''

# import logging
# logging.basic.Config(level=logging.DEBUG)
import sys
repo_path = '/export/home/mort/programs/dev'
if repo_path not in sys.path:
	sys.path.append(repo_path)
import numpy as np
import pandas as pd

import organization as O
import EEGdata

from bokeh.plotting import Figure, gridplot, hplot, vplot
from bokeh.models import Plot, Segment, ColumnDataSource, CustomJS, \
					BoxSelectTool, TapTool, GridPlot, \
				BoxZoomTool, ResetTool, PanTool, WheelZoomTool

from bokeh.models.widgets import VBox, Slider, TextInput, VBoxForm, Select, CheckboxGroup, \
				RadioButtonGroup, Button
from bokeh.io import curdoc, curstate


exp_path = '/processed_data/mt-files/vp3/suny/ns/a-session/vp3_3_a1_40025009_avg.h1'
eeg_exp = EEGdata.avgh1( exp_path )
eeg = eeg_exp

data_source = eeg.make_data_source()
pick_source = ColumnDataSource( data= dict( x=[], y=[], width=[], height=[],
								 start=[], finish=[], bots=[], tops=[] ))


text = TextInput( title="file", name='file', value=exp_path)

case_toggle = CheckboxGroup( labels=eeg.case_list, inline=True,
				active=[n for n in range(len(eeg.case_list))] )

case_chooser = RadioButtonGroup( labels=eeg.case_list, active=0 )
peak_chooser = RadioButtonGroup( labels=['P3','P4','N1','N2','N3','N4'], active=0)

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

plot_props = {'width':200, 'height':120,
				 'extra_bottom_height':40, # for bottom row
				'min_border':2}
gridplots = eeg.selected_cases_by_channel(cases='all',
			channels=['FZ','CZ','PZ','F3','C3','P3'],
			props=plot_props,  mode='server',
			source=data_source,
			tool_gen=[box_generator,BoxZoomTool, WheelZoomTool, ResetTool, PanTool]
			)

pick_starts = Segment(x0='start',x1='start',y0='bots',y1='tops',
				line_width=1,line_alpha=0.65,line_color='#FF6666')
pick_finishes = Segment(x0='finish',x1='finish',y0='bots',y1='tops',
				line_width=1,line_alpha=0.65,line_color='#FF6666')
gridplots[0][1].add_glyph(pick_source,pick_starts)
gridplots[0][1].add_glyph(pick_source,pick_finishes)

def update_data():
	pass

def input_change(attr,old,new):
	pass

def apply_handler():
	print('Apply')
	#print( dir() )
	print(pick_source.to_df())

def checkbox_handler(active):
    for n,nm in enumerate(eeg.case_list):
    	label = nm+'_line'
    	selections=grid.select(dict(name=label))
    	for sel in selections:
    		sel.glyph.line_alpha= 1 if n in active else 0


def input_change(attr, old, new):
	s.update_data()
	s.plot.title = s.text.value


case_toggle.on_click(checkbox_handler)
apply_button.on_click(apply_handler)

text.on_change('value', input_change)



inputs= VBox( children=[ text, case_chooser, peak_chooser, 
 					apply_button, save_button, case_toggle ])

curdoc().add_root(inputs)
grid = gridplot( gridplots ) # gridplot works properly outside of curdoc