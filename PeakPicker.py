'''HBNL dashboard

to start:
	bokeh-server --script PeakPicker.py
point browser to:
	http://localhost:5006/bokeh/PeakPicker/

'''

# import logging
# logging.basic.Config(level=logging.DEBUG)

import numpy as np
import pandas as pd

from bokeh.plotting import figure
from bokeh.models import Plot, ColumnDataSource, CustomJS, BoxSelectTool, TapTool, Rect, GridPlot
from bokeh.properties import Instance
from bokeh.server.app import bokeh_app
from bokeh.server.utils.plugins import object_page
from bokeh.models.widgets import VBox, Slider, TextInput, VBoxForm, Select, CheckboxGroup

#import master_info as mi
import organization as O
import EEGdata

class PickerApp(VBox):

	extra_generated_classes = [["PickerApp", "PickerApp","VBox"]]
	inputs = Instance(VBoxForm)
	text = Instance(TextInput)

	case_toggle = Instance(CheckboxGroup)

	gridplot = Instance(GridPlot)
	data_source = Instance(ColumnDataSource)

	exp_path = '/processed_data/mt-files/vp3/suny/ns/a-session/vp3_3_a1_40025009_avg.h1'
	eeg_exp = EEGdata.avgh1( exp_path )

	@classmethod
	def create(cls):
		obj = cls()
		eeg = PickerApp.eeg_exp

		obj.data_source = eeg.make_data_source()
		
		obj.text = TextInput( title="title", name='title', value=PickerApp.exp_path)

		obj.case_toggle = CheckboxGroup( labels=eeg.case_list, inline=True,
				active=[n for n in range(len(eeg.case_list))] )


		toolset = "crosshair,pan,reset,resize,save,wheel_zoom"
		tap_callback = CustomJS( args=dict(source=obj.data_source), code="""
		 	alert('clicked');
		 	""" )
		# box_callback = CustomJS(args=dict(source=obj.rect_source), code="""
		#         // get data source from Callback args
		#         console.dir(cb_data)
		#         var data = source.get('data');

		#         /// get BoxSelectTool dimensions from cb_data parameter of Callback
		#         var geometry = cb_data['geometry'];

		#         /// calculate Rect attributes
		#         var width = geometry['x1'] - geometry['x0'];
		#         var height = geometry['y1'] - geometry['y0'];
		#         var x = geometry['x0'] + width/2;
		#         var y = geometry['y0'] + height/2;

		#         /// update data source with new Rect attributes
		#         data['x'].push(x);
		#         data['y'].push(y);
		#         data['width'].push(width);
		#         data['height'].push(height);

		#         // trigger update of data source
		#         source.trigger('change');
		#     """)
		# box = BoxSelectTool( callback=box_callback )
		tap = TapTool( callback=tap_callback )

		obj.gridplot = eeg.selected_cases_by_channel(cases='all',
					channels=['FZ','CZ','PZ','F3','C3','P3'],
					props={'width':200,'height':165}, mode='server',
					source=obj.data_source)

		obj.inputs= VBoxForm( 
				children=[ obj.text, obj.case_toggle ])
		obj.children.append( obj.inputs )
		obj.children.append( obj.gridplot )
		return obj

	def setup_events(s):

		super( PickerApp, s).setup_events()
		if not s.text:
			return

		s.case_toggle.on_click(s.checkbox_handler)

		s.text.on_change('value', s, 'input_change')

	def checkbox_handler(s,active):

	    for n,nm in enumerate(PickerApp.eeg_exp.case_list):
	    	label = nm+'_line'
	    	selections=s.gridplot.select(dict(name=label))
	    	for sel in selections:
	    		sel.glyph.line_alpha= 1 if n in s.case_toggle.active else 0


	def input_change(s, obj, attrname, old, new):
		s.update_data()
		s.plot.title = s.text.value

	def update_data(s):

		pass


@bokeh_app.route("/bokeh/PeakPicker/")
@object_page("picker")
def make_picker():
	app= PickerApp.create()
	print(app.children)
	return app