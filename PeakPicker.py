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
from bokeh.models import Plot, ColumnDataSource, CustomJS, BoxSelectTool, TapTool, Rect
from bokeh.properties import Instance
from bokeh.server.app import bokeh_app
from bokeh.server.utils.plugins import object_page
from bokeh.models.widgets import VBox, Slider, TextInput, VBoxForm, Select, CheckboxGroup

#import master_info as mi
import organization as O

class DashApp(VBox):

	slider_steps= 200

	extra_generated_classes = [["DashApp", "DashApp","VBox"]]
	inputs = Instance(VBoxForm)
	text = Instance(TextInput)

	toggle = Instance(CheckboxGroup)
	# Aage_min = Instance(Slider)
	# Aage_max = Instance(Slider)
	# phase = Instance(Slider)
	# freq = Instance(Slider)
	
	plot = Instance(Plot)
	data_source = Instance(ColumnDataSource)
	rect_source = Instance(ColumnDataSource)

	two_sessions_query = O.Mdb.subjects.find( {'b-age':{ '$gt': 0 }} )
	EEGsubjects = pd.DataFrame( list( two_sessions_query ) )

	@classmethod
	def create(cls):
		obj = cls()
		df = DashApp.EEGsubjects
		print(df.head())
		two_sesDF = df[['a-age','b-age']]
		two_sesDF.columns= ['Aage','Bage']
		#obj.source = ColumnDataSource( data= dict(x=[],y=[]))
		#obj.source = ColumnDataSource( data= df[['a-age','b-age']] 
		obj.data_source = ColumnDataSource( data = dict( Aage=df['a-age'].tolist(), 
															Bage=df['b-age'].tolist() ))
		obj.rect_source = ColumnDataSource( data= dict( x=[], y=[], width=[], height=[] ))
		
		obj.text = TextInput( title="title", name='title', value='first two session ages')

		obj.toggle = CheckboxGroup( labels=["Case 1","Case 2","Case 3"],active=[0],inline=True)

		# Aage_range= [ df['a-age'].min(), df['a-age'].max() ]
		# obj.Aage_min = Slider(title="a-age_min", name='a-age_min',
		# 				value=Aage_range[0], start=Aage_range[0], end = Aage_range[1],
		# 				 step=np.diff(Aage_range)[0]/DashApp.slider_steps)
		# obj.Aage_max = Slider(title="a-age_max", name='a-age_max',
		# 				value=Aage_range[1], start=Aage_range[0], end=Aage_range[1])
		# obj.phase = Slider(title="phase", name='phase',
		# 				value=0.0, start=0.0, end=2*np.pi)
		# obj.freq = Slider(title="frequency", name='frequency',
		# 				value=1.0, start=0.1, end=5.1)

		select = Select(title="Choose",value='that',
		 				options=['this','that','other','what'])

		toolset = "crosshair,pan,reset,resize,save,wheel_zoom"
		tap_callback = CustomJS( args=dict(source=obj.data_source), code="""
		 	alert('clicked');
		 	""" )
		box_callback = CustomJS(args=dict(source=obj.rect_source), code="""
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

		        // trigger update of data source
		        source.trigger('change');
		    """)
		box = BoxSelectTool( callback=box_callback )
		tap = TapTool( callback=tap_callback )

		plot = figure( title_text_font_size="12pt",
					plot_height=400, plot_width=400,
					tools=[box,tap,'wheel_zoom'],#toolset,
					title=obj.text.value,
					x_range=[0, 100], y_range=[0,100])


		rect = Rect(x='x',
            y='y',
            width='width',
            height='height',
            fill_alpha=0.3,
            fill_color='#CC5555')
		plot.add_glyph( obj.rect_source, rect, selection_glyph=rect, nonselection_glyph=rect )
		#print(obj.source)
		plot.circle('Aage','Bage', source=obj.data_source)# line_width=3, line_alpha=0.6)

		obj.plot = plot
		#obj.update_data()

		obj.inputs= VBoxForm( 
				children=[ obj.text, obj.toggle ])
		obj.children.append( obj.inputs )
		obj.children.append( obj.plot )

		return obj

	def setup_events(s):

		super( DashApp, s).setup_events()
		if not s.text:
			return

		s.text.on_change('value', s, 'input_change')


		# for w in ['Aage_min', 'Aage_max', 'phase', 'freq']:
		# 	getattr( s, w).on_change('value', s, 'input_change')

	def input_change(s, obj, attrname, old, new):
		s.update_data()
		s.plot.title = s.text.value

	def update_data(s):
		# N=200

		a_min= 0#s.Aage_min.value
		a_max= 80#s.Aage_max.value
		# w= s.phase.value
		# k= s.freq.value

		# x = np.linspace(0, 4*np.pi, N)
		# y = a*np.sin(k*x + w) + b
		df = DashApp.EEGsubjects
		data = df[ ( df['a-age'] >= a_min ) & ( df['a-age'] <= a_max ) ]#[['a-age','b-age']]
		#two_sesDF = data[['a-age','b-age','c-age']]
		#two_sesDF.columns= ['Aage','Bage','Cage']
		#logging.debug("PARAMS: offset: %s amplitude: %s", s.offset.value,
	#					s.amplitude.value)
		#sd = data.to_dict()
		#print(sd.keys())
		#D = s.source.data
		#print([ (k,len(v)) for k,v in D.items()])
		s.data_source.data= dict( Aage=data['a-age'].tolist(), Bage=data['b-age'].tolist() )
							#	x=D['x'], y=D['y'], width=D['width'], height=D['height'] )

@bokeh_app.route("/bokeh/PeakPicker/")
@object_page("dashtest")
def make_sliders():
	app= DashApp.create()
	return app