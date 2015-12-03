'''HBNL dashboard
'''

# import logging
# logging.basic.Config(level=logging.DEBUG)

import numpy as np
import pandas as pd

from bokeh.plotting import figure
from bokeh.models import Plot, ColumnDataSource
from bokeh.properties import Instance
from bokeh.server.app import bokeh_app
from bokeh.server.utils.plugins import object_page
from bokeh.models.widgets import HBox, Slider, TextInput, VBoxForm, Select

#import master_info as mi
import organization as O

class DashApp(HBox):

	slider_steps= 200

	extra_generated_classes = [["DashApp", "DashApp","HBox"]]
	inputs = Instance(VBoxForm)
	text = Instance(TextInput)

	Aage_min = Instance(Slider)
	Aage_max = Instance(Slider)
	phase = Instance(Slider)
	freq = Instance(Slider)
	
	plot = Instance(Plot)
	source = Instance(ColumnDataSource)

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
		#obj.source = ColumnDataSource( data= df[['a-age','b-age']] )
		obj.source = ColumnDataSource( data = dict( Aage=df['a-age'].tolist(), Bage=df['b-age'].tolist() ))
		obj.text = TextInput( title="title", name='title', value='first two session ages')

		Aage_range= [ df['a-age'].min(), df['a-age'].max() ]
		obj.Aage_min = Slider(title="a-age_min", name='a-age_min',
						value=Aage_range[0], start=Aage_range[0], end = Aage_range[1],
						 step=np.diff(Aage_range)[0]/DashApp.slider_steps)
		obj.Aage_max = Slider(title="a-age_max", name='a-age_max',
						value=Aage_range[1], start=Aage_range[0], end=Aage_range[1])
		obj.phase = Slider(title="phase", name='phase',
						value=0.0, start=0.0, end=2*np.pi)
		obj.freq = Slider(title="frequency", name='frequency',
						value=1.0, start=0.1, end=5.1)

		select = Select(title="Choose",value='that',
		 				options=['this','that','other','what'])

		toolset = "crosshair,pan,reset,resize,save,wheel_zoom"

		plot = figure( title_text_font_size="12pt",
					plot_height=400, plot_width=400,
					tools=toolset,
					title=obj.text.value,
					x_range=[0, 100], y_range=[0,100])
		plot.circle('Aage','Bage', source=obj.source)# line_width=3, line_alpha=0.6)

		obj.plot = plot
		obj.update_data()

		obj.inputs= VBoxForm( 
				children=[ obj.text, obj.Aage_min, obj.Aage_max, 
						obj.phase, obj.freq, select ])
		obj.children.append( obj.inputs )
		obj.children.append( obj.plot )

		return obj

	def setup_events(s):

		super( DashApp, s).setup_events()
		if not s.text:
			return

		s.text.on_change('value', s, 'input_change')


		for w in ['Aage_min', 'Aage_max', 'phase', 'freq']:
			getattr( s, w).on_change('value', s, 'input_change')

	def input_change(s, obj, attrname, old, new):
		s.update_data()
		s.plot.title = s.text.value

	def update_data(s):
		# N=200

		a_min= s.Aage_min.value
		a_max= s.Aage_max.value
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
		s.source.data= dict( Aage=data['a-age'].tolist(), Bage=data['b-age'].tolist() )

@bokeh_app.route("/bokeh/HBNLdash/")
@object_page("dashtest")
def make_sliders():
	app= DashApp.create()
	return app