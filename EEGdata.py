'''reading and handling EEG data
'''

import numpy as np
import h5py, os
import bokeh
from bokeh.plotting import figure, output_notebook, show, gridplot
from bokeh.models import FixedTicker, CustomJS, TapTool, Range1d
from bokeh.palettes import brewer

import study_info as SI
import file_handling as FH

class avgh1:

	def __init__(s,filepath):

		s.filepath = filepath
		s.filename = os.path.split(s.filepath)[1]
		s.file_info = FH.parse_filename(s.filename)
		#s.cases = SI.experiments_parts[s.file_info['experiment']]
		s.loaded = h5py.File(s.filepath,'r')
		s.electrodes = [ s.decode() for s in list(s.loaded['file']['run']['run'])[0][-2] ]

		s.samp_freq = 256

	def show_file_heirarchy(s):
		def disp_node( name, node):
		    print(name)
		    indent = name.count('/')*5*' '
		    if 'items' in dir(node):
		        for k,v in node.items():
		            print( indent, k, ': ', v)
		    else: 
		        print( indent, node)
		        if type(node) == h5py._hl.dataset.Dataset:
		            print( indent, node.dtype)


		s.loaded.visititems(disp_node)

	def extract_case_data(s):
		if 'cases' not in dir(s):
			case_info = s.loaded['file']['run']['case']['case']
			s.cases = {}
			for vals in case_info.value:
				#print([type(v) for v in vals])
				dvals = [ v[0].decode() if type(v[0]) == np.bytes_ else v[0] for v in vals   ]
				caseD = { n:v for n,v in zip(case_info.dtype.names, dvals) }
				s.cases[ caseD['case_num'] ] = caseD
		else:
			returns

	def prepare_plot_data(s):
		
		potentials = s.loaded['zdata']['zdata']
		times = np.array(range(potentials.shape[2]))/s.samp_freq

		return times,potentials

	def butterfly_channels_by_case(s,channel_list=['FZ','CZ','PZ'], offset=0):
		s.extract_case_data()

		# edata = s.loaded['zdata']['zdata']

		# tms = np.array(range(edata.shape[2]))/samp_freq

		tms,pot = s.prepare_plot_data()

		colors = brewer['Spectral'][len(channel_list)]
		plots = []
		for cs in range(pot.shape[0]):
			callback = CustomJS( code="alert('clicked')" )
			#callback = CustomJS( code="function(){ var data = source.get('data'); console.dir(data); }" )
			tap = TapTool( callback=callback )
			splot = figure(width=550, height=350, title=s.cases[cs+1]['descriptor'], tools=[tap])
			tick_locs=[]
			for cnt,ch in enumerate(channel_list):
				y_level = offset*cnt
				tick_locs.append(y_level)
				ch_ind = s.electrodes.index(ch)
				splot.line(x=tms,y=pot[cs,ch_ind,:]+y_level,color=colors[cnt],
				           line_width=3, line_alpha=0.85, legend=ch)
			splot.legend.orientation='top_left'
			splot.legend.background_fill_alpha = 0
			splot.legend.label_standoff = 0
			splot.legend.legend_padding = 2
			splot.legend.legend_spacing = 2
			splot.yaxis[0].ticker=FixedTicker(ticks=[])#tick_locs,tags=channel_list)
			splot.xaxis.axis_label="Time (s)"
			plots.append(splot)
		g=gridplot([plots])
		show(g)

	def selected_cases_by_channel(s,case_list,props={}):
		default_props = {'width':250,
						'height':150,
						'min_border':2,
						'extra_bottom_height':50}

		default_props.update(props)
		props = default_props

		s.extract_case_data()
		case_num_map = { D['case_type']:k for k,D in s.cases.items() }
		#case_num_map = { cs:num for   }
		tms,pot = s.prepare_plot_data()
		min_val = int(np.floor(np.min(pot)))
		max_val = int(np.ceil(np.max(pot)))

		if len(case_list) > 3:
			colors = brewer['Spectral'][len(case_list)]
		else: colors=['#2222DD','#DD2222','#66DD66']

		callback = CustomJS( code="alert('clicked')" )
		#callback = CustomJS( code="function(){ var data = source.get('data'); console.dir(data); }" )

		plots = []
		n_elec = pot.shape[1]
		n_per_row = int(np.ceil(n_elec**0.5))

		for eind in range(n_elec):
			if eind % n_per_row == 0:
				plots.append([])
			electrode = s.electrodes[eind]
			tap = TapTool( callback=callback )
			if n_elec - eind < n_per_row+1:
				height = props['height']+props['extra_bottom_height']
			else:
				height = props['height']
			splot = figure(width=props['width'], height=height, 
				title=s.electrodes[eind], tools=[tap],
				min_border=props['min_border'])
			splot.y_range = Range1d(min_val,max_val)
			splot.title_text_font_size = '8pt'
			splot.xaxis.axis_label_text_font_size = '12pt'

			for cs_ind,case in enumerate(case_list):
				case_ind = case_num_map[case]-1
				leg = None
				if eind == 0:
					leg = case
				splot.line( x=tms, y=pot[case_ind,eind,:], color=colors[cs_ind],
							line_width=3, line_alpha=0.85, legend=leg)

			if eind ==0:
				splot.legend.orientation='top_left'
				splot.legend.label_text_font_size = '8pt'
				splot.legend.background_fill_alpha = 0
				splot.legend.label_standoff = 0
				splot.legend.legend_padding = 2
				splot.legend.legend_spacing = 0
			splot.yaxis[0].ticker=FixedTicker(ticks=[])#tick_locs,tags=channel_list)
			if n_elec - eind < n_per_row+1:
				splot.xaxis.axis_label="Time (s)"
			else: 
				splot.xaxis[0].ticker = FixedTicker(ticks=[])
			plots[-1].append(splot)
		g=gridplot(plots,border_space=-40)
		show(g)
