'''reading and handling EEG data
'''

import numpy as np
import pandas as pd
import h5py, os
import bokeh
from bokeh.plotting import figure, output_notebook, show, gridplot
from bokeh.models import FixedTicker, CustomJS, TapTool, Range1d, ColumnDataSource, GridPlot
from bokeh.palettes import brewer
from collections import OrderedDict

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
		s.peak = OrderedDict()

	def show_file_hierarchy(s):
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

	def extract_subject_data(s):
		if 'subject' not in dir(s):
			sub_info = s.loaded['file']['subject']['subject'][0]
			dvals = [ v[0].decode() if type(v) == np.ndarray else v for v in sub_info]
			s.subject = { n:v for n,v in zip(sub_info.dtype.names, dvals) }
		else:
			return

	def extract_exp_data(s):
		if 'exp' not in dir(s):
			exp_info = s.loaded['file']['experiment']['experiment'][0]
			dvals = [ v[0].decode() if type(v) == np.ndarray else v for v in exp_info]
			s.exp = { n:v for n,v in zip(exp_info.dtype.names, dvals) }
		else:
			return

	def extract_transforms_data(s):
		if 'transforms' not in dir(s):
			transforms_info = s.loaded['file']['transforms']['transforms'][0]
			dvals = [ v[0].decode() if type(v[0]) == np.bytes_ else v[0] for v in transforms_info]
			s.transforms = { n:v for n,v in zip(transforms_info.dtype.names, dvals) }
		else:
			return

	def extract_case_data(s):
		if 'cases' not in dir(s):
			case_info = s.loaded['file']['run']['case']['case']
			s.cases = {}
			s.case_num_map = {}
			s.case_list = []
			for vals in case_info.value:
				dvals = [ v[0].decode() if type(v[0]) == np.bytes_ else v[0] for v in vals   ]
				caseD = { n:v for n,v in zip(case_info.dtype.names, dvals) }
				s.cases[ caseD['case_num'] ] = caseD
				s.case_list.append(caseD['case_type'])
				s.case_num_map[ caseD['case_type'] ] = caseD['case_num']

		else:
			return

	def build_mt(s):
		s.extract_subject_data()
		s.extract_exp_data()
		s.extract_transforms_data()
		s.extract_case_data()
		s.build_mt_header()
		s.build_mt_body()
		s.mt = s.mt_header + s.mt_body

	def build_mt_header(s):
		chans = s.electrodes[0:31]+s.electrodes[32:62]
		peaks = ['N1','P3'] # for testing
		n_peaks = len(peaks)
		s.mt_header = ''
		s.mt_header += '#nchans ' + str(len(chans)) + ';\n'
		for case in s.cases.keys():
			s.mt_header += '#case ' + str(case) + ' (' + s.cases[case]['case_type'] + '); npeaks ' + str(n_peaks) + ';\n'
		s.mt_header += '#hipass ' + str(s.transforms['hi_pass_filter']) + '\n'
		s.mt_header += '#lopass ' + str(s.transforms['lo_pass_filter']) + '\n'
		s.mt_header += '#thresh ' + str(s.exp['threshold_value']) + '\n'

	def build_mt_body(s):
		# indices
		sid 	= s.subject['subject_id']
		expname = s.exp['exp_name']
		expver 	= s.exp['exp_version']
		gender 	= s.subject['gender']
		age 	= int(s.subject['age'])
		cases 	= list(s.cases.keys())
		chans 	= s.electrodes[0:31]+s.electrodes[32:62] # only head chans
		peaks 	= ['N1','P3'] # test case
		indices = [ [sid], [expname], [expver], [gender], [age],
					cases, chans, peaks ]
		index 	= pd.MultiIndex.from_product(indices,
					names=FH.mt_file.columns[:-3])

		# data
		n_lines = len(cases) * len(chans) * len(peaks)
		amp 	= np.random.normal(10,5,n_lines) # test case
		lat 	= np.random.normal(300,100,n_lines) # test case
		rt = []
		for case in s.cases.keys():
		    rt.extend([s.cases[case]['mean_resp_time']]*len(peaks)*len(chans))
		data = {'amplitude':amp, 'latency':lat, 'mean_rt':rt}
		
		# making CSV structure
		df = pd.DataFrame(data,index=index)
		mt_string = df.to_csv(path_or_buf=None, sep=' ', na_rep='NaN',
			float_format='%.3f', header=False, index=True,
			index_label=None, line_terminator='\n')
		s.mt_body = mt_string

	def apply_peak(s, case, peak):
		pass

	def prepare_plot_data(s):
		
		s.extract_exp_data()
		
		potentials = s.loaded['zdata']['zdata']
		# times = np.array(range(potentials.shape[2]))/s.samp_freq
		start_ms = -s.exp['pre_stim_time_ms']
		end_ms = s.exp['post_stim_time_ms']
		times = np.linspace(start_ms, end_ms, potentials.shape[2] + 1)[:-1]

		return times, potentials

	def find_peak(s):
		# erps is cases x chans x pts
		lats,erps = s.prepare_plot_data()
		
		case = 0; start_ms = 200; end_ms = 600 # test case
		peak_polarity = 'p' # test case
		chan_scope = 'all' # test case

		start_pt = np.argmin(np.fabs( lats-start_ms ))
		end_pt   = np.argmin(np.fabs( lats-end_ms ))
			
		if chan_scope == 'one': # find peak for one chan
			chan 	= 0 # test case
			erpa 	= erps[case,chan,:]
		elif chan_scope == 'all': # find peak for all chans
			erpa = erps[case,:,:]
			erpa = erpa.swapaxes(0,1)
		else:
			return # error, the range is not correctly specified

		if peak_polarity == 'p': # find the max
			peak_val = np.max(erpa[start_pt:end_pt+1], axis=0)
			peak_pt  = np.argmax(erpa[start_pt:end_pt+1], axis=0) + start_pt
		elif peak_polarity == 'n': # find the min
			peak_val = np.min(erpa[start_pt:end_pt+1], axis=0)
			peak_pt  = np.argmin(erpa[start_pt:end_pt+1], axis=0) + start_pt
		else:
			return # error, the peak polarity is not correctly specified

		if chan_scope == 'one': #test
			if peak_pt == start_pt or peak_pt == end_pt:
				pass # peak is at an edge
		elif chan_scope == 'all':
			if any(peak_pt == start_pt) or any(peak_pt == end_pt):
				pass # at least one peak is at an edge

		peak_ms = lats[peak_pt] # convert to ms if necessary

		return peak_val, peak_ms
			

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

	def selected_cases_by_channel(s,cases='all',channels='all',props={}, 
			mode='notebook',source=None):

		# Setup properties for plots 
		default_props = {'width':250,
						'height':150,
						'min_border':2,
						'extra_bottom_height':50,
						'font size':'8pt'}

		default_props.update(props)
		props = default_props

		s.extract_case_data()
		if cases == 'all':
			cases = s.case_list

		tms,potentials = s.prepare_plot_data()

		props['times'] = tms
		min_val = int(np.floor(np.min(potentials)))
		max_val = int(np.ceil(np.max(potentials)))
		props['yrange'] = [min_val,max_val]

		if len(cases) > 3:
			props['colors'] = brewer['Spectral'][len(cases)]
		else: props['colors'] = ['#2222DD','#DD2222','#66DD66']

		callback = CustomJS( code="alert('clicked')" )

		if channels ==  'all':
			channels = s.electrodes

		
		n_plots = len( channels ) #potentials.shape[1]
		n_per_row = int( np.ceil(n_plots**0.5) )

		plots = []
		for plot_ind,electrode in enumerate(channels):
			eind= s.electrodes.index(electrode)
			if plot_ind % n_per_row == 0:
				plots.append([])
			
			tap = TapTool( callback=callback )
			tools=[tap]
			if n_plots - plot_ind < n_per_row+1:
				bot_flag = True
			else:
				bot_flag = False
			if plot_ind == 0:
				leg_flag = True
			else:
				leg_flag = False

			splot = s.make_plot_for_channel(potentials,eind,props,cases,tools,
										bottom_label=bot_flag,legend=leg_flag, mode=mode,
										source=source)
			plots[-1].append(splot)
		
		g=gridplot(plots,border_space=-40)
		if mode == 'server':
			return g
		else:
			show(g)

	def make_data_source(s,channels='all'):
		times, potentials = s.prepare_plot_data()
		s.extract_case_data()

		source_dict = dict(
						times=times)

		if channels == 'all':
			channels = s.electrodes

		for chan in channels:
			ch_ind = s.electrodes.index(chan)
			for cs_ind,cs in s.cases.items():
				source_dict[chan+'_'+cs['case_type'] ] = potentials[cs_ind-1,ch_ind,:]

		source = ColumnDataSource(
			data = source_dict )

		return source


	def make_plot_for_channel(s,pot,el_ind,props,case_list,tools,
						mode='notebook',bottom_label=False,legend=False,
						source=None):

		if bottom_label:
			height = props['height']+props['extra_bottom_height']
		else:
			height = props['height'] 

		electrode = s.electrodes[el_ind]

		plot = figure(width=props['width'], height=height, 
			title=electrode, #tools=tools,
			min_border=props['min_border'])
		plot.y_range = Range1d(*props['yrange'])
		plot.title_text_font_size = props['font size']
		plot.xaxis.axis_label_text_font_size = props['font size']

		for cs_ind,case in enumerate(case_list):
			case_ind = s.case_num_map[case]-1
			leg = None
			if legend:
				leg = case
			if mode == 'server':
				print(case)
				plot.line( x='times', y=electrode+'_'+case, color=props['colors'][cs_ind],
						line_width=3, line_alpha=0.85, name=case+'_line', legend=leg, source=source)
			else: #notebook for now
				plot.line( x=props['times'], y=pot[case_ind,el_ind,:], color=props['colors'][cs_ind],
						line_width=3, line_alpha=0.85, name=case, legend=leg)

		if legend:
			plot.legend.orientation='top_left'
			plot.legend.label_text_font_size = props['font size']
			#plot.legend.background_fill_color = '#444' # fill not working
			#plot.legend.background_fill_alpha = 0.2
			plot.legend.label_text_align = 'left'
			#plot.legend.label_text_baseline = 'top'
			plot.legend.label_width = 20
			plot.legend.label_height = 12
			plot.legend.label_standoff = 10
			plot.legend.legend_padding = 2
			plot.legend.legend_spacing = 2
			#plot.legend.glyph_height = 10
			plot.legend.glyph_width = 15
			plot.legend.glyph_height= 12

		plot.yaxis[0].ticker=FixedTicker(ticks=[])#tick_locs,tags=channel_list)
		if bottom_label:
			plot.xaxis.axis_label="Time (s)"
		else: 
			plot.xaxis[0].ticker = FixedTicker(ticks=[])
		
		return plot