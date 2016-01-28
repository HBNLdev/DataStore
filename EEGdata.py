'''reading and handling EEG data
'''

import numpy as np
import pandas as pd
import h5py, os
import bokeh
from bokeh.plotting import figure, gridplot
from bokeh.models import ( FixedTicker, CustomJS, TapTool, Range1d, 
				ColumnDataSource, Plot, Line, LinearAxis, 
				CompositeTicker, AdaptiveTicker, SingleIntervalTicker, Grid )
from bokeh.palettes import brewer
from collections import OrderedDict

import study_info as SI
import file_handling as FH

class avgh1:

	def __init__(s,filepath):

		s.filepath = filepath
		s.filename = os.path.split(s.filepath)[1]
		s.file_info = FH.parse_filename(s.filename)
		# s.cases = SI.experiments_parts[s.file_info['experiment']]
		s.loaded = h5py.File(s.filepath,'r')
		s.electrodes = [ s.decode() for s in list(s.loaded['file']['run']['run'])[0][-2] ]
		s.electrodes_61	= s.electrodes[0:31]+s.electrodes[32:62]
		s.samp_freq = 256
		# s.peak = OrderedDict()

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
			dvals = [ v[0].decode() if type(v) == np.ndarray else v for v in sub_info ]
			s.subject = { n:v for n,v in zip(sub_info.dtype.names, dvals) }
		else:
			return

	def extract_exp_data(s):
		if 'exp' not in dir(s):
			exp_info = s.loaded['file']['experiment']['experiment'][0]
			dvals = [ v[0].decode() if type(v) == np.ndarray else v for v in exp_info ]
			s.exp = { n:v for n,v in zip(exp_info.dtype.names, dvals) }
		else:
			return

	def extract_transforms_data(s):
		if 'transforms' not in dir(s):
			transforms_info = s.loaded['file']['transforms']['transforms'][0]
			dvals = [ v[0].decode() if type(v[0]) == np.bytes_ else v[0] for v in transforms_info ]
			s.transforms = { n:v for n,v in zip(transforms_info.dtype.names, dvals) }
		else:
			return

	def extract_case_data(s):
		if 'cases' not in dir(s):
			case_info = s.loaded['file']['run']['case']['case']
			s.cases = OrderedDict()
			s.case_num_map = {}
			s.case_list = []
			for vals in case_info.value:
				dvals = [ v[0].decode() if type(v[0]) == np.bytes_ else v[0] for v in vals ]
				caseD = { n:v for n,v in zip(case_info.dtype.names, dvals) }
				s.cases[ caseD['case_num'] ] = caseD
				s.case_list.append(caseD['case_type'])
				s.case_num_map[ caseD['case_type'] ] = caseD['case_num']

		else:
			return

	def build_mt(s, cases, peaks, amp, lat):
		s.extract_subject_data()
		s.extract_exp_data()
		s.extract_transforms_data()
		s.extract_case_data()
		s.build_mt_header(cases, peaks)
		s.build_mt_body(cases, peaks, amp, lat)
		s.mt = s.mt_header + s.mt_body

	def build_mt_header(s, cases, peaks):
		chans = s.electrodes[0:31]+s.electrodes[32:62]
		n_peaks = len(peaks)
		s.mt_header = ''
		s.mt_header += '#nchans ' + str(len(chans)) + '; '
		s.mt_header += 'filter ' + str(s.transforms['hi_pass_filter']) + '-' \
			+ str(s.transforms['lo_pass_filter']) + '; '
		s.mt_header += 'thresh ' + str(s.exp['threshold_value']) + ';\n'
		for case in cases:
			s.mt_header += '#case ' + str(case) + ' (' + s.cases[case]['case_type'] + '); npeaks ' + str(n_peaks) + ';\n'

	def build_mt_body(s, cases, peaks, amp, lat):
		# indices
		sid 	= s.subject['subject_id']
		expname = s.exp['exp_name']
		expver 	= s.exp['exp_version']
		gender 	= s.subject['gender']
		age 	= int(s.subject['age'])
		# cases 	= list(s.cases.keys())
		chans 	= s.electrodes_61 # only head chans
		# peaks 	= ['N1','P3'] # test case
		indices = [ [sid], [expname], [expver], [gender], [age],
					cases, chans, peaks ]
		index 	= pd.MultiIndex.from_product(indices,
					names=FH.mt_file.columns[:-3])

		# data
		n_lines = len(cases) * len(chans) * len(peaks)
		# amp 	= np.random.normal(10,5,n_lines) # test case
		# lat 	= np.random.normal(300,100,n_lines) # test case
		rt = []
		for case in cases:
		    rt.extend( [ s.cases[case]['mean_resp_time'] ] * len(peaks) * len(chans) )
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

	def find_peak(s, case, start_ms = 200, end_ms = 600,
		 chan_scope='all', chan=0, peak_polarity='p'):
		# erps is cases x chans x pts
		caseN = s.case_list.index(case)
		lats,erps = s.prepare_plot_data()
		
		 # test case
		peak_polarity = 'p' # test case
		chan_scope = 'all' # test case

		start_pt = np.argmin(np.fabs( lats-start_ms ))
		end_pt   = np.argmin(np.fabs( lats-end_ms ))
		
		# get data
		if chan_scope == 'one': # find peak for one chan
			chan 	= 0 # test case
			erpa 	= erps[caseN,chan,:]
		elif chan_scope == 'all': # find peak for all chans
			erpa = erps[caseN,:,:]
			erpa = erpa.swapaxes(0,1)
		else:
			return # error, the range is not correctly specified

		# find min/max in range
		if peak_polarity == 'p': # find the max
			peak_val = np.max(erpa[start_pt:end_pt+1], axis=0)
			peak_pt  = np.argmax(erpa[start_pt:end_pt+1], axis=0) + start_pt
		elif peak_polarity == 'n': # find the min
			peak_val = np.min(erpa[start_pt:end_pt+1], axis=0)
			peak_pt  = np.argmin(erpa[start_pt:end_pt+1], axis=0) + start_pt
		else:
			return # error, the peak polarity is not correctly specified

		# check if at edge
		if chan_scope == 'one': #test
			if peak_pt == start_pt or peak_pt == end_pt:
				pass # peak is at an edge
		elif chan_scope == 'all':
			if any(peak_pt == start_pt) or any(peak_pt == end_pt):
				pass # at least one peak is at an edge

		peak_ms = lats[peak_pt] # convert to ms if necessary

		return peak_val, peak_ms
			

	def get_yscale(s, potentials, channels):
		# get full list of display channels
		ind_lst = []
		for chan in channels:
			ind_lst.append( s.electrodes.index(chan) )

		# find means / stds
		pots = np.take(potentials, ind_lst, 1)
		pots_mean = np.mean(pots, axis=2)
		pots_mean2 = np.mean(pots_mean, axis=1)
		pots_meanstd = np.std(pots_mean, axis=1)

		# find indices of outlying channels
		n_std = 2 # num of stdev away from mean to detect
		out_means = np.logical_or(
			pots_mean.T <  pots_mean2 - n_std*pots_meanstd,
			pots_mean.T >  pots_mean2 + n_std*pots_meanstd)
		out_finds = out_means.any(axis=1)
		out_inds = np.nonzero(out_finds)[0]
		
		# remove outlying chans from calculation
		for ind in out_inds:
			ind_lst.remove( s.electrodes.index(channels[ind]) )
		disp_pots = np.take(potentials, ind_lst, 1)

		# calculate min / max for y limits
		min_val = int(np.floor(np.min(disp_pots)))
		max_val = int(np.ceil(np.max(disp_pots)))
		print('Yscale: ',min_val,max_val)
		return min_val, max_val

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
		g=GridPlot([plots])
		show(g)

	def selected_cases_by_channel(s,cases='all', channels='all', props={}, 
			mode='notebook', source=None,
			tools=[], tool_gen=[], style='grid'):

		# Setup properties for plots 
		default_props = {'width':250,
						'height':150,
						'min_border':2,
						'extra_bottom_height':20,
						'font size':8,
						'axis alpha':0,
						'outline alpha':1,
						'grid alpha':0.75}

		default_props.update(props)
		props = default_props

		s.extract_case_data()
		if cases == 'all':
			cases = s.case_list

		tms,potentials = s.prepare_plot_data()

		props['times'] = tms

		min_val, max_val = s.get_yscale(potentials, channels)

		props['yrange'] = [min_val,max_val]
		props['xrange'] = [-100,700]

		if len(cases) > 3:
			props['colors'] = brewer['Spectral'][len(cases)]
		else: props['colors'] = ['#DD2222','#66DD66','#2222DD']

		if channels ==  'all':
			channels = s.electrodes
		elif channels == 'core_31':
			channels = s.electrodes[0:31]
			channels.append(s.electrodes[63])

		if style == 'grid':
			n_plots = len( channels ) #potentials.shape[1]
			n_per_row = int( np.ceil(n_plots**0.5) )

			plots = []
			for plot_ind,electrode in enumerate(channels):
				eind= s.electrodes.index(electrode)
				if plot_ind % n_per_row == 0:
					plots.append([])

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
											source=source, tool_gen=tool_gen)
				plots[-1].append(splot)

		elif style == 'layout':
			layout = []
			layout.append( [None, 'FP1', 'Y',  'FP2', 'X']  )
			layout.append( ['F7', 'AF1', None, 'AF2', 'F8']  )
			layout.append( [None, 'F3', 'FZ',  'F4',  None]  )
			layout.append( ['FC5','FC1', None, 'FC2', 'FC6'] )
			layout.append( ['T7', 'C3', 'CZ',  'C4',  'T8']  )
			layout.append( ['CP5','CP1', None, 'CP2', 'CP6'] )
			layout.append( [None, 'P3', 'PZ',  'P4',  None]  )
			layout.append( ['P7', 'PO1', None, 'PO2', 'P8']  )
			layout.append( [None, 'O1',  None, 'O2',  None]  )

			plots = []
			for row in layout:
				plots.append([])
				for cell in row:
					if cell == None:
						plots[-1].append(None)
					else:
						eind= s.electrodes.index(cell)
						splot = s.make_plot_for_channel(potentials,eind,props,
							cases,tools, mode=mode, source=source,
							tool_gen=tool_gen)
						plots[-1].append(splot)

		if mode == 'server':
			return plots
		else:
			g=gridplot( plots, border_space=-40)#, tools=[TapTool()])#tools )
			show(g)

	def extract_mt_data(s):
		if 'mt_data' not in dir(s):
			mt_dir = os.path.split(s.filepath)[0]
			h1_name = os.path.split(s.filepath)[1]
			s.mt_name = os.path.splitext(h1_name)[0] + '.mt'
			s.mt_defaultpath = os.path.splitext(s.filepath)[0] + '.mt'
			if os.path.isfile( s.mt_defaultpath ):
				mt = FH.mt_file( s.mt_defaultpath )
				mt.parse_file()
				s.mt_data = mt.data
				s.case_peaks = mt.data.keys()
				peak_lst = []
				for c_pk in s.case_peaks:
				    peak_lst.append(c_pk[1])
				s.peaks = list(set(peak_lst))
			else:
				return
		else:
			return

	def make_data_sources(s,channels='all'):
		times, potentials = s.prepare_plot_data()
		s.extract_case_data()
		s.extract_mt_data()

		pot_source_dict = OrderedDict( times=times )

		if channels == 'all':
			channels = s.electrodes
	
		#peaks
		peak_sourcesD = {}
		for case in s.case_list:
			peak_source_dict = dict( peaks = []  )
			for chan in channels:
				peak_source_dict[ chan+'_pot'] = []
				peak_source_dict[ chan+'_time'] = []
			peak_sourcesD[ case ] = peak_source_dict

		if 'mt_data' in dir(s):
			for case, peak in s.case_peaks:
				c_pk = (case, peak)
				case_name = s.case_list[int(case)-1]
				peak_sourcesD[case_name]['peaks'].append(peak)
				for chan in channels:
					if chan not in ['X','Y','BLANK']:
						peak_sourcesD[case_name][ chan+'_pot'].append( float(s.mt_data[c_pk][chan][0]) )
						peak_sourcesD[case_name][ chan+'_time'].append( float(s.mt_data[c_pk][chan][1]) )
					else:
						peak_sourcesD[case_name][ chan+'_pot'].append( np.nan )
						peak_sourcesD[case_name][ chan+'_time'].append( np.nan )

		peak_sources = { case:ColumnDataSource( data = D ) for case,D in peak_sourcesD.items() }				

		# potentials
		for chan in channels:
			ch_ind = s.electrodes.index(chan)
			for cs_ind,cs in s.cases.items():
				pot_source_dict[chan+'_'+cs['case_type'] ] = potentials[cs_ind-1,ch_ind,:]

		pot_source = ColumnDataSource(
			data = pot_source_dict )


		return pot_source, peak_sources

	def update_peak_source(s, psD, case, peak, pot_vals, times ):
		''' assumes all channels
		'''
		psD[ 'peaks' ].append( peak )
		for chan, val, tm in zip(s.electrodes,pot_vals,times):
			psD[ chan+'_pot' ].append( val )
			psD[ chan+'_time' ].append( tm )

	def make_plot_for_channel(s,pot,el_ind,props,case_list,tools,
						mode='notebook',bottom_label=False,legend=False,
						source=None,tool_gen=None):

		if bottom_label:
			height = props['height']+props['extra_bottom_height']
		else:
			height = props['height'] 

		electrode = s.electrodes[el_ind]

		plot = Plot(#plot_width=props['width'], plot_height=height, 
			title=electrode,
			tools=tools
			)
		plot.title_standoff = 0
		plot.title_text_align='left'
		plot.title_text_baseline='top'
		#plot.border_fill='#888'

		#plot.min_border = props['min_border']
		plot.min_border_left = props['min_border']
		plot.min_border_right = props['min_border']
		plot.min_border_top = props['min_border']
		plot.min_border_bottom = props['min_border']
		plot.plot_width = props['width']
		plot.plot_height = height
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

		#yTicker.desired_num_ticks = 2
		#yTicker.num_minor_ticks = 0
		
		yAxis.axis_label_text_font_size = str(props['font size'])+'pt'
		yAxis.major_label_text_font_size = str(props['font size']-2)+'pt'
		#yAxis.major_label_text_align = 'right'
		yAxis.major_label_standoff = 2
		yAxis.minor_tick_line_color = None
		yAxis.major_tick_out = 0
		yAxis.major_tick_in = 4
		plot.add_layout(yAxis,'left')
		#yAxis = LinearAxis()
		'''
		plot.xaxis.axis_label_text_font_size = str(props['font size'])+'pt'
		#plot.grid.grid_line_alpha = props['grid alpha']

		plot.xaxis.major_label_text_font_size = str(props['font size']-2)+'pt'
		plot.xaxis.major_label_text_align = 'right'
		plot.xaxis.major_label_standoff = 0

		plot.yaxis.major_label_text_font_size = str(props['font size']-2)+'pt'

		plot.outline_line_color = None

		plot.yaxis[0].ticker.desired_num_ticks=2
		plot.yaxis.major_tick_line_width = 1 
		plot.yaxis.minor_tick_line_color = None
		plot.yaxis.major_tick_in = 4
		plot.yaxis.major_tick_out = 0

		plot.ygrid.grid_line_color = None
		
		plot.xgrid.grid_line_alpha = 0.35
		plot.xaxis.minor_tick_line_color = None
		plot.xaxis.major_tick_out = 0
		plot.xaxis.major_tick_in = 2
		'''
		if tool_gen:
			plot.add_tools(*[g() for g in tool_gen])


		for cs_ind,case in enumerate(case_list):
			case_ind = s.case_num_map[case]-1
			leg = None
			if legend:
				leg = case
			if mode == 'server':
				#print(case)
				line= Line( x='times', y=electrode+'_'+case, line_color=props['colors'][cs_ind],
						line_width=1.5, line_alpha=0.85, name=case+'_line')
				plot.add_glyph(source,line)
			else: #notebook for now
				plot.line( x=props['times'], y=pot[case_ind,el_ind,:], color=props['colors'][cs_ind],
						line_width=2, line_alpha=0.85, name=case, legend=leg)

		if legend:
			plot.legend.location='top_left'
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

		# if bottom_label:
		# 	plot.xaxis.axis_label="Time (ms)"
		# else: 
		# 	plot.xaxis[0].ticker = FixedTicker(ticks=[0,400])
		# 	plot.grid.ticker = plot.xaxis[0].ticker
		# plot.axis.axis_line_alpha = props['axis alpha']

		return plot