'''reading and handling EEG data
'''

import os, sys
from collections import OrderedDict

import h5py
import numpy as np
import pandas as pd
from bokeh.palettes import brewer
from scipy.signal import argrelextrema

try:
    from db.utils.filename_parsing import parse_filename
except:
    sys.path.append( os.path.join(os.path.split(__file__)[0],'../db') )
    from file_handling import parse_Filename

split_grid_layout = [[None, 'FP1', 'Y', 'FP2', 'X'],
                      ['F7', 'AF1', None, 'AF2', 'F8'],
                      [None, 'F3', 'FZ', 'F4', None],
                      ['FC5', 'FC1', None, 'FC2', 'FC6'],
                      ['T7', 'C3', 'CZ', 'C4', 'T8'],
                      ['CP5', 'CP1', None, 'CP2', 'CP6'],
                      [None, 'P3', 'PZ', 'P4', None],
                      ['P7', 'PO1', None, 'PO2', 'P8'],
                      [None, 'O1', None, 'O2', None],
                      ['AF7', 'FPZ', 'AFZ', None, 'AF8'],
                      ['F5', 'F1', None, 'F2', 'F6'],
                      ['FT7', 'FC3', 'FCZ', 'FC4', 'FT8'],
                      ['C5', 'C1', None, 'C2', 'C6'],
                      ['TP7', 'CP3', 'CPZ', 'CP4', 'TP8'],
                      ['P5', 'P1', 'POZ', 'P2', 'P6'],
                      [None, 'PO7', 'OZ', 'PO8', None]]

full_grid_layout = [[None,None,None,'FP1','FPZ','FP2',None,None,None],
                    [None,None,'AF7','AF1','AFZ','AF2','AF8',None,None],
                    ['F7','F5','F3','F1','FZ','F2','F4','F6','F8'],
                    ['FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8'],
                    ['T7','C5','C3','C1','CZ','C2','C4','C6','T8'],
                    ['TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8'],
                    ['P7','P5','P3','P1','PZ','P2','P4','P6','P8'],
                    [None,None,'PO7','PO1','POZ','PO2','PO8',None,None],
                    [None,None,None,'O1','OZ','O2',None,None,None] ]

class avgh1:
    save_elec_order = ['FP1', 'FP2', 'F7', 'F8', 'AF1', 'AF2', 'FZ', 'F4', 'F3', 'FC6', 'FC5', 'FC2',
                       'FC1', 'T8', 'T7', 'CZ', 'C3', 'C4', 'CP5', 'CP6', 'CP1', 'CP2', 'P3', 'P4', 'PZ',
                       'P8', 'P7', 'PO2', 'PO1', 'O2', 'O1', 'AF7', 'AF8', 'F5', 'F6', 'FT7', 'FT8',
                       'FPZ', 'FC4', 'FC3', 'C6', 'C5', 'F2', 'F1', 'TP8', 'TP7', 'AFZ', 'CP3', 'CP4',
                       'P5', 'P6', 'C1', 'C2', 'PO7', 'PO8', 'FCZ', 'POZ', 'OZ', 'P2', 'P1', 'CPZ']

    def __init__(s, filepath):

        s.filepath = filepath
        s.filename = os.path.split(s.filepath)[1]
        s.file_info = parse_filename(s.filename)
        # s.cases = SI.experiments_parts[s.file_info['experiment']]
        s.loaded = h5py.File(s.filepath, 'r')
        s.electrodes = [st.decode() for st in list(s.loaded['file']['run']['run'])[0][-2]]
        s.electrodes_61 = s.electrodes[0:31] + s.electrodes[32:62]
        s.samp_freq = 256

    # s.peak = OrderedDict()

    def show_file_hierarchy(s):
        def disp_node(name, node):
            print(name)
            indent = name.count('/') * 5 * ' '
            if 'items' in dir(node):
                for k, v in node.items():
                    print(indent, k, ': ', v)
            else:
                print(indent, node)
                if type(node) == h5py._hl.dataset.Dataset:
                    print(indent, node.dtype)

        s.loaded.visititems(disp_node)

    def extract_subject_data(s):
        if 'subject' not in dir(s):
            sub_info = s.loaded['file']['subject']['subject'][0]
            dvals = [v[0].decode() if type(v) == np.ndarray else v for v in sub_info]
            s.subject = {n: v for n, v in zip(sub_info.dtype.names, dvals)}
        else:
            return

    def subject_data(s):
        s.extract_subject_data()
        return s.subject

    def extract_run_data(s):
        if 'run_info' not in dir(s):
            run_data = s.loaded['file/run/run']
            rvals = [v[0].decode() if isinstance(v, type(np.array(1))) \
                                      and 'decode' in dir(v[0]) else v for v in run_data[0]]
            s.run_info = {n: v for n, v in zip(run_data.dtype.names, rvals)}
        else:
            return

    def run_data(s):
        s.extract_run_data()
        return s.run_info

    def extract_exp_data(s):
        if 'exp' not in dir(s):
            exp_info = s.loaded['file']['experiment']['experiment'][0]
            dvals = [v[0].decode() if type(v) == np.ndarray else v for v in exp_info]
            s.exp = {n: v for n, v in zip(exp_info.dtype.names, dvals)}
        else:
            return

    def exp_data(s):
        s.extract_exp_data()
        return s.exp

    def extract_transforms_data(s):
        if 'transforms' not in dir(s):
            transforms_info = s.loaded['file']['transforms']['transforms'][0]
            dvals = [v[0].decode() if type(v[0]) == np.bytes_ else v[0] for v in transforms_info]
            s.transforms = {n: v for n, v in zip(transforms_info.dtype.names, dvals)}
        else:
            return

    def transforms_data(s):
        s.extract_transforms_data()
        return s.transforms

    def extract_case_data(s, output=False):
        if 'cases' not in dir(s):
            case_info = s.loaded['file']['run']['case']['case']
            s.cases = OrderedDict()
            s.case_num_map = {}
            s.case_ind_map = {}
            s.case_list = []
            for c_ind, vals in enumerate(case_info.value):
                dvals = [v[0].decode() if type(v[0]) == np.bytes_ else v[0] for v in vals]
                caseD = {n: v for n, v in zip(case_info.dtype.names, dvals)}
                s.cases[caseD['case_num']] = caseD
                s.case_list.append(caseD['case_type'])
                s.case_num_map[caseD['case_type']] = caseD['case_num']
                s.case_ind_map[caseD['case_type']] = c_ind
            s.num_case_map = {v: k for k, v in s.case_num_map.items()}
            s.case_ind_D = caseD
        else:
            return

    def case_data(s):
        s.extract_case_data()
        outD = {cD['case_type']: cD for cN, cD in s.cases.items()}
        return outD

    def build_mt(s, cases, peaks_by_case, amp, lat):
        s.extract_subject_data()
        s.extract_exp_data()
        s.extract_transforms_data()
        s.extract_case_data()
        s.build_mt_header(cases, peaks)
        s.build_mt_body(cases, peaks, amp, lat)
        s.mt = s.mt_header + s.mt_body

    def build_mt_header(s, cases, peaks_by_case):
        chans = s.electrodes[0:31] + s.electrodes[32:62]
        s.mt_header = ''
        s.mt_header += '#nchans ' + str(len(chans)) + '; '
        s.mt_header += 'filter ' + str(s.transforms['hi_pass_filter']) + '-' \
                       + str(s.transforms['lo_pass_filter']) + '; '
        s.mt_header += 'thresh ' + str(s.exp['threshold_value']) + ';\n'
        for case in cases:
            s.mt_header += '#case ' + str(case) + ' (' + s.cases[case]['case_type'] + '); npeaks ' +\
             str(len(peaks_by_case[case])) + ';\n'

    def build_mt_body(s, cases, peaks_by_case, amp, lat):
        # indices
        sid = s.subject['subject_id']
        expname = s.exp['exp_name']
        expver = s.exp['exp_version']
        gender = s.subject['gender']
        age = int(s.subject['age'])
        # cases 	= list(s.cases.keys())
        chans = s.electrodes_61  # only head chans
        # peaks 	= ['N1','P3'] # test case
        peak_list = []
        for case, peaks in peaks_by_case.items():
            peak_list.extend(peaks)
        all_peaks = sorted(list(set(peak_list)))
        indices = [[sid], [expname], [expver], [gender], [age],
                   cases, chans, all_peaks]
        index = pd.MultiIndex.from_product(indices,
                                           names=MT_File.columns[:-3])

        # data
        rt = []
        for case in cases:
            rt.extend([s.cases[case]['mean_resp_time']] * len(all_peaks) * len(chans))
        data = {'amplitude': amp, 'latency': lat, 'mean_rt': rt}

        # making CSV structure
        df = pd.DataFrame(data, index=index)

        dfR = df.reset_index()
        elecIndex = dict(zip(s.save_elec_order, range(len(s.save_elec_order))))
        dfR['elec_rank'] = dfR['electrode'].map(elecIndex)
        dfR.sort_values(['case_num', 'elec_rank', 'peak'], inplace=True)
        dfR.drop('elec_rank', 1, inplace=True)

        dfR.dropna(inplace=True)
        mt_string = dfR.to_string(buf=None, header=False, na_rep='NaN',
                                  float_format='%.3f', index=False,
                                  formatters={'mean_rt': lambda x: '%.1f' % x})

        s.mt_body = mt_string

    def apply_peak(s, case, peak):
        pass

    def prepare_plot_data(s, time_range='all'):
        ''' time_range can be a list of start,finish
        '''

        s.extract_exp_data()

        potentials = s.loaded['zdata']['zdata']
        # times = np.array(range(potentials.shape[2]))/s.samp_freq
        start_ms = -s.exp['pre_stim_time_ms']
        end_ms = s.exp['post_stim_time_ms']
        times = np.linspace(start_ms, end_ms, potentials.shape[2] + 1)[:-1]
        delta = times[1] - times[0]
        times = times - delta

        if time_range != 'all':
            start_ind = np.argmin(abs(times - time_range[0]))
            fin_ind = np.argmin(abs(times - time_range[1]))
            times = times[start_ind:fin_ind]
            potentials = potentials[:, :, start_ind:fin_ind]

        return times, potentials

    def find_peaks(s, case, chan_list, starts_ms, ends_ms, polarity='p'):

        def peak_alg(erp_array, polarity):
            if polarity == 'p':
                comparator = np.greater
                fallback_func = np.argmax
            elif polarity == 'n':
                comparator = np.less
                fallback_func = np.argmin
            local_extreme_inds = argrelextrema(erp_array, comparator)[0]
            if local_extreme_inds.shape[0] == 0:  # no local extremum
                ext_lmi = fallback_func(erp_array)
            else:
                local_extreme_vals = erp_array[local_extreme_inds]
                if polarity == 'p':
                    local_extreme_tmp_ind = np.argmax(local_extreme_vals)
                elif polarity == 'n':
                    local_extreme_tmp_ind = np.argmin(local_extreme_vals)
                ext_lmi = local_extreme_inds[local_extreme_tmp_ind]
            return ext_lmi

        caseN = s.case_ind_map[case]
        lats, erps = s.prepare_plot_data()
        n_tms = lats.shape[0]

        ch_inds = [s.electrodes.index(c) for c in chan_list]
        # erps_ch = np.squeeze(erps[ caseN, ch_inds, : ])

        n_chans = len(chan_list)
        latmat = np.matrix(lats).repeat(n_chans, axis=0)

        if type(starts_ms) == list:
            starts_ms = np.matrix(starts_ms).transpose()
        if type(ends_ms) == list:
            ends_ms = np.matrix(ends_ms).transpose()

        startsmat = np.matrix(starts_ms).repeat(n_tms, axis=1)
        endsmat = np.matrix(ends_ms).repeat(n_tms, axis=1)

        start_pts = np.argmin(np.fabs(latmat - startsmat), axis=1)
        end_pts = np.argmin(np.fabs(latmat - endsmat), axis=1)

        # find min/max in range
        peak_vals = []
        peak_pts = []
        for ci, ch in enumerate(chan_list):
            erpa = np.squeeze(erps[caseN, ch_inds[ci], :])
            start_pt = start_pts.item(ci)
            end_pt = end_pts.item(ci)

            erp = erpa[start_pt:end_pt + 1]
            peak_pt = peak_alg(erp, polarity)
            peak_vals.append(erp[peak_pt])
            peak_pts.append(int(peak_pt + start_pt))

        peaks_ms = lats[peak_pts]

        return peak_vals, peaks_ms

    def find_peak(s, case, start_ms=200, end_ms=600,
                  chan_scope='all', chan=0, peak_polarity='p'):
        # erps is cases x chans x pts
        caseN = s.case_list.index(case)
        lats, erps = s.prepare_plot_data()

        start_pt = np.argmin(np.fabs(lats - start_ms))
        end_pt = np.argmin(np.fabs(lats - end_ms))

        # get data
        if chan_scope == 'one':  # find peak for one chan
            erpa = erps[caseN, chan, :]
        elif chan_scope == 'all':  # find peak for all chans
            erpa = erps[caseN, :, :]
            erpa = erpa.swapaxes(0, 1)
        else:
            return  # error, the range is not correctly specified

        # find min/max in range
        if peak_polarity == 'p':  # find the max
            peak_val = np.max(erpa[start_pt:end_pt + 1], axis=0)
            peak_pt = np.argmax(erpa[start_pt:end_pt + 1], axis=0) + start_pt
        elif peak_polarity == 'n':  # find the min
            peak_val = np.min(erpa[start_pt:end_pt + 1], axis=0)
            peak_pt = np.argmin(erpa[start_pt:end_pt + 1], axis=0) + start_pt
        else:
            return  # error, the peak polarity is not correctly specified

        '''
        # check if at edge
        if chan_scope == 'one':  # test
            if peak_pt == start_pt or peak_pt == end_pt:
                pass  # peak is at an edge
        elif chan_scope == 'all':
            if any(peak_pt == start_pt) or any(peak_pt == end_pt):
                pass  # at least one peak is at an edge
        '''

        peak_ms = lats[peak_pt]  # convert to ms if necessary

        return peak_val, peak_ms

    def case_letter_from_number(s, number):
        return s.cases[int(number)]['case_type']

    def get_yscale(s, potentials=None, channels=None, cases=None):
        if potentials is None:
            dummy, potentials = s.prepare_plot_data()
        # get full list of display channels
        if channels is None:
            channels = s.electrodes
        if cases is None:
            cases = s.case_list
        ch_ind_lst = []
        # print('electrodes:', s.filepath, s.electrodes)
        for chan in channels:
            ch_ind_lst.append(s.electrodes.index(chan))
        cs_ind_lst = []
        for case in cases:
            cs_ind_lst.append(s.case_list.index(case))

        # find means / stds
        pots = np.take(potentials, ch_ind_lst, 1)
        pots = np.take(pots,cs_ind_lst,0)
        pots_mean = np.mean(pots, axis=2)
        pots_mean2 = np.mean(pots_mean, axis=1)
        pots_meanstd = np.std(pots_mean, axis=1)

        # find indices of outlying channels
        n_std = 2  # num of stdev away from mean to detect
        out_means = np.logical_or(
            pots_mean.T < pots_mean2 - n_std * pots_meanstd,
            pots_mean.T > pots_mean2 + n_std * pots_meanstd)
        out_finds = out_means.any(axis=1)
        out_inds = np.nonzero(out_finds)[0]

        # remove outlying chans from calculation
        for ind in out_inds:
            ch_ind_lst.remove(s.electrodes.index(channels[ind]))
        disp_pots = np.take(potentials, ch_ind_lst, 1)
        disp_pots = np.take(disp_pots,cs_ind_lst,0)

        # calculate min / max for y limits
        min_val = int(np.floor(np.min(disp_pots)))
        max_val = int(np.ceil(np.max(disp_pots)))
        #print('Yscale: ', min_val, max_val)
        return min_val, max_val

    def butterfly_channels_by_case(s, channel_list=['FZ', 'CZ', 'PZ'], offset=0):
        s.extract_case_data()

        # edata = s.loaded['zdata']['zdata']

        # tms = np.array(range(edata.shape[2]))/samp_freq

        tms, pot = s.prepare_plot_data()

        colors = brewer['Spectral'][len(channel_list)]
        plots = []
        for cs in range(pot.shape[0]):
            callback = CustomJS(code="alert('clicked')")
            # callback = CustomJS( code="function(){ var data = source.get('data'); console.dir(data); }" )
            tap = TapTool(callback=callback)
            splot = figure(width=550, height=350, title=s.cases[cs + 1]['descriptor'], tools=[tap])
            tick_locs = []
            for cnt, ch in enumerate(channel_list):
                y_level = offset * cnt
                tick_locs.append(y_level)
                ch_ind = s.electrodes.index(ch)
                splot.line(x=tms, y=pot[cs, ch_ind, :] + y_level, color=colors[cnt],
                           line_width=3, line_alpha=0.85, legend=ch)
            splot.legend.orientation = 'top_left'
            splot.legend.background_fill_alpha = 0
            splot.legend.label_standoff = 0
            splot.legend.legend_padding = 2
            splot.legend.legend_spacing = 2
            splot.yaxis[0].ticker = FixedTicker(ticks=[])  # tick_locs,tags=channel_list)
            splot.xaxis.axis_label = "Time (s)"
            plots.append(splot)
        g = GridPlot([plots])
        show(g)

    def selected_cases_by_channel(s, cases='all', channels='all', props={},
                                  mode='notebook', source=None, time_range='all',
                                  tools=[], tool_gen=[], style='grid'):

        # Setup properties for plots
        default_props = {'width': 250,
                         'height': 150,
                         'min_border': 2,
                         'extra_bottom_height': 20,
                         'font size': 8,
                         'axis alpha': 0,
                         'outline alpha': 1,
                         'grid alpha': 0.75}

        default_props.update(props)
        props = default_props

        s.extract_case_data()
        if cases == 'all':
            cases = s.case_list
        if channels == 'all':
            channels = s.electrodes
        elif channels == 'core_31':
            channels = s.electrodes[0:31]
            channels.append(s.electrodes[63])

        tms, potentials = s.prepare_plot_data(time_range=time_range)

        props['times'] = tms

        min_val, max_val = s.get_yscale(potentials, channels, cases)

        props['yrange'] = [min_val, max_val]
        props['xrange'] = [-100, 700]

        if len(cases) > 3:
            props['colors'] = brewer['Spectral'][len(cases)]
        else:
            props['colors'] = ['#DD2222', '#66DD66', '#2222DD']

        if style == 'grid':
            n_plots = len(channels)  # potentials.shape[1]
            n_per_row = int(np.ceil(n_plots ** 0.5))

            plots = []
            for plot_ind, electrode in enumerate(channels):
                eind = s.electrodes.index(electrode)
                if plot_ind % n_per_row == 0:
                    plots.append([])

                if n_plots - plot_ind < n_per_row + 1:
                    bot_flag = True
                else:
                    bot_flag = False
                if plot_ind == 0:
                    leg_flag = True
                else:
                    leg_flag = False

                splot = s.prepare_plot_for_channel(potentials, eind, props, cases, tools,
                                                   bottom_label=bot_flag, legend=leg_flag, mode=mode,
                                                   source=source, tool_gen=tool_gen)
                plots[-1].append(splot)

        elif style == 'layout':
            layout = split_grid_layout
            # if 'FP1' in chans:
            # elif 'FPZ' in chans:
            # layout.append( [None, 'FPZ', None,  None, None]  )
            # layout.append( [None,'CP3', 'CPZ', 'CP4', None] )

            plots = []
            for row in layout:
                plots.append([])
                for cell in row:
                    if cell is None or cell not in s.electrodes:
                        plots[-1].append(None)
                    else:
                        eind = s.electrodes.index(cell)
                        splot = s.prepare_plot_for_channel(potentials, eind, props,
                                                           cases, tools, mode=mode, source=source,
                                                           tool_gen=tool_gen)
                        plots[-1].append(splot)

        if mode == 'server':
            return plots
        else:
            g =  GridPlot(plots, border_space=-40)  # , tools=[TapTool()])#tools )
            show(g)

    def extract_mt_data(s):

        if 'mt_data' not in dir(s):
            # mt_dir = os.path.split(s.filepath)[0]
            h1_name = os.path.split(s.filepath)[1]
            s.mt_name = os.path.splitext(h1_name)[0] + '.mt'
            s.mt_defaultpath = os.path.splitext(s.filepath)[0] + '.mt'
            #print(s.mt_defaultpath)
            if os.path.isfile(s.mt_defaultpath):
                mt = MT_File(s.mt_defaultpath)
                mt.parse_file()
                s.mt_data = mt.mt_data
                s.case_peaks = mt.mt_data.keys()
                peak_lst = []
                for c_pk in s.case_peaks:
                    peak_lst.append(c_pk[1])
                s.peaks = list(set(peak_lst))
                s.data_loaded = True
            else:
                s.data_loaded = False

        return s.data_loaded

    def make_data_sources(s, channels='all', empty_flag=False, time_range='all'):
        times, potentials = s.prepare_plot_data(time_range=time_range)
        s.extract_case_data()
        s.extract_mt_data()

        times_use = times
        if empty_flag:
            times_use = []
        pot_source_dict = OrderedDict(times=times_use)

        if channels == 'all':
            channels = s.electrodes

        # peaks
        peak_sourcesD = {}
        for case in s.case_list:
            peak_source_dict = dict(peaks=[])
            for chan in channels:
                peak_source_dict[chan + '_pot'] = []
                peak_source_dict[chan + '_time'] = []
            peak_sourcesD[case] = peak_source_dict

        if 'mt_data' in dir(s):
            for case, peak in s.case_peaks:
                c_pk = (case, peak)
                case_name = s.case_list[int(case) - 1]
                peak_sourcesD[case_name]['peaks'].append(peak)
                for chan in channels:
                    if chan not in ['X', 'Y', 'BLANK']:
                        peak_sourcesD[case_name][chan + '_pot'].append(float(s.mt_data[c_pk][chan][0]))
                        peak_sourcesD[case_name][chan + '_time'].append(float(s.mt_data[c_pk][chan][1]))
                    else:
                        peak_sourcesD[case_name][chan + '_pot'].append(0)  # np.nan )
                        peak_sourcesD[case_name][chan + '_time'].append(-500)  # np.nan )

        # potentials
        for chan in channels:
            ch_ind = s.electrodes.index(chan)
            for cs_num, cs in s.cases.items():
                case_name = cs['case_type']
                cs_ind = s.case_ind_map[case_name]
                if empty_flag:
                    pot_source_dict[chan + '_' + case_name] = []
                else:
                    pot_source_dict[chan + '_' + case_name] = potentials[cs_ind, ch_ind, :]

        # return peak_sourcesD
        return pot_source_dict, peak_sourcesD

    def get_peak_data(s, channel, case, peak):
        '''return (amplitude,latency) for channel, case, peak
        '''
        # print s.mt_data()
        if type(case) == str:
            pass

        return s.mt_data[(case, peak)][channel]

    def prepare_plot_for_channel(s, pot, el_ind, props, case_list, tools,
                                 mode='notebook', bottom_label=False, legend=False,
                                 source=None, tool_gen=None):
        PS = {'props': props, 'case list': case_list, 'tools': tools}  # plot_setup

        if bottom_label:
            height = props['height'] + props['extra_bottom_height']
        else:
            height = props['height']

        PS['electrode'] = s.electrodes[el_ind]
        PS['adjusted height'] = height

        PS['tool generators'] = tool_gen

        return PS

class MT_File:
    ''' manually picked files from eeg experiments
        initialization only parses the filename, call parse_file to load data
    '''
    columns = ['subject_id', 'experiment', 'version', 'gender', 'age', 'case_num',
               'electrode', 'peak', 'amplitude', 'latency', 'reaction_time']

    cases_peaks_by_experiment = {'aod': {(1, 'tt'): ['N1', 'P3'],
                                         (2, 'nt'): ['N1', 'P2']
                                         },
                                 'vp3': {(1, 'tt'): ['N1', 'P3'],
                                         (2, 'nt'): ['N1', 'P3'],
                                         (3, 'nv'): ['N1', 'P3']
                                         },
                                 'ant': {(1, 'a'): ['N4', 'P3'],
                                         (2, 'j'): ['N4', 'P3'],
                                         (3, 'w'): ['N4', 'P3'],
                                         # (4, 'p'): ['P3', 'N4']
                                         }
                                 }

    # string for reference
    data_structure = '{(case#,peak):{electrodes:(amplitude,latency),reaction_time:time} }'

    ant_cases_types_lk = [((1, 'A', 'Antonym'),
                           (2, 'J', 'Jumble'),
                           (3, 'W', 'Word'),
                           (4, 'P', 'Prime')),
                          ((1, 'T', 'jumble'),
                           (2, 'T', 'prime'),
                           (3, 'T', 'antonym'),
                           (4, 'T', 'other')),
                          ((1, 'T', 'jumble'),
                           (2, 'T', ' prime'),
                           (3, 'T', ' antonym'),
                           (4, 'T', ' other')),
                          ((1, 'T', 'jumble'),
                           (2, 'T', 'prime'),
                           (3, 'T', 'antonym'),
                           (4, 'T', 'word'))]

    case_fields = ['case_num', 'case_type', 'descriptor']

    ant_case_convD = {0: {1: 1, 2: 2, 3: 3, 4: 4},  # Translates case0 to each case
                      1: {1: 3, 2: 1, 3: 4, 4: 2},
                      2: {1: 3, 2: 1, 3: 4, 4: 2},
                      3: {1: 3, 2: 1, 3: 4, 4: 2}}

    # 4:{1:1,2:2,3:3,4:4} }
    case_nums2names = {'aod': {1: 't', 2: 'nt'},
                       'vp3': {1: 't', 2: 'nt', 3: 'nv'},
                       'ant': {1: 'j', 2: 'p', 3: 'a', 4: 'w'},
                       'cpt': {1: 'g', 2: 'c', 3: 'cng',
                               4: 'db4ng', 5: 'ng', 6: 'dad'},
                       'stp': {1: 'c', 2: 'i'},
                       }

    query_fields = ['id', 'session', 'experiment']

    def normAntCase(s):
        query = {k: v for k, v in s.file_info.items() if k in s.query_fields}
        doc = D.Mdb['avgh1s'].find_one(query)
        avgh1_path = doc['filepath']
        case_tup = extract_case_tuple(avgh1_path)
        case_type = MT_File.ant_cases_types_lk.index(case_tup)
        return MT_File.ant_case_convD[case_type]

    def __init__(s, filepath):
        s.fullpath = filepath
        s.filename = os.path.split(filepath)[1]
        s.header = {'cases_peaks': {}}

        s.parse_fileinfo()

        s.data = dict()
        s.data['uID'] = s.file_info['id'] + '_' + s.file_info['session']

        if s.file_info['experiment'] == 'ant':
            s.normed_cases_calc()
        s.parse_header()

    def parse_fileinfo(s):
        s.file_info = parse_filename(s.filename)

    def __repr__(s):
        return '<mt-file object ' + str(s.file_info) + ' >'

    def parse_header(s):
        of = open(s.fullpath, 'r')
        reading_header = True
        s.header_lines = 0
        while reading_header:
            file_line = of.readline()
            if len(file_line) < 2 or file_line[0] != '#':
                reading_header = False
                continue
            s.header_lines += 1

            line_parts = [pt.strip() for pt in file_line[1:-1].split(';')]
            if 'nchans' in line_parts[0]:
                s.header['nchans'] = int(line_parts[0].split(' ')[1])
            elif 'case' in line_parts[0]:
                cs_pks = [lp.split(' ') for lp in line_parts]
                if cs_pks[1][0] != 'npeaks':
                    s.header['problems'] = True
                else:
                    case = int(cs_pks[0][1])
                    if 'normed_cases' in dir(s):
                        case = s.normed_cases[case]
                    s.header['cases_peaks'][case] = int(cs_pks[1][1])

        of.close()

    def normed_cases_calc(s):
        try:
            norm_dict = s.normAntCase()
            s.normed_cases = norm_dict
        except:
            s.normed_cases = MT_File.ant_case_convD[0]
            s.norm_fail = True

    def parse_fileDB(s, general_info=False):
        s.parse_file()
        exp = s.file_info['experiment']
        ddict = {}
        for k in s.mt_data:  # for
            case_convdict = s.case_nums2names[exp]
            case = case_convdict[int(k[0])]
            peak = k[1]
            inner_ddict = {}
            for chan, amp_lat in s.mt_data[k].items():  # chans
                if type(amp_lat) is tuple:  # if amp / lat tuple
                    inner_ddict.update(
                        {chan: {'amp': float(amp_lat[0]),
                                'lat': float(amp_lat[1])}}
                    )
            ddict[case + '_' + peak] = inner_ddict
        ddict['filepath'] = s.fullpath
        ddict['run'] = s.file_info['run']
        ddict['version'] = s.file_info['version']

        if general_info:
            s.data.update(s.file_info)
            del s.data['experiment']
            del s.data['run']
            del s.data['version']
            s.data['ID'] = s.data['id']

        s.data[exp] = ddict

    def parse_file(s):
        of = open(s.fullpath, 'r')
        data_lines = of.readlines()[s.header_lines:]
        of.close()
        s.mt_data = OrderedDict()
        for L in data_lines:
            Ld = {c: v for c, v in zip(s.columns, L.split())}
            if 'normed_cases' in dir(s):
                Ld['case_num'] = s.normed_cases[int(Ld['case_num'])]
            key = (int(Ld['case_num']), Ld['peak'])
            if key not in s.mt_data:
                s.mt_data[key] = OrderedDict()
            s.mt_data[key][Ld['electrode'].upper()] = (
                Ld['amplitude'], Ld['latency'])
            if 'reaction_time' not in s.mt_data[key]:
                s.mt_data[key]['reaction_time'] = Ld['reaction_time']
        return

    def parse_fileDF(s):
        s.dataDF = pd.read_csv(s.fullpath, delim_whitespace=True,
                               comment='#', names=s.columns)

    def check_peak_order(s):
        ''' Pandas Dataframe based '''
        if 'dataDF' not in dir(s):
            s.parse_fileDF()
        if 'normed_cases' in dir(s):
            case_lk = {v: k for k, v in s.normed_cases.items()}
        probs = {}
        # peaks by case number
        case_peaks = {k[0]: v for k, v in
                      s.cases_peaks_by_experiment[s.file_info['experiment']].items()}
        cols_use = ['electrode', 'latency']
        for case in s.dataDF['case_num'].unique():
            cDF = s.dataDF[s.dataDF['case_num'] == case]
            if 'normed_cases' in dir(s):
                case_norm = case_lk[case]
            else:
                case_norm = case
            if case_norm in case_peaks:
                pk = case_peaks[case_norm][0]
                ordDF = cDF[cDF['peak'] == pk][cols_use]
                ordDF.rename(columns={'latency': 'latency_' + pk}, inplace=True)
                peak_track = [pk]
                delta_cols = []
                if case in case_peaks:
                    for pk in case_peaks[case][1:]:
                        pkDF = cDF[cDF['peak'] == pk][cols_use]
                        pkDF.rename(columns={'latency': 'latency_' + pk}, inplace=True)
                        # return (ordDF, pkDF)
                        ordDF = ordDF.join(pkDF, on='electrode', rsuffix=pk)
                        delta_col = pk + '_' + peak_track[-1] + '_delta'
                        ordDF[delta_col] = \
                            ordDF['latency_' + pk] - ordDF['latency_' + peak_track[-1]]
                        peak_track.append(pk)
                        delta_cols.append(delta_col)

                for dc in delta_cols:
                    wrong_order = ordDF[ordDF[dc] < 0]
                    if len(wrong_order) > 0:
                        case_name = s.case_nums2names[s.file_info['experiment']][case_norm]
                        probs[case_name + '_' + dc] = list(wrong_order['electrode'])

        if len(probs) == 0:
            return True
        else:
            return probs

    def check_max_latency(s, latency_thresh=1000):
        ''' Pandas Dataframe based '''
        if 'dataDF' not in dir(s):
            s.parse_fileDF()
        high_lat = s.dataDF[s.dataDF['latency'] > latency_thresh]
        if len(high_lat) == 0:
            return True
        else:
            return high_lat[['case_num', 'electorde', 'peak', 'amplitude', 'latency']]

    def build_header(s):
        if 'mt_data' not in dir(s):
            s.parse_file()
        cases_peaks = list(s.mt_data.keys())
        cases_peaks.sort()
        header_data = OrderedDict()
        for cp in cases_peaks:
            if cp[0] not in header_data:
                header_data[cp[0]] = 0
            header_data[cp[0]] += 1

        # one less for reaction_time
        s.header_text = '#nchans ' + \
                        str(len(s.mt_data[cases_peaks[0]]) - 1) + '\n'
        for cs, ch_count in header_data.items():
            s.header_text += '#case ' + \
                             str(cs) + '; npeaks ' + str(ch_count) + ';\n'

        print(s.header_text)

    def build_file(s):
        pass

    def check_header_for_experiment(s):
        expected = s.cases_peaks_by_experiment[s.file_info['experiment']]
        if len(expected) != len(s.header['cases_peaks']):
            return 'Wrong number of cases'
        case_problems = []
        for pknum_name, pk_list in expected.items():
            if s.header['cases_peaks'][pknum_name[0]] != len(pk_list):
                case_problems.append(
                    'Wrong number of peaks for case ' + str(pknum_name))
        if case_problems:
            return str(case_problems)

        return True

    def check_peak_identities(s):
        if 'mt_data' not in dir(s):
            s.parse_file()
        for case, peaks in s.cases_peaks_by_experiment[s.file_info['experiment']].items():
            if (case[0], peaks[0]) not in s.mt_data:
                return False, 'case ' + str(case) + ' missing ' + peaks[0] + ' peak'
            if (case[0], peaks[1]) not in s.mt_data:
                return False, 'case ' + str(case) + ' missing ' + peaks[1] + ' peak'
        return True

    def check_peak_orderNmax_latency(s, latency_thresh=1000):
        if 'mt_data' not in dir(s):
            s.parse_file()
        for case, peaks in s.cases_peaks_by_experiment[s.file_info['experiment']].items():
            try:
                latency1 = float(s.mt_data[(case[0], peaks[0])]['FZ'][1])
                latency2 = float(s.mt_data[(case[0], peaks[1])]['FZ'][1])
            except:
                print(s.fullpath + ': ' +
                      str(s.mt_data[(case[0], peaks[0])].keys()))
            if latency1 > latency_thresh:
                return (
                    False,
                    str(case) + ' ' + peaks[0] + ' ' + 'exceeds latency threshold (' + str(latency_thresh) + 'ms)')
            if latency2 > latency_thresh:
                return (
                    False,
                    str(case) + ' ' + peaks[1] + ' ' + 'exceeds latency threshold (' + str(latency_thresh) + 'ms)')
            if latency1 > latency2:
                return False, 'Wrong order for case ' + str(case)
        return True
