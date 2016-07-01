''' represents intermediate results from processing pipeline:
    does import, analyses, plotting '''

from glob import glob
import os
import itertools

import h5py
import dask.array as da

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne

# values are tuples of h5py fieldname and datatype
opt_info = {'Coordinates file': ('coords_file', 'text'),
            'Batch ID': ('batch_id', 'text'),

            'Condition labels': ('case_label', 'cell'),
            'Measures available': ('measures', 'cell'),
            'Coherence pair subset labels': ('pair_indlbls', 'cell'),

            'Sampling rate': ('rate', 'array'),
            '# of timepoints': ('n_samps', 'array'),
            'Temporal limits': ('epoch_lims', 'array'),
            'Frequency limits': ('freq_lims', 'array'),
            'TF scales': ('wavelet_scales', 'array'),
            'TF time-downsample ratio': ('tf_timedownsamp_ratio', 'array'),
            'CSD matrix': ('csd_G', 'array'),
            'Coherence pairs': ('coherence_pairs', 'array'),
            'Coherence pair subset index': ('pair_inds', 'array'),
            }

# text parsing


def uid_frompath(fp):
    ''' given HBNL-formatted filepath, returns ID, session tuple '''
    path_pieces = fp.split('/')
    file_pieces = path_pieces[-1].split('_')
    return file_pieces[3], file_pieces[2][0]

# dataframe-oriented functions


def uid_inds(df):
    ''' set ID and session as dataframe indices if they exist '''
    def_inds = ['ID', 'session']
    for ind in def_inds:
        if ind in df.columns:
            do_append = df.index.name != None
            df.set_index(ind, append=do_append, inplace=True)

# h5py parsing functions


def parse_text(dset, dset_field):
    ''' parse .mat-style h5 field that contains text '''
    dset_ref = dset[dset_field]
    return ''.join(chr(c[0]) for c in dset_ref[:])

def parse_cell(dset, dset_field):
    ''' parse .mat-style h5 field that contains a cell array '''
    try:
        dset_ref = dset[dset_field]
        refs = [t[0] for t in dset_ref[:]]
        out_lst = [''.join(chr(c) for c in dset[ref][:]) for ref in refs]
        return out_lst
    except:
        return []

def parse_array(dset, dset_field):
    ''' parse .mat-style h5 field that contains a numerical array.
        not in use yet. '''
    contents = dset[dset_field][:]
    if contents.shape == (1, 1):
        return contents[0][0]
    elif contents == np.array([0, 0], dtype=np.uint64):
        return None
    else:
        return contents

ftypes_funcs = {'text': parse_text, 'cell': parse_cell, 'array': None}

def handle_parse(dset, dset_field, field_type):
    ''' given file pointer, field, and datatype, apply appropriate parser '''
    func = ftypes_funcs[field_type]
    return func(dset, dset_field) if func else dset[dset_field][:]


# array functions
def baseline_amp(array, pt_lims, along_dim=-1):
    ''' baseline array in a subtractive way '''
    return array - array.take(range(pt_lims[0], pt_lims[1]+1), axis=along_dim)\
                        .mean(axis=along_dim, keepdims=True)

def baseline_tf(array, pt_lims, along_dim=-1):
    ''' baseline array in a divisive way '''
    return 10 * np.log10(array / array.take(range(pt_lims[0], pt_lims[1] + 1),
                                            axis=along_dim)
                         .mean(axis=along_dim, keepdims=True))

def convert_ms(time_array, ms):
    ''' given time array, find index nearest to given time value '''
    return np.argmin(np.fabs(time_array - ms))

def compound_take(a, vals, dims):
    ''' given array, apply multiple indexing operations '''
    def apply_take(a, v, d):
        if isinstance(v, int):
            return a.take([v], d)
        else:
            return a.take(v, d)
    print(a.shape)
    for v, d in zip(vals, dims):
        if isinstance(v, tuple):
            for v_stage, d_stage in zip(v, d):
                a = apply_take(a, v_stage, d_stage)
        else:
            a = apply_take(a, v, d)
        print(a.shape)
    return np.squeeze(a)


# plotting functions


def subplot_heuristic(n):
    ''' for n subplots, determine best grid layout dimensions '''
    def isprime(n):
        for x in range(2, int(np.sqrt(n)) + 1):
            if n % x == 0:
                return False
        return True
    if n > 6 and isprime(n):
        n += 1
    num_lst, den_lst = [n], [1]
    for x in range(2, int(np.sqrt(n)) + 1):
        if n % x == 0:
            den_lst.append(x)
            num_lst.append(n // x)
    ratios = np.array([a / b for a, b in zip(num_lst, den_lst)])
    best_ind = np.argmin(ratios - 1.1618)  # most golden
    if den_lst[best_ind] < num_lst[best_ind]:
        return den_lst[best_ind], num_lst[best_ind]
    else:
        return num_lst[best_ind], den_lst[best_ind]


class Results:
    ''' represents HDF-compatible .mat's as dask stacks '''

    def __init__(s, optpath, csvpath=None):
        s.opt = h5py.File(optpath)
        s.init_filedf()
        if csvpath:
            s.add_demogs(csvpath)
        s.get_params()
        s.make_scales()

    def init_filedf(s):
        ''' initialize dataframe of .mat files in the opt's "outpath"
            indexed by ID+session '''
        datapath = parse_text(s.opt, 'opt/outpath')
        files = glob(datapath + '/*.mat')
        uid_fromfiles = [uid_frompath(fp) for fp in files]
        uid_index = pd.MultiIndex.from_tuples(
            uid_fromfiles, names=['ID', 'session'])
        s.file_df = pd.DataFrame({'path': pd.Series(files, index=uid_index)})

    def add_demogs(s, csvpath):
        ''' read demographics, set ID+session as index, join to files '''
        demog_df = pd.read_csv(csvpath)
        uid_inds(demog_df)
        s.demog_df = demog_df.join(s.file_df)

    def get_params(s):
        ''' extract data parameters from opt, relying upon opt_info. '''
        prefix = 'opt/'
        s.params = {}
        for param, info in opt_info.items():
            s.params.update({param:
                             handle_parse(s.opt, prefix + info[0], info[1])})

    def make_scales(s):
        ''' populate attributes describing channels and units '''
        
        # channels
        use_ext = '.sfp'
        path, file = os.path.split(s.params['Coordinates file'])
        fn, ext = file.split('.')
        s.montage = mne.channels.read_montage(os.path.join(path, fn) + use_ext)

        # units and suggested limits; CSD transform or not
        if s.params['CSD matrix'].shape[0] > 2:
            s.pot_units = r'$\mu$V / $cm^2$'
            s.pot_lims = [-.2, .2]
        else:
            s.pot_units = r'$\mu$V'
            s.pot_lims = [-10, 16]
        s.db_units = 'decibels (dB)'
        s.db_lims = [-3, 3]
        s.itc_lims = [-.06, 0.3]

        # ERP times
        s.srate = s.params['Sampling rate'][0][0]
        ep_lims = s.params['Temporal limits']
        n_timepts = s.params['# of timepoints']
        s.time = np.linspace(ep_lims[0], ep_lims[1], n_timepts + 1)[1:]

        # TF times / freqs
        n_timepts_tf = int(n_timepts / s.params['TF time-downsample ratio'])
        s.time_tf = np.linspace(ep_lims[0], ep_lims[1], n_timepts_tf + 1)[1:]
        s.freq = np.array([2 * s.srate / scale[0]
                           for scale in s.params['TF scales'][::-1]]) #rev
        s.freq_ticks_pt = range(0, len(s.freq), 2)
        s.freq_ticks_hz = ['{0:.1f}'.format(f) for f in s.freq[::2]]

        s.time_ticks() # additional helper function for ticks

    def time_ticks(s, interval=200):
        ''' time ticks for plots '''
        ms_start_plot = np.round(
            s.time[0] / 100) * 100  # start at a round number
        first_tick = int(ms_start_plot + ms_start_plot % interval)
        time_ticks_ms = list(range(first_tick, int(s.time[-1]), interval))

        s.time_ticks_pt_erp = [convert_ms(s.time, ms) for ms in time_ticks_ms]
        s.time_ticks_pt_tf = [convert_ms(s.time_tf, ms)
                              for ms in time_ticks_ms]

        # for display purposes
        s.time_ticks_ms = [n / 1000 for n in time_ticks_ms]

    def load(s, measure, dset_name, transpose_lst):
        # need to know: name of h5 dataset, transpose list
        if measure in dir(s):
            print(measure, 'already loaded')
            return

        dsets = [h5py.File(fn, 'r')[dset_name]
                    for fn in s.demog_df['path'].values]
        arrays = [da.from_array(dset, chunks=dset.shape) for dset in dsets]
        stack = da.stack(arrays, axis=-1)  # concatenate along last axis
        stack = stack.transpose(transpose_lst) # do transposition
        data = np.empty(stack.shape)
        da.store(stack, data)
        print(data.shape)
        return data


    def load_erp(s, lp_cutoff=16, bl_window=[-100, 0]):
        ''' load, filter, and subtractively baseline ERP data '''

        erp = s.load('erp', 'erp', [3, 0, 1, 2])
        # erp is (subjects, conditions, channels, timepoints,)

        # filter
        erp_filt = mne.filter.low_pass_filter(erp,
                                              s.params['Sampling rate'],
                                              lp_cutoff)
        # baseline
        erp_filt_bl = baseline_amp(erp_filt, (convert_ms(s.time, bl_window[0]),
                                              convert_ms(s.time, bl_window[1])))

        s.erp = erp_filt_bl
        s.erp_dims = ('subject', 'condition', 'channel', 'timepoint')
        s.erp_dim_lsts = (s.demog_df.index.values, s.params['Condition labels'],
                          s.montage.ch_names, s.time)

    def prepare_mne(s):
        ''' prepare an mne EvokedArray object from erp data '''
        if 'erp' not in dir(s):
            s.load_erp()

        # create info
        info = mne.create_info(s.montage.ch_names, s.params['Sampling rate'],
                               'eeg', s.montage)
        # EvokedArray
        chan_erps = s.erp.mean(axis=(0, 1)) / 1e6  # subject/condition mean
        return mne.EvokedArray(chan_erps, info,
                               tmin=s.params['Temporal limits'][0] / 1000)

    def load_power(s, bl_window=[-500, -200]):
        ''' load (total) power data and divisively baseline-normalize '''

        power = s.load('power', 'wave_totpow', [4, 0, 2, 1, 3])
        power = power[:, :, :, ::-1, :]  # reverse the freq dimension
        # power is (subjects, conditions, channels, freq, timepoints,)

        # divisively baseline normalize
        power_bl = baseline_tf(power, (convert_ms(s.time_tf, bl_window[0]),
                                       convert_ms(s.time_tf, bl_window[1]),))

        s.power = power_bl
        s.tf_dims = ('subject', 'condition', 'channel',
                        'frequency', 'timepoint')
        s.tf_dim_lsts = (s.demog_df.index.values,
                            s.params['Condition labels'],
                            s.montage.ch_names, s.freq, s.time_tf)

    def load_itc(s, bl_window=[-500, -200]):
        ''' load phase data, take absolute() of, and subtractively baseline '''

        dsets = [h5py.File(fn, 'r')['wave_evknorm']
                 for fn in s.demog_df['path'].values]
        arrays = [dset.value.view(np.complex) for dset in dsets]
        stack = np.stack(arrays, axis=-1)  # concatenate along last axis
        print(stack.shape)
        stack = stack.transpose([4, 0, 2, 1, 3])  # subject dimension to front

        # itc is (subjects, conditions, channels, freq, timepoints,)
        itc = np.absolute(stack)
        itc = itc[:, :, :, ::-1, :]  # reverse the freq dimension
        print(itc.shape)

        # baseline normalize
        itc_bl = baseline_amp(itc, (convert_ms(s.time_tf, bl_window[0]),
                                    convert_ms(s.time_tf, bl_window[1]),))

        s.itc = itc_bl
        s.tf_dims = ('subject', 'condition', 'channel',
                        'frequency', 'timepoint')
        s.tf_dim_lsts = (s.demog_df.index.values,
                            s.params['Condition labels'],
                            s.montage.ch_names, s.freq, s.time_tf)


    def plot_erp(s, figure_by=('channel', ['FZ', 'CZ', 'PZ']),
                 subplot_by=('POP', None),
                 glyph_by=('condition', None),
                 savedir=None):
        ''' plot ERPs as lines '''
        ptype = 'line'
        measure = 'erp'

        if 'erp' not in dir(s):
            s.load_erp()
        d_dims = s.erp_dims
        d_dimlst = s.erp_dim_lsts

        f_dim, f_vals, f_lbls = s.handle_by(figure_by, d_dims, d_dimlst)
        sp_dim, sp_vals, sp_lbls = s.handle_by(subplot_by, d_dims, d_dimlst)
        g_dim, g_vals, g_lbls = s.handle_by(glyph_by, d_dims, d_dimlst)

        sp_dims = subplot_heuristic(len(sp_vals))
        for fi, fval in enumerate(f_vals):
            f, axarr = plt.subplots(sp_dims[0], sp_dims[1],
                                    sharex=True, sharey=True, figsize=(12, 5))
            f.suptitle(f_lbls[fi])
            axarr = axarr.ravel()
            for spi, spval in enumerate(sp_vals):
                for gi, gval in enumerate(g_vals):
                    line = compound_take(s.erp, [fval, spval, gval],
                                         [f_dim[fi], sp_dim[spi], g_dim[gi]])
                    print(line.shape)
                    while len(line.shape) > 1:
                        line = line.mean(axis=0)
                        print(line.shape)
                    axarr[spi].plot(np.arange(len(line)), line,
                                    label=g_lbls[gi])
                axarr[spi].grid(True)
                axarr[spi].set_title(sp_lbls[spi])
                axarr[spi].legend(loc='upper left')
                axarr[spi].set_xticks(s.time_ticks_pt_erp)
                axarr[spi].set_xticklabels(s.time_ticks_ms)
                axarr[spi].set_xlabel('Time (s)')
                axarr[spi].set_ylabel('Potential (' + s.pot_units + ')')
            if savedir:
                s.save_fig(savedir, ptype, measure, f_lbls[fi])

    def plot_topo(s, data='erp', times=list(range(0, 501, 125)),
                     figure_by=('POP', ['C']),
                     row_by=('condition', None)):
        ''' plot data as topographic maps at specific timepoints '''

        if data in ['erp', 'power', 'itc']:
            data, d_dims, d_dimlst, units, lims, cmap = s.get_plotparams(data)
        else:
            print('data not recognized')
            return

        final_dim = d_dims.index('channel')
        final_dimlen = len(d_dimlst[final_dim])
        info = mne.create_info(s.montage.ch_names, s.params['Sampling rate'],
                               'eeg', s.montage)

        f_dim, f_vals, f_lbls = s.handle_by(figure_by, d_dims, d_dimlst)
        r_dim, r_vals, r_lbls = s.handle_by(row_by, d_dims, d_dimlst)
        time_by = ('timepoint', times)
        t_dim, t_vals, t_lbls = s.handle_by(time_by, d_dims, d_dimlst)

        sp_dims = (len(r_vals), len(times))
        for fi, fval in enumerate(f_vals):
            f, axarr = plt.subplots(sp_dims[0], sp_dims[1],
                                    sharex=True, sharey=True, figsize=(12, 5))
            f.suptitle(f_lbls[fi])
            axarr = axarr.ravel()
            ax_dum = -1
            for ri, rval in enumerate(r_vals):
                for ti, tval in enumerate(t_vals):
                    topo = compound_take(data, [fval, rval, tval],
                                         [f_dim[fi], r_dim[ri], t_dim[ti]])
                    print(topo.shape)
                    mean_dims = np.where([d!=final_dimlen for d in topo.shape])
                    topo = topo.mean(axis=tuple(dim for dim in mean_dims[0]))
                    print(topo.shape)
                    ax_dum += 1
                    im, cn = mne.viz.plot_topomap(topo, info,
                                        vmin=lims[0], vmax=lims[1],
                                        cmap=cmap, axes=axarr[ax_dum],
                                        show=False)
                    ''' labels '''
                    axarr[ax_dum].text(-.5, .5, t_lbls[ti])  # time label
                    if ti == 0:
                        axarr[ax_dum].text(-1.5, 0, r_lbls[ri])  # row label
            ''' colorbar '''
            plt.subplots_adjust(right=0.85)
            cbar_ax = f.add_axes([0.9, 0.15, 0.03, 0.75])
            cbar = plt.colorbar(im, cax=cbar_ax)
            cbar.ax.set_ylabel(units, rotation=270)

    def plot_tf(s, data='power', figure_by=[('POP', None), ('channel', ['FZ'])],
                                 subplot_by=('condition', None)):
        ''' plot time-frequency data as a rectangular contour image '''

        if data in ['power', 'itc']:
            data, d_dims, d_dimlst, units, lims, cmap = s.get_plotparams(data)
        else:
            print('data not recognized')
            return

        f_dim, f_vals, f_lbls = s.handle_by(figure_by, d_dims, d_dimlst)
        sp_dim, sp_vals, sp_lbls = s.handle_by(subplot_by, d_dims, d_dimlst)

        sp_dims = subplot_heuristic(len(sp_vals))
        for fi, fval in enumerate(f_vals):
            f, axarr = plt.subplots(sp_dims[0], sp_dims[1],
                                    sharex=True, sharey=True, figsize=(12, 5))
            f.suptitle(f_lbls[fi])
            axarr = axarr.ravel()
            for spi, spval in enumerate(sp_vals):
                rect = compound_take(data, [fval, spval],
                                     [f_dim[fi], sp_dim[spi]])
                print(rect.shape)
                while len(rect.shape) > 2:
                    rect = rect.mean(axis=0)
                    print(rect.shape)
                ''' contour '''
                c = axarr[spi].contourf(rect, 8, cmap=cmap,
                                        vmin=lims[0], vmax=lims[1])
                # c = axarr[spi].contour(rect, cmap=plt.cm.RdBu_r,
                #                         vmin=-4, vmax=4)
                # plt.clabel(c, inline=1, fontsize=9)
                ''' ticks and grid '''
                axarr[spi].set_xticks(s.time_ticks_pt_tf)
                axarr[spi].set_xticklabels(s.time_ticks_ms)
                axarr[spi].set_yticks(s.freq_ticks_pt)
                axarr[spi].set_yticklabels(s.freq_ticks_hz)
                axarr[spi].grid(True)
                ''' labels and title '''
                axarr[spi].set_xlabel('Time (s)')
                axarr[spi].set_ylabel('Frequency (Hz)')
                axarr[spi].set_title(sp_lbls[spi])
            ''' colorbar '''
            plt.subplots_adjust(right=0.85)
            cbar_ax = f.add_axes([0.88, 0.12, 0.03, 0.75])
            cbar = plt.colorbar(c, cax=cbar_ax)
            cbar.ax.set_ylabel(units, rotation=270)
            # plt.colorbar(c, ax=axarr[spi])

    # plot assistance
    ''' dictionary mapping measures to their object info '''
    measure_pps = {'erp':   {'data': 'erp', 'd_dims': 'erp_dims',
                             'd_dimlst': 'erp_dim_lsts', 'units': 'pot_units',
                             'lims': 'pot_lims', 'cmap': plt.cm.RdBu_r,
                             'load': 'load_erp'},
                   'power': {'data': 'power', 'd_dims': 'tf_dims',
                             'd_dimlst': 'tf_dim_lsts', 'units': 'db_units',
                             'lims': 'db_lims', 'cmap': plt.cm.RdBu_r,
                             'load': 'load_power'},
                   'itc':   {'data': 'itc', 'd_dims': 'tf_dims',
                             'd_dimlst': 'tf_dim_lsts', 'units': 'ITC',
                             'lims': 'itc_lims', 'cmap': plt.cm.Purples,
                             'load': 'load_itc'}
                   }
    
    def get_plotparams(s, measure):
        ''' given a measure, retrieve its data and plot info '''

        measure_d = s.measure_pps[measure]
        if measure_d['data'] not in dir(s):
            getattr(s, measure_d['load'])()
        data = getattr(s, measure_d['data'])
        d_dims = getattr(s, measure_d['d_dims'])
        d_dimlst = getattr(s, measure_d['d_dimlst'])
        units = getattr(s, measure_d['units'])
        lims = getattr(s, measure_d['lims'])
        cmap = measure_d['cmap']
        return data, d_dims, d_dimlst, units, lims, cmap

    def save_fig(s, savedir, ptype, measure, label, form='svg'):
        figname = s.gen_figname(ptype, measure, label)+'.'+form
        outpath = os.path.join(savedir, figname)
        plt.savefig(outpath, format=form, dpi=1000)

    def gen_figname(s, ptype, measure, label):
        return '_'.join([s.params['Batch ID'], ptype, measure, label])

    def handle_by(s, by_stage, d_dims, d_dimlst):
        ''' handle a 'by' argument, which tells a plotting functions what parts
            of the data will be distributed across a plotting object.
            returns lists of the dimension, indices, and labels requested.
            if given a list, create above lists as products of requests '''
        if isinstance(by_stage, list):
            # create list versions of the dim, vals, and labels
            tmp_dims, tmp_vals, tmp_labels = [], [], []
            for bs in by_stage:
                dims, vals, labels = s.interpret_by(bs, d_dims, d_dimlst)
                tmp_dims.append(dims)
                tmp_vals.append(vals)
                tmp_labels.append(labels)
            all_dims = list(itertools.product(*tmp_dims))
            all_vals = list(itertools.product(*tmp_vals))
            all_labels = list(itertools.product(*tmp_labels))
            return all_dims, all_vals, all_labels
        else:
            return s.interpret_by(by_stage, d_dims, d_dimlst)

    def interpret_by(s, by_stage, data_dims, data_dimlst):
        ''' by_stage: 2-tuple of variable name and levels
            data_dims: n-tuple describing the n dimensions of the data
            data_dimlst: n-tuple of lists describing levels of each dim '''

        print('by stage is', by_stage[0])
        if by_stage[0] in data_dims:  # if variable is in data dims
            dim = data_dims.index(by_stage[0])
            print('data in dim', dim)
            if by_stage[1]:
                labels = by_stage[1]
                if isinstance(data_dimlst[dim], np.ndarray):
                    vals = []
                    for lbl in labels:
                        if isinstance(lbl, list):
                            if len(lbl) == 2:
                                tmp_inds = range(np.argmin(np.fabs(\
                                    data_dimlst[dim] - lbl[0])),
                                                np.argmin(np.fabs(\
                                    data_dimlst[dim] - lbl[1]))+1)
                            else:
                                tmp_inds = [np.argmin(np.fabs(\
                                    data_dimlst[dim] - lp)) for lp in lbl]
                        else:
                            tmp_inds = np.argmin(np.fabs(\
                                data_dimlst[dim] - lbl))
                        vals.append(tmp_inds)
                else:
                    vals = [data_dimlst[dim].index(lbl) for lbl in labels]
                print('vals to iterate on are', vals)
            else:
                labels = data_dimlst[dim]
                vals = list(range(len(labels)))
                print('iterate across available vals including', vals)
        elif by_stage[0] in s.demog_df.columns:  # if variable in demog dims
            dim = data_dims.index('subject')
            print('demogs in', dim)
            if by_stage[1]:
                labels = by_stage[1]
                vals = [np.where(s.demog_df[by_stage[0]] == lbl)[0]
                        for lbl in labels]
                print('vals to iterate on are', vals)
            else:
                labels = s.demog_df[by_stage[0]].unique()
                vals = [np.where(s.demog_df[by_stage[0]].values == lbl)[0]
                        for lbl in labels]
                print('iterate across available vals including', vals)
        else:
            print('variable not found in data or demogs')
            raise
        dims = [dim] * len(vals)
        return dims, vals, labels