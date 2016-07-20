''' represents intermediate results from processing pipeline:
    does import, analyses, plotting '''

from glob import glob
import os
import itertools

import h5py
import dask.array as da

import numpy as np
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
from plot import MidpointNormalize

import mne

# values are tuples of h5py fieldname and datatype
opt_info = {'Options path':                 ('optpath', 'text'),
            'Data path':                    ('outpath', 'text'),
            'Batch ID':                     ('batch_id', 'text'),
            'Coordinates file':             ('coords_file', 'text'),

            'Condition labels':             ('case_label', 'cell'),
            'Measures available':           ('measures', 'cell'),
            'Coherence pair subsets':       ('pair_indlbls', 'cell'),

            'Sampling rate':                ('rate', 'array'),
            '# of timepoints':              ('n_samps', 'array'),
            'Temporal limits':              ('epoch_lims', 'array'),
            
            'TF scales':                    ('wavelet_scales', 'array'),
            'TF time-downsample ratio':     ('tf_timedownsamp_ratio', 'array'),
            
            'CSD matrix':                   ('csd_G', 'array'),
            'Coherence pairs':              ('coherence_pairs', 'array'),
            'Coherence pair subset index':  ('pair_inds', 'array'),
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
        if isinstance(v, int) or isinstance(v, np.int64):
            return a.take([v], d)
        else:
            return a.take(v, d)
    # print(a.shape)
    for v, d in zip(vals, dims):
        if isinstance(v, tuple): # reserved for itertools.product-takes
            for vp, dp in zip(v, d):
                if isinstance(vp, list):
                    if '-' in vp:
                        a = apply_take(a, vp[0], dp) - apply_take(a, vp[2], dp)
                        # print('level subtraction')
                else:
                    a = apply_take(a, vp, dp)
        elif isinstance(v, list): # reserved for operation-takes (subtractions)
            if '-' in v:
                a = apply_take(a, v[0], d) - apply_take(a, v[2], d)
                # print('level subtraction')
        else:
            a = apply_take(a, v, d)
        # print(a.shape)
    return a

def basic_slice(a, in_dimval_tups):
    ''' given array a and list of (dim, val) tuples, basic-slice '''
    slicer = [slice(None)]*len(a.shape) # initialize slicer
    
    # if the elements of the tuples are tuples, unpack them
    dimval_tups = []
    for dvt in in_dimval_tups:
        if isinstance(dvt[0], tuple):
            for d, v in zip(*dvt):
                dimval_tups.append((d, v))
        else:
            dimval_tups.append(dvt)
    
    # build the slice list
    dimval_tups.sort(reverse=True) # sort descending by dims
    for d, v in dimval_tups:
        try:
            v[1] # for non-singleton vals
            slicer[d] = v
        except: # for singleton vals
            slicer[d] = v
            slicer.insert(d+1, np.newaxis)
    
    # print(slicer)
    return a[tuple(slicer)]

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
    best_ind = np.argmin(ratios - 1.618)  # most golden
    if den_lst[best_ind] < num_lst[best_ind]: # always have more rows than cols
        return den_lst[best_ind], num_lst[best_ind]
    else:
        return num_lst[best_ind], den_lst[best_ind]

def figsize_heuristic(sp_dims):
    ''' given subplot dims (rows, columns), return (width, height) of figure '''
    max_width, max_height = 16, 10
    r = sp_dims[1] / sp_dims[0]
    phi = 1.618
    if r >= phi:
        out_width = max_width
        sp_width = out_width / sp_dims[1]
        sp_height = sp_width / phi
        out_height = sp_height * sp_dims[0]
    else:
        out_height = max_height
        sp_height = max_height / sp_dims[0]
        sp_width = sp_height * phi
        out_width = sp_width * sp_dims[1]
    return out_width, out_height


class Results:
    ''' represents HDF-compatible .mat's as dask stacks '''

    def __init__(s, optpath, csvpath=None, trial_thresh=15):
        s.opt = h5py.File(optpath, 'r')
        s.get_params()
        s.init_filedf()
        s.add_rejinfo()
        s.add_demogs(csvpath)
        s.apply_rej(trial_thresh)
        s.save_demogs()
        s.make_scales()

    def get_params(s):
        ''' extract data parameters from opt, relying upon opt_info. '''
        prefix = 'opt/'
        s.params = {param: handle_parse(s.opt, prefix + info[0], info[1])
                        for param, info in opt_info.items()}

    def init_filedf(s):
        ''' initialize dataframe of .mat files in the opt's "outpath"
            indexed by ID+session '''
        files = glob(s.params['Data path'] + '/*.mat')
        uid_fromfiles = [uid_frompath(fp) for fp in files]
        uid_index = pd.MultiIndex.from_tuples(
            uid_fromfiles, names=['ID', 'session'])
        s.file_df = pd.DataFrame({'path': pd.Series(files, index=uid_index)})

    def add_rejinfo(s):
        ''' for each file, get # of accepted trials and interpolated chans '''

        trials = s.load('trials', 'n_trials', [2, 0, 1], 'file_df')
        trial_df = pd.DataFrame(trials.squeeze(),
            columns=['trials_'+cond for cond in s.params['Condition labels']],
            index=s.file_df.index)
        s.file_df = s.file_df.join(trial_df)
        interpchans = s.load('interpchans', 'n_interpchans',
                                              [2, 0, 1], 'file_df')
        interpchan_df = pd.DataFrame(interpchans.squeeze(),
            columns=['# of interpolated channels'],
            index=s.file_df.index)
        s.file_df = s.file_df.join(interpchan_df)

    def add_demogs(s, csvpath):
        ''' read demographics file with ID/session columns, join to file_df '''

        if csvpath:
            demog_df = pd.read_csv(csvpath)
            demog_df['ID'] = demog_df['ID'].apply(str)
            uid_inds(demog_df) # ID and session as indices if they exist

            # note: this will exclude subs from demogs file with no data found
            s.demog_df = s.file_df.join(demog_df).sort_index()
        else:
            s.demog_df = s.file_df

    def apply_rej(s, trial_thresh, interpchan_thresh=12):
        ''' remove subs having too few trials or too many interp'd chans '''

        trial_cols = [col for col in s.demog_df.columns if 'trials_' in col]
        s.demog_df = s.demog_df[(s.demog_df[trial_cols] >= \
            trial_thresh).all(axis=1)]
        s.demog_df = s.demog_df[(s.demog_df['# of interpolated channels'] <= \
            interpchan_thresh)]

    def save_demogs(s):
        ''' save the current demog_df '''
        s._demog_bkup = s.demog_df.copy()

    def load_demogs(s):
        ''' load the most recently saved demog_df '''
        s.demog_df = s._demog_bkup.copy()

    def make_scales(s):
        ''' populate attributes describing channels and units '''
        
        # channels
        use_ext = '.sfp'
        path, file = os.path.split(s.params['Coordinates file'])
        fn, ext = file.split('.')
        s.montage = mne.channels.read_montage(os.path.join(path, fn) + use_ext)

        # units and suggested limits; CSD transform or not
        if s.params['CSD matrix'].shape[0] > 2:
            s.pot_lims, s.pot_units = [-.2, .2], r'$\mu$V / $cm^2$'
        else:
            s.pot_lims, s.pot_units = [-10, 16], r'$\mu$V'
        s.db_lims, s.db_units = [-3, 3], 'decibels (dB)'
        s.itc_lims, s.itc_units = [-.06, 0.3], 'ITC'
        s.coh_lims, s.coh_units = [-.06, 0.3], 'ISPS'
        s.phi_lims, s.phi_units = [-np.pi, np.pi], 'Radians'

        # ERP times
        s.srate = s.params['Sampling rate'][0][0]
        ep_lims = s.params['Temporal limits']
        n_timepts = s.params['# of timepoints']
        s.time = np.linspace(ep_lims[0], ep_lims[1], n_timepts + 1)[1:]
        s.zero = convert_ms(s.time, 0)

        # TF times / freqs
        n_timepts_tf = int(n_timepts / s.params['TF time-downsample ratio'])
        s.time_tf = np.linspace(ep_lims[0], ep_lims[1], n_timepts_tf + 1)[1:]
        s.zero_tf = convert_ms(s.time_tf, 0)
        s.freq = np.array([2 * s.srate / scale[0]
                           for scale in s.params['TF scales'][::-1]]) #reverse
        s.freq_ticks_pt = range(0, len(s.freq), 2)
        s.freq_ticks_hz = ['{0:.1f}'.format(f) for f in s.freq[::2]]

        s.time_ticks() # additional helper function for ticks

        # time plot limits
        time_plotlims_ms = [-200, 800]
        s.time_plotlims = [convert_ms(s.time, ms) for ms in time_plotlims_ms]
        s.time_tf_plotlims = [convert_ms(s.time_tf, ms)
                                        for ms in time_plotlims_ms]

        # coherence pairs
        if len(s.params['Coherence pairs'].shape) > 1:
            s.cohpair_lbls = ['~'.join([s.montage.ch_names[int(chan_num)-1]
                                for chan_num in pair])
                                    for pair in s.params['Coherence pairs'].T]

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

    def load(s, measure, dset_name, transpose_lst, df_attr='demog_df'):
        ''' given measure, h5 dataset name, transpose list: load data '''

        df = getattr(s, df_attr)

        if measure in dir(s):
            print(measure, 'already loaded')
            if df.shape[0] != getattr(s, measure).shape[0]:
                print('shape of loaded data does not match demogs, reloading')
            else:
                return

        dsets = [h5py.File(fn, 'r')[dset_name] for fn in df['path'].values]
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
        # erp is now (subjects, conditions, channels, timepoints,)

        # filter
        erp_filt = mne.filter.low_pass_filter(erp, s.params['Sampling rate'],
                                                   lp_cutoff)
        # baseline
        erp_filt_bl = baseline_amp(erp_filt, (convert_ms(s.time, bl_window[0]),
                                              convert_ms(s.time, bl_window[1])))

        s.erp = erp_filt_bl
        s.erp_dims = ('subject', 'condition', 'channel', 'timepoint')
        s.erp_dim_lsts = (s.demog_df.index.values,
                          np.array(s.params['Condition labels'], dtype=object),
                          np.array(s.montage.ch_names, dtype=object),
                          s.time)

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
                        np.array(s.params['Condition labels'], dtype=object),
                        np.array(s.montage.ch_names, dtype=object),
                        s.freq, s.time_tf)

    def load_itc(s, bl_window=[-500, -200]):
        ''' load phase data, take absolute() of, and subtractively baseline '''

        dsets = [h5py.File(fn, 'r')['wave_evknorm']
                 for fn in s.demog_df['path'].values]
        arrays = [dset.value.view(np.complex) for dset in dsets]
        stack = np.stack(arrays, axis=-1)  # concatenate along last axis
        print(stack.shape)
        stack = stack.transpose([4, 0, 2, 1, 3])  # subject dimension to front

        # itc is (subjects, conditions, channels, freq, timepoints,)
        stack = stack[:, :, :, ::-1, :] # reverse the freq dimension
        itc = np.absolute(stack)
        print(itc.shape)

        # baseline normalize
        itc_bl = baseline_amp(itc, (convert_ms(s.time_tf, bl_window[0]),
                                    convert_ms(s.time_tf, bl_window[1]),))

        s.itc = itc_bl
        s.tf_dims = ('subject', 'condition', 'channel',
                        'frequency', 'timepoint')
        s.tf_dim_lsts = (s.demog_df.index.values,
                        np.array(s.params['Condition labels'], dtype=object),
                        np.array(s.montage.ch_names, dtype=object),
                        s.freq, s.time_tf)

    def load_coh(s, bl_window=[-500, -200]):

        dsets = [h5py.File(fn, 'r')['coh']
                 for fn in s.demog_df['path'].values]
        arrays = [dset.value.view(np.complex) for dset in dsets]
        stack = np.stack(arrays, axis=-1)  # concatenate along last axis
        print(stack.shape)
        stack = stack.transpose([4, 1, 0, 2, 3])  # subject dimension to front

        # coh is (subjects, conditions, pairs, freq, timepoints,)
        stack = stack[:, :, :, ::-1, :] # reverse the freq dimension
        coh = np.absolute(stack)
        print(coh.shape)

        # baseline normalize
        coh_bl = baseline_amp(coh, (convert_ms(s.time_tf, bl_window[0]),
                                    convert_ms(s.time_tf, bl_window[1]),))

        s.coh = coh_bl
        s.coh_dims = ('subject', 'condition', 'pair', 'frequency', 'timepoint')
        s.coh_dim_lsts = (s.demog_df.index.values,
                        np.array(s.params['Condition labels'], dtype=object),
                        np.array(s.cohpair_lbls, dtype=object),
                        s.freq, s.time_tf)

    def load_phi(s):
        ''' load phase data, take angle() of '''

        dsets = [h5py.File(fn, 'r')['wave_evknorm']
                 for fn in s.demog_df['path'].values]
        arrays = [dset.value.view(np.complex) for dset in dsets]
        stack = np.stack(arrays, axis=-1)  # concatenate along last axis
        print(stack.shape)
        stack = stack.transpose([4, 0, 2, 1, 3])  # subject dimension to front

        # itc is (subjects, conditions, channels, freq, timepoints,)
        stack = stack[:, :, :, ::-1, :] # reverse the freq dimension
        phi = np.angle(stack)
        print(phi.shape)

        s.phi = phi
        s.tf_dims = ('subject', 'condition', 'channel',
                        'frequency', 'timepoint')
        s.tf_dim_lsts = (s.demog_df.index.values,
                        np.array(s.params['Condition labels'], dtype=object),
                        np.array(s.montage.ch_names, dtype=object),
                        s.freq, s.time_tf)


    def plot_erp(s, figure_by=('channel', ['FZ', 'CZ', 'PZ']),
                    subplot_by=('POP', None),
                    glyph_by=('condition', None),
                    savedir=None):
        ''' plot ERPs as lines '''
        ptype = 'line'
        measure = 'erp'

        if 'erp' not in dir(s):
            s.load_erp()
        data = s.erp
        d_dims = s.erp_dims
        d_dimlvls = s.erp_dim_lsts

        f_dim, f_vals, f_lbls = s.handle_by(figure_by, d_dims, d_dimlvls)
        sp_dim, sp_vals, sp_lbls = s.handle_by(subplot_by, d_dims, d_dimlvls)
        g_dim, g_vals, g_lbls = s.handle_by(glyph_by, d_dims, d_dimlvls)

        sp_dims = subplot_heuristic(len(sp_vals))
        figsize = figsize_heuristic(sp_dims)
        for fi, fval in enumerate(f_vals):
            f, axarr = plt.subplots(sp_dims[0], sp_dims[1],
                                    sharex=True, sharey=True, figsize=figsize)
            f.suptitle(f_lbls[fi])
            axarr = axarr.ravel()
            for spi, spval in enumerate(sp_vals):
                for gi, gval in enumerate(g_vals):
                    vals = [fval, spval, gval]
                    dims = [f_dim[fi], sp_dim[spi], g_dim[gi]]
                    try:
                        dimval_tups = [(d,v) for d,v in zip(dims, vals)]
                        line = basic_slice(data, dimval_tups)
                        print('slice')
                    except:
                        line = compound_take(data, vals, dims)
                        print('compound take')
                    while len(line.shape) > 1:
                        line = line.mean(axis=0)
                        err = ss.sem(line, axis=0)
                        # err = np.std(line, axis=0, ddof=1)
                        print(line.shape)
                    l, = axarr[spi].plot(np.arange(len(line)), line,
                                    label=g_lbls[gi])
                    axarr[spi].fill_between(np.arange(len(line)),
                                    line - err, line + err,
                                    alpha=0.5, linewidth=0,
                                    facecolor=l.get_color())
                axarr[spi].grid(True)
                axarr[spi].set_title(sp_lbls[spi])
                axarr[spi].legend(loc='upper left')
                axarr[spi].set_xticks(s.time_ticks_pt_erp)
                axarr[spi].set_xticklabels(s.time_ticks_ms)
                axarr[spi].set_xlabel('Time (s)')
                axarr[spi].set_ylabel('Potential (' + s.pot_units + ')')
                axarr[spi].axhline(0, color='k', linestyle='--')
                axarr[spi].axvline(s.zero, color='k', linestyle='--')
                axarr[spi].set_xlim(s.time_plotlims)
            if savedir:
                s.save_fig(savedir, ptype, measure, f_lbls[fi])


    def plot_topo(s, measure='erp', times=list(range(0, 501, 125)),
                     figure_by=('POP', ['C']),
                     row_by=('condition', None),
                     lims='absmax', cmap_override=None,
                     savedir=None):
        ''' plot data as topographic maps at specific timepoints '''
        ptype = 'topo'
        final_dim = 'channel'

        if measure in ['erp', 'power', 'itc', 'phi']:
            data, d_dims, d_dimlvls, units, lims, cmap = \
                s.get_plotparams(measure, lims, cmap_override)
        else:
            print('data not recognized')
            return

        info = mne.create_info(s.montage.ch_names, s.params['Sampling rate'],
                               'eeg', s.montage)

        f_dim, f_vals, f_lbls = s.handle_by(figure_by, d_dims, d_dimlvls)
        r_dim, r_vals, r_lbls = s.handle_by(row_by, d_dims, d_dimlvls)
        time_by = ('timepoint', times)
        t_dim, t_vals, t_lbls = s.handle_by(time_by, d_dims, d_dimlvls)

        sp_dims = (len(r_vals), len(times))
        figsize = figsize_heuristic(sp_dims)
        for fi, fval in enumerate(f_vals):
            f, axarr = plt.subplots(sp_dims[0], sp_dims[1],
                                    sharex=True, sharey=True, figsize=figsize)
            f.suptitle(f_lbls[fi])
            axarr = axarr.ravel()
            ax_dum = -1
            topo_lst = []
            im_lst = []
            for ri, rval in enumerate(r_vals):
                for ti, tval in enumerate(t_vals):
                    vals = [fval, rval, tval]
                    dims = [f_dim[fi], r_dim[ri], t_dim[ti]]
                    # try:
                    dimval_tups = [(d,v) for d,v in zip(dims, vals)]
                    try:
                        topo = basic_slice(data, dimval_tups)
                        print('slice')
                    except:
                        topo = compound_take(data, vals, dims)
                        print('compound take')
                    mean_dims = np.where([d!=final_dim for d in d_dims])
                    topo = topo.mean(axis=tuple(mean_dims[0]))
                    print(topo.shape)
                    ax_dum += 1
                    im, cn = mne.viz.plot_topomap(topo, info,
                                        cmap=cmap, axes=axarr[ax_dum],
                                        show=False)
                    topo_lst.append(topo)
                    im_lst.append(im)
                    ''' labels '''
                    axarr[ax_dum].text(-.5, .5, t_lbls[ti])  # time label
                    if ti == 0:
                        axarr[ax_dum].text(-1.5, 0, r_lbls[ri])  # row label
            
            norm = None
            topos = np.stack(topo_lst, axis=-1)
            if lims == 'absmax':
                vmax = np.max(np.fabs(topos))
                vmin = -vmax
            elif lims == 'minmax':
                vmax, vmin = np.max(topos), np.min(topos)
            else:
                vmax, vmin = lims[1], lims[0]
            
            if vmin != -vmax:
                norm = MidpointNormalize(vmin, vmax, 0)
            print('vmin', vmin, 'vmax', vmax)

            for im in im_lst:
                im.set_clim(vmin, vmax)
                if norm:
                    im.set_norm(norm)

            ''' colorbar '''
            plt.subplots_adjust(right=0.85)
            cbar_ax = f.add_axes([0.9, 0.15, 0.03, 0.75])
            cbar = plt.colorbar(im, cax=cbar_ax)
            cbar.ax.set_ylabel(units, rotation=270)
            if savedir:
                s.save_fig(savedir, ptype, measure, f_lbls[fi])


    def plot_tf(s, measure='power',
                   figure_by=[('POP', None), ('channel', ['FZ'])],
                   subplot_by=('condition', None),
                   lims='absmax', cmap_override=None,
                   savedir=None):
        ''' plot time-frequency data as a rectangular contour image '''
        ptype = 'tf'

        if measure in ['power', 'itc', 'phi', 'coh']:
            data, d_dims, d_dimlvls, units, lims, cmap = \
                s.get_plotparams(measure, lims, cmap_override)
        else:
            print('data not recognized')
            return

        f_dim, f_vals, f_lbls = s.handle_by(figure_by, d_dims, d_dimlvls)
        sp_dim, sp_vals, sp_lbls = s.handle_by(subplot_by, d_dims, d_dimlvls)

        sp_dims = subplot_heuristic(len(sp_vals))
        figsize = figsize_heuristic(sp_dims)
        for fi, fval in enumerate(f_vals):
            f, axarr = plt.subplots(sp_dims[0], sp_dims[1],
                                    sharex=True, sharey=True, figsize=figsize)
            f.suptitle(f_lbls[fi])
            axarr = axarr.ravel()
            rect_lst = []
            c_lst = []
            for spi, spval in enumerate(sp_vals):
                vals = [fval, spval]
                dims = [f_dim[fi], sp_dim[spi]]
                try:
                    dimval_tups = [(d,v) for d,v in zip(dims, vals)]
                    rect = basic_slice(data, dimval_tups)
                    print('slice')
                except:
                    rect = compound_take(data, vals, dims)
                    print('done compound take')
                while len(rect.shape) > 2:
                    rect = rect.mean(axis=0)
                    print(rect.shape)
                ''' contour '''
                c = axarr[spi].contourf(rect, 8, cmap=cmap)
                                        # vmin=lims[0], vmax=lims[1])
                c_lst.append(c)
                rect_lst.append(rect)
                # c = axarr[spi].contour(rect, cmap=plt.cm.RdBu_r,
                #                         vmin=-4, vmax=4)
                # plt.clabel(c, inline=1, fontsize=9)
                # cbar = plt.colorbar(c, ax=axarr[spi])
                # cbar.ax.set_ylabel(units, rotation=270)
                ''' ticks and grid '''
                axarr[spi].set_xticks(s.time_ticks_pt_tf)
                axarr[spi].set_xticklabels(s.time_ticks_ms)
                axarr[spi].set_yticks(s.freq_ticks_pt)
                axarr[spi].set_yticklabels(s.freq_ticks_hz)
                axarr[spi].grid(True)
                axarr[spi].axvline(s.zero_tf, color='k', linestyle='--')
                axarr[spi].set_xlim(s.time_tf_plotlims)
                ''' labels and title '''
                axarr[spi].set_xlabel('Time (s)')
                axarr[spi].set_ylabel('Frequency (Hz)')
                axarr[spi].set_title(sp_lbls[spi])

            norm = None
            rects = np.stack(rect_lst, axis=-1)
            if lims == 'absmax':
                vmax = np.max(np.fabs(rects))
                vmin = -vmax
            elif lims == 'minmax':
                vmax, vmin = np.max(rects), np.min(rects)
            else:
                vmax, vmin = lims[1], lims[0]
            
            if vmin != -vmax:
                norm = MidpointNormalize(vmin, vmax, 0)
            print('vmin', vmin, 'vmax', vmax)

            for c, spi in zip(c_lst, range(len(sp_vals))):
                c.set_clim(vmin, vmax)
                if norm:
                    c.set_norm(norm)
                cbar = plt.colorbar(c, ax=axarr[spi])
                cbar.ax.set_ylabel(units, rotation=270)

            ''' colorbar '''
            # plt.subplots_adjust(right=0.85)
            # cbar_ax = f.add_axes([0.88, 0.12, 0.03, 0.75])
            # cbar = plt.colorbar(c, cax=cbar_ax)
            # cbar.ax.set_ylabel(units, rotation=270)
            # plt.colorbar(c, ax=axarr[spi])
            if savedir:
                s.save_fig(savedir, ptype, measure, f_lbls[fi])


    # plot assistance
    ''' dictionary mapping measures to their object info '''
    measure_pps = {'erp':   {'data': 'erp', 'd_dims': 'erp_dims',
                             'd_dimlvls': 'erp_dim_lsts', 'units': 'pot_units',
                             'lims': 'pot_lims', 'cmap': plt.cm.RdBu_r,
                             'load': 'load_erp'},
                   'power': {'data': 'power', 'd_dims': 'tf_dims',
                             'd_dimlvls': 'tf_dim_lsts', 'units': 'db_units',
                             'lims': 'db_lims', 'cmap': plt.cm.RdBu_r,
                             'load': 'load_power'},
                   'itc':   {'data': 'itc', 'd_dims': 'tf_dims',
                             'd_dimlvls': 'tf_dim_lsts', 'units': 'itc_units',
                             'lims': 'itc_lims', 'cmap': plt.cm.Purples,
                             'load': 'load_itc'},
                   'coh':   {'data': 'coh', 'd_dims': 'coh_dims',
                             'd_dimlvls': 'coh_dim_lsts', 'units': 'coh_units',
                             'lims': 'coh_lims', 'cmap': plt.cm.Purples,
                             'load': 'load_coh'},
                   'phi':   {'data': 'phi', 'd_dims': 'tf_dims',
                             'd_dimlvls': 'tf_dim_lsts', 'units': 'phi_units',
                             'lims': 'phi_lims', 'cmap': plt.cm.Purples,
                             'load': 'load_phi'},
                   }
    
    def get_plotparams(s, measure, lims=None, cmap_override=None):
        ''' given a measure, retrieve data and get plotting info '''

        measure_d = s.measure_pps[measure]
        if measure_d['data'] not in dir(s):
            getattr(s, measure_d['load'])()

        data        = getattr(s, measure_d['data'])
        d_dims      = getattr(s, measure_d['d_dims'])
        d_dimlvls   = getattr(s, measure_d['d_dimlvls'])
        units       = getattr(s, measure_d['units'])

        if lims:
            if isinstance(lims, str):
                if lims not in ['absmax', 'minmax']:
                    print('lims incorrectly specified')
                    raise
            elif isinstance(lims, list):
                if len(lims) != 2:
                    print('lims incorrectly specified')
                    raise
            else:
                lims = [-lims, lims] # numeric, set symmetric
        else:
            lims = getattr(s, measure_d['lims'])

        if cmap_override:
            cmap = cmap_override
        else:
            cmap = measure_d['cmap']

        return data, d_dims, d_dimlvls, units, lims, cmap

    def save_fig(s, savedir, ptype, measure, label, form='svg'):
        ''' name and save the current figure to a target directory '''
        figname = s.gen_figname(ptype, measure, label)+'.'+form
        outpath = os.path.join(savedir, figname)
        plt.savefig(outpath, format=form, dpi=1000)

    def gen_figname(s, ptype, measure, label):
        ''' generate a figure name from the plot type, measure, and label '''
        return '_'.join([s.params['Batch ID'], ptype, measure, label])

    def handle_by(s, by_stage, d_dims, d_dimlvls):
        ''' handle a 'by' argument, which tells a plotting functions what parts
            of the data will be distributed across a plotting object.
            returns lists of the dimension, indices, and labels requested.
            if given a list, create above lists as products of requests '''
        if len(by_stage) > 1:
            # create list versions of the dim, vals, and labels
            tmp_dims, tmp_vals, tmp_labels = [], [], []
            for bs in by_stage.items():
                dims, vals, labels = s.interpret_by(bs, d_dims, d_dimlvls)
                tmp_dims.append(dims)
                tmp_vals.append(vals)
                tmp_labels.append(labels)
            all_dims = list(itertools.product(*tmp_dims))
            all_vals = list(itertools.product(*tmp_vals))
            all_labels = list(itertools.product(*tmp_labels))
            return all_dims, all_vals, all_labels
        else:
            return s.interpret_by(tuple(by_stage.items())[0], d_dims, d_dimlvls)

    def interpret_by(s, by_stage, data_dims, data_dimlvls):
        ''' by_stage: 2-tuple of variable name and requested levels
            data_dims: n-tuple describing the n dimensions of the data
            data_dimlvls: n-tuple of lists describing levels of each dim '''

        print('by stage is', by_stage)
        print('by stage is', by_stage[0])
        if by_stage[0] in data_dims:  # if variable is in data dims
            dim = data_dims.index(by_stage[0])
            print('data in dim', dim)
            if by_stage[1]:
                labels = by_stage[1]
                if data_dimlvls[dim].dtype == np.float64: # if array data
                    vals = []
                    for lbl in labels:
                        if isinstance(lbl, list):
                            if len(lbl) == 2:
                                tmp_inds = range(np.argmin(np.fabs(\
                                    data_dimlvls[dim] - lbl[0])),
                                                np.argmin(np.fabs(\
                                    data_dimlvls[dim] - lbl[1]))+1)
                            else:
                                tmp_inds = [np.argmin(np.fabs(\
                                    data_dimlvls[dim] - lp)) for lp in lbl]
                        else:
                            tmp_inds = np.argmin(np.fabs(\
                                data_dimlvls[dim] - lbl))
                        vals.append(tmp_inds)
                else:
                    vals = []
                    for lbl in labels:
                        if '-' in lbl:
                            vals.append([np.where(data_dimlvls[dim]==lbl[0])[0],
                                        '-',
                                        np.where(data_dimlvls[dim]==lbl[2])[0]])
                        else:
                            vals.append(np.where(data_dimlvls[dim]==lbl)[0])
                print('vals to iterate on are', vals)
            else:
                labels = data_dimlvls[dim]
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