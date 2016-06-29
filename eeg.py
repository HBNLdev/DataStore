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
opt_info = {'Coordinates file':             ('coords_file',         'text'),

            'Condition labels':             ('case_label',          'cell'),
            'Measures available':           ('measures',            'cell'),
            'Coherence pair subset labels': ('pair_indlbls',        'cell'),

            'Sampling rate':                ('rate',                'array'),
            '# of timepoints':              ('n_samps',             'array'),
            'Temporal limits':              ('epoch_lims',          'array'),
            'Frequency limits':             ('freq_lims',           'array'),
            'TF scales':                    ('wavelet_scales',      'array'),
            'TF time-downsample ratio':     ('tf_timedownsamp_ratio', 'array'),
            'CSD matrix':                   ('csd_G',               'array'),
            'Coherence pairs':              ('coherence_pairs',     'array'),
            'Coherence pair subset index':  ('pair_inds',           'array'),
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
    return array - array.take(range(pt_lims[0], pt_lims[1]+1), axis=along_dim)\
                        .mean(axis=along_dim, keepdims=True)

def baseline_tf(array, pt_lims, along_dim=-1):
    return 10 * np.log10( array / array.take(range(pt_lims[0], pt_lims[1]+1),
                                                axis=along_dim)\
                                        .mean(axis=along_dim, keepdims=True) )

def convert_ms(time_array, ms):
    return np.argmin(np.fabs(time_array - ms))

def compound_take(a, vals, dims):
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
        for x in range(2, int(np.sqrt(n))+1):
            if n%x==0:
                return False
        return True
    if n > 6 and isprime(n):
        n += 1
    num_lst, den_lst = [n], [1]
    for x in range(2, int(np.sqrt(n))+1):
        if n%x == 0:
            den_lst.append(x)
            num_lst.append(n//x)
    ratios = np.array([a/b for a,b in zip(num_lst, den_lst)])
    best_ind = np.argmin(ratios - 1.1618) # most golden
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
            s.params.update( {param:
                              handle_parse(s.opt, prefix + info[0], info[1])} )

    def make_scales(s):
        ''' populate attributes describing channels and units '''
        use_ext = '.sfp'
        path, file = os.path.split(s.params['Coordinates file'])
        fn, ext = file.split('.')
        s.montage = mne.channels.read_montage(os.path.join(path, fn) + use_ext)

        # Units
        if s.params['CSD matrix'].shape[0] > 2:
            s.pot_units = r'$\mu$V / $cm^2$'
        else:
            s.pot_units = r'$\mu$V'
        s.srate = s.params['Sampling rate'][0][0]

        # ERP
        ep_lims     = s.params['Temporal limits']
        n_timepts   = s.params['# of timepoints']
        s.time      = np.linspace(ep_lims[0], ep_lims[1], n_timepts + 1)[1:]

        # TF
        n_timepts_tf = int(n_timepts / s.params['TF time-downsample ratio'])
        s.time_tf = np.linspace(ep_lims[0], ep_lims[1], n_timepts_tf + 1)[1:]
        s.freq = np.array([2*s.srate/scale[0]
                            for scale in s.params['TF scales']])
        s.freq_ticks_pt = range(0, len(s.freq), 2)
        s.freq_ticks_hz = ['{0:.1f}'.format(f) for f in reversed(s.freq[1::2])]

        s.time_ticks()


    def time_ticks(s, interval=200):
        ''' time ticks for plots '''
        ms_start_plot = np.round(s.time[0]/100)*100 # start at a round number
        first_tick = int(ms_start_plot + ms_start_plot % interval)
        time_ticks_ms = list(range(first_tick, int(s.time[-1]), interval))

        s.time_ticks_pt_erp = [convert_ms(s.time, ms) for ms in time_ticks_ms]
        s.time_ticks_pt_tf = [convert_ms(s.time_tf, ms) for ms in time_ticks_ms]
        
        s.time_ticks_ms = [n/1000 for n in time_ticks_ms] # for display purposes


    def load_erp(s, lp_cutoff=16, bl_window=(-100, 0)):
        ''' load, filter, and subtractively baseline ERP data '''
        dsets = [h5py.File(fn)['erp'] for fn in s.file_df['path'].values]
        arrays = [da.from_array(dset, chunks=dset.shape) for dset in dsets]
        stack = da.stack(arrays, axis=-1)  # concatenate along last axis
        stack = stack.transpose([3, 0, 1, 2]) # move subject dimension to front
        
        erp = np.empty(stack.shape)
        da.store(stack, erp)
        # erp is (subjects, conditions, channels, timepoints,)
        print(erp.shape)

        # filter
        erp_filt = mne.filter.low_pass_filter(erp, 
                        s.params['Sampling rate'], lp_cutoff)
        # baseline
        erp_filt_bl = baseline_amp(erp_filt, (convert_ms(s.time, bl_window[0]),
                                              convert_ms(s.time, bl_window[1])))

        s.erp = erp_filt_bl
        s.erp_dims = ('subject', 'condition', 'channel', 'timepoint')
        s.erp_dim_lsts = (s.file_df.index.values, s.params['Condition labels'],
            s.montage.ch_names, s.time)

    def load_power(s, bl_window=(-500, -200)):
        ''' load and divisively baseline-normalize total power data '''
        dsets = [h5py.File(fn)['wave_totpow']
                    for fn in s.file_df['path'].values]
        arrays = [da.from_array(dset, chunks=dset.shape) for dset in dsets]
        stack = da.stack(arrays, axis=-1)  # concatenate along last axis
        print(stack.shape)
        stack = stack.transpose([4, 0, 2, 1, 3]) # subject dimension to front

        power = np.empty(stack.shape)
        da.store(stack, power)
        # power is (subjects, conditions, channels, freq, timepoints,)
        power = power[:,:,:,::-1,:] # reverse the freq dimension
        print(power.shape)

        # baseline normalize
        power_bl = baseline_tf(power, (convert_ms(s.time_tf, bl_window[0]),
                                       convert_ms(s.time_tf, bl_window[1]),))

        s.power = power_bl
        s.power_dims = ('subject', 'condition', 'channel',
                            'timepoint', 'frequency')
        s.power_dim_lsts = (s.file_df.index.values,
            s.params['Condition labels'], s.montage.ch_names, s.time, s.freq)

    def prepare_mne(s):
        ''' prepare an mne EvokedArray object from erp data '''
        if 'erp' not in dir(s):
            s.load_erp()

        # create info
        info = mne.create_info(s.montage.ch_names, s.params['Sampling rate'],
                                    'eeg', s.montage)
        # EvokedArray
        chan_erps = s.erp.mean(axis=(0,1))/1000000 # subject/condition mean
        return mne.EvokedArray(chan_erps, info,
                    tmin=s.params['Temporal limits'][0]/1000)

    def plot_erp(s, figure_by=('channel', ['FZ', 'CZ', 'PZ']),
                    subplot_by=('group', None),
                    glyph_by=('condition', None) ):

        if 'erp' not in dir(s):
            s.load_erp()

        d_dims = s.erp_dims
        d_dimlst = s.erp_dim_lsts

        f_dim, f_vals, f_lbls = s.interpret_by(figure_by, d_dims, d_dimlst)
        sp_dim, sp_vals, sp_lbls = s.interpret_by(subplot_by, d_dims, d_dimlst)
        g_dim, g_vals, g_lbls = s.interpret_by(glyph_by, d_dims, d_dimlst)

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
                axarr[spi].legend()
                axarr[spi].set_xticks(s.time_ticks_pt_erp)
                axarr[spi].set_xticklabels(s.time_ticks_ms)
                axarr[spi].set_xlabel('Time (s)')
                axarr[spi].set_ylabel('Potential ('+s.pot_units+')')


    def plot_ersp(s, figure_by=[('POP', None),
                                ('channel', ['FZ'])],
                     subplot_by=('condition', None)):

        if 'power' not in dir(s):
            s.load_power()

        d_dims = s.power_dims
        d_dimlst = s.power_dim_lsts

        f_dim, f_vals, f_lbls = s.interpret_by(figure_by, d_dims, d_dimlst)
        sp_dim, sp_vals, sp_lbls = s.interpret_by(subplot_by, d_dims, d_dimlst)

        sp_dims = subplot_heuristic(len(sp_vals))
        for fi, fval in enumerate(f_vals):
            f, axarr = plt.subplots(sp_dims[0], sp_dims[1],
                sharex=True, sharey=True, figsize=(12, 5))
            f.suptitle(f_lbls[fi])
            axarr = axarr.ravel()
            for spi, spval in enumerate(sp_vals):
                rect = compound_take(s.power, [fval, spval],
                        [f_dim[fi], sp_dim[spi]])
                print(rect.shape)
                while len(rect.shape) > 2:
                    rect = rect.mean(axis=0)
                    print(rect.shape)
                ''' contour '''
                c = axarr[spi].contourf(rect, 10, cmap=plt.cm.RdBu_r,
                                        vmin=-4, vmax=4)
                # c = axarr[spi].contour(rect, cmap=plt.cm.RdBu_r,
                #                         vmin=-4, vmax=4)
                # plt.clabel(c, inline=1, fontsize=9)
                ''' colorbar '''
                plt.subplots_adjust(right=0.85)
                cbar_ax = f.add_axes([0.88, 0.12, 0.03, 0.75])
                cbar = plt.colorbar(c, cax=cbar_ax)
                cbar.ax.set_ylabel('dB', rotation=270)
                # plt.colorbar(c, ax=axarr[spi])
                ''' ticks and grid '''
                axarr[spi].set_xticks(s.time_ticks_pt_tf)
                axarr[spi].set_xticklabels(s.time_ticks_ms)
                axarr[spi].set_yticks(s.freq_ticks_pt)
                axarr[spi].set_yticklabels(s.freq_ticks_hz)
                # axarr[spi].set_zlabel('Potential ('+s.pot_units+')')
                axarr[spi].grid(True)
                ''' labels and title '''
                axarr[spi].set_xlabel('Time (s)')
                axarr[spi].set_ylabel('Frequency (Hz)')
                axarr[spi].set_title(sp_lbls[spi])

    # plot helpers
    def interpret_by(s, by_stage, d_dims, d_dimlst):
        ''' interpret a 'by' stage, which tells plotting functions which parts
            of the data will be distributed across that plotting object '''

        def interpret_one(s, by_stage, data_dims, data_dimlst):
            ''' sub-function '''
            print('by stage is', by_stage[0])
            if by_stage[0] in data_dims: # if variable is in data dims
                dim = data_dims.index(by_stage[0])
                print('data in dim', dim)
                if by_stage[1]:
                    labels = by_stage[1]
                    vals = [data_dimlst[dim].index(lbl) for lbl in labels]
                    print('vals to iterate on are', vals)
                else:
                    labels = data_dimlst[dim]
                    vals = list(range(len(labels)))
                    print('iterate across available vals including', vals)
            elif by_stage[0] in s.demog_df.columns: # if variable in demog dims
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
            dims = [dim]*len(vals)
            return dims, vals, labels

        ''' main function '''
        if isinstance(by_stage, list):
            # create list versions of the dim, vals, and labels
            tmp_dims, tmp_vals, tmp_labels = [], [], []
            for bs in by_stage:
                dims, vals, labels = interpret_one(s, bs, d_dims, d_dimlst)
                tmp_dims.append(dims)
                tmp_vals.append(vals)
                tmp_labels.append(labels)
            all_dims = list(itertools.product(*tmp_dims))
            all_vals = list(itertools.product(*tmp_vals))
            all_labels = list(itertools.product(*tmp_labels))
            return all_dims, all_vals, all_labels
        else:
            return interpret_one(s, by_stage, d_dims, d_dimlst)

class ERP:

    def __init__(s, results_obj):
        s.data = results_obj.erp
        s.dims = results_obj.erp_dims
        s.montage = results_obj

    def plot_erp(s, channels=['FZ', 'CZ', 'PZ'],
            line_by='group', subplot_by='condition', figure_by='channel'):
        pass