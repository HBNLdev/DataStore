''' represents intermediate results from MATLAB processing pipeline '''

import os
from glob import glob
from datetime import datetime

import h5py
import dask.array as da
import numpy as np
import pandas as pd
import mne

from array_utils import (convert_ms, baseline_sub, baseline_div, handle_by,
                             basic_slice, compound_take)
from plot import measure_pps, get_plotparams
from plot_utils import nested_strjoin

import compilation as C

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

ftypes_funcs = {'text': parse_text, 'cell': parse_cell, 'array': None}

def handle_parse(dset, dset_field, field_type):
    ''' given file pointer, field, and datatype, apply appropriate parser '''
    func = ftypes_funcs[field_type]
    return func(dset, dset_field) if func else dset[dset_field][:]


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

    def retrieve_behavior(s, experiment):
        ''' given 3-letter experiment designation, retrieve behavioral data '''
        wanted_fields = ['ID', 'session', experiment]
        proj = {wf:1 for wf in wanted_fields}
        proj.update({'_id':0})

        s.demog_df = C.join_collection(s.demog_df, 'EEGbehavior', add_proj=proj,
            left_join_inds=['ID', 'session'], right_join_inds=['ID', 'session'],
            prefix='')

    def export_demogs(s, savedir='~'):
        ''' export the current demog_df as a csv '''
        batch_id = s.params['Batch ID']
        today = datetime.now().strftime('%m-%d-%Y')
        now = datetime.now().strftime('%H%M')

        outname = '_'.join([batch_id, today, now])+'.csv'
        outpath = os.path.expanduser(os.path.join(savedir, outname))

        print(outpath)
        s.demog_df.to_csv(outpath)

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
        s.cohpair_inds = [ [int(chan_num)-1 for chan_num in pair] # MATLAB index
                                for pair in s.params['Coherence pairs'].T ]
        if len(s.params['Coherence pairs'].shape) > 1:
            s.cohpair_lbls = ['~'.join([s.montage.ch_names[chan_ind]
                                        for chan_ind in pair])
                                            for pair in s.cohpair_inds]
            s.cohpair_sets = {}
            s.cohchan_sets = {}
            for pind, pset in enumerate(s.params['Coherence pair subsets']):
                tmp_pairs = np.where(
                        s.params['Coherence pair subset index']==pind+1)[1]
                setpairs = [s.cohpair_lbls[pair] for pair in tmp_pairs]
                s.cohpair_sets[pset] = setpairs
                setchans = np.unique(np.array([s.cohpair_inds[p]
                                                    for p in tmp_pairs]))
                s.cohchan_sets[pset] = [s.montage.ch_names[i] for i in setchans]

    def time_ticks(s, interval=200):
        ''' time ticks for plots '''
        ms_start_plot = np.round(s.time[0] / 100) * 100  # start at round number
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
                return 'loaded already'

        dsets = [h5py.File(fn, 'r')[dset_name] for fn in df['path'].values]
        arrays = [da.from_array(dset, chunks=dset.shape) for dset in dsets]
        stack = da.stack(arrays, axis=-1)  # concatenate along last axis
        stack = stack.transpose(transpose_lst) # do transposition

        data = np.empty(stack.shape)
        da.store(stack, data)
        print(data.shape)
        return data

    def load_erp(s, filt=True, lp_cutoff=16,
                    baseline=True, bl_window=[-100, 0]):
        ''' load, filter, and subtractively baseline ERP data '''

        erp = s.load('erp', 'erp', [3, 0, 1, 2])
        if erp == 'loaded already':
            return
        # erp is now (subjects, conditions, channels, timepoints,)

        # filter
        if filt:
            erp_filt = mne.filter.low_pass_filter(erp,
                s.params['Sampling rate'], lp_cutoff)

        s.erp = erp_filt
        s.erp_dims = ('subject', 'condition', 'channel', 'timepoint')
        s.erp_dim_lsts = (s.demog_df.index.values,
                          np.array(s.params['Condition labels'], dtype=object),
                          np.array(s.montage.ch_names, dtype=object),
                          s.time)

        # baseline
        if baseline:
            s.baseline('erp', 'subtractive', bl_window)

    def baseline(s, measure, method, window):

        if measure not in measure_pps:
            print('measure incorrectly specified'); raise

        if method not in ['subtractive', 'divisive']:
            print('method incorrectly specified'); raise

        data = getattr(s, measure_pps[measure]['data'])
        pt_lims = (convert_ms(s.time, window[0]), convert_ms(s.time, window[1]))

        print('baselining {} from {} to {} ms using {} method'.format(
                    measure, window[0], window[1], method))

        if method == 'subtractive':
            setattr(s, measure_pps[measure]['data'],
                baseline_sub(data, pt_lims))
        elif method == 'divisive':
            setattr(s, measure_pps[measure]['data'],
                baseline_div(data, pt_lims))        
        
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

    def load_power(s, baseline=True, bl_window=[-500, -200]):
        ''' load (total) power data and divisively baseline-normalize '''

        power = s.load('power', 'wave_totpow', [4, 0, 2, 1, 3])
        if power == 'loaded already':
            return
        power = power[:, :, :, ::-1, :]  # reverse the freq dimension
        # power is (subjects, conditions, channels, freq, timepoints,)

        s.power = power
        s.tf_dims = ('subject', 'condition', 'channel',
                        'frequency', 'timepoint')
        s.tf_dim_lsts = (s.demog_df.index.values,
                        np.array(s.params['Condition labels'], dtype=object),
                        np.array(s.montage.ch_names, dtype=object),
                        s.freq, s.time_tf)

        # divisively baseline normalize
        if baseline:
            s.baseline('power', 'divisive', bl_window)

    def load_itc(s, baseline=True, bl_window=[-500, -200]):
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

        s.itc = itc
        s.tf_dims = ('subject', 'condition', 'channel',
                        'frequency', 'timepoint')
        s.tf_dim_lsts = (s.demog_df.index.values,
                        np.array(s.params['Condition labels'], dtype=object),
                        np.array(s.montage.ch_names, dtype=object),
                        s.freq, s.time_tf)

        # divisively baseline normalize
        if baseline:
            s.baseline('itc', 'subtractive', bl_window)

    def load_coh(s, baseline=True, bl_window=[-500, -200]):

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

        s.coh = coh
        s.coh_dims = ('subject', 'condition', 'pair', 'frequency', 'timepoint')
        s.coh_dim_lsts = (s.demog_df.index.values,
                        np.array(s.params['Condition labels'], dtype=object),
                        np.array(s.cohpair_lbls, dtype=object),
                        s.freq, s.time_tf)

        # divisively baseline normalize
        if baseline:
            s.baseline('coh', 'subtractive', bl_window)

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


    def save_mean(s, measure, spec_dict):
        ''' save data-means and add as columns in s.demog_df '''
        final_dim = 'subject'

        if measure in measure_pps.keys():
            data, d_dims, d_dimlvls, units, lims, cmap = \
                get_plotparams(s, measure)

        # TODO: verify spec_dict accords with data_dims
        
        dims, vals, lbls = handle_by(s, spec_dict, d_dims, d_dimlvls)

        amean_lst = []
        for dim_set, val_set, lbl_set in zip(dims, vals, lbls):
            dimval_tups = [(d, v) for d, v in zip(dim_set, val_set)]
            try:
                amean = basic_slice(data, dimval_tups)
                print('done slice')
            except:
                amean = compound_take(data, dimval_tups)
                print('done compound take')
            
            mean_dims = np.where([d!=final_dim for d in d_dims])
            amean = amean.mean(axis=tuple(mean_dims[0]))
            
            amean_lst.append(amean)

        amean_stack = np.stack(amean_lst, axis=-1)
        lbl_lst = [measure+'_'+nested_strjoin(lbl_set) for lbl_set in lbls]

        amean_df = pd.DataFrame(amean_stack,
                        index=s.demog_df.index, columns=lbl_lst)

        s.demog_df = pd.concat([s.demog_df, amean_df], axis=1)