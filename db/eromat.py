''' representing and working with .mat's containing 3D ERO data '''

import os

import h5py
import numpy as np
import pandas as pd
from scipy import interpolate
from tqdm import tqdm

import db.database as D
from .cnth1_to_stmat import build_paramstr
from .compilation import join_collection
from .stmat_to_eromat import version_info, chan_mapping, powertype_mapping
from .utils.compilation import join_allcols
from .utils.math import convert_scale
from .utils.matlab_h5 import handle_parse

opt_info = {'Add baseline': ('add_baseline', 'array'),
            'Calculation type': ('calc_type', 'array'),
            'Condition': ('case_name', 'text'),
            'Channel sort': ('channel_sort', 'array'),
            'Do baseline': ('do_baseline', 'array'),
            'Channels': ('elec_array', 'text_array'),
            'Experiment': ('exp_name', 'text'),
            # 'Original frequencies':   ('f_vec',           'array'),
            'Frequencies': ('f_vec_ds', 'array'),
            'File ID': ('file_id', 'text'),
            # 'File name':              ('file_name',       'text'),
            'Mat file': ('filenm', 'text'),
            'Run': ('file_run', 'array'),
            'Session': ('file_session', 'text'),
            # 'File index': ('i_file', 'array'),
            'Natural log': ('ln_calc', 'array'),
            '# of channels': ('n_chans_present', 'array'),
            'Output type': ('out_type', 'array'),
            'Output type name': ('out_type_name', 'text'),
            # 'Output text':            ('output_text',     'text'),
            'Sampling rate': ('rate', 'array'),
            'S-transform type': ('st_type', 'array'),
            'S-transform type name': ('st_type_name', 'text'),
            'Time downsample ratio': ('time_ds_factor', 'array'),
            # 'Original times':         ('time_vec',        'array'),
            'Times': ('time_vec_ds', 'array'),
            }

chans_61 = ['FP1', 'FP2', 'F7', 'F8', 'AF1', 'AF2', 'FZ',
            'F4', 'F3', 'FC6', 'FC5', 'FC2', 'FC1', 'T8', 'T7', 'CZ',
            'C3', 'C4', 'CP5', 'CP6', 'CP1', 'CP2', 'P3', 'P4', 'PZ',
            'P8', 'P7', 'PO2', 'PO1', 'O2', 'O1', 'X', 'AF7', 'AF8',
            'F5', 'F6', 'FT7', 'FT8', 'FPZ', 'FC4', 'FC3', 'C6', 'C5',
            'F2', 'F1', 'TP8', 'TP7', 'AFZ', 'CP3', 'CP4', 'P5',
            'P6', 'C1', 'C2', 'PO7', 'PO8', 'FCZ', 'POZ', 'OZ',
            'P2', 'P1', 'CPZ']

chans_19 = ['O1', 'F4', 'F8', 'P3', 'C4', 'P7', 'T8', 'T7', 'P8', 'C3',
            'P4', 'F7', 'F3', 'O2', 'FP1', 'FZ', 'FP2', 'CZ', 'PZ', ]

chans_31 = ['FP1', 'FP2', 'F7', 'F8', 'AF1', 'AF2', 'FZ', 'F4', 'F3', 'FC6',
            'FC5', 'FC2', 'FC1', 'T8', 'T7', 'CZ', 'C3', 'C4', 'CP5', 'CP6',
            'CP1', 'CP2', 'P3', 'P4', 'PZ', 'P8', 'P7', 'PO2', 'PO1', 'O2', 'O1']


# prep functions

def prepare_erodf(ero_df):
    ''' given an ero_df prepared by EROStack.tf_mean_multiwin_chans,
        that is row-index by ID, session, powertype, experiment, and condition,
        and column-indexed by TFROI and channel,
        reorganize the dataframe to be in an export-friendly format '''

    ero_df2 = ero_df.unstack(['powertype', 'experiment', 'condition'])
    ero_df2.columns = ero_df2.columns.reorder_levels(['powertype', 'experiment', 'condition', 'TFROI', 'channel'])
    ero_df3 = ero_df2.dropna(axis=1, how='all')
    ero_df3.sortlevel(0, axis=1, inplace=True)
    collapsed_cols = pd.Series(ero_df3.columns.tolist()).apply(pd.Series).apply(join_allcols, axis=1, args=['_'])
    ero_df3.columns = collapsed_cols
    return ero_df3


def add_eropaths(df, proc_type, exp_cases, power_types=['total', 'evoked'], n_chans=['64', '32', '21']):
    ''' given a dataframe indexed by ID and session, a desired ERO processing type,
        a dict mapping desired experiments to desired cases within each,
        and a list of desired power types,
        return a dataframe with added columns that locate the corresponding paths to 3d ero mats '''

    uIDs = [ID + '_' + session for ID, session in df.index.tolist()]
    query = {'uID': {'$in': uIDs}, 'experiment': {'$in': list(exp_cases.keys())}}
    proj = {'_id': 0, 'ID': 1, 'session': 1, 'n_chans': 1}
    nchans_df = join_collection(df, 'cnth1s',
                                add_query=query, add_proj=proj,
                                left_join_inds=['ID', 'session'], right_join_inds=['ID', 'session'])

    nchans_df.dropna(subset=['cnt_n_chans'], inplace=True)
    groups = nchans_df.groupby(level=nchans_df.index.names)
    nchans_df_nodupes = groups.last()
    nchans_df_nodupes_selectchans = nchans_df_nodupes[nchans_df_nodupes['cnt_n_chans'].isin(n_chans)]
    df_out = df.join(nchans_df_nodupes_selectchans['cnt_n_chans'])

    prc_ver = proc_type[1]
    parent_dir = version_info[prc_ver]['storage path']

    df_out = df_out[df_out['cnt_n_chans'].notnull()]

    for exp, cases in exp_cases.items():
        for case in cases:
            for ptype in power_types:
                ptype_short = powertype_mapping[ptype]
                apply_args = [parent_dir, proc_type, exp, case, ptype_short]
                pathcol_name = '_'.join([proc_type, exp, case, ptype])
                df_out[pathcol_name] = df_out.apply(gen_path, axis=1, args=apply_args)

    return df_out


def gen_path(rec, parent_dir, proc_type, exp, case, power_type_short):
    ''' apply function designed to operate on a dataframe indexed by ID and session.
        given processing version, parameter string, number of channels in the raw data, experiment,
        case, power type, ID, and session, generate the path to the expected 3d ero mat '''

    ID = rec.name[0]
    session = rec.name[1]
    raw_chans = rec['cnt_n_chans']
    try:
        param_str = build_paramstr(proc_type, raw_chans, exp)
    except KeyError:
        return np.nan  # also consider returning a default value here
    if 'center9' in proc_type:
        n_chans = '20'
    else:
        n_chans = chan_mapping[raw_chans]

    path_start = os.path.join(parent_dir, param_str, n_chans, exp)
    fname = '_'.join([ID, session, exp, case, power_type_short])
    ext = '.mat'

    path = os.path.join(path_start, fname + ext)

    return path


# math function for interpolation

def interp_freqdomain(a, t, f1, f2):
    ''' given time-frequency array a that is of shape (t.size, f1.size),
        timepoint vector t, and two frequency vectors, f1 and f2,
        use 2d interpolation to produce output array that is of size (t.size, f2.size) '''
    f = interpolate.interp2d(f1, t, a)
    return f(f2, t)


def interp_freqdomain_fast(a, t, f1, f2):
    ''' faster version of above that uses a cubic spline procedure '''
    f = interpolate.RectBivariateSpline(t, f1, a)
    return f(t, f2)


# main classes

class EmptyStackError(Exception):
    def __init__(s):
        print('all files in the stack were missing')


class EROStack:
    ''' represents a list of .mat's as a stack of arrays '''

    def __init__(s, path_lst, touch_db=False):
        s.init_df(path_lst, touch_db=touch_db)
        s.get_params()
        s.survey_vecs()

    def init_df(s, path_lst, touch_db=False):
        ''' given, a list of paths, intializes the dataframe that represents each existent EROmat.
            pulls info out of the path of each. '''

        row_lst = []
        pointer_lst = []

        missing_count = 0
        for fp in path_lst:
            if os.path.exists(fp):
                em = EROMat(fp)

                if touch_db:
                    em.prepare_row()

                try:
                    pointer_lst.append(h5py.File(fp, 'r'))
                    row_lst.append(em.info)
                except OSError:
                    missing_count += 1
            else:
                missing_count += 1

        if row_lst:
            print(missing_count, 'files missing')
        else:
            raise EmptyStackError

        s.data_df = pd.DataFrame.from_records(row_lst)
        s.data_df.set_index(['ID', 'session', 'powertype', 'experiment', 'condition'], inplace=True)
        s.data_df.sort_index(inplace=True)
        s.pointer_lst = pointer_lst

    def get_params(s):
        ''' get the ERO file parameters, using the first file as an assumed exemplar '''

        em = EROMat(s.data_df.ix[0, 'path'])
        # em = EROMat(s.data_df['path'][0])
        em.get_params()
        s.params = em.params

    def survey_vecs(s):
        ''' survey all .mat's for their frequency and channel vectors,
            and create data structures to keep track of them for later steps '''

        # freqs /chans
        freqlen_lst = []
        chanlen_lst = []
        freqlen_dict = {}
        chanlen_dict = {}
        for fp in s.pointer_lst:
            f_vec = handle_parse(fp, 'opt/f_vec_ds', 'array')
            chans = handle_parse(fp, 'opt/elec_array', 'text_array')
            freqlen_lst.append(f_vec.shape[1])
            chanlen_lst.append(len(chans))
            if f_vec.shape[1] not in freqlen_dict:
                freqlen_dict[f_vec.shape[1]] = f_vec
            if len(chans) not in chanlen_dict:
                chanlen_dict[len(chans)] = chans

        s.data_df['freq_len'] = pd.Series(freqlen_lst, index=s.data_df.index)
        s.data_df['chan_len'] = pd.Series(chanlen_lst, index=s.data_df.index)
        s.freqlen_dict = freqlen_dict
        s.chanlen_dict = chanlen_dict
        s.params['Frequencies'] = freqlen_dict[max(freqlen_dict)]
        s.params['Channels'] = chanlen_dict[max(chanlen_dict)]

    def survey_conds(s):
        ''' survey what conditions are available among all ERO .mat's
            so that later it's possible to build a results package-style data array,
            in which conditions is its own dimension '''

        cond_lst = list(set(s.data_df.index.get_level_values('condition')))
        n_conds = len(cond_lst)
        g = s.data_df.reset_index('condition').groupby(level=['ID', 'session'])
        cond_counts = g['condition'].count()
        cond_deficient_uIDs = cond_counts.index[(cond_counts < n_conds).values]
        data_df_fullconds = s.data_df.drop(cond_deficient_uIDs)
        data_df_fullconds_uID = data_df_fullconds.reset_index().set_index(['ID', 'session'])
        fullcond_uIDs = list(set(data_df_fullconds_uID.index.values))
        data_df_fullconds_uIDconds = data_df_fullconds_uID.set_index('condition', append=True)
        g = data_df_fullconds_uID.groupby(level=data_df_fullconds_uID.index.names)
        data_df_fullconds_uID_nodupes = g.first()

        s.cond_lst = cond_lst
        s.fullcond_uIDs = fullcond_uIDs
        s.data_df_fullconds = data_df_fullconds
        s.data_df_fullconds_uIDconds = data_df_fullconds_uIDconds
        s.data_df_fullconds_uID_nodupes = data_df_fullconds_uID_nodupes

    # all of the below functions use a low memory style which, rather than stacking all of the 3d ERO arrays in memory,
    # takes the mean of each and removes the array from memory before continuing

    def tfmean(s, times, freqs):
        ''' given 2-tuples of start and end times and frequencies,
            return a dataframe containing the mean value in that TFROI
            for each .mat in the stack '''

        time_lims = [convert_scale(s.params['Times'], time) for time in times]
        freq_lims = dict()
        for freq_len, freq_array in s.freqlen_dict.items():
            freq_lims[freq_len] = [convert_scale(freq_array, freq) for freq in freqs]
        print(time_lims)
        print(freq_lims)

        n_chans = int(s.params['# of channels'][0][0])
        n_files = len(s.data_df['path'].values)
        mean_array = np.empty((n_files, n_chans))

        fpi = 0
        for fp, fl in tqdm(zip(s.pointer_lst, s.data_df['freq_len'].values)):
            mean_array[fpi, :] = fp['data'][:, time_lims[0]:time_lims[1] + 1,
                                 freq_lims[fl][0]:freq_lims[fl][1] + 1].mean(axis=(1, 2))
            fpi += 1

        time_lbl = '-'.join([str(t) for t in times]) + 'ms'
        freq_lbl = '-'.join([str(f) for f in freqs]) + 'Hz'
        lbls = ['_'.join([chan, time_lbl, freq_lbl])
                for chan in s.params['Channels']]

        mean_df = pd.DataFrame(mean_array, index=s.data_df.index, columns=lbls)
        return mean_df

    def tfmean_multiwin(s, windows):
        ''' given a list of tf windows, which are 4-tuples of (t1, t2, f1, f2),
            calculate means in those windows for all channels and subjects '''

        win_inds = []
        win_lbls = []
        for win in windows:
            winds = dict()
            for freq_len, freq_array in s.freqlen_dict.items():
                winds[freq_len] = (convert_scale(s.params['Times'], win[0]),
                                   convert_scale(s.params['Times'], win[1]),
                                   convert_scale(freq_array, win[2]),
                                   convert_scale(freq_array, win[3]))
            time_lbl = '-'.join([str(t) for t in win[0:2]]) + 'ms'
            freq_lbl = '-'.join([str(f) for f in win[2:4]]) + 'Hz'
            lbls = ['_'.join([chan, time_lbl, freq_lbl])
                    for chan in s.params['Channels']]
            print(winds)
            win_inds.append(winds)
            win_lbls.extend(lbls)

        n_files = len(s.data_df['path'].values)
        n_windows = len(windows)
        n_chans = int(s.params['# of channels'][0][0])
        mean_array = np.empty((n_files, n_windows, n_chans))

        fpi = 0
        for fp, fl in tqdm(zip(s.pointer_lst, s.data_df['freq_len'].values)):
            for wi, winds in enumerate(win_inds):
                mean_array[fpi, wi, :] = fp['data'][:, winds[fl][0]:winds[fl][1] + 1,
                                         winds[fl][2]:winds[fl][3] + 1].mean(axis=(1, 2))
            fpi += 1

        mean_array_reshaped = np.reshape(mean_array, (n_files, n_windows * n_chans))

        mean_df = pd.DataFrame(mean_array_reshaped, index=s.data_df.index, columns=win_lbls)
        return mean_df

    def tfmean_multiwin_chans(s, windows, channels):
        ''' same as tfmean_multiwin but only compiles for a subset of channels,
            supplied by name as a list '''

        win_inds = []
        win_lbls = []
        for win in windows:
            winds = dict()
            for freq_len, freq_array in s.freqlen_dict.items():
                winds[freq_len] = (convert_scale(s.params['Times'], win[0]),
                                   convert_scale(s.params['Times'], win[1]),
                                   convert_scale(freq_array, win[2]),
                                   convert_scale(freq_array, win[3]))
            time_lbl = 'to'.join([str(t) for t in win[0:2]]) + 'ms'
            freq_lbl = 'to'.join([str(f) for f in win[2:4]]) + 'Hz'
            win_lbl = '_'.join([time_lbl, freq_lbl])
            print(winds)
            win_inds.append(winds)
            win_lbls.append(win_lbl)

        cinds = dict()
        minds = dict()
        for chan_len, chan_lst in s.chanlen_dict.items():
            minds[chan_len] = False
            tmp_cinds = []
            for c in channels:
                try:
                    tmp_cinds.append(chan_lst.index(c))
                except ValueError:
                    print(c, 'not present for', chan_len, '--> filling with nans')
                    minds[chan_len] = True
            cinds[chan_len] = sorted(tmp_cinds)
        max_chans = max(s.chanlen_dict.keys())
        max_chans_labels = s.chanlen_dict[max_chans]
        chan_lbls = [max_chans_labels[ci] for ci in cinds[max_chans]]  # this effectively sorts the channel labels

        # because the indexing order for the 21-channel montage is so different, it needs to be specially handled
        if 20 in s.chanlen_dict.keys():
            c20_newind_oldind = dict()
            for clbl_ind, clbl in enumerate(chan_lbls):
                try:
                    c20_newind_oldind[clbl_ind] = s.chanlen_dict[20].index(clbl)
                except ValueError:
                    pass

        n_files = len(s.data_df['path'].values)
        n_windows = len(windows)
        n_chans = len(chan_lbls)
        mean_array = np.empty((n_files, n_windows, n_chans))
        mean_array.fill(np.nan)  # fill with nan

        fpi = 0
        for fp, fl, cl in tqdm(zip(s.pointer_lst, s.data_df['freq_len'].values, s.data_df['chan_len'].values)):
            for wi, winds in enumerate(win_inds):
                if minds[cl] or cl == 20:
                    if cl == 20:
                        data_vec = fp['data'][:, winds[fl][0]:winds[fl][1] + 1,
                                   winds[fl][2]:winds[fl][3] + 1].mean(axis=(1, 2))
                        for new_ind, old_ind in c20_newind_oldind.items():
                            mean_array[fpi, wi, new_ind] = data_vec[old_ind]
                    else:
                        mean_array[fpi, wi, :len(cinds[cl])] = fp['data'][cinds[cl], winds[fl][0]:winds[fl][1] + 1,
                                                               winds[fl][2]:winds[fl][3] + 1].mean(axis=(1, 2))
                else:
                    mean_array[fpi, wi, :] = fp['data'][cinds[cl], winds[fl][0]:winds[fl][1] + 1,
                                             winds[fl][2]:winds[fl][3] + 1].mean(axis=(1, 2))
            fpi += 1

        mean_array_reshaped = np.reshape(mean_array, (n_files, n_windows * n_chans))

        column_mi = pd.MultiIndex.from_product([win_lbls, chan_lbls], names=['TFROI', 'channel'])

        mean_df = pd.DataFrame(mean_array_reshaped, index=s.data_df.index, columns=column_mi)
        return mean_df

    def tfmean_multi_winchantups(s, windows_channels):
        ''' given a list of 5-tuples of (t1, t2, f1, f2, chans),
            where chans is a list of channels that you want to calculate for those windows,
            calculate TFROI means '''

        winchan_inds = []
        winchan_lbl_tups = []
        channels = set()
        for winchans in windows_channels:
            win = winchans[:4]
            win_chans = winchans[4]
            winds = dict()
            for freq_len, freq_array in s.freqlen_dict.items():
                winds[freq_len] = (convert_scale(s.params['Times'], win[0]),
                                   convert_scale(s.params['Times'], win[1]),
                                   convert_scale(freq_array, win[2]),
                                   convert_scale(freq_array, win[3]))
            time_lbl = 'to'.join([str(t) for t in win[0:2]]) + 'ms'
            freq_lbl = 'to'.join([str(f) for f in win[2:4]]) + 'Hz'
            win_lbl = '_'.join([time_lbl, freq_lbl])
            print(winds)
            winchan_inds.append((winds, win_chans))
            for chan in win_chans:
                winchan_lbl_tups.append((win_lbl, chan))
                channels.add(chan)

        cinds = dict()
        minds = dict()
        for chan_len, chan_lst in s.chanlen_dict.items():
            minds[chan_len] = False
            tmp_cinds = dict()
            for c in channels:
                try:
                    tmp_cinds[c] = chan_lst.index(c)
                except ValueError:
                    print(c, 'not present for', chan_len, '--> filling with nans')
                    minds[chan_len] = True
            cinds[chan_len] = tmp_cinds

        n_files = len(s.data_df['path'].values)
        n_winchans = len(winchan_lbl_tups)
        mean_array = np.empty((n_files, n_winchans))
        mean_array.fill(np.nan)  # fill with nan

        fpi = 0
        for fp, fl, cl in tqdm(zip(s.pointer_lst, s.data_df['freq_len'].values, s.data_df['chan_len'].values)):
            data_vec = fp['data']
            wci = 0
            for winds, win_chans in winchan_inds:
                data_vec_win = data_vec[:, winds[fl][0]:winds[fl][1] + 1,
                               winds[fl][2]:winds[fl][3] + 1].mean(axis=(1, 2))
                for chan in win_chans:
                    try:
                        nchans = data_vec.shape[0] # there are some discrepancies with cl determined above
                        chan_ind = cinds[ nchans ][chan]
                        mean_array[fpi, wci] = data_vec_win[chan_ind]
                    except KeyError:
                        pass

                    wci += 1
            fpi += 1

        column_mi = pd.MultiIndex.from_tuples(winchan_lbl_tups, names=['TFROI', 'channel'])

        mean_df = pd.DataFrame(mean_array, index=s.data_df.index, columns=column_mi)
        return mean_df

    def tfmean_multi_winchantups_bl(s, windows_channels, bl_window=[-187.5, -100]):
        ''' same as tfmean_multi_winchantups but uses the ERSP-style baseline normalization
            in a given baseline window'''

        bl_time1 = convert_scale(s.params['Times'], bl_window[0])
        bl_time2 = convert_scale(s.params['Times'], bl_window[1])

        winchan_inds = []
        winchan_lbl_tups = []
        channels = set()
        for winchans in windows_channels:
            win = winchans[:4]
            win_chans = winchans[4]
            winds = dict()
            for freq_len, freq_array in s.freqlen_dict.items():
                winds[freq_len] = (convert_scale(s.params['Times'], win[0]),
                                   convert_scale(s.params['Times'], win[1]),
                                   convert_scale(freq_array, win[2]),
                                   convert_scale(freq_array, win[3]))
            time_lbl = 'to'.join([str(t) for t in win[0:2]]) + 'ms'
            freq_lbl = 'to'.join([str(f) for f in win[2:4]]) + 'Hz'
            win_lbl = '_'.join([time_lbl, freq_lbl])
            print(winds)
            winchan_inds.append((winds, win_chans))
            for chan in win_chans:
                winchan_lbl_tups.append((win_lbl, chan))
                channels.add(chan)

        cinds = dict()
        minds = dict()
        for chan_len, chan_lst in s.chanlen_dict.items():
            minds[chan_len] = False
            tmp_cinds = dict()
            for c in channels:
                try:
                    tmp_cinds[c] = chan_lst.index(c)
                except ValueError:
                    print(c, 'not present for', chan_len, '--> filling with nans')
                    minds[chan_len] = True
            cinds[chan_len] = tmp_cinds

        n_files = len(s.data_df['path'].values)
        n_winchans = len(winchan_lbl_tups)
        mean_array = np.empty((n_files, n_winchans))
        mean_array.fill(np.nan)  # fill with nan

        fpi = 0
        for fp, fl, cl in tqdm(zip(s.pointer_lst, s.data_df['freq_len'].values, s.data_df['chan_len'].values)):
            data_vec = fp['data']
            to_div = data_vec[:, bl_time1:bl_time2 + 1, :].mean(axis=1, keepdims=True)
            data_vec_bl = 10 * np.log10(data_vec / to_div)
            wci = 0
            for winds, win_chans in winchan_inds:
                data_vec_win = data_vec_bl[:, winds[fl][0]:winds[fl][1] + 1,
                               winds[fl][2]:winds[fl][3] + 1].mean(axis=(1, 2))
                for chan in win_chans:
                    try:
                        chan_ind = cinds[cl][chan]
                        mean_array[fpi, wci] = data_vec_win[chan_ind]
                    except KeyError:
                        pass
                    wci += 1
            fpi += 1

        column_mi = pd.MultiIndex.from_tuples(winchan_lbl_tups, names=['TFROI', 'channel'])

        mean_df = pd.DataFrame(mean_array, index=s.data_df.index, columns=column_mi)
        return mean_df

    def sub_mean(s):
        ''' calculate a grand (cross-subject) mean.
            creates a numpy array of shape (chans, freqs, times) '''

        n_files = len(s.data_df['path'].values)
        n_chans = int(s.params['# of channels'][0][0])
        n_times = s.params['Times'].shape[1]
        n_freqs = s.params['Frequencies'].shape[1]
        print('{} files, {} chans, {} times, {} freqs'.format(n_files, n_chans, n_times, n_freqs))

        sum_array = np.zeros((n_chans, n_times, n_freqs))

        shape_mismatch_count = 0
        for fpi, fp in enumerate(s.pointer_lst):
            data = fp['data'][:]
            if data.shape != sum_array.shape:
                print('shape mismatch on file', fpi)
                print('data', data.shape)
                print('sum array', sum_array.shape)
                shape_mismatch_count += 1
                continue
            sum_array += data

        mean_array = sum_array / (n_files - shape_mismatch_count)

        print('total mismatched:', shape_mismatch_count)

        return mean_array

    def sub_mean_interp(s):
        ''' calculate a grand (cross-subject) mean. creates a numpy array of shape (chans, freqs, times)
            if interpolation is necessary '''

        n_files = len(s.data_df['path'].values)
        n_chans = int(s.params['# of channels'][0][0])
        time_vec = s.params['Times'][0]
        n_times = time_vec.size
        n_freqs = s.params['Frequencies'].shape[1]
        print('{} files, {} chans, {} times, {} freqs'.format(n_files, n_chans, n_times, n_freqs))

        sum_array = np.zeros((n_chans, n_times, n_freqs))

        max_freqsize = max(s.freqlen_dict)

        fpi = 0
        for fp, fl in tqdm(zip(s.pointer_lst, s.data_df['freq_len'].values)):
            data = fp['data'][:]
            if fl != max_freqsize:
                new_data = np.empty((n_chans, n_times, max_freqsize))
                for ctfi, chan_tf in enumerate(data):
                    new_data[ctfi, :, :] = interp_freqdomain_fast(chan_tf, time_vec,
                                                                  s.freqlen_dict[fl][0],
                                                                  s.freqlen_dict[max_freqsize][0])
                data = new_data
            sum_array += data
            fpi += 1

        mean_array = sum_array / n_files

        return mean_array

    def load_data_interp(s):
        ''' load all data (memory-intensive) and stack it, interpolating if necessary '''

        n_files = len(s.data_df['path'].values)
        n_chans = int(s.params['# of channels'][0][0])
        time_vec = s.params['Times'][0]
        n_times = time_vec.size
        max_freqsize = max(s.freqlen_dict)
        n_freqs = s.freqlen_dict[max_freqsize].shape[1]
        print('{} files, {} chans, {} times, {} freqs'.format(n_files, n_chans, n_times, n_freqs))

        # subjects, conditions, channels, freq, timepoints
        data_array = np.empty((n_files, 1, n_chans, n_times, n_freqs))

        fpi = 0
        for fp, fl in tqdm(zip(s.pointer_lst, s.data_df['freq_len'].values)):
            data = fp['data'][:]  # is chans x times x freq
            if fl != max_freqsize:
                new_data = np.empty((n_chans, n_times, max_freqsize))
                for ctfi, chan_tf in enumerate(data):
                    new_data[ctfi, :, :] = interp_freqdomain_fast(chan_tf, time_vec,
                                                                  s.freqlen_dict[fl][0],
                                                                  s.freqlen_dict[max_freqsize][0])
                data = new_data
            data_array[fpi, 0, :, :, :] = data
            fpi += 1

        return data_array

    def load_data_interp_conds(s):
        ''' same as load_data_interp, but transpose conditions into its own dimension '''

        if 'cond_lst' not in dir(s):
            s.survey_conds()

        n_subjects = len(s.fullcond_uIDs)
        n_conds = len(s.cond_lst)
        n_chans = int(s.params['# of channels'][0][0])
        time_vec = s.params['Times'][0]
        n_times = time_vec.size
        max_freqsize = max(s.freqlen_dict)
        n_freqs = s.freqlen_dict[max_freqsize].shape[1]
        print('{} subjects, {} conds, {} chans, {} times, {} freqs'.format(
            n_subjects, n_conds, n_chans, n_times, n_freqs))

        # subjects, conditions, channels, freq, timepoints
        data_array = np.empty((n_subjects, n_conds, n_chans, n_times, n_freqs))

        uidi = -1
        for ID, session in s.data_df_fullconds_uID_nodupes.index:
            uidi += 1
            for ci, cond in enumerate(s.cond_lst):
                ID_ses_cond = (ID, session, cond)
                # print(ID_ses_cond)
                row = s.data_df_fullconds_uIDconds.loc[ID_ses_cond]
                file_path = row['path']
                fl = row['freq_len']
                pointer = h5py.File(file_path, 'r')
                data = pointer['data'][:]  # is chans x times x freq
                if fl != max_freqsize:
                    new_data = np.empty((n_chans, n_times, max_freqsize))
                    for ctfi, chan_tf in enumerate(data):
                        new_data[ctfi, :, :] = interp_freqdomain_fast(chan_tf, time_vec,
                                                                      s.freqlen_dict[fl][0],
                                                                      s.freqlen_dict[max_freqsize][0])
                    data = new_data
                data_array[uidi, ci, :, :, :] = data

        return data_array


class EROMat:
    def __init__(s, path):
        ''' represents h5-compatible mat containing 3d ERO data and options '''
        s.filepath = path
        s.parse_path()
        # s.get_params()

    def parse_path(s):
        ''' parse file info from new 3d ERO .mat path '''
        filedir, filename = os.path.split(s.filepath)

        dirparts = filedir.split(os.path.sep)
        n_chans = dirparts[-2]
        param_string = dirparts[-3]

        name, ext = os.path.splitext(filename)
        ID, session, experiment, condition, powertype = name.split('_')

        s.info = {'path': s.filepath,
                  'n_chans': n_chans,
                  'param_string': param_string,
                  'ID': ID,
                  'session': session,
                  'experiment': experiment,
                  'condition': condition,
                  'powertype': powertype}

    def prepare_row(s):
        ''' get info for row in dataframe from database '''
        s.info.update({'_id': s._id,
                       'path': s.filepath,
                       'matpath': s.params['Mat file']})

    def get_params(s):
        ''' extract data parameters from opt, relying upon opt_info. '''
        mat = h5py.File(s.filepath, 'r')
        prefix = 'opt/'
        s.params = {}
        for param, info in opt_info.items():
            try:
                s.params.update({param:
                                     handle_parse(mat, prefix + info[0], info[1])})
            except KeyError:
                print('param extraction failed for', param, 'for', s.filepath)
        mat.close()
        s.params['Times'] = s.params['Times'].astype('float64')

    def load_data(s):
        ''' prepare 3d ERO data '''
        s.data = h5py.File(s.filepath, 'r')['data'][:]

    def tfmean(s, times, freqs):
        ''' given 2-tuples of start and end times and frequencies,
            return a dataframe containing the mean value in that TFROI '''

        time_lims = [convert_scale(s.params['Times'], time) for time in times]
        freq_lims = [convert_scale(s.params['Frequencies'], freq)
                     for freq in freqs]
        print(time_lims)
        print(freq_lims)
        tf_slice = s.data[:, time_lims[0]:time_lims[1] + 1,
                   freq_lims[0]:freq_lims[1] + 1]
        tfmean = tf_slice.mean(axis=2)
        out_array = tfmean.mean(axis=1)

        mean_df = pd.DataFrame(out_array, index=chans_61,
                               columns=[s.filepath[-10:]])
        return mean_df


# Post processing tools to manage output and compare with old compilations

def find_column_groups(column_list, exclusions=['trial', 'age'], var_ind=-1):
    ''' group columns with similar values across all fields except the one specified by var_ind

        this is meant to parse old style labellings to set up groups for comparison
    '''
    process_list = column_list.copy()
    for ex in exclusions:
        process_list = [c for c in process_list if ex not in c]

    col_lists = [c.split('_') for c in process_list]
    [cl.pop(var_ind) for cl in col_lists]
    col_tups = [tuple(cl) for cl in col_lists]
    col_types = set(col_tups)
    col_groups = {ct: [] for ct in col_types}
    for ct, field in zip(col_tups, process_list):
        col_groups[ct].append(field)

    return col_groups


# Representational class to parse and produce labels

def num_drop_units(st):
    stD = st
    found_unit = None
    for unit in ['ms', 'Hz']:
        if unit in stD:
            stD = stD.replace(unit, '')
            found_unit = unit
    return float(stD), found_unit


def split_label(label, sep='_'):
    parts = label.split(sep)
    return parts


class EROpheno_label_set:
    '''normally  a set for one phenotype across electordes
    '''
    # Default example descirption - used for filling gaps - these are overriden by 
    # 'implicit_values' argument on initialization
    field_desc = {'experiment': ['vp3'],
                  'case': ['target'],
                  'time': [(300, 700)],
                  'frequency': [(3.0, 7.0)],
                  'channels': ['FZ', 'PZ', 'CZ', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4'],
                  'power': ['total']}
    def_order = ['experiment', 'case', 'frequency', 'time', 'power', 'channels']

    experiments = D.Mdb.STinverseMats.distinct('experiment')
    case_name_aliases = {'tt': 'target', 'nv': 'novel', 'nt': 'nontarget'}
    cases_db = D.Mdb.STinverseMats.distinct('case')
    cases = []
    for case in cases_db:
        cases.append(case)
        if case in case_name_aliases:
            cases.append(case_name_aliases[case])
    powers = ['tot-pwr', 'evo-pwr', 'total', 'evoked']

    method_descriptor_type = type(str.lower)
    function_type = type(lambda x: x)

    def __init__(s, description={}, fields=None, implicit_values={}, sep='_'):
        s.in_sep = sep
        s.units = {}
        s.implicit_values = implicit_values

        if fields is None:
            fields = s.def_order

        if type(description) == list:
            s.in_cols = description.copy()
            parse_out = s.parse_col_group(s.in_cols, fields, sep='_')
            if type(parse_out) != tuple:
                print(parse_out)
                return
            else:
                field_order, options, units = parse_out
            s.units = units
        # need to normalize custom description

        s.parsed_desc = {fd: desc for fd, desc in zip(field_order, options)}

        s.parsed_order = field_order

    def parse_col_group(s, col_list, field_names, sep='_'):
        all_pieces = [split_label(c, sep) for c in col_list]
        num_parts = set([len(pcs) for pcs in all_pieces])
        if len(num_parts) > 1:
            return 'non_uniform'
        else:
            num_parts = list(num_parts)[0]
        options = [set() for i in range(num_parts)]
        for col_pcs in all_pieces:
            [options[i].add(pc) for i, pc in enumerate(col_pcs)]
        options = [list(o) for o in options]

        part_names = field_names.copy()  # set([k for k in def_desc.keys()])
        # identify part meanings
        part_order = []
        parsed_opts = []
        unitsD = {}
        for opts in options:
            if len(part_names) == 1:  # handle the final remaining field
                part_order.append(list(part_names)[0])
                parsed_opts.append(opts)
            elif opts[0] in s.cases:
                part_order.append('case')
                part_names.remove('case')
                parsed_opts.append(opts)
            elif opts[0] in s.experiments or opts[0] in [e.upper() for e in s.experiments]:
                part_order.append('experiment')
                part_names.remove('experiment')
                parsed_opts.append(opts)
            elif opts[0] in s.powers:
                part_order.append('power')
                part_names.remove('power')
                parsed_opts.append(opts)
            elif '-' in opts[0] and 'pwr' not in opts[0]:
                nums_pieces = opts[0].split('-')
                nums = []
                this_unit = ''
                for numS in nums_pieces:
                    val, unit = num_drop_units(numS)
                    nums.append(val)
                    if unit:
                        this_unit = unit
                if nums[1] - nums[0] < 40:
                    part_order.append('frequency')
                    part_names.remove('frequency')

                else:
                    part_order.append('time')
                    part_names.remove('time')

                parsed_opts.append([tuple(nums)])
                unitsD[part_order[-1]] = this_unit

            else:
                part_order.append('channels')
                part_names.remove('channels')
                parsed_opts.append(opts)

        return part_order, parsed_opts, unitsD

    def produce_column_names(s, order=def_order, units=False, dec_funs={'frequency': "%.1f", 'time': "%.0f"},
                             translators={'experiment': str.lower}, sep='_'):
        '''Output list of column names with 
            translators is nested dict by fields and values
        '''
        if order == 'parsed':
            order = s.parsed_order

        # note input column order 
        sort_fields = [f for f, opts in s.parsed_desc.items() if len(opts) > 1]
        if 'in_cols' in dir(s):
            in_orders = {fd: [] for fd in sort_fields}
            in_sort = []
            for incol in s.in_cols:
                parsed = {fd: val for fd, val in zip(s.parsed_order, incol.split(s.in_sep))}
                for sf in sort_fields:
                    in_sort.append(tuple([parsed[fd] for fd in sort_fields]))

        full_desc = s.field_desc.copy()
        full_desc.update(s.implicit_values)
        full_desc.update(s.parsed_desc)
        # print(full_desc)

        opt_lens = [len(v) for k, v in full_desc.items()]
        n_cols = np.product(opt_lens)
        Wcols = n_cols * ['']
        out_sorts = [[] for n in range(n_cols)]
        joiner = ''
        for field in order:
            translator = None
            if field in translators:
                translator = translators[field]
            if field in s.parsed_desc:
                opts = s.parsed_desc[field]
            else:
                opts = full_desc[field]
            if field in ['time', 'frequency']:
                nums_st = []
                for v in opts[0]:
                    # print(v,  "dec_places[field])+"f")
                    nums_st.append(dec_funs[field] % v)
                opt_st = '-'.join(nums_st)
                if units and field in s.units:
                    opt_st = opt_st + s.units[field]
                optL = [opt_st]

            else:
                optL = opts
            if len(optL) == 1:
                optL = n_cols * optL

            # print(optL)
            cols = []
            ind = -1
            for cs, opt in zip(Wcols, optL):
                ind += 1
                if field in sort_fields:
                    out_sorts[ind].append(opt)
                if type(translator) in [s.method_descriptor_type, s.function_type]:
                    opt = translator(opt)
                elif type(translator) == dict:
                    if opt in translator:
                        opt = translator[opt]

                cols.append(cs + joiner + opt)
            Wcols = cols.copy()
            joiner = sep

        # preserve input column order
        if 'in_cols' in dir(s):
            out_cols = []
            out_sortD = {tuple(osl): col for osl, col in zip(out_sorts, cols)}
            for sort_tup in in_sort:
                out_cols.append(out_sortD[sort_tup])

        else:
            out_cols = cols

        return out_cols


# Compilation functions for use with processing module
standard_cols = ['uID', 'nearest session to fMRI', 'Sl. No.', 'NYU fMRI ID',
                 'fMRI test date', 'followup', 'POP', 'alc_dep_dx',
                 'alc_dep_ons', 'PH', 'fhd_dx4_ratio', 'fhd_dx4_sum',
                 'n_rels', 'date', 'sex', 'handedness', 'current_age',
                 'session_age', 'alc_dep_dx_f', 'alc_dep_dx_m', 'fID', 'mID',
                 'famID', 'famtype', 'DNA', 'SmS', 'genoID', 'rel2pro',
                 'ruID', 'self-reported', 'core-race', 'site', 'system',
                 'twin', 'genotyped', 'methylation']


def PR_load_session_file(path, cols=standard_cols):
    df = pd.read_csv(path, converters={'ID': str})
    df.set_index(['ID', 'session'], inplace=True)

    cols_use = df.columns.intersection(cols)
    comp_df = df[cols_use]

    return comp_df


PR_load_session_file.store_name = 'db/eromat.load_session_file'


def PR_stack_from_mat_lst(df, proc_type):
    path_cols = [c for c in df.columns if proc_type in c]
    mat_list = []
    for pc in path_cols:
        mat_list.extend(list(df[pc].dropna().values))
    stack = EROStack(mat_list)
    return stack


PR_stack_from_mat_lst.store_name = 'db/eromat.stack_from_mat_lst'


def PR_get_tfmeans(erostack, tf_windows, electrodes):
    return erostack.tfmean_multiwin_chans(tf_windows, electrodes)


PR_get_tfmeans.store_name = 'db/eromat.get_tfmeans'
