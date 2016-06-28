''' represents intermediate results from processing pipeline:
    does import, analyses, plotting '''

from glob import glob
import os
import h5py
import numpy as np
import pandas as pd
import dask.array as da
import mne
import matplotlib.pyplot as plt

# values are tuples of h5py fieldname and datatype
opt_info = {'Coordinates file':             ('coords_file',         'text'),

            'Condition labels':             ('case_label',          'cell'),
            'Measures available':           ('measures',            'cell'),

            'Sampling rate':                ('rate',                'array'),
            '# of timepoints':              ('n_samps',             'array'),
            'Temporal limits':              ('epoch_lims',          'array'),
            'Frequency limits':             ('freq_lims',           'array'),
            'TF scales':                    ('wavelet_scales',      'array'),
            'TF pad ratio':                 ('padratio',            'array'),
            'TF time-downsample ratio':     ('tf_timedownsamp_ratio', 'array'),
            'Coherence pairs':              ('coherence_pairs',     'array'),
            'Coherence pair subset labels': ('pair_indlbls',        'array'),
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
    dset_ref = dset[dset_field]
    refs = [t[0] for t in dset_ref[:]]
    out_lst = [''.join(chr(c) for c in dset[ref][:]) for ref in refs]
    return out_lst

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
def baseline(array, pt_lims, along_dim=-1):
    return array - array.take(range(pt_lims[0], pt_lims[1]+1), axis=along_dim)\
                        .mean(axis=along_dim, keepdims=True)

def convert_ms(time, ms_tuple):
    pt_start    = np.argmin(np.fabs(time - ms_tuple[0]))
    pt_end      = np.argmin(np.fabs(time - ms_tuple[1]))
    return pt_start, pt_end

def compound_take(a, vals, dims):
    print(a.shape)
    for v, d in zip(vals, dims):
        if isinstance(v, int):
            a = a.take([v], d)
        else:
            a = a.take(v, d)
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
    num_lst = [n]
    den_lst = [1]
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
        s.params = {param: handle_parse(s.opt, prefix + info[0], info[1])
                    for param, info in opt_info.items()}

    def make_scales(s):
        # read montage - this is a bit hackish, but works
        use_ext = '.sfp'
        path, file = os.path.split(s.params['Coordinates file'])
        fn, ext = file.split('.')
        s.montage = mne.channels.read_montage(os.path.join(path, fn) + use_ext)

        # ERP
        ep_lims     = s.params['Temporal limits']
        n_pts       = s.params['# of timepoints']
        s.time      = np.linspace(ep_lims[0], ep_lims[1], n_pts + 1)[1:]

        # TF


    def load_erp(s, lp_cutoff=16, bl_window=(-100, 0)):
        ''' load ERP data, filter it, and baseline '''
        dsets = [h5py.File(fn)['erp'] for fn in s.file_df['path'].values]
        arrays = [da.from_array(dset, chunks=dset.shape) for dset in dsets]
        stack = da.stack(arrays, axis=-1)  # concatenate along last axis
        stack = stack.transpose([3, 0, 1, 2]) # move subjects dimension to front
        
        erp = np.empty(stack.shape)
        da.store(stack, erp)
        # erp is (subjects, conditions, channels, timepoints,)
        print(erp.shape)

        # filter
        erp_filt = mne.filter.low_pass_filter(erp, 
                        s.params['Sampling rate'], lp_cutoff)
        # baseline
        erp_filt_bl = baseline(erp_filt, convert_ms(s.time, bl_window))

        s.erp = erp_filt_bl
        s.erp_dims = ('subject', 'condition', 'channel', 'timepoint')
        s.erp_dim_lsts = (s.file_df.index.values, s.params['Condition labels'],
            s.montage.ch_names, s.time)

    def prepare_mne(s):
        # create info
        info = mne.create_info(s.montage.ch_names, s.params['Sampling rate'],
                                    'eeg', s.montage)
        # EvokedArray
        chan_erps = s.erp.mean(axis=(0,1))/1000000 # subject grand mean for now
        return mne.EvokedArray(chan_erps, info,
                    tmin=s.params['Temporal limits'][0]/1000)

    def plot_erp(s, figure_by=('channel', ['FZ', 'CZ', 'PZ']),
                    subplot_by=('condition', None),
                    line_by=('group', None) ):

        figure_dim, figure_vals, figure_lbls = s.interpret_by(figure_by)
        subplot_dim, subplot_vals, subplot_lbls = s.interpret_by(subplot_by)
        line_dim, line_vals, line_lbls = s.interpret_by(line_by)

        sp_dims = subplot_heuristic(len(subplot_vals))
        for fi, fval in enumerate(figure_vals):
            f, axarr = plt.subplots(sp_dims[0], sp_dims[1])
            f.suptitle(figure_lbls[fi])
            axarr = axarr.ravel()
            for spi, spval in enumerate(subplot_vals):
                for li, lval in enumerate(line_vals):
                    line = compound_take(s.erp, [fval, spval, lval],
                            [figure_dim, subplot_dim, line_dim])
                    print(line.shape)
                    while len(line.shape) > 1:
                        line = line.mean(axis=0)
                        print(line.shape)
                    axarr[spi].plot(np.arange(len(line)), line,
                            label=line_lbls[li])
                axarr[spi].set_title(subplot_lbls[spi])
                axarr[spi].legend()


    # plot helpers
    def interpret_by(s, by_stage):
        print('by stage is', by_stage[0])
        if by_stage[0] in s.erp_dims:
            dim = s.erp_dims.index(by_stage[0])
            print('data in dim', dim)
            if by_stage[1]:
                labels = by_stage[1]
                vals = [s.erp_dim_lsts[dim].index(lbl) for lbl in labels]
                print('vals to iterate on are', vals)
            else:
                labels = s.erp_dim_lsts[dim]
                vals = list(range(len(labels)))
                print('iterate across available vals including', vals)
        elif by_stage[0] in s.demog_df.columns:
            dim = s.erp_dims.index('subject')
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
        return dim, vals, labels

class ERP:

    def __init__(s, results_obj):
        s.data = results_obj.erp
        s.dims = results_obj.erp_dims
        s.montage = results_obj

    def plot_erp(s, channels=['FZ', 'CZ', 'PZ'],
            line_by='group', subplot_by='condition', figure_by='channel'):
        pass