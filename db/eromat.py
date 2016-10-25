''' representing and working with .mat's containing 3D ERO data '''

import os

from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py
import dask.array as da

from .organization import Mdb
# import compilation as C

opt_info = {'Add baseline':             ('add_baseline',    'array'),
            'Calculation type':         ('calc_type',       'array'),
            'Condition':                ('case_name',       'text'),
            'Channel sort':             ('channel_sort',    'array'),
            'Do baseline':              ('do_baseline',     'array'),
            'Channels':                 ('elec_array',      'text_array'),
            'Experiment':               ('exp_name',        'text'),
            # 'Original frequencies':   ('f_vec',           'array'),
            'Frequencies':              ('f_vec_ds',        'array'),
            'File ID':                  ('file_id',         'text'),
            # 'File name':              ('file_name',       'text'),
            'Mat file':                 ('filenm',          'text'),
            'Run':                      ('file_run',        'array'),
            'Session':                  ('file_session',    'text'),
            'File index':               ('i_file',          'array'),
            'Natural log':              ('ln_calc',         'array'),
            '# of channels':            ('n_chans_present', 'array'),
            'Output type':              ('out_type',        'array'),
            'Output type name':         ('out_type_name',   'text'),
            # 'Output text':            ('output_text',     'text'),
            'Sampling rate':            ('rate',            'array'),
            'S-transform type':         ('st_type',         'array'),
            'S-transform type name':    ('st_type_name',    'text'),
            'Time downsample ratio':    ('time_ds_factor',  'array'),
            # 'Original times':         ('time_vec',        'array'),
            'Times':                    ('time_vec_ds',     'array'),
           }

chans = ['FP1', 'FP2', 'F7' , 'F8' , 'AF1', 'AF2', 'FZ' ,
 'F4' , 'F3' , 'FC6', 'FC5', 'FC2', 'FC1', 'T8' , 'T7' , 'CZ' ,
  'C3' , 'C4' , 'CP5', 'CP6', 'CP1', 'CP2', 'P3' , 'P4' , 'PZ' ,
   'P8' , 'P7' , 'PO2', 'PO1', 'O2' , 'O1' , 'X'  , 'AF7', 'AF8',
    'F5' , 'F6' , 'FT7', 'FT8', 'FPZ', 'FC4', 'FC3', 'C6' , 'C5' ,
     'F2' , 'F1' , 'TP8', 'TP7', 'AFZ', 'CP3', 'CP4', 'P5' ,
      'P6' , 'C1' , 'C2' , 'PO7', 'PO8', 'FCZ', 'POZ', 'OZ' ,
       'P2' , 'P1' , 'CPZ']

# h5py parsing functions

def parse_text(dset, dset_field):
    ''' parse .mat-style h5 field that contains text '''
    dset_ref = dset[dset_field]
    return ''.join(chr(c[0]) for c in dset_ref[:])

def parse_textarray(dset, dset_field):
    ''' parse .mat-style h5 field that contains a text array '''
    dset_ref = dset[dset_field]
    array = dset_ref[:]    
    return [''.join([chr(arg) for arg in args]).rstrip() for args in 
                zip(*array)]

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

ftypes_funcs = {'text': parse_text, 'cell': parse_cell, 'array': None,
                'text_array': parse_textarray}

def handle_parse(dset, dset_field, field_type):
    ''' given file pointer, field, and datatype, apply appropriate parser '''
    func = ftypes_funcs[field_type]
    return func(dset, dset_field) if func else dset[dset_field][:]

# dataframe functions

def join_columns(row, columns):
    return '_'.join([row[col] for col in columns])

# array functions

def convert_scale(scale_array, scale_val):
    ''' given array, find index nearest to given value '''
    return np.argmin(np.fabs(scale_array - scale_val))

# main classes

class EROStack:
    ''' represents a list of .mat's as a dask stack '''

    def __init__(s, path_lst, touch_db=False):
        s.init_df(path_lst, touch_db=touch_db)
        s.get_params()

    def init_df(s, path_lst, touch_db=False):
        ''' given, a list of paths, intializes the dataframe that represents each existent EROmat.
            pulls info out of the path of each. '''
        row_lst = []
        missing_count = 0
        for fp in path_lst:
            if os.path.exists(fp):
                em = EROMat(fp)
                if touch_db:
                    em.prepare_row()
                row_lst.append(em.info)
            else:
                missing_count += 1
                
        print(missing_count, 'files missing')

        s.data_df = pd.DataFrame.from_records(row_lst)
        s.data_df.set_index(['ID', 'session', 'experiment'], inplace=True)

    def get_params(s):
        em = EROMat(s.data_df.ix[0, 'path'])
        em.get_params()
        s.params = em.params

    def load_stack(s):
        dsets = [h5py.File(fp, 'r')['data'] for fp in s.data_df['path'].values]
        arrays = [da.from_array(dset, chunks=dset.shape) for dset in dsets]
        s.stack = da.stack(arrays, axis=-1)
        print(s.stack.shape)

    def load_stack_np(s):
        arrays = [h5py.File(fp, 'r')['data'][:] for fp in s.data_df['path'].values]
        s.stack_np = np.stack(arrays, axis=-1)
        print(s.stack_np.shape)

    def tf_mean(s, times, freqs):
        time_lims = [convert_scale(s.params['Times'], time) for time in times]
        freq_lims = [convert_scale(s.params['Frequencies'], freq)
                                for freq in freqs]
        print(time_lims)
        print(freq_lims)
        tf_slice = s.stack[:, time_lims[0]:time_lims[1]+1,
                              freq_lims[0]:freq_lims[1]+1, :]
        tf_mean = tf_slice.mean(axis=2)
        tf_mean = tf_mean.mean(axis=1)
        out_array = np.empty(tf_mean.shape)
        da.store(tf_mean, out_array)

        time_lbl = '-'.join([str(t) for t in times])+'ms'
        freq_lbl = '-'.join([str(f) for f in freqs])+'Hz'
        lbls = ['_'.join([chan, time_lbl, freq_lbl])
                        for chan in s.params['Channels']]

        mean_df = pd.DataFrame(out_array.T, index=s.data_df.index, columns=lbls)
        return mean_df

    def tf_mean_np(s, times, freqs):
        time_lims = [convert_scale(s.params['Times'], time) for time in times]
        freq_lims = [convert_scale(s.params['Frequencies'], freq)
                                for freq in freqs]
        print(time_lims)
        print(freq_lims)
        tf_slice = s.stack_np[:, time_lims[0]:time_lims[1]+1,
                              freq_lims[0]:freq_lims[1]+1, :]
        print('slice', tf_slice.shape)
        tf_mean = tf_slice.mean(axis=(1,2))
        print('mean', tf_mean.shape)

        time_lbl = '-'.join([str(t) for t in times])+'ms'
        freq_lbl = '-'.join([str(f) for f in freqs])+'Hz'
        lbls = ['_'.join([chan, time_lbl, freq_lbl])
                        for chan in s.params['Channels']]

        mean_df = pd.DataFrame(tf_mean.T, index=s.data_df.index, columns=lbls)
        return mean_df

    def tf_mean_lowmem(s, times, freqs):
        time_lims = [convert_scale(s.params['Times'], time) for time in times]
        freq_lims = [convert_scale(s.params['Frequencies'], freq)
                                for freq in freqs]
        print(time_lims)
        print(freq_lims)

        n_chans = int(s.params['# of channels'][0][0])
        n_files = len(s.data_df['path'].values)
        mean_array = np.empty((n_files, n_chans))

        for fpi, fp in enumerate(s.data_df['path'].values):
            file_pointer = h5py.File(fp, 'r')
            mean_array[fpi, :] = file_pointer['data']\
                                [:, time_lims[0]:time_lims[1]+1,
                                    freq_lims[0]:freq_lims[1]+1].mean(axis=(1,2))
            file_pointer.close()

        time_lbl = '-'.join([str(t) for t in times])+'ms'
        freq_lbl = '-'.join([str(f) for f in freqs])+'Hz'
        lbls = ['_'.join([chan, time_lbl, freq_lbl])
                        for chan in s.params['Channels']]

        mean_df = pd.DataFrame(mean_array, index=s.data_df.index, columns=lbls)
        return mean_df

    def tf_mean_lowmem_multiwin(s, windows):
        ''' given a list of tf windows, which are 4-tuples of (t1, t2, f1, f2),
            calculate means in those windows for all channels and subjects '''
        win_inds = []
        win_lbls = []
        for win in windows:
            winds = ( convert_scale(s.params['Times'], win[0]),
                      convert_scale(s.params['Times'], win[1]),
                      convert_scale(s.params['Frequencies'], win[2]),
                      convert_scale(s.params['Frequencies'], win[3]) )
            time_lbl = '-'.join([str(t) for t in win[0:2]])+'ms'
            freq_lbl = '-'.join([str(f) for f in win[2:4]])+'Hz'
            lbls = ['_'.join([chan, time_lbl, freq_lbl])
                        for chan in s.params['Channels']]
            print(winds)
            win_inds.append(winds)
            win_lbls.extend(lbls)

        n_files = len(s.data_df['path'].values)
        n_windows = len(windows)
        n_chans = int(s.params['# of channels'][0][0])
        mean_array = np.empty((n_files, n_windows, n_chans))

        for fpi, fp in tqdm(enumerate(s.data_df['path'].values)):
            file_pointer = h5py.File(fp, 'r')
            for wi, winds in enumerate(win_inds):
                mean_array[fpi, wi, :] = file_pointer['data']\
                                    [:, winds[0]:winds[1]+1,
                                        winds[2]:winds[3]+1].mean(axis=(1,2))
            file_pointer.close()

        mean_array_reshaped = np.reshape(mean_array, (n_files, n_windows*n_chans))

        mean_df = pd.DataFrame(mean_array_reshaped, index=s.data_df.index, columns=win_lbls)
        return mean_df

    def tf_mean_nosubs(s, times, freqs):
        pass

    def retrieve_dbdata(s, times, freqs):
        # for mat in s.data_df.iterrows():
        #     proj = C.format_EROprojection()
        pass

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
        ID, session, experiment, condition, measure = name.split('_')
        
        s.info = {'path': s.filepath,
                  'n_chans': n_chans,
                  'param_string': param_string,
                  'ID': ID,
                  'session': session,
                  'experiment': experiment,
                  'condition': condition,
                  'measure': measure}

    def prepare_row(s):
        ''' get info for row in dataframe from database '''
        s.retrieve_csvdoc()
        s.info.update({'_id':   s._id,
                       'path':  s.filepath,
                       'matpath': s.params['Mat file']})
    
    def get_params(s):
        ''' extract data parameters from opt, relying upon opt_info. '''
        mat = h5py.File(s.filepath, 'r')
        prefix = 'opt/'
        s.params = {}
        for param, info in opt_info.items():
            s.params.update({param:
                             handle_parse(mat, prefix + info[0], info[1])})
        mat.close()
        
    def load_data(s):
        ''' prepare 3d ERO data '''
        s.data = h5py.File(s.filepath, 'r')['data'][:]
    
    def tf_mean(s, times, freqs):
        time_lims = [convert_scale(s.params['Times'], time) for time in times]
        freq_lims = [convert_scale(s.params['Frequencies'], freq)
                                for freq in freqs]
        print(time_lims)
        print(freq_lims)
        tf_slice = s.data[:, time_lims[0]:time_lims[1]+1,
                              freq_lims[0]:freq_lims[1]+1]
        tf_mean = tf_slice.mean(axis=2)
        out_array = tf_mean.mean(axis=1)
        
        mean_df = pd.DataFrame(out_array, index=chans,
                                          columns=[s.filepath[-10:]])
        return mean_df

    def retrieve_csvdoc(s, just_id=True):
        ''' retrieve matching CSV document from the EROpheno collection '''

        uID = '_'.join([s.info['ID'], s.info['session'], s.info['experiment']])
        if just_id:
            c = Mdb['EROpheno'].find({'uID': uID}, {'_id': 1})
        else:
            c = Mdb['EROpheno'].find({'uID': uID})

        if c.count() == 0:
            print(uID, 'not found in collection')
            s.csv_doc = None
            s._id = None
            return
        elif c.count() > 1:
            print(uID, 'had more than one matching doc')
        
        s.csv_doc = next(c)
        s._id = s.csv_doc['_id']
        # extract _id and store in dataframe?