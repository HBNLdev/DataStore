''' representing and working with .mat's containing 3D ERO data '''

import os

from tqdm import tqdm
import numpy as np
from scipy import interpolate
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
    return '_'.join([row[field] for field in columns])

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
        s.open_pointers()
        s.survey_freqvecs()

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

    def open_pointers(s):
        s.pointer_lst = [h5py.File(fp, 'r') for fp in s.data_df['path'].values]

    def survey_freqvecs(s):

        freqlen_lst = []
        freqlen_dict = {}
        for fp in s.pointer_lst:
            f_vec = handle_parse(fp, 'opt/f_vec_ds', 'array')
            freqlen_lst.append(f_vec.shape[1])
            if f_vec.shape[1] not in freqlen_dict:
                freqlen_dict[f_vec.shape[1]] = f_vec

        s.data_df['freq_len'] = pd.Series(freqlen_lst, index=s.data_df.index)
        s.freqlen_dict = freqlen_dict
        s.params['Frequencies'] = freqlen_dict[max(freqlen_dict)]

    def load_stack(s):
        dsets = [fp['data'] for fp in s.pointer_lst]
        arrays = [da.from_array(dset, chunks=dset.shape) for dset in dsets]
        s.stack = da.stack(arrays, axis=-1)
        print(s.stack.shape)

    def load_stack_np(s):
        arrays = [fp['data'][:] for fp in s.pointer_lst]
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
            mean_array[fpi, :] = fp['data']\
                                [:, time_lims[0]:time_lims[1]+1,
                                    freq_lims[fl][0]:freq_lims[fl][1]+1].mean(axis=(1,2))
            fpi += 1

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
            winds = dict()
            for freq_len, freq_array in s.freqlen_dict.items():
                winds[freq_len] = ( convert_scale(s.params['Times'], win[0]),
                                    convert_scale(s.params['Times'], win[1]),
                                    convert_scale(freq_array, win[2]),
                                    convert_scale(freq_array, win[3]) )
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

        fpi = 0
        for fp, fl in tqdm(zip(s.pointer_lst, s.data_df['freq_len'].values)):
            for wi, winds in enumerate(win_inds):
                mean_array[fpi, wi, :] = fp['data']\
                                    [:, winds[fl][0]:winds[fl][1]+1,
                                        winds[fl][2]:winds[fl][3]+1].mean(axis=(1,2))
            fpi += 1

        mean_array_reshaped = np.reshape(mean_array, (n_files, n_windows*n_chans))

        mean_df = pd.DataFrame(mean_array_reshaped, index=s.data_df.index, columns=win_lbls)
        return mean_df

    def sub_mean_lowmem(s):
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

    def sub_mean_lowmem_interp(s):
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
                        s.freqlen_dict[fl][0], s.freqlen_dict[max_freqsize][0])
                data = new_data
            sum_array += data
            fpi += 1

        mean_array = sum_array / n_files

        return mean_array

    def retrieve_dbdata(s, times, freqs):
        # for mat in s.data_df.iterrows():
        #     proj = C.format_EROprojection()
        pass


def interp_freqdomain(a, t, f1, f2):
    ''' given time-frequency array a that is of shape (t.size, f1.size),
        timepoint vector t, and two frequency vectors, f1 and f2,
        use 2d interpolation to produce output array that is of size (t.size, f2.size) '''
    f = interpolate.interp2d(f1, t, a)
    return f(f2, t)


def interp_freqdomain_fast(a, t, f1, f2):
    ''' faster version of above '''
    f = interpolate.RectBivariateSpline(t, f1, a)
    return f(t, f2)


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


# Post processing tools to manage output and compare with old compilations

def find_column_groups(column_list,exclusions=['trial','age'],var_ind=-1):
    ''' group columns with similar values across all fields except the one specified by var_ind

        this is meant to parse old style labellings to set up groups for comparison
    '''
    process_list = column_list.copy()
    for ex in exclusions:
        process_list = [ c for c in process_list if ex not in c]
    
    col_lists = [ c.split('_') for c in process_list ]
    [ cl.pop(var_ind) for cl in col_lists  ]
    col_tups = [ tuple(cl) for cl in col_lists ]
    col_types = set(col_tups)
    col_groups = { ct:[] for ct in col_types }
    for ct, field in zip(col_tups,process_list):
        col_groups[ct].append(field)
    
    return col_groups

    # Tools to extract order from column names and produce them flexibly - aiming at more generalization
        # Information to guide processing 
experiments = Mdb.STinverseMats.distinct('experiment')
case_name_aliases = {'tt':'target','nv':'novel','nt':'nontarget'}
cases_db = Mdb.STinverseMats.distinct('case')
cases = []
for case in cases_db:
    cases.append(case)
    if case in case_name_aliases:
        cases.append(case_name_aliases[case])
        
powers = ['tot-pwr','evo-pwr','total','evoked']
units = ['ms','Hz']
method_descriptor_type = type(str.lower)
function_type = type(lambda x: x)

        # utility functions
def num_drop_units(st):
    stD = st
    found_unit = None
    for unit in units:
        if unit in stD:
            stD = stD.replace(unit,'')
            found_unit = unit
    return float(stD), found_unit

def split_label( label, sep='_' ):
    parts = label.split(sep)
    return parts


def parse_col_group(col_list,field_names,sep='_'):

    all_pieces = [ split_label(c,sep) for c in col_list ]
    num_parts = set([ len(pcs) for pcs in all_pieces ])
    if len(num_parts) > 1:
        return 'non_uniform'
    else: num_parts = list(num_parts)[0]
    options = [ set() for  i in range(num_parts)]
    for col_pcs in all_pieces:
        [ options[i].add(pc) for i,pc in enumerate(col_pcs) ]
    options = [ list(o) for o in options ]


    part_names = field_names.copy() #set([k for k in def_desc.keys()])
    #identify part meanings
    part_order = []
    parsed_opts = []
    unitsD = {}
    for opts in options:
        if len(part_names) == 1: # handle the final remaining field
            part_order.append( list(part_names)[0] )
            parsed_opts.append( opts )
        elif opts[0] in cases:
            part_order.append('case')
            part_names.remove('case')
            parsed_opts.append(opts)
        elif opts[0] in experiments or opts[0] in [ e.upper() for e in experiments ]:
            part_order.append('experiment')
            part_names.remove('experiment')
            parsed_opts.append(opts)
        elif opts[0] in powers:
            part_order.append('power')
            part_names.remove('power')
            parsed_opts.append( opts )            
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

            parsed_opts.append( [ tuple(nums) ] )
            unitsD[part_order[-1]] = this_unit

        else:
            part_order.append('channels')
            part_names.remove('channels')
            parsed_opts.append( opts )

    return part_order, parsed_opts, unitsD


        # Representational class to parse and produce labels
class EROpheno_label_set:
    '''normally  a set for one phenotype across electordes
    '''
    # Default example descirption - used for filling gaps - these are overriden by 
    # 'implicit_values' argument on initialization
    field_desc = {'experiment':['vp3'],
                'case':['target'],
               'time':[(300,700)],
                'frequency':[(3.0,7.0)],
               'channels':['FZ','PZ','CZ','F3','F4','C3','C4','P3','P4'],
               'power':['total']}
    def_order = ['experiment','case','frequency','time','power','channels']
    
    
    def __init__(s,description={}, fields = None, implicit_values = {}, sep='_' ):
        s.in_sep = sep
        s.units = {}
        s.implicit_values = implicit_values
        
        if fields is None:
            fields = s.def_order
        
        if type(description) == list:
            s.in_cols = description.copy()
            parse_out = parse_col_group(s.in_cols, fields, sep='_')
            if type(parse_out) != tuple:
                print(parse_out)
                return
            else:
                field_order, options, units = parse_out
            s.units = units
        # need to normalize custom description
        
        s.parsed_desc = { fd:desc for fd,desc in zip(field_order,options) }
        
        s.parsed_order = field_order 
        
    def produce_column_names(s,order=def_order, units=False, dec_funs={'frequency':"%.1f", 'time':"%.0f"},
                            translators = {'experiment':str.lower}, sep='_'):
        '''Output list of column names with 
            translators is nested dict by fields and values
        '''
        if order == 'parsed':
            order = s.parsed_order
        
        # note input column order 
        sort_fields = [ f for f,opts in s.parsed_desc.items() if len(opts)>1 ]   
        if 'in_cols' in dir(s):
            in_orders = {fd:[] for fd in sort_fields}
            in_sort = []
            for incol in s.in_cols:
                parsed = { fd:val for fd,val in zip(s.parsed_order,incol.split(s.in_sep)) }
                for sf in sort_fields:
                    in_sort.append( tuple([ parsed[fd] for fd in sort_fields ]) )

        full_desc = s.field_desc.copy()
        full_desc.update(s.implicit_values)
        full_desc.update(s.parsed_desc)
        #print(full_desc)
        
        opt_lens = [ len(v) for k,v in  full_desc.items() ]
        n_cols = np.product(opt_lens)
        Wcols = n_cols*[ '' ]
        out_sorts = [ [] for n in range(n_cols) ]
        joiner = ''
        for field in order:
            translator = None
            if field in translators:
                translator = translators[field]
            if field in s.parsed_desc:
                opts = s.parsed_desc[field]
            else:
                opts = full_desc[field]
            if field in ['time','frequency']:
                nums_st = []
                for v in opts[0]:
                    #print(v,  "dec_places[field])+"f")
                    nums_st.append( dec_funs[field] % v )
                opt_st = '-'.join(nums_st)
                if units and field in s.units:
                    opt_st = opt_st+s.units[field]
                optL = [opt_st]
                
            else:
                optL = opts
            if len(optL) == 1:
                optL = n_cols*optL

            #print(optL)
            cols = []; ind = -1
            for cs,opt in zip(Wcols,optL):
                ind+=1
                if field in sort_fields:
                    out_sorts[ind].append(opt)
                if type(translator) in [method_descriptor_type, function_type]:
                    opt = translator(opt)
                elif type(translator) == dict:
                    if opt in translator:
                        opt = translator[opt]
                    
                cols.append(cs+joiner+opt)
            Wcols = cols.copy()
            joiner = sep
        
        # preserve input column order
        if 'in_cols' in dir(s):
            out_cols = []
            out_sortD = { tuple(osl):col for osl,col in zip(out_sorts,cols) }
            for sort_tup in in_sort:
                out_cols.append( out_sortD[sort_tup] )

        else: out_cols = cols

        return out_cols

'''
Usage example for post processing tools:

v4_sep16_old = pd.read_csv('/active_projects/mort/custom/SmokescreenGWAS_ERO_v4_update_9-2016.csv',
                             na_values=['.'], converters={'ID':str})
v4_sep16_old.set_index(['ID','session'], inplace=True)


v4_ERO_power_frames = {}
for group,phenos in v4_pheno_groups.items():
    tf_wins = [ tuple( [ float(val) for val in \
                pheno[2].split('-') + pheno[1].split('-') ] ) \
                for pheno in phenos ]
    print( group, tf_wins)
    mat_lst = list(v4_ses_df[group].values)
    stack = EM.EROStack(mat_lst)
    v4_ERO_power_frames[group] = stack.tf_mean_lowmem_multiwin(tf_wins)
    del stack  


v4_comb_set = v4_sep16_old.copy()
v4_old_groups = find_column_groups(v4_sep16_old.columns)
v4_old_col_set = EM.EROpheno_label_set( next( iter (v4_old_groups.values()) ) )
print(v4_old_col_set.parsed_order)
v4_all_rn = []
for k,df in v4_ERO_power_frames.items():
    exp = k.split('_')[0]
    exp_renames = {}
    if len(df.index.levels) > 2: # drop uniform experiment index
        df.index = df.index.droplevel(2)  
    col_groups = find_column_groups( df.columns, var_ind=0 )
    for cgk,group in col_groups.items():
        this_col_set = EM.EROpheno_label_set( group , implicit_values = {'experiment':[exp],'power':['total']})
        renamed_cols = this_col_set.produce_column_names( order=v4_old_col_set.parsed_order,#order=['experiment','frequency','time','case','power','channels']
                                    translators={'experiment':str.upper,
                                    'power':{'total':'tot-pwr','evoked':'evo-pwr'} })
        rename_dict = { old:new for old,new in zip(group,renamed_cols) }
        exp_renames.update(rename_dict)
    #print(exp_renames)
    dfRN = df.rename(columns=exp_renames)
    v4_all_rn.append(dfRN.copy())
    comb_set = comb_set.combine_first(dfRN)
    c_cols = comb_set.columns
    u_cols = dfRN.columns
    print( len(c_cols), len(u_cols), len(set(c_cols).intersection(u_cols)) )
v4_comb_set.describe()

v4_full_new = pd.concat(v4_all_rn,join='inner',axis=1)


'''