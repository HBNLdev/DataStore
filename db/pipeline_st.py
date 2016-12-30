''' tools for batch creation of st-mats from cnt-h1's '''

import os
import subprocess
import time
from collections import defaultdict

import numpy as np

from .compilation import join_collection

proctype_info = {'v40center9': {'ruby script': '/active_projects/ERO_scripts/calc_hdf1_st_v4.0_custom.rb',
                                 'old storage path': '/processed_data/mat-files-v40/',
                                 'storage path': '/processed_data/mat-files-ps-v40/',
                                 'experiments': ['vp3', 'aod', 'ant']},
                 'v60center9': {'ruby script': '/active_projects/ERO_scripts/calc_hdf1_st_v6.0_custom.rb',
                                 'old storage path': '/processed_data/mat-files-v60/',
                                 'storage path': '/processed_data/mat-files-ps-v60/',
                                 'experiments': ['vp3', 'aod', 'ant', 'ans', 'cpt', 'ern', 'err', 'gng', 'stp']},
                 'v60all': {'ruby script': '/active_projects/ERO_scripts/calc_hdf1_st_v6.0_custom.rb',
                             'old storage path': '/processed_data/mat-files-v60/',
                             'storage path': '/processed_data/mat-files-ps-v60/',
                             'experiments': ['vp3', 'aod', 'ant', 'ans', 'cpt', 'ern', 'err', 'gng', 'stp']},
                 }

# ruby file parameter reference

# [-b norm_baseline_type_to_use <1=mean default, 2=rms, 3=geomean, 4=harmmean>] 
# [-c case_number <e.g., 1=tt, 2=nt, 3=nv>] 
# [-e elec_list_type <1=19 default, 2=31, 3=61>] 
# [-g use_grid <0=no default, 1=yes>] 
# [--hi hp_filter <0.05 default>] 
# [--lo lp_filter <55.0 default>]
# [-k pre_stim_time_ms <positive value!>] 
# [-n out_min_trials <10 default>] 
# [-o out_max_trials <100 default>] 
# [-q response_window_max_time_ms <0=old value default>]
# [-s threshold_electrodes <first 31 default, 61, own comma separated electrode list>] 
# [-t threshold_value <-1=threshold off, 0=old value default, e.g. 100>] 
# [-u threshold_min_time_ms <0=old value default; real values, e.g. -100>] 
# [-v threshold_max_time_ms <0=old value default; real values, e.g. 600>] 
# [-y st_baseline_time_min_ms <real values, i.e., negative if pre stimulus>] 
# [-z st_baseline_time_max_ms <real values; i.e., negative if pre stimulus>]\n
# [-f filelist_filename]

# parameters only present in v60

# [--tf accepted_trial_value_files <0=no default, 1=yes>] 
# [--tp trial_amplitude_plot <0=no default, 1=yes>] 
# [-p response_window_min_time_ms <200 default>]

# default parameters across all calls
default_params = {'n': '10',
                  't': '100',
                  'v': '800',
                  'k': '500'}  # NOTE: k parameter is prestimulus time!

# maps from number of channels to the 'e' parameter
nchans_to_eparam = {'21': '1', '32': '2', '64': '3'}

# maps from experiments to their default set of parameters
exp_params = {
    'cpt': {'-hi': '0.3', '-lo': '45', 'n': '15', 'o': '25', 'p': '100', 'u': '-187.5', 'y': '-187.5', 'z': '-50'},
    'ern': {'p': '100'},
    'err': {'n': '15', 'p': '100'},
    'gng': {'-hi': '0.3', '-lo': '45', 'n': '15', 'o': '25', 'p': '100', 'u': '-187.5', 'y': '-187.5', 'z': '-50'},
    'stp': {'-hi': '0.3', '-lo': '45', 'n': '15', 'o': '25', 'p': '100', 'u': '-187.5', 'y': '-187.5', 'z': '-50'},
}
# 'cpt2': {'k': '300', 'n': '15', 'u': '-200', 'v': '1000', 'y': '-300', 'z': '-50'} # second set of cpt params

# maps for experiments to the list of conditions
exp_cases = {'ant': ['a', 'j', 'w', 'p'],  # need renaming (for masscomp files, tttt --> ajwp)
             'aod': ['tt', 'nt'],  # need renaming (t --> tt)
             'vp3': ['tt', 'nt', 'nv'],
             'ans': ['r1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8'],
             'cpt': ['cg', 'c', 'cn', 'un', 'dn', 'dd'],
             'ern': ['n50', 'n10', 'p10', 'p50'],
             'err': ['p', 'n'],
             'gng': ['g', 'ng'],
             'stp': ['c', 'in'], }

# maps experiments to possible #s of chans
exp_nchans = {'ant': ['21', '32', '64'],
              'aod': ['21', '32', '64'],
              'vp3': ['21', '32', '64'],
              'ans': ['32', '64'],
              'ern': ['32', '64'],
              'err': ['32', '64'],
              'gng': ['64'],
              'cpt': ['64'],
              'stp': ['64'], }

# maps from new condition order for ant to the old (fix)
ant_cind_map = {'1': '3', '2': '1', '3': '4', '4': '2'}


def run_calls_cwds(call_edir_lst, proc_lim=10):
    ''' given a list of 2-tuples containing (complete command-line call strings, directories in which to execute them),
        administer them into a pool of processes with at most proc_lim processes at one time. '''

    processes = set()

    call_ind = -1
    for call, edir in call_edir_lst:

        if ~os.path.isdir(edir):
            os.makedirs(edir, mode=0o755, exist_ok=True)

        processes.add(subprocess.Popen(call, cwd=edir, shell=True))
        call_ind += 1
        print(call_ind)
        print(call)

        time.sleep(2)

        if len(processes) >= proc_lim:
            os.wait()
            processes.difference_update(
                [p for p in processes if p.poll() is not None])


def get_stmat_calls(docs, file_lim=None):
    ''' main function. given a cursor of mongo docs and a file limit, create a list of 2-tuples that are
        (command line call string, current working directory in which to execute it).
        this list can be passed to run_calls_cwds below '''

    call_edir_lst = []

    batch_dict = organize_docs(docs)
    culled_bd = cull_batch(batch_dict)

    for pt_rt_exp_case, cnth1_lst in culled_bd.items():

        if file_lim is None:
            n_files = len(cnth1_lst)
        else:
            n_files = file_lim

        proc_type, rec_type, exp, case = pt_rt_exp_case

        params = get_params(proc_type, rec_type, exp, case)

        list_file_path = make_cnth1listfile(pt_rt_exp_case, cnth1_lst, n_files)
        params['f'] = list_file_path  # adds -f param (file list)

        call = construct_call(proc_type, params)
        edir = build_eventualdir(proc_type, rec_type, exp, case)

        call_edir_lst.append((call, edir))

    return call_edir_lst

def add_stpaths(df, proc_type, exp_cases, age='old'):
    ''' given a dataframe indexed by ID and session, a desired ERO processing type,
        and a dict mapping desired experiments to desired cases within each,
        return a dataframe with added columns that locate the corresponding paths to ST-mats,
        if they exist. '''

    uIDs = [ID + '_' + session for ID, session in df.index.tolist()]
    query = {'uID': {'$in': uIDs}, 'experiment': {'$in': list(exp_cases.keys())}}
    proj = {'_id': 0, 'ID': 1, 'session': 1, 'rec_type':1, 'filepath': 1, 'site': 1}
    nchans_df = join_collection(df, 'cnth1s',
                                add_query=query, add_proj=proj,
                                left_join_inds=['ID', 'session'], right_join_inds=['ID', 'session'])

    nchans_df.dropna(subset=['cnt_filepath'], inplace=True)
    groups = nchans_df.groupby(level=nchans_df.index.names)
    nchans_df_nodupes = groups.last()
    df_out = df.join(nchans_df_nodupes[['cnt_rec_type', 'cnt_filepath', 'cnt_site']])

    if age is 'old':
        apply_func = build_oldpath_apply
    else:
        apply_func = build_path_apply

    for exp, cases in exp_cases.items():
        for case in cases:
            apply_args = [proc_type, exp, case]
            pathcol_name = '_'.join([proc_type, exp, case])
            df_out[pathcol_name] = df_out.apply(apply_func, axis=1, args=apply_args)

    return df_out

def build_oldpath_apply(rec, proc_type, exp, case):
    rec_type = rec['cnt_rec_type']
    cnth1_fp = rec['cnt_filepath']
    site = rec['cnt_site']
    try:
        fp = build_old_eventualpath(proc_type, rec_type, exp, case, cnth1_fp, site)
        if os.path.exists(fp):
            return fp
        else:
            return np.nan
    except:
        return np.nan

def build_path_apply(rec, proc_type, exp, case):
    rec_type = rec['cnt_rec_type']
    cnth1_fp = rec['cnt_filepath']
    try:
        fp = build_eventualpath(proc_type, rec_type, exp, case, cnth1_fp)
        if os.path.exists(fp):
            return fp
        else:
            return np.nan
    except:
        return np.nan

def organize_docs(docs):
    ''' given a mongo cursor of cnth1 docs, organize them into a dict where the keys are 4-tuples of
        (processing type, recording type, experiment, case), and the values are lists of 2-tuples of
        (cnth1 filepath, eventual stmat filepath) '''

    batch_dict = defaultdict(list)

    # build the full lists first, then cull them down based on whether the paths exist
    for d in docs:
        try:
            cnth1_fp = d['filepath']
            exp = d['experiment']
        except KeyError:
            print('a doc was missing a key')
        fp_parts = cnth1_fp.split(os.path.sep)
        rec_type = fp_parts[6]
        if rec_type not in ['mc16-21', 'mc16-32', 'ns16-32', 'mc16-64', 'ns16-64', 'ns32-64']:
            print('recording type not recognized for', cnth1_fp)
            continue
        for proc_type in proctype_info.keys():
            if exp not in proctype_info[proc_type]['experiments']:
                continue
            for case in exp_cases[exp]:
                stmat_fp = build_eventualpath(proc_type, rec_type, exp, case, cnth1_fp)
                batch_dict[(proc_type, rec_type, exp, case)].append((cnth1_fp, stmat_fp))

    return batch_dict


def cull_batch(batch_dict):
    ''' given a batch dictionary returned by organized_docs (above), cull it into a new batch dictionary that is same
        except the lists contain only cnth1s for which the corresponding stmat does not yet exist '''

    new_bd = {}
    for k, lst in batch_dict.items():
        new_bd[k] = [cnth1_fp for cnth1_fp, stmat_fp in lst if ~os.path.exists(stmat_fp)]

    return new_bd


def make_cnth1listfile(pt_rt_exp_case, file_list, limit=None):
    ''' given a batch-identifying 4-tuple, a list of cnth-h1 files,
        and a file limit: create a text file with those cnth-h1 files,
        and return its path '''

    if limit is None:
        limit = len(file_list)
        lim_flag = ''
    else:
        lim_flag = '_L' + str(limit)

    batch_id = '-'.join([p for p in pt_rt_exp_case])

    tstamp = str(int(time.time() * 1000))
    list_path = '/active_projects/ERO_scripts/cnth1_lists/' + \
                batch_id + '_cnth1s-' + tstamp + lim_flag + '.lst'
    with open(list_path, 'w') as list_file:
        list_file.writelines([L + '\n' for L in file_list[:limit]])

    return list_path


def construct_call(proc_type, pdict):
    ''' given a processing type and a parameter dictionary, return a full command line call '''

    try:
        program = proctype_info[proc_type]['ruby script']
    except KeyError:
        print('processing type not recognized')
        return

    params_chain = ''
    for p, v in pdict.items():
        params_chain += '-' + p + ' ' + v + ' '

    call = program + ' ' + params_chain

    return call


def get_params(proc_type, rec_type, exp, case):
    ''' given the elements of a batch-unique 4-tuple, create a dict of appropriate parameters.
        note this implements necessary parameters for correct functions, but also uses default params
        defined for each experiment in the dict exp_params '''

    pdict = default_params.copy()

    # channels
    n_chans = rec_type[-2:]
    if proc_type in ['v40center9', 'v60center9']:
        pdict.update({'e': '1'})
        if n_chans == '21':
            pdict.update({'s': '2,4,5,10,11,13,16,18,19'})
        elif n_chans in ['32', '64']:
            pdict.update({'s': '7,8,9,16,17,18,23,24,25'})
        else:
            print('number of channels unexpected')
            return
    elif proc_type == 'v60all':
        try:
            pdict.update({'e': nchans_to_eparam[n_chans]})
        except KeyError:
            print('number of channels unexpected')
            return

    # experiment-specific params
    if exp in exp_params:
        pdict.update(exp_params[exp])

    # condition number
    try:
        case_ind = str(exp_cases[exp].index(case) + 1)
        pdict.update({'c': case_ind})
    except KeyError:
        print('experiment unexpected')
        return
    except ValueError:
        print('case unexpected')
        return

    # fix ANT problem
    if exp == 'ant' and rec_type[:2] == 'mc':
        pdict.update({'c': ant_cind_map[case_ind]})

    # remove p param from v40 calls
    if proc_type == 'v40center9' and 'p' in pdict:
        del pdict['p']

    return pdict

def build_eventualpath(proc_type, rec_type, exp, case, cnth1_fp):
    ''' given processing type, recording type, experiment, case, and cnth1 path, return the eventual st mat path '''

    parent_dir = proctype_info[proc_type]['storage path']
    n_chans = rec_type[-2:]
    param_str = build_paramstr(proc_type, n_chans, exp)
    cnth1_fn = os.path.split(cnth1_fp)[1]
    fname = '.'.join([cnth1_fn, case, 'st', 'mat'])
    fp = os.path.join(parent_dir, rec_type, exp, exp + '-' + case, param_str, fname)
    return fp

def build_old_eventualpath(proc_type, rec_type, exp, case, cnth1_fp, site):
    ''' given processing type, recording type, experiment, case, and cnth1 path, return the eventual st mat path '''

    parent_dir = proctype_info[proc_type]['old storage path']
    n_chans = rec_type[-2:]
    param_str = build_paramstr(proc_type, n_chans, exp)
    cnth1_fn = os.path.split(cnth1_fp)[1]
    fname = '.'.join([cnth1_fn, case, 'st', 'mat'])
    fp = os.path.join(parent_dir, rec_type, exp, exp + '-' + case, param_str, site, fname)
    return fp

def build_eventualdir(proc_type, rec_type, exp, case):
    ''' given processing type, recording type, experiment, case, and cnth1 path, return the eventual st mat dir '''

    parent_dir = proctype_info[proc_type]['storage path']
    n_chans = rec_type[-2:]
    param_str = build_paramstr(proc_type, n_chans, exp)

    edir = os.path.join(parent_dir, rec_type, exp, exp + '-' + case, param_str)

    return edir


def build_paramstr(proc_type, n_chans, exp):
    ''' given processing type, # of channels, and experiment, return correct parameter string '''

    if proc_type == 'v40center9':
        return 'e1-n10-s9-t100-v800'
    if proc_type == 'v60center9':
        ps = 'e1'
    elif proc_type == 'v60all':
        ps = 'e' + nchans_to_eparam[n_chans]
    else:
        print('processing type not recognized, the following are acceptable:')
        print(['v40center9', 'v60center9' 'v60all'])
        return

    if exp in ['ans', 'ant', 'aod', 'vp3']:
        ps += '-n10-s9-t100-v800'
    elif exp in ['cpt', 'gng', 'stp']:
        ps += '-hi03-lo45-k187-n15-o25-p100-s9-t100-u187-y187-z50'
    elif exp == 'ern':
        ps += '-n10-p100-s9-t100-v800'
    elif exp == 'err':
        ps += '-n15-p100-s9-t100-v800'
    else:
        print('experiment not recognized')
        return

    if proc_type == 'v60all':
        ps = ps.replace('-s9-', '-')

    return ps
