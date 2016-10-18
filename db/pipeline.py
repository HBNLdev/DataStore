''' tools to aid batch processing of ERO data, specifically aimed at
    converting ST-inverse mats from Niklas' pipeline into a newer,
    3D representation of the ERO results '''

import os
import time
import subprocess
from collections import defaultdict

from tqdm import tqdm

from .organization import Mdb
from .compilation import get_subjectdocs


# processing parameters cheatsheet

# v4.0_center9  = prc_ver='4', e4-n10-s9-t100-v800
# v6.0_all      = prc_ver='6', e1-n10-t100-v800
# v6.0_center9  = prc_ver='6', e4-n10-s9-t100-v800

# for reference, here is the v6-all part of the bash script

# /active_projects/mort/ERO_scripts/extract_st_bands_$version.rb
# -d $dd -o new -p 1 -q 0 -v /export/home/mort/projects/freq-for_$tw.txt
# -x $t1 -y $t2 -e 1 -f $tmp.3

# v6-center9 part of bash script

# /active_projects/mort/ERO_scripts/extract_st_bands_$version.rb
# -d $dd -o new -p 1 -q 0 -v /export/home/mort/projects/freq-for_$tw.txt
# -x $t1 -y $t2 -e /export/home/mort/projects/electrodes-for_21.txt -f $tmp.3

# v4-center9 part of bash script

# /active_projects/mort/ERO_scripts/extract_st_bands_$version.rb
# -d $dd -o new -p 1 -q 0 -v /export/home/mort/projects/freq-for_$tw.txt
# -x $t1 -y $t2 -e /export/home/mort/projects/electrodes-for_21.txt -f $tmp.3

# notice the only difference is the text file

# logfile="/vol01/processed_data/EROdbLogs/csv-files-$versionshort-$Type-$expns-$subfolder-$parameter-$site-$tw-$pwr.log"
# echo "logging to $logfile"
# echo "process date, write time: $today,$tmp" >> $logfile
# echo "-d power_type: $dd" >> $logfile
# echo "-p calc_type: 1" >> $logfile
# echo "-q add_baseline: 0" >> $logfile
# echo "-v curve_band: $tw" >> $logfile
# echo "-x min_win_time: $t1" >> $logfile
# echo "-y max_win_time: $t2" >> $logfile
# echo "-----" >> $logfile

# can ignore
# --
# o is output text, can ignore
# p is power_out_type, should always be 1 (linear power)
# x and y are time window lims, can ignore

# fixed but cannot ignore
# --
# d is power type, should iterate over 1 and 2 for total/evoked
# q is add_baseline, should always be 0
# v is frequencies to iterate, should always be 5 (do 1 frequency)

# variable?
# --
# e is electrode list:
#   in v6-all, it's always 1
#   in v6-center9, it's a static text file path
#   in v4-center9, it's a static text file path
# f is st_mat_filelist_filename, a text file listing the stmats to use

default_params = {'-p': '1',
                  '-d': None,
                  '-q': '0',
                  '-v': '5',
                  '-e': None,
                  '-f': None
                  }

### Info ###

version_info = {'4': {'ruby script': '/active_projects/mort/ERO_scripts/extract_st_bands_v4.0_custom.rb',
                      'storage path': '/processed_data/ero-mats-V4/'},
                '6': {'ruby script': '/active_projects/mort/ERO_scripts/extract_st_bands_v6.0_custom.rb',
                      'storage path': '/processed_data/ero-mats-V6/'},
                }

center9_text = '/export/home/mort/projects/electrodes-for_21.txt'

extract_st_bands_params = {
        'b': {'name': 'apply_baseline','options': { '0':'no default', '1':'yes'} },
        'd': {'name': 'st_type_to_use','options': { '1':'total default', '2':'evoked',
                         '3':'total-evoked'} },
        'e': {'name': 'electrode_list_type','options': { '1':'old elec_list default',
                         '2':'rows across head', '3':'rows within 6 regions', 
                         'custom':'own elec_list'} },
        'f': {'name': 'st_mat_filelist_filename'},      
        'm': {'name': 'calc_val_type ','options': { '1':'mean default', '2':'max',
                         '3':'centroid', '4':'maxfreq', '5':'maxtime', '6':'sum'} },
        'o': {'name': 'output_text'}, #,'options': { 'additional information in the ps-file name'} },
        'p': {'name': 'power_out_type','options': { '1':'power (lin) default',
                         '2':'amplitude (lin)', '3':'power (ln)', '4':'amplitude (ln)'} },
        'q': {'name': 'add_baseline_values','options': { '0':'no', '1':'yes default'} },
        'v': {'name': 'curve_band_type','options': { '1':'bands default',
                         '2':'single frequencies', '3':'low frequencies',
                          '4':'high frequencies', '5': 'one frequency',
                          'custom':'freq_file'} },
        'x': {'name': 'min_win_time_ms','options': { 'int':'300 default'} },
        'y': {'name': 'max_win_time_ms','options': { 'int':'500 default'} }

        }



### Utilities ###

def txt2list(path):
    ''' given path to text file, return a list of its lines '''
    with open(path, 'r') as f:
        lst = [line.strip() for line in f]
    return lst


def make_STlistfile(exp_case, file_list, limit=None):
    ''' given an experiment-case name, a list of st-inverse-mat files,
        and a file limit, create a text file with those st-inverse-mats,
        and return its path '''

    if limit is None:
        limit = len(file_list)
        lim_flag = ''
    else:
        lim_flag = '_L' + str(limit)

    tstamp = str(int(time.time() * 1000))
    list_path = '/processed_data/EROprc_lists/' + \
                exp_case + '_mats-' + tstamp + lim_flag + '.lst'
    with open(list_path, 'w') as list_file:
            list_file.writelines([L + '\n' for L in file_list[:limit]])

    return list_path

def hasbeen_calculated(d):
    ''' given a doc containing info about an STinverseMat, determine if its already been calculated '''

    parent_dir = version_info[d['prc_ver']]['storage path']
    path_start = os.path.join(parent_dir, d['param_string'], d['n_chans'], d['experiment'])
    fname_start = '_'.join( [ d['id'], d['session'], d['experiment'], d['case'] ] )
    target_paths = (os.path.join(path_start, fname_start + '_tot.mat'),
                    os.path.join(path_start, fname_start + '_evo.mat'))
    if os.path.exists(target_paths[0]) and os.path.exists(target_paths[1]):
        return True
    else:
        return False

def organize_docs(docs):
    ''' given a mongo cursor of STinverseMats docs, organize them into a dictionary whose keys are 4-tuples of
        (processing version, parameter string, experiment, case) combinations and whose values are lists
        of file paths in that category '''
    batch_dict = defaultdict(list)
    for d in docs:
        if not hasbeen_calculated(d):
            batch_dict[(d['prc_ver'],
                        d['param_string'],
                        d['experiment'],
                        d['case'])].append(d['path'])
    return batch_dict

def add_stringparams(params, param_str):
    ''' given a params dictionary and a param string, add any parameters that need to be added.
        currently just checks to see if center 9 or not and changes the e param. '''
    if '-s9-' in param_str:
        params.update({'-e': center9_text})
    else:
        params.update({'-e': '1'})
    return params

### Main functions ###

def create_3dmats(docs, run_now=False, file_lim=None, proc_lim=10):

    processes = set()

    batch_dict = organize_docs(docs)

    for ver_ps_exp_case, STmat_lst in batch_dict.items():
        if file_lim is None:
            file_lim = len(STmat_lst)
            lim_flag = ''
        else:
            lim_flag = '_L' + str(file_lim)

        if not STmat_lst:  # if empty, continue to next
            continue

        version, param_str, exp, case = ver_ps_exp_case

        ruby_file = version_info[version]['ruby script']

        params = default_params.copy()
        params = add_stringparams(params, param_str)  # adds -e param (center 9 or not)

        ec_st = exp + '-' + case
        list_file_path = make_STlistfile(ec_st, STmat_lst, file_lim)
        params['-f'] = list_file_path  # adds -f param (file list)

        for pwr_type in ['1', '2']:  # total, evoked

            params['-d'] = pwr_type  # adds -d param (power type)

            paramL = [flag + ' ' + val for flag, val in params.items()]
            paramS = ' '.join(paramL)
            call = [ruby_file, paramS]
            print(' '.join(call))

            # queue the process
            if run_now:
                processes.add(subprocess.Popen(' '.join(call), shell=True))
                if len(processes) >= proc_lim:
                    os.wait()
                    processes.difference_update(
                        [p for p in processes if p.poll() is not None])


def process_ero_mats_all_params( STinvList, params_by_exp_case, run_now=False,
                         file_lim=None, proc_lim=10, version='6' ):
    ''' convert a list of STinverseMat documents to ero mats, using lists of 
        parameters for each based on the experiment and case
    '''
    processes = set()
    
    file_lists_by_exp_case = assemble_file_lists('',STinvList)

    for exp_case, mat_files in tqdm(file_lists_by_exp_case.items()):
        if file_lim is None:
            file_lim = len(mat_files)
            lim_flag = ''
        else:
            lim_flag = '_L' + str(file_lim)

        ec_st = exp_case[0] + '-' + exp_case[1]
        # doesn't need to be a loop like this
        for param_set in params_by_exp_case[ exp_case ]: # but would have to assemble params elsehow

            for pwr_type in ['1','2']: #total, evoked

                params = [ (p[0],p[1]) for p in list(param_set)] # should be dict?

                params.append( ('-d',pwr_type) )

                list_file_path = make_STlistfile(ec_st, mat_files, file_lim)
                params.append( ('-f', list_file_path) )

                paramL = [flag + ' ' + val for flag, val in params]
                paramS = ' '.join(paramL)
                call = [rubyscripts_byversion[version], paramS]
                print(' '.join(call))

                # queue the process
                if run_now:
                    processes.add(subprocess.Popen(' '.join(call), shell=True))
                    if len(processes) >= proc_lim:
                        os.wait()
                        processes.difference_update(
                            [p for p in processes if p.poll() is not None])


def assemble_file_lists(study, STinv_mats=None, existing_paths=None):
    ''' given a study string or a list of STinverseMat docs, create a dict
        of lists where the key is an (experiment, case) tuple,
        and the value is a list of the corresponding ST-inverse mats '''

    if STinv_mats is None:
        subs = get_subjectdocs(study)
        ids = [s['ID'] for s in subs]
        inv_mats = list(Mdb['STinverseMats'].find({'id': {'$in': ids}}))
    else:
        inv_mats = STinv_mats

    exp_cases = set([(d['experiment'], d['case']) for d in inv_mats])
    mat_lists = {ec:
                 [d['path'] for d in inv_mats
                  if '/' + ec[0] + '-' + ec[1] + '/' in d['path']]
                 for ec in exp_cases}

    # if existing_paths is not None:
    #     for ec,lst in mat_lists:
    #         mat_lists[ec] =  [ p for p in lst if ]

    return mat_lists

'''
code for compiling custom list for High Risk sample:

HRsubs = C.get_subjectdocs('HighRisk')
HRids = [s['ID'] for s in HRsubs]
HRinv_mats = list( O.Mdb['STinverseMats'].find( {'id':{'$in':HRids}}) )
HRses = set([ (im['id'], im['session']) for im in HRinv_mats ]) # 3889 sessions

def get_age( id_ses ):
    docs = list(O.Mdb['sessions'].find( {'ID':id_ses[0],'session':id_ses[1]} ))
    if len(docs) == 1:
        return docs[0]['age']
    elif len(docs) == 0:
        return 0
    else:
        return -1

# 1840 in range 17-31, 9 missing sessions(0), 0 others(-1)
HR_ses_ages = [ get_age(s) for s in HRses ]

HRses_age17t31 = [ ses for ses,age in zip(HRses,HR_ses_ages) if 17 < age < 31 ]

STinv_HR17t31 = []
for ses in HRses_age17t31:
    STs = list( O.Mdb['STinverseMats'].find({'id':ses[0],'session':ses[1]}) )
    STinv_HR17t31.extend( STs )

P.process_ero_mats_study(STinv_HR17t31,run_now=True,proc_lim=12)


'''

# def process_ero_mats_study(study_or_STinvList, run_now=False,
#                            file_lim=None, proc_lim=10):
#     ''' given a sample string or list of STinverseMat file paths,
#         convert the corresponding ST inverse mats to new ERO mats '''

#     # assemble logs to lookup processing parameters
#     log_dir = '/processed_data/EROdbLogs/'
#     ero_logs = os.listdir(log_dir)

#     processes = set()

#     # assemble file lists
#     if type(study_or_STinvList) == str:
#         file_lists_by_exp_case = assemble_file_lists(study_or_STinvList)
#     else:
#         file_lists_by_exp_case = assemble_file_lists(
#             '', STinv_mats=study_or_STinvList)

#     # for each experiment, case combo
#     for exp_case, mat_files in tqdm(file_lists_by_exp_case.items()):
#         if file_lim is None:
#             file_lim = len(mat_files)
#             lim_flag = ''
#         else:
#             lim_flag = '_L' + str(file_lim)

#         ec_st = exp_case[0] + '-' + exp_case[1]

#         # need loop over versions here
#         version = '6'
#         tstamp = str(int(time.time() * 1000))
#         list_file_path = '/processed_data/EROprc_lists/' + \
#             ec_st + '_mats-' + tstamp + lim_flag + '.lst'
#         with open(list_file_path, 'w') as list_file:
#             list_file.writelines([L + '\n' for L in mat_files[:file_lim]])

#         # read logs to determine processing parameters
#         # repeats for sites, so just take first
#         log_path = [fp for fp in ero_logs if ec_st in fp][0]
#         logDs = read_ero_log(os.path.join(log_dir, log_path))

#         paramD = logDs[0]['parameters']
#         paramD['-f'] = list_file_path
#         if '-e' not in paramD:
#             paramD['-e'] = '1'  # old_elec list

#         # construct command
#         paramL = [k + ' ' + v for k, v in paramD.items()]
#         paramS = ' '.join(paramL)
#         call = [ruby_scripts_byversion[version], paramS]
#         print(' '.join(call))

#         # queue the process
#         if run_now:
#             processes.add(subprocess.Popen(' '.join(call), shell=True))
#             if len(processes) >= proc_lim:
#                 os.wait()
#                 processes.difference_update(
#                     [p for p in processes if p.poll() is not None])
