''' tools to aid batch processing of ERO data, specifically aimed at
    converting ST-inverse mats from Niklas' pipeline into a newer,
    3D representation of the ERO results '''

import os
import time
import subprocess
from collections import defaultdict

from .organization import Mdb

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
# x and y are time window lims, can ignore

# fixed but cannot ignore
# --
# d is power type, should iterate over 1 and 2 for total/evoked
# q is add_baseline, should always be 0
# v is frequencies to iterate, should always be 5 (do 1 frequency)
# p is power_out_type, should always be 1 (linear power)

# variable
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

version_info = {'4': {'ruby script': '/active_projects/ERO_scripts/extract_st_bands_v4.0_custom.rb',
                      'storage path': '/processed_data/ero-mats-V4/'},
                '6': {'ruby script': '/active_projects/ERO_scripts/extract_st_bands_v6.0_custom.rb',
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

# maps expected number of chans to the calculated number
chan_mapping = {'21': '20',
                '32': '32',
                '64': '62'}

powertype_mapping = {'total': 'tot',
                     'evoked': 'evo'}

### Utilities ###

def txt2list(path):
    ''' given path to text file, return a list of its lines '''
    with open(path, 'r') as f:
        lst = [line.strip() for line in f]
    return lst

def gen_path(rec, prc_ver, param_str, raw_chans, exp, case, power_type):
    ''' apply function designed to operate on a dataframe indexed by ID and session.
        given processing version, parameter string, number of channels in the raw data, experiment,
        case, power type, ID, and session, generate the path to the expected 3d ero mat '''

    ID = rec.name[0]
    session = rec.name[1]

    parent_dir = version_info[prc_ver]['storage path']

    # handle the expected number of channels
    if '-s9-' in param_str:
        n_chans = '20'
    else:
        n_chans = chan_mapping[raw_chans]

    path_start = os.path.join(parent_dir, param_str, n_chans, exp)
    fname = '_'.join( [ ID, session, exp, case, powertype_mapping[power_type] ] )
    ext = '.mat'

    path = os.path.join(path_start, fname + ext)

    return path

def gen_path_stdf(rec, power_type):
    ''' apply function designed to operate on a dataframe indexed by ID and session.
        given processing version, parameter string, number of channels in the raw data, experiment,
        case, power type, ID, and session, generate the path to the expected 3d ero mat '''

    try:
        ID = rec.name[0]
        session = rec.name[1]

        parent_dir = version_info[rec['prc_ver']]['storage path']

        # handle the expected number of channels
        if '-s9-' in rec['param_string']:
            n_chans = '20'
        else:
            n_chans = chan_mapping[rec['n_chans']]

        path_start = os.path.join(parent_dir, rec['param_string'], n_chans, rec['experiment'])
        fname = '_'.join( [ ID, session, rec['experiment'], rec['case'], powertype_mapping[power_type] ] )
        ext = '.mat'

        path = os.path.join(path_start, fname + ext)

        return path
    except KeyError:
        print(ID, session, 'had a key missing')
        return None
    except IndexError:
        print(ID, 'had an index missing')

def doc_exists(path):
    ''' given a path to a new ero-mat, determine if a corresponding STinverseMats doc exists '''
    filedir, filename = os.path.split(path)
    dirparts = filedir.split(os.path.sep)

    param_string = dirparts[-3]
    prc_ver = dirparts[-4][-1]
    
    name, ext = os.path.splitext(filename)
    ID, session, experiment, condition, measure = name.split('_')

    query = {'ID': ID, 'session': session, 'experiment': experiment,
             'prc_ver': prc_ver, 'param_string': param_string}

    docs = Mdb['STinverseMats'].find(query)

    if docs.count() > 1:
        print('multiple docs matched', path)
        return True
    elif docs.count() == 1:
        return True
    else:
        return False

def hasbeen_calculated(d):
    ''' given a doc containing info about an STinverseMat, determine if its already been calculated '''

    parent_dir = version_info[d['prc_ver']]['storage path']

    # handle the expected number of channels
    if '-s9-' in d['param_string']:
        n_chans = '20'
    else:
        n_chans = chan_mapping[d['n_chans']]

    path_start = os.path.join(parent_dir, d['param_string'], n_chans, d['experiment'])
    fname_start = '_'.join( [ d['id'], d['session'], d['experiment'], d['case'] ] )
    target_paths = (os.path.join(path_start, fname_start + '_tot.mat'),
                    os.path.join(path_start, fname_start + '_evo.mat'))
    if os.path.exists(target_paths[0]) and os.path.exists(target_paths[1]):
        return True
    else:
        return False

def organize_docs(docs):
    ''' given a mongo cursor of STinverseMats docs, organize them into a dictionary whose keys are 5-tuples of
        (processing version, parameter string, experiment, case, n_chans) combinations and whose values are lists
        of file paths in that category '''
    batch_dict = defaultdict(list)
    for d in docs:
        if not hasbeen_calculated(d):
            batch_dict[(d['prc_ver'],
                        d['param_string'],
                        d['experiment'],
                        d['case'],
                        d['n_chans'])].append(d['path'])
    return batch_dict

def add_stringparams(params, param_str):
    ''' given a params dictionary and a param string, add any parameters that need to be added.
        currently just checks to see if center 9 or not and changes the e param. '''
    if '-s9-' in param_str:
        params.update({'-e': center9_text})
    else:
        params.update({'-e': '1'})
    return params

def make_STlistfile(ver_ps_exp_case_nchans, file_list, limit=None):
    ''' given a batch-identifying 5-tuple, a list of st-inverse-mat files,
        and a file limit: create a text file with those st-inverse-mats,
        and return its path '''

    if limit is None:
        limit = len(file_list)
        lim_flag = ''
    else:
        lim_flag = '_L' + str(limit)

    batch_id = '-'.join([p for p in ver_ps_exp_case_nchans])

    tstamp = str(int(time.time() * 1000))
    list_path = '/processed_data/EROprc_lists/' + \
                batch_id + '_mats-' + tstamp + lim_flag + '.lst'
    with open(list_path, 'w') as list_file:
            list_file.writelines([L + '\n' for L in file_list[:limit]])

    return list_path

### Main functions ###

def create_3dmats(docs, file_lim=None, run_now=False, proc_lim=10):
    ''' given a pymongo cursor of STinverseMat docs, create lists of at most <file_lim> files, organized by
        (ERO version, parameter string, experiment, case, number of chans) tuples. then construct the command-line
        calls to execute those calculations. if run_now is True, administer those calls into a process pool with
        at most proc_lim processes running simultaneously. returns a list of the command-line calls when done '''

    processes = set()
    call_lst = []

    batch_dict = organize_docs(docs)

    for ver_ps_exp_case_nchans, STmat_lst in batch_dict.items():
        
        if file_lim is None:
            n_files = len(STmat_lst)
        else:
            n_files = file_lim

        version, param_str, exp, case, n_chans = ver_ps_exp_case_nchans

        ruby_file = version_info[version]['ruby script']

        params = default_params.copy()
        params = add_stringparams(params, param_str)  # adds -e param (center 9 or not)

        list_file_path = make_STlistfile(ver_ps_exp_case_nchans, STmat_lst, n_files)
        params['-f'] = list_file_path  # adds -f param (file list)

        for pwr_type in ['1', '2']:  # total, evoked

            params['-d'] = pwr_type  # adds -d param (power type)

            paramL = [flag + ' ' + val for flag, val in params.items()]
            paramS = ' '.join(paramL)
            call = [ruby_file, paramS]
            call_string = ' '.join(call)
            print(call_string)

            # queue the process
            call_lst.append(call_string)
            if run_now:
                processes.add(subprocess.Popen(call_string, shell=True))
                time.sleep(2)
                if len(processes) >= proc_lim:
                    os.wait()
                    processes.difference_update(
                        [p for p in processes if p.poll() is not None])
                
    return call_lst


def run_calls(call_lst, proc_lim=10):
    ''' given a list of complete command-line call strings, administer them into a pool of processes.
        use when a list of calls is accessible but the mongo db is not (e.g. on mp3 or mp6). '''
    
    processes = set()

    for call_ind, call_string in enumerate(call_lst):

        processes.add(subprocess.Popen(call_string, shell=True))
        print(call_ind)
        print(call_string)

        time.sleep(2)

        if len(processes) >= proc_lim:
            os.wait()
            processes.difference_update(
                [p for p in processes if p.poll() is not None])


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
