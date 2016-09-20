''' tools to aid batch processing of ERO data, specifically aimed at
    converting ST-inverse mats from Niklas' pipeline into a newer,
    3D representation of the ERO results '''

import organization as O
import compilation as C

import os
import time
import subprocess
from tqdm import tqdm

### Info ###

ruby_scripts_byversion = {'6': ('/active_projects/mort/ERO_scripts/'
                                'extract_st_bands_v6.0_custom.rb') }

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
                          '4':'high frequencies', 'custom':'freq_file'} },
        'x': {'name': 'min_win_time_ms','options': { 'int':'300 default'} },
        'y': {'name': 'max_win_time_ms','options': { 'int':'500 default'} }

        }

log_dir = '/processed_data/EROdbLogs/'
ero_logs = [ p for p in os.listdir(log_dir) if 'test' not in p]





### Utilities ###


def read_ero_log(filepath, sep='-----'):
    ''' given an ERO log, returns the processing parameters'''

    with open(filepath, 'r') as of:
        lines = of.read().splitlines()
    entries = [{'parameters': {}}]
    for line in lines:
        if line:
            if line == sep:
                entries.append({'parameters': {}})
            else:
                if line[0] == '-':
                    flag, entry = line.split(':')
                    entries[-1]['parameters'][flag.split(' ')[0]] = \
                        entry.strip()
                else:
                    names_vals = [fd.split(',') for fd in line.split(': ')]
                    uD = {k: val for k, val in zip(*names_vals)}
                    entries[-1].update(uD)

    return entries[:-1]

def make_file_list_file(name,file_list,limit=None):

    if limit is None:
        limit = len(file_list)
        lim_flag = ''
    else:
        lim_flag = '_L' + str(limit)

    tstamp = str(int(time.time() * 1000))
    file_path = '/processed_data/EROprc_lists/' + \
        name + '_mats-' + tstamp + lim_flag + '.lst'
    with open(file_path, 'w') as list_file:
            list_file.writelines([L + '\n' for L in file_list[:limit]])

    return file_path


### Main functions ###

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

        for param_set in params_by_exp_case[ exp_case ]:

            for pwr_type in ['1','2']: #total, evoked

                params = [ (p[0],p[1]) for p in list(param_set)]

                params.append( ('-d',pwr_type) )

                list_file_path = make_file_list_file(ec_st, mat_files, file_lim)
                params.append( ('-f', list_file_path) )

                paramL = [flag + ' ' + val for flag, val in params]
                paramS = ' '.join(paramL)
                call = [ruby_scripts_byversion[version], paramS]
                print(' '.join(call))

                # queue the process
                if run_now:
                    processes.add(subprocess.Popen(' '.join(call), shell=True))
                    if len(processes) >= proc_lim:
                        os.wait()
                        processes.difference_update(
                            [p for p in processes if p.poll() is not None])


def process_ero_mats_study(study_or_STinvList, run_now=False,
                           file_lim=None, proc_lim=10):
    ''' given a sample string or list of STinverseMat file paths,
        convert the corresponding ST inverse mats to new ERO mats '''

    # assemble logs to lookup processing parameters
    log_dir = '/processed_data/EROdbLogs/'
    ero_logs = os.listdir(log_dir)

    processes = set()

    # assemble file lists
    if type(study_or_STinvList) == str:
        file_lists_by_exp_case = assemble_file_lists(study_or_STinvList)
    else:
        file_lists_by_exp_case = assemble_file_lists(
            '', STinv_mats=study_or_STinvList)

    # for each experiment, case combo
    for exp_case, mat_files in tqdm(file_lists_by_exp_case.items()):
        if file_lim is None:
            file_lim = len(mat_files)
            lim_flag = ''
        else:
            lim_flag = '_L' + str(file_lim)

        ec_st = exp_case[0] + '-' + exp_case[1]

        # need loop over versions here
        version = '6'
        tstamp = str(int(time.time() * 1000))
        list_file_path = '/processed_data/EROprc_lists/' + \
            ec_st + '_mats-' + tstamp + lim_flag + '.lst'
        with open(list_file_path, 'w') as list_file:
            list_file.writelines([L + '\n' for L in mat_files[:file_lim]])

        # read logs to determine processing parameters
        # repeats for sites, so just take first
        log_path = [fp for fp in ero_logs if ec_st in fp][0]
        logDs = read_ero_log(os.path.join(log_dir, log_path))

        paramD = logDs[0]['parameters']
        paramD['-f'] = list_file_path
        if '-e' not in paramD:
            paramD['-e'] = '1'  # old_elec list

        # construct command
        paramL = [k + ' ' + v for k, v in paramD.items()]
        paramS = ' '.join(paramL)
        call = [ruby_scripts_byversion[version], paramS]
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
        subs = C.get_subjectdocs(study)
        ids = [s['ID'] for s in subs]
        inv_mats = list(O.Mdb['STinverseMats'].find({'id': {'$in': ids}}))
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
