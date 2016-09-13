'''processing based on hbnl database
'''

import organization as O
import compilation as C

import os, time, subprocess
from tqdm import tqdm

### Utilities ###

def read_ero_log(filepath, sep='-----'):
    ''' logs written by extract_st_bands_v???.rb
    '''
    with open(filepath,'r') as of:
        lines = of.read().splitlines()
    entries = [{'parameters':{}}]
    for line in lines:
        if line:
            if line == sep:
                entries.append({'parameters':{}})
            else:
                if line[0] == '-':
                    flag, entry = line.split(':')
                    entries[-1]['parameters'][flag.split(' ')[0]] = entry.strip()
                else:
                    names_vals = [ fd.split(',') for fd in line.split(': ') ]
                    uD = { k:val for k,val in zip(*names_vals) }
                    entries[-1].update(uD)
            
    return entries[:-1]


### ERO mats by study ###
rb_extract_fns_by_ver = {'6':'/active_projects/mort/ERO_scripts/extract_st_bands_v6.0_custom.rb'
                        }
def process_ero_mats_study(study_or_STinvList,run_now=False,file_lim=None,proc_lim=10):
    ''' first input can be either a study name string or a list of STinverseMat file paths
    '''
    # assemble logs to lookup processing parameters
    log_dir = '/processed_data/EROdbLogs/'
    ero_logs = os.listdir(log_dir)

    processes = set()

    if type(study_or_STinvList) == str:
        file_lists_by_exp_case = assemble_file_lists(study_or_STinvList)
    else: 
        file_lists_by_exp_case = assemble_file_lists('',STinv_mats = study_or_STinvList)
    for exp_case,mat_files in tqdm(file_lists_by_exp_case.items()):
        if file_lim is None:
            file_lim = len(mat_files)
            lim_flag = ''
        else: 
            lim_flag = '_L'+str(file_lim)

        ec_st = exp_case[0]+'-'+exp_case[1]
        #need loop over versions here
        version = '6'
        tstamp=str(int(time.time()*1000))
        list_file_path = '/processed_data/EROprc_lists/'+ec_st+'_mats-'+tstamp+lim_flag+'.lst'
        with open(list_file_path,'w') as list_file:
            list_file.writelines( [ L+'\n' for L in mat_files[:file_lim] ] )

        log_path =[ fp for fp in ero_logs if ec_st in fp ][0] #repeats for sites, so take first
        logDs = read_ero_log( os.path.join(log_dir,log_path) )
        #print(ec_st,logDs)
        paramD = logDs[0]['parameters']
        paramD['-f'] = list_file_path
        if '-e' not in paramD:
            paramD['-e'] = '1' # old_elec list

        paramL = [ k+' '+v for k,v in paramD.items() ]
        paramS = ' '.join(paramL)
        call = [ rb_extract_fns_by_ver[version] , paramS ]
        print(' '.join(call))
        if run_now:
            processes.add( subprocess.Popen(' '.join(call), shell=True) )
            if len(processes) >= proc_lim:
                os.wait()
                processes.difference_update(
                    [p for p in processes if p.poll() is not None] )
            #os.system( ' '.join(call) )

        #subprocess.call( call )

def assemble_file_lists(study,STinv_mats = None, existing_paths=None):
    '''
    '''

    if STinv_mats is None:
        subs = C.get_subjectdocs(study)
        ids = [s['ID'] for s in subs]
        inv_mats = list( O.Mdb['STinverseMats'].find( {'id':{'$in':ids}}) ) 
    else: 
        inv_mats = STinv_mats

    exp_cases = set([(d['experiment'],d['case']) for d in inv_mats])
    mat_lists = { ec:[d['path'] for d in inv_mats if '/'+ec[0]+'-'+ec[1]+'/' in d['path'] ]\
                    for ec in exp_cases }

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

HR_ses_ages = [ get_age(s) for s in HRses ] # 1840 in range 17-31, 9 missing sessions(0), 0 others(-1)

HRses_age17t31 = [ ses for ses,age in zip(HRses,HR_ses_ages) if 17 < age < 31 ]

STinv_HR17t31 = []
for ses in HRses_age17t31:
    STs = list( O.Mdb['STinverseMats'].find({'id':ses[0],'session':ses[1]}) )
    STinv_HR17t31.extend( STs )

P.process_ero_mats_study(STinv_HR17t31,run_now=True,proc_lim=12)


'''