'''EEGmaster
'''

import numbers
import os
from datetime import datetime

import pandas as pd

from .utils.compilation import ID_nan_strint
from .utils.dates import calc_date_w_Qs
from .utils.filename_parsing import site_fromIDstr

master_path = '/processed_data/master-file/EEG-master-file-29.csv'
access_path = '/processed_data/master-file/masterFile_10-2018_cl.csv'

subjects_sparser_sub = \
    ['famID', 'mID', 'fID', 'DNA', 'rel2pro', 'famtype', 'POP',
     'DOB', 'twin', 'EEG', 'System', 'Wave12', 'Wave12-fam',
     'fMRI subject', 'Wave3', 'Phase4-session', 'Phase4-testdate',
     'Phase4-age', '4500', 'ccGWAS', 'AAfamGWAS', 'ExomeSeq',
     'EAfamGWAS', 'EAfamGWAS-fam', 'wave12-race', '4500-race',
     'ccGWAS-race', 'core-race', 'COGA11k-fam', 'COGA11k-race',
     'COGA11k-fam-race', 'ruID', 'genoID', 'SmS', 'CA/CO',
     'a-session', 'b-session', 'c-session', 'd-session',
     'e-session', 'f-session', 'g-session', 'h-session',
     'i-session', 'j-session', 'k-session',
     'a-raw', 'b-raw', 'c-raw', 'd-raw', 'e-raw', 'f-raw', 'g-raw',
     'h-raw', 'i-raw', 'j-raw', 'k-raw', 'missing-EEG' 'remarks']

subjects_sparser_add = \
    ['ID', 'sex', 'handedness', 'Self-reported-race', 'alc_dep_dx',
     'alc_dep_ons', 'a-age', 'b-age', 'c-age', 'd-age', 'e-age',
     'f-age', 'g-age', 'h-age', 'i-age', 'j-age', 'k-age',
     'famID', 'mID', 'fID', 'POP', 'alc_dep_dx_f', 'alc_dep_dx_m']

session_sadd = [field for field in subjects_sparser_add if 'age' not in field]
session_sadd.extend(['session', 'followup', 'age', 'date'])

def load_access():
    return pd.read_csv(access_path,converters={'ID':str},na_values=['.'])

def load_master(preloaded=None, force_reload=False, custom_path=None):
    ''' load the most current HBNL master file as a pandas dataframe '''

    master = None

    if type(preloaded) == pd.core.frame.DataFrame and not custom_path:
        master = preloaded
        return

    if custom_path:
        master_path_use = custom_path
    else:
        master_path_use = master_path

    if not type(master) == pd.core.frame.DataFrame or force_reload:

        # read as csv
        master = pd.read_csv(master_path_use,
                             converters={'ID': ID_nan_strint, 'famID': ID_nan_strint,
                                         'mID': ID_nan_strint, 'fID': ID_nan_strint,
                                         'genoID': ID_nan_strint,
                                         'EAfamGWAS-fam': ID_nan_strint,
                                         'COGA11k-fam': ID_nan_strint},
                             na_values='.', low_memory=False)
        master.set_index('ID', drop=False, inplace=True)

        for dcol in ['DOB'] + [col for col in master.columns if '-date' in col]:
            master[dcol] = master[dcol].map(calc_date_w_Qs)

        master['site'] = master['ID'].apply(site_fromIDstr)

        for pcol in ['mID', 'fID']:
            jDF = pd.merge(master[['ID', pcol]], master[['ID', 'alc_dep_dx']],
                           how='left', left_on=[pcol], right_on=['ID'])
            jDF.set_index('ID_x', inplace=True)
            jDF.index.names = ['ID']
            master['alc_dep_dx_' + pcol[0]] = jDF['alc_dep_dx']

    # check date modified on master file
    master_mtime = datetime.fromtimestamp(os.path.getmtime(master_path_use))

    return master, master_mtime


def masterYOB():
    '''Writes _YOB version of master file in same location
    '''
    master, mod_time = load_master()
    masterY = master.copy()
    masterY['DOB'] = masterY['DOB'].apply(lambda d: d.year)
    masterY.rename(columns={'DOB': 'YOB'}, inplace=True)

    outname = master_path.replace('.csv', '_YOB.csv')
    masterY.to_csv(
        outname, na_rep='.', index=False, date_format='%m/%d/%Y', float_format='%.5f')

    return masterY


def ids_with_exclusions(master):
    return master[master['no-exp'].notnull()]['ID'].tolist()


def excluded_experiments(master, ID):
    ex_str = master.ix[ID]['no-exp']
    excluded = []
    if type(ex_str) == str:
        excluded = [tuple(excl.split('-')) for excl in ex_str.split('_')]
    return excluded


def subjects_for_study(study, return_series=False):
    '''for available studies, see study_info module
    '''

    study_symbol_dispatch = {'Wave12': ('Wave12', 'x'),
                             'COGA4500': ('4500', 'x'),  # could also be 'x-'
                             'ccGWAS': ('ccGWAS', 'ID number'),
                             'EAfamGWAS': ('EAfamGWAS', 'x'),
                             'COGA11k': ('COGA11k-fam', 'ID number'),
                             'ExomeSeq': ('ExomeSeq', 'x'),
                             'AAfamGWAS': ('AAfamGWAS', 'x'),
                             'PhaseIV': ('Phase4-session', ('a', 'b', 'c', 'd')),
                             'fMRI-NKI-bd1': ('fMRI', ('1a', '1b')),
                             'fMRI-NKI-bd2': ('fMRI', ('2a', '2b')),
                             'fMRI-NYU-hr': ('fMRI', ('3a', '3b')),
                             'a-subjects': 'use ID',
                             'c-subjects': 'use ID',
                             'h-subjects': 'use ID',
                             'p-subjects': 'use ID',
                             'smokeScreen': ('SmS', 'x')
                             }

    study_identifier_info = study_symbol_dispatch[study]
    if study_identifier_info == 'use ID':
        # works for -subjects studies
        id_series = master['ID'].str.contains(study[0])
    elif study_identifier_info[1] == 'ID number':
        id_series = master[study_identifier_info[0]] > 0
    elif type(study_identifier_info[1]) == tuple:
        id_series = master['ID'] == 'dummy'
        for label in study_identifier_info[1]:
            id_series = id_series | (master[study_identifier_info[0]] == label)
    else:
        id_series = master[
                        study_identifier_info[0]] == study_identifier_info[1]

    if return_series:
        return id_series
    else:
        return master.ix[id_series]['ID'].tolist()


def frame_for_study(master, study):
    id_series = subjects_for_study(study)

    study_frame = master.ix[id_series]

    return study_frame


session_letters = 'abcdefghijk'


def sessions_for_subject_experiment(master, subject_id, experiment):
    sessions = []
    subject = master.ix[subject_id]
    excluded = excluded_experiments(subject_id)
    next_session = True
    ses_ind = 0
    while not pd.isnull(next_session) and ses_ind < len(session_letters):
        ses_letter = session_letters[ses_ind]
        next_session = subject[ses_letter + '-run']
        if not pd.isnull(next_session) and (ses_letter, experiment) not in excluded:
            sessions.append(
                (next_session, subject[ses_letter + '-raw'], subject[ses_letter + '-date']))
        ses_ind += 1

    return sessions


def protect_dates(filepath):
    path, filename = os.path.split(filepath)
    newname = filename.replace('.', '_protectedDates.')

    inf = open(filepath)
    outf = open(os.path.join(path, newname), 'w')
    for line in inf:
        lparts = line.split(',')
        newline = []
        for part in lparts:
            if '/' in part:
                newline.append("'" + part)
            else:
                newline.append(part)
        newline = ','.join(newline)
        outf.write(newline)

    inf.close()
    outf.close()


# columnn check functions

def famID_ck(stck):
    try:
        assert len(stck) == 5
        assert stck[0] in '1234567'
        return True
    except AssertionError:
        return False


def selfReported_ck(stck):
    try:
        assert stck[0] in 'hnu'
        assert stck[1] in '1234689'
    except AssertionError:
        return False


column_guides = {
    'ID': ['start in', '1234567achp'],
    'famID': famID_ck,
    'mID': ['start in', '1234567'],
    'fID': ['start in', '1234567'],
    'DNA': ['in', 'x'],
    'rel2pro': ['in', ['bil', 'fath', 'gof', 'hsib', 'm1c', 'm1c1r', 'ma', 'mate', 'mgf', 'mgm',
                       'mgn', 'mhs', 'mu', 'moth', 'niece', 'neph', 'off', 'olaw', 'p1c', 'p1c1r',
                       'pa', 'pgf', 'pgm', 'pgn', 'phs', 'pu', 'self', 'sib', 'sil', 'slaw']],
    'famtype': ['in', ['PILOT', 'I', 'IV', 'CONTROL', 'I-S-III', 'I-S-IV', 'II', 'L-II',
                       'III', 'IV-H', 'I-S-II', 'A-I', 'KIDS', 'I-S-IV-H', 'A-II', 'IRPG',
                       'IRPG-CTRL', 'L-IV', 'CTRL-K', 'CL-IV', 'A-IV-H', 'A-III',
                       'CL-SPEC', 'SPEC', 'CHAL-CON']],
    'POP': ['in', ['Pilot', 'COGA', 'COGA-Ctl', 'Relia', 'IRPG', 'IRPG-Ctl', 'Alc-chal',
                   'A', 'C', 'H', 'P']],
    'sex': ['in', ['m', 'f', '(m)', '(f)']],
    'handedness': ['in', ['b', 'l', 'r', 'w', 'u']],
    'DOB': 'date',
    'twin': ['in', [0, 1, 2]],
    'EEG': ['in', ['x', 'e', '-']],
    'system': ['in', ['es', 'es-ns', 'mc', 'mc-es', 'mc-ns', 'ns']],
    'Wave12': ['in', ['P', 'p', 'x']],
    'Wave12-fam': ['in', [1, 2]],
    'fMRI': ['in', ['1a', '1b']],
    'Wave3': ['in', ['x', 'rm']],
    'Phase4-session': ['in', ['a', 'b', 'c', 'd', 'p', 'pa', 'pb', 'x']],
    'Phase4-testdate': ['date'],
    'Phase4-age': 'numeric',
    '4500': ['in', ['x', 'x-', 'rm']],
    'ccGWAS': 'numeric',
    'AAfamGWAS': ['in', ['x', 'f']],
    'ExomeSeq': ['in', ['x']],
    'EAfamGWAS': ['x', 'xx', 'f'],
    'EEfamGWAS-fam': [famID_ck],
    'self-reported': [selfReported_ck],
    'wave12-race': ['in', ['Mixed-EA', 'Mixed-?', 'Mixed-other', 'White-EA', 'Black-AA',
                           'Mixed-AA', 'PacIs']],
    '4500-race': ['in', ['EA_0', 'OTHER_0', 'EA_1', 'AA_0', 'AA_1', 'AA_.', 'OTHER_1',
                         'EA_.', 'OTHER_.']],
    'ccGWAS-race': ['in', ['AA', 'EA', 'other']],
    'core-race': ['in', ['n6', 'n9', 'h6', 'n4', 'n2', 'h4', 'h9', 'n8', 'u6', 'h2']],
    'COGA11k-fam': famID_ck,
    'COGA11k-race': ['in', ['AA', 'EA', 'other']],
    'COGA11k-fam-race': ['in', ['black', 'white', 'other']],
    'ruID': ['start in', ['AA', 'PG']],
    'genoID': ['start in', '1234567achp'],
    'SmS': ['in', ['x']],
    'alc_dep_dx': ['in', [0, 1]],
    'alc_dep_ons': 'numeric',
    'CA/CO': ['in', ['CA', '0', '1', 'CO', 'CO(CA)', '(CA)']],
}


def check_column_contents(filepath):
    M, store_date = load_master(custom_path=filepath)
    failures = []

    for col in M.columns:
        date_col = False
        vals = [v for v in M[col].tolist() if not pd.isnull(v)]
        if col in column_guides:
            guide = column_guides[col]
            if guide == 'numeric':
                tests = [isinstance(v, numbers.Number) for v in vals]
            elif guide == 'date':
                date_col = True
            elif type(guide) == list:
                if guide[0] == 'start in':
                    start_len = len(guide[1][0])
                    tests = [v[0:start_len] in guide[1] for v in vals]
                elif guide[0] == 'in':
                    tests = [v in guide[1] for v in vals]
            elif callable(guide):
                tests = [guide(v) for v in vals]

        else:
            if '-date' in col or col in ['DOB', 'Phase4-testdate']:
                date_col = True

        if date_col:
            print('date types for ' + col, set([type(v) for v in vals]))

        if False in tests:
            failures.append((col, [v for v, t in zip(vals, tests) if t == False]))

    return M, failures


def prprint(message,priority,threshold = 3):
    if priority > threshold:
        if isinstance(message,list):
            message = [message]
        print( *message )


def check_exclusions(master,noexp_listDFs):
    ''' Takes a master object and a dictionary of dataframes 
    of not included experiments by their paths.
    Returns lists of exclusions and those missing from the master'''

    subject_exclusions = {}
    for f,df in noexp_listDFs.items():
        for ix,subject in df.iterrows():
            sub_col = df.columns[0]
            ex_col = df.columns[4]
            ID = str(subject[sub_col]).strip()
            exclusions = str(subject[ex_col]).strip().split('_')
            if ID in subject_exclusions:
                subject_exclusions[ID][0].extend(exclusions)
                subject_exclusions[ID][1].append(f)
            else:
                subject_exclusions[ID] = (exclusions,[f])

    missing = {}
    for subID,ex_files in subject_exclusions.items():
        ex, files = ex_files
        if subID in master.index:
            master_exclusions = [ '-'.join(e) for e in excluded_experiments(master,subID)]
            diff = set(ex).difference(master_exclusions)
        else: diff = ex
        if len(diff) > 0:
            missing[subID] = list(diff)

    return subject_exclusions, missing

def update(master_path,access_path, verbose=True, log='auto'):

    start= datetime.now()

    if verbose:
        Pthresh = 0

    #increment file number
    MasterN = int(mpath.split('.')[0].split('_')[-1])
    newMasterN = MasterN+1

    #check header

    #date format

    #masscomp raw-files use repeating letters to indicate session repeats - add zero pads

    # filter access rows

    # correcting dates, all first session
    date_corrD = {    "10123124":"06/24/1994", 
                "20042001":"12/14/1993",
                "30022021":"01/26/1993",
                "30162010":"05/14/2001",
                "30179005":"05/24/1996",
                "40003004":"08/01/1991",
                "40003005":"08/01/1991",
                "40003022":"10/09/1991",
                "40003023":"03/05/1993",
                "40037001":"01/22/1992",
                "50040007":"06/13/1997",
                "50049020":"08/21/2003",
                "50169010":"03/30/1994",
                "60230001":"02/19/1997",
                "63166005":"11/12/1993",
                "c0000207":"05/22/1992",
                "c0000500":"08/28/1996",
                "c0000510":"10/08/1996",
                "h0000300":"10/29/1998",
                "h0000306":"03/10/1999"}

    # correcting raw-file folders
        #issues like nodat
    raw_corrD = {   ("10006001","1"):"00103",
                   ("10006005","0"):"00103",
                   ("10016005","0"):"00001",
                   ("10024007","0"):"00093",
                   ("10171024","0"):"00093",
                   ("10190006","0"):"00095",
                   ("10190006","1"):"ns010",
                   ("10200004","0"):"00093",
                   ("10217001","0"):"ns037",
                   ("30156068","0"):"ns043",
                   ("40002001","0"):"00110",
                   ("40003007","0"):"00005",
                   ("40003008","0"):"00007",
                   ("40005001","0"):"00018",
                   ("40020005","0"):"00055",
                   ("40225004","1"):"00580",
                   ("40259006","1"):"00587",
                   ("40295005","1"):"ns064",
                   ("49425038","0"):"ns120",
                   ("50004023","0"):"00001",
                   ("50019007","0"):"00042",
                   ("50032005","0"):"00034",
                   ("51000711","0"):"00040",
                   ("53009804","0"):"00015",
                   ("c0000830","0"):"ns171",
                   ("h0000351","0"):"00587",
                   ("h0000352","0"):"00587",
                   ("h0000353","0"):"00588",
                   ("h0000354","0"):"00588" }

    # change to multi - first session only
    mchange_list = ["10001001","10006005","10199014","20042001","30022018","30022020",
                    "30022021","30087005","30543531","40003004","40003005","40003022",
                    "40003023","40012004","40018003","40036014","40037001","40049025",
                    "40036019","40059001","40059003","40059011","40059014","40086008",
                    "40092056","40102010","40102015","40112001","40114001","40115013",
                    "40115014","40115015","40115016","40115017","40115027","40115028",
                    "40124002","40132015","40134003","40146011","40222011","40222013",
                    "49026010","49026018","49090001","49367004","49367005","49369003",
                    "49393004","49429001","49462001","50036011","50169214","51000718",
                    "53900903","53900904","58002004","58002006","60038012","60045061",
                    "60213008","60230001","60251204","a0000135","a0000155","a0000296",
                    "a0000808","a0000885","a0000886","c0000207","c0000408","c0000498",
                    "c0000537","c0000539","c0000558","c0000703","c0000704","c0000737",
                    "c0000830","h0000223","h0000242","h0000300","h0000306","p0000250",
                    "p0000432"]       

