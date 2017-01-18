'''EEGmaster
'''

master_path = '/processed_data/master-file/EEG-master-file-22.csv'

import os
from numbers import Number
from datetime import datetime

import pandas as pd
import numpy as np

from .file_handling import site_hash

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

def calc_date_w_Qs(dstr):
    ''' assumes date of form mm/dd/yyyy
    '''
    dstr = str(dstr)
    if dstr == 'nan':
        return np.nan
    if '?' in dstr:
        if dstr[:2] == '??':
            if dstr[3:5] == '??':
                if dstr[5:7] == '??':
                    return None
                else:
                    dstr = '6/1' + dstr[5:]
            else:
                dstr = '6' + dstr[2:]
        else:
            if dstr[3:5] == '??':
                dstr = dstr[:2] + '/15' + dstr[5:]
    try:
        return datetime.strptime(dstr, '%m/%d/%Y')
    except:
        print('problem with date: ' + dstr)
        return np.nan


def site_fromIDstr(IDstr):
    if IDstr[0].isnumeric():
        return site_hash[IDstr[0]]
    else:
        return 'suny'

def build_parentID(rec, famID_col, parentID_col):
    try:
        return rec[famID_col] + '{0:03d}'.format(int(rec[parentID_col]))
    except ValueError:
        return np.nan

def ID_nan_strintfloat_COGA(v):
    try:
        return str(int(float(v))) # will work for fully numeric IDs
    except ValueError:
        return np.nan

def ID_nan_strint(v):
    try:
        return str(int(v)) # will work for fully numeric IDs
    except ValueError:
        if v[0].isalpha(): # (if an ACHP ID)
            return str(v)
        else: # if is a missing val ('.')
            return np.nan

def load_master(preloaded=None, force_reload=False, custom_path=None):
    
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
                                         'mID': ID_nan_strint, 'fID': ID_nan_strint},
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
    master_mtime = datetime.fromtimestamp(
        os.path.getmtime(master_path_use))	

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


def excluded_experiments(master,id):
    ex_str = master.ix[id]['no-exp']
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

def famID(stck):
    
    try:
        assert len(stck) == 5
        assert stck[0] in '1234567'
        return True
    except AssertionError:
        return False


column_guides = {
    'ID': ['start in',['1234567achp']] ,
    'famID': famID,
    'mID': ['start in',['1234567']],
    'fID': ['start in',['1234567']],
    'DNA': ['in','x'],
    'rel2pro': ['in',['bil','fath','gof','hsib','m1c','m1c1r','ma','mate','mgf','mgm',
                    'mgn','mhs','mu','moth','niece','neph','off','olaw','p1c','p1c1r',
                    'pa','pgf','pgm','pgn','phs','pu','self','sib','sil','slaw']],
    'famtype': ['in',['PILOT', 'I', 'IV', 'CONTROL', 'I-S-III', 'I-S-IV', 'II', 'L-II',
       'III', 'IV-H', 'I-S-II', 'A-I', 'KIDS', 'I-S-IV-H', 'A-II', 'IRPG',
       'IRPG-CTRL','L-IV', 'CTRL-K', 'CL-IV', 'A-IV-H', 'A-III',
       'CL-SPEC', 'SPEC', 'CHAL-CON'] ],
    'POP': ['in',['Pilot', 'COGA', 'COGA-Ctl', 'Relia', 'IRPG', 'IRPG-Ctl','Alc-chal', 
            'A', 'C', 'H', 'P']],
    'sex': ['in',['m','f','(m)','(f)']],
    'handedness':['in',['b','l','r','w']] ,
    'DOB': ['date'],
    'twin': ['in',[0,1,2]],
    'EEG': ['in',['x','e','-']],
    'system': ['in',['es','es-ns','mc','mc-es','mc-ns','ns']],
    'Wave12': ['in',['P','x']],
    'Wave12-fam': ['in',[1,2]],
    'fMRI': ['in',['1a','1b']],
    'Wave3': ['in',['x','rm']],
    'Phase4-session': ['in',['a','b','c','d']],
    'Phase4-testdate': ['date'],
    'Phase4-age': 'numeric',
    '4500': ['in',['x','x-','rm']],
    'ccGWAS': 'numeric',
    'AAfamGWAS': ['in',['x','f']],
    'ExomeSeq': ['in',['x']],
    'EAfamGWAS':['x','xx','f'] ,
    'EEfamGWAS-fam': [famID],
    'self-reported': ['in',['u8', 'h6', 'n4', 'n3', 'u2', 'h2', 'n2', 'h4', 'u6','u1',
                    'n1', 'h8', 'n9', 'n8', 'u4', 'u3', 'u9', 'h9', 'h1', 'h3']],
    'wave12-race': ['in',['Mixed-EA', 'Mixed-?', 'Mixed-other', 'White-EA', 'Black-AA',
                    'Mixed-AA', 'PacIs']],
    '4500-race': ['in',['EA_0', 'OTHER_0', 'EA_1', 'AA_0', 'AA_1', 'AA_.', 'OTHER_1',
                    'EA_.', 'OTHER_.']],
    'ccGWAS-race': ['in',['AA','EA']],
    'core-race': ['in',['n6', 'n9', 'h6', 'n4', 'n2', 'h4', 'h9', 'n8', 'u6', 'h2' ]],
    'COGA11k-fam': famID,
    'COGA11k-race': ['in',['AA','EA','other']],
    'COGA11k-fam-race': ['in',['black','white','other']],
    'ruID': [ 'start in',['AA', 'PG'] ],
    'genoID': ['skip'],
    'SmS': ['in',['x']],
    'alc_dep_dx': ['in',[0,1]],
    'alc_dep_ons': 'numeric',
    'CA/CO': ['in',['CA', '0', '1', 'CO', 'CO(CA)', '(CA)']],
    # 'a-run': ,
    # 'a-raw': ,
    # 'a-date': ,
    # 'a-age': ,
    # 'b-run': ,
    # 'b-raw': ,
    # 'b-date': ,
    # 'b-age': ,
    # 'c-run': ,
    # 'c-raw': ,
    # 'c-date': ,
    # 'c-age': ,
    # 'd-run': ,
    # 'd-raw': ,
    # 'd-date': ,
    # 'd-age': ,
    # 'e-run': ,
    # 'e-raw': ,
    # 'e-date': ,
    # 'e-age': ,
    # 'f-run': ,
    # 'f-raw': ,
    # 'f-date': ,
    # 'f-age': ,
    # 'g-run': ,
    # 'g-raw': ,
    # 'g-date': ,
    # 'g-age': ,
    # 'h-run': ,
    # 'h-raw': ,
    # 'h-date': ,
    # 'h-age': ,
    # 'i-run': ,
    # 'i-raw': ,
    # 'i-date': ,
    # 'i-age': ,
    # 'j-run': ,
    # 'j-raw': ,
    # 'j-date': ,
    # 'j-age': ,
    # 'k-run': ,
    # 'k-raw': ,
    # 'k-date': ,
    # 'k-age': ,
    # 'no-exp': ,
    # 'remarks'

}

def check_column_contents(filepath):
    M, store_date = load_master(filepath)

    for col in M.columns:
        vals = M[col].tolist()
        if col in column_guides:
            guide = column_guides[col]
            if guide == 'numeric':
                tests = [ isinstance(v,numbers.Number) for v in vals ]
            elif callable(guide):
                tests = [ guide(v) for v in vals ]