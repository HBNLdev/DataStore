'''EEGmaster
'''

# master_path = '/export/home/mike/projects/mongo/EEG-master-file-14.csv'
master_path = '/processed_data/master-file/EEG-master-file-16.csv'
master = None

import os
from datetime import datetime

import pandas as pd
import numpy as np

sparser_sub = ['famID', 'mID', 'fID', 'DNA', 'rel2pro', 'famtype', 'POP',
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

sparser_add = ['ID', 'sex', 'handedness', 'Self-reported-race', 'alc_dep_dx',
               'alc_dep_ons', 'a-age', 'b-age', 'c-age', 'd-age', 'e-age',
               'f-age', 'g-age', 'h-age', 'i-age', 'j-age', 'k-age']


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


def load_master(preloaded=None, force_reload=False, custom_path=None):
    global master
    if type(preloaded) == pd.core.frame.DataFrame and not custom_path:
        master = preloaded
        return

    if custom_path:
        master_path_use = custom_path
    else:
        master_path_use = master_path

    if not type(master) == pd.core.frame.DataFrame or force_reload:
        # check date modified on master file
        master_mtime = datetime.fromtimestamp(
            os.path.getmtime(master_path_use))
        # read as csv
        master = pd.read_csv(master_path_use, converters={'ID': str},
                             na_values=['.'], low_memory=False)

        # verify_integrity=True)
        master.set_index('ID', drop=False, inplace=True)
        for dcol in ['DOB'] + [col for col in master.columns if '-date' in col]:
            master[dcol] = master[dcol].map(calc_date_w_Qs)

    return master_mtime


def masterYOB():
    '''Writes _YOB version of master file in same location
    '''
    load_master()
    masterY = master.copy()
    masterY['DOB'] = masterY['DOB'].apply(lambda d: d.year)
    masterY.rename(columns={'DOB': 'YOB'}, inplace=True)

    outname = master_path.replace('.csv', '_YOB.csv')
    masterY.to_csv(
        outname, na_rep='.', index=False, date_format='%m/%d/%Y', float_format='%.5f')

    return masterY


def ids_with_exclusions():
    return master[master['no-exp'].notnull()]['ID'].tolist()


def excluded_experiments(id):
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


def frame_for_study(study):

    id_series = subjects_for_study(study)

    study_frame = master.ix[id_series]

    return study_frame


session_letters = 'abcdefghijk'


def sessions_for_subject_experiment(subject_id, experiment):
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
