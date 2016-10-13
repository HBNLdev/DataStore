''' tools for compilation '''

import os
from datetime import datetime, timedelta
from glob import glob

import numpy as np

from .file_handling import site_hash

# single value or column apply functions

def conv_nan(v):
    ''' convert nans into the string 'unknown' '''
    if np.isnan(v):
        return 'unknown'
    else:
        return v

def collapse_POP(v):
    ''' given a value for a population column, return a version that collapses
        the four possibilities into just COGA or CTL '''
    if v in ['COGA', 'IRPG']:
        return 'COGA'
    elif v in ['COGA-Ctl', 'IRPG-Ctl']:
        return 'CTL'
    else:
        return np.nan

def calc_currentage(dob):
    ''' given a DOB, calculate current age '''
    return (datetime.now() - dob).days / 365.25

def convert_date(datestr, dateform='%m/%d/%Y'):
    ''' convert a date string with a given format '''
    try:
        return datetime.strptime(datestr, dateform)
    except:
        return np.nan

# row-apply functions that usually operate on
# a subject/session-collection-based dataframe
# so certain columns are expected to exist

def calc_PH(row):
    ''' calculate parental history (POP should be a column) '''
    if (row['POP']=='COGA' or row['POP']=='IRPG') and \
    (row['alc_dep_dx_f']==1.0 or row['alc_dep_dx_m']==1.0):
        return 1
    elif (row['POP']=='COGA-Ctl' or row['POP']=='IRPG-Ctl') and \
    row['alc_dep_dx_f']==0.0 and row['alc_dep_dx_m']==0.0:
        return 0
    else:
        return np.nan

def find_exppath(rec, exp, ext, rawfolder_colname='raw_folder'):
    ''' given an experiment name, file extension, and the name of a column
        containing the raw data path, find the path to the raw data for
        that experiment, with that file extension '''
    try:
        glob_expr = rec[rawfolder_colname] + '/' + exp + '*.' + ext
        match_files = glob(glob_expr)
        if len(match_files) > 2:
            print('duplicates, taking first')
        return match_files[0]
    except:
        return np.nan

def calc_nearestsession(row):
    ''' given the name of a column containing dates, return the letter of the
        nearest session '''
    try:
        session_letters = 'abcdefgh'
        date_cols = [c+'-date' for c in session_letters]
        diff_lst = []
        for dcol in date_cols:
            if type(row[dcol]) == datetime:
                diff_lst.append(abs(row[dcol] - row['fMRI test date']))
            else:
                diff_lst.append(timedelta(100000))
        if all(np.equal(timedelta(100000), diff_lst)):
            best_match = np.nan
        else:
            min_ind = np.argmin(diff_lst)
            best_match = session_letters[min_ind]
        return best_match
    except:
        return np.nan

def determine_session(row, df, datediff_col, target_col):
    ''' given a row, dataframe, a date difference column, and a target column,
        determine the nearest session '''
    ID, session = row.name
    ID_df = df.loc[ID].reset_index()
    row_ind = ID_df[datediff_col].argmin()
    correct_fup = ID_df.loc[row_ind, target_col]
    return correct_fup


# the following functions operate on a subject/session-collection-based
# dataframe that is indexed by ID and session

def eeg_system(rec):
    ''' determine the EEG system that was used '''
    session = rec.name[1]
    folder = rec[session+'-raw']
    try:
        folder = int(folder)
    except:
        pass
    if isinstance(folder, str):
        return 'neuroscan'
    elif isinstance(folder, int):
        return 'masscomp'
    else:
        return np.nan
    
def raw_folder(rec):
    ''' determine the path to the folder containing raw data '''
    base_path = '/raw_data/'
    ID = rec.name[0]
    site = site_hash[ID[0]]
    session = rec.name[1]
    folder = rec[session+'-raw']
    try:
        folder = int(folder)
    except:
        pass
    if isinstance(folder, str):
        system = 'neuroscan'
    elif isinstance(folder, int):
        system = 'masscomp'
    else:
        return np.nan
    if system is 'neuroscan':
        try:
            fullpath = os.path.join(base_path,system,site,folder,ID)
            return fullpath
        except:
            return np.nan
    else:
        pop = rec['POP'][:4].lower()
        try:
            fullpath = os.path.join(base_path,system,pop,site,ID)
            return fullpath
        except:
            return np.nan

