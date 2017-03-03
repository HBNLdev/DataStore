''' tools for compilation '''

import os
from datetime import datetime, timedelta
from glob import glob

import numpy as np
import pandas as pd

from .filenames import site_hash


# groupby functions


def groupby_followup(df):
    df['followup'] = df.index.get_level_values('followup')
    g = df.groupby(level=df.index.names[0])  # ID
    df.drop('followup', axis=1, inplace=True)
    return g


# dataframe filter functions


def latest_session(in_df):
    ''' given a datframe with an (ID, session) index, return a version in which
        only the latest sessions is kept for each '''
    out_df = in_df.copy()
    out_df['session'] = out_df.index.get_level_values('session')
    g = out_df.groupby(level=out_df.index.names[0])  # ID
    out_df = g.last()
    out_df.set_index('session', append=True, inplace=True)
    return out_df


def mark_latest(in_df):
    ''' given a datframe with an (ID, session) index, add a column that contains x's
        for rows that are the latest session '''

    out_df = in_df.copy()
    out_df['session'] = out_df.index.get_level_values('session')
    g = out_df.groupby(level=out_df.index.names[0])  # ID
    uIDs = g.last().set_index('session', append=True).index.values
    out_df['latest_session'] = np.nan
    out_df.loc[uIDs, 'latest_session'] = 'x'
    out_df.drop('session', axis=1, inplace=True)

    return out_df


# single value or column apply functions


def ID_nan_strintfloat_COGA(v):
    ''' convert a COGA ID (fully numeric) to string '''

    try:
        return str(int(float(v)))  # will work for fully numeric IDs
    except ValueError:
        return np.nan


def ID_nan_strint(v):
    ''' convert an HBNL ID (not fully numeric) to string '''

    try:
        return str(int(v))  # will work for fully numeric IDs
    except ValueError:
        if v[0].isalpha():  # (if an ACHP ID)
            return str(v)
        else:  # if is a missing val ('.')
            return np.nan


def mark_overthresh(v, thresh):
    if v > thresh:
        return 'x'
    else:
        return np.nan


def neg_toNaN(v):
    ''' convert negative numbers to nans '''
    if v < 0:
        return np.nan
    else:
        return v


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


def conv_159(v):
    ''' convert COGA coding to regular coding '''
    if v == 1:
        return 0
    elif v == 5:
        return 1
    else:
        return np.nan


def num_chans(h1_path):
    ''' given the path to an h1-file (avgh1 or cnth1), extract the number of channels '''
    try:
        path_parts = os.path.split(h1_path)
        chan_part = path_parts[-2]
        n_chans = int(chan_part[-2:])
        return n_chans
    except:
        return np.nan


def convert_internalizing_columns(cols):
    col_tups = []

    for col in cols:
        pieces = col.split('_')
        info = '_'.join(pieces[:-2])
        fup = '_'.join(pieces[-2:])

        col_tups.append((info, fup))

    return pd.MultiIndex.from_tuples(col_tups, names=('', 'followup'))


def convert_intfup(fup):
    if fup[:2] == 's2':
        fup_rn = 'p2'
    else:
        fup_rn = int(fup[-1])

    return fup_rn


def convert_questname(v):
    if v[0] == 'c':
        return 'cssaga'
    elif v[1] == 's':
        return 'ssaga'


def extract_session_fromuID(v):
    return v.split('_')[0][0]


# row-apply functions that usually operate on
# a subject/session-collection-based dataframe
# so certain columns are expected to exist

def build_parentID(rec, famID_col, parentID_col):
    ''' dataframe-apply function that uses partial ID info from
        a family part column and a parent part column
        to construct the full ID of the parent '''

    try:
        return rec[famID_col] + '{0:03d}'.format(int(rec[parentID_col]))
    except ValueError:
        return np.nan


def join_ufields(row, exp=None):
    if exp:
        return '_'.join([row['ID'], row['session'], exp])
    else:
        return '_'.join([row['ID'], row['session']])


def harmonize_fields_max(row, field1, field2):
    ''' harmonize two columns by return the max of the two '''
    if row[field1] == row[field2] or \
            (np.isnan(row[field1]) and np.isnan(row[field2])):
        return row[field1]
    else:
        return max(row[field1], row[field2])


def harmonize_fields_left(row, field1, field2):
    ''' harmonize two columns such that field1 overrules field2, but field2 is accepted if field1 is missing '''
    if np.isnan(row[field1]):
        return row[field2]
    else:
        return row[field1]


def calc_followupcol(row):
    ''' return the Phase 4 followup # '''
    if row['Phase4-session'] is np.nan or row['Phase4-session'] not in 'abcd':
        return np.nan
    else:
        return ord(row['session']) - ord(row['Phase4-session'])


def join_allcols(rec, sep='_'):
    ''' dataframe apply function that simply joins the whole rows contents (should be strings),
        using sep as the separator '''
    return sep.join(rec)


def ID_in_x(rec, ck_set):
    ID = rec.name[0]
    if ID in ck_set:
        return 'x'
    else:
        return np.nan


def calc_PH(row):
    ''' calculate parental history (POP should be a column) '''
    if (row['POP'] == 'COGA' or row['POP'] == 'IRPG') and \
            (row['alc_dep_dx_f'] == 1.0 or row['alc_dep_dx_m'] == 1.0):
        return 1
    elif (row['POP'] == 'COGA-Ctl' or row['POP'] == 'IRPG-Ctl') and \
                    row['alc_dep_dx_f'] == 0.0 and row['alc_dep_dx_m'] == 0.0:
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
        date_cols = [c + '-date' for c in session_letters]
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


# the following functions operate on a subject/session-collection-based
# dataframe that is indexed by ID and session

def eeg_system(rec):
    ''' determine the EEG system that was used '''
    session = rec.name[1]
    folder = rec[session + '-raw']
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
    try:
        site = site_hash[ID[0]]
    except:
        print('site not recognized')
        return np.nan
    session = rec.name[1]
    folder = rec[session + '-raw']
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
            fullpath = os.path.join(base_path, system, site, folder, ID)
            return fullpath
        except:
            return np.nan
    else:
        pop = rec['POP'][:4].lower()
        try:
            fullpath = os.path.join(base_path, system, pop, site, ID)
            return fullpath
        except:
            return np.nan


def raw_folder_achp(rec):
    ''' version of the above that is safe to use on the SUNY Brain Dysfunction (achp) subjects '''
    base_path = '/raw_data/'
    ID = rec.name[0]
    site = 'suny'
    session = rec.name[1]
    folder = rec[session + '-raw']
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
            fullpath = os.path.join(base_path, system, site, folder, ID)
            return fullpath
        except:
            return np.nan
    else:
        pop = rec['POP'][:4].lower()
        try:
            fullpath = os.path.join(base_path, system, pop, site, ID)
            return fullpath
        except:
            return np.nan


# functions that operate on a dataframe

def get_restingcnts(df):
    ''' given dataframe indexed by ID/session, find paths of resting CNTs '''

    df_out = df.copy()

    for cnt_exp in ['eec', 'eeo']:
        df_out[cnt_exp + '_cnt_path'] = df_out.apply(find_exppath, axis=1, args=[cnt_exp, 'cnt'])

    return df_out


def get_bestsession(df, datediff_col='session_datediff'):
    ''' given a dataframe with columns of ID, session, and session_datediff,
        add a column called session_best, which is a copy of session, except
        for duplicated ID/session combinations,
        only the one with the smallest date difference is kept '''

    in_inds = df.index.names

    df_uID = df.reset_index().set_index(['ID', 'session'])
    df_uID['session_best'] = df_uID.index.get_level_values('session')

    df_uID_sort = df_uID.sort_values(datediff_col).sort_index()
    df_uID_sort.ix[df_uID_sort.index.duplicated(keep='first'), 'session_best'] = np.nan

    return df_uID_sort.reset_index().set_index(in_inds)


def keep_bestsession(df):
    ''' given a dataframe with a session_best column,
        return a version such that only the best sessions are present '''

    df['session'] = df.index.get_level_values('session')
    out_df = df.ix[df['session'] == df['session_best']]
    out_df.drop(['session', 'session_best'], axis=1, inplace=True)

    return out_df


# dealing with columns

def drop_frivcols(df):
    ''' given df, drop typically frivolous columns '''

    friv_cols = [col for col in df if '_id' in col or 'insert_time' in col \
                 or '_technique' in col or '_subject' in col \
                 or '_questname' in col or '_ADM' in col or '_IND_ID' in col]
    dcols = ['4500', '4500-race', 'AAfamGWAS', 'CA/CO', 'COGA11k-fam', 'COGA11k-fam-race',
             'COGA11k-race', 'EAfamGWAS', 'EEfamGWAS-fam', 'ExomeSeq',
             'Wave12', 'Wave12-fam', 'Wave3', '_ID', 'ccGWAS', 'ccGWAS-race',
             'wave12-race', 'no-exp', 'remarks'] + friv_cols
    return df.drop(dcols, axis=1)


def reorder_columns(df, beginning_order):
    ''' given a dataframe and a list of columns that you would like to be
        at the beginning of the columns (in order), reorder them '''

    cols = df.columns.tolist()
    for col in reversed(beginning_order):
        cols.insert(0, cols.pop(cols.index(col)))
    return df[cols]


def my_strptime(v, dateform='%Y-%m-%d'):
    ''' applymap function to convert all dataframe elements based on a date format '''

    try:
        return datetime.strptime(v, dateform)
    except:
        return np.nan


def prepare_datedf(df):
    ''' prepare a date dataframe to be used for finding date means '''

    return df.dropna(how='all').applymap(my_strptime)
