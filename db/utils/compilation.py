''' tools for compilation '''

import os
from datetime import datetime, timedelta
from glob import glob

import numpy as np
import pandas as pd

from .filename_parsing import site_hash


# csv import functions


def readcsv_uID(csv_path, inds=['ID', 'session']):
    ''' read a CSV in as a pandas dataframe, convert the ID column to string, and
        (defaultly) set ID and sessions as indices '''

    df = pd.read_csv(csv_path)
    df['ID'] = df['ID'].apply(int).apply(str)
    df.set_index(inds, inplace=True)
    return df


def df_fromcsv(fullpath, id_lbl='ind_id', na_val=''):
    ''' convert csv into dataframe, converting ID column to standard '''

    # read csv in as dataframe
    try:
        df = pd.read_csv(fullpath, na_values=na_val)
    except pd.parser.EmptyDataError:
        print('csv file was empty, continuing')
        return pd.DataFrame()

    # convert id to str and save as new column
    df[id_lbl] = df[id_lbl].apply(int).apply(str)
    df['ID'] = df[id_lbl]
    df.set_index('ID', drop=False, inplace=True)

    return df


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
    ''' if v is larger than thresh, return an 'x'. used to create a new columns marking supra-threshold rows '''

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


def column_split(v, ind=0, sep='_'):
    ''' split value by sep and take the indth element of the result'''

    return v.split(sep)[ind]


def extract_session_fromuID(v):
    ''' given an HBNL-style uID (e. a1_10010001), extract the session '''

    return v.split('_')[0][0]


# dealing with columns


def reorder_columns(df, beginning_order):
    ''' given a dataframe and a list of columns that you would like to be
        at the beginning of the columns (in order), reorder them '''

    cols = df.columns.tolist()
    for col in reversed(beginning_order):
        try:
            cols.insert(0, cols.pop(cols.index(col)))
        except ValueError:
            print(col, 'was missing, will not be re-ordered')
    return df[cols]


# row-apply functions that usually operate on
# a subject/session-collection-based dataframe
# so certain columns are expected to exist


def fix_fhdratio(row, k=10):
    ''' given a dataset with the columns fhd_dx4_ratio and n_rels,
        return a version of the fhd ratio so that none of the values are 0 and 1 exactly.
        this is a preparatory step to apply logarithm-based transforms to the values. '''

    ratio = row['fhd_dx4_ratio']
    n_rels = row['n_rels']

    if ratio == 0:
        return 1 / (k * n_rels)
    elif ratio == 1:
        return 1 - (1 / (k * n_rels))
    else:
        return ratio


def join_columns(row, columns, sep='_'):
    ''' given list of columns, join all (string-compatible) values from those columns, using sep as delimiter '''

    return sep.join([row[field] for field in columns])


def build_parentID(rec, famID_col, parentID_col):
    ''' dataframe-apply function that uses partial ID info from
        a family part column and a parent part column
        to construct the full ID of the parent '''

    try:
        return rec[famID_col] + '{0:03d}'.format(int(rec[parentID_col]))
    except ValueError:
        return np.nan


def join_ufields(row, exp=None):
    ''' join ID and session columns into a uID. if exp is True, also join experiment name '''

    if exp:
        return '_'.join([row['ID'], row['session'], exp])
    else:
        return '_'.join([row['ID'], row['session']])


def eq_fields(row, field1, field2):
    ''' create a boolean that represents whether field1 and field2 are equal '''

    if row[field1] == row[field2] or (np.isnan(row[field1]) and np.isnan(row[field1])):
        return True
    else:
        return False


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
    if not isinstance(row['Phase4-session'],str) or row['Phase4-session'] not in 'abcd':
        return np.nan
    else:
        return ord(row['session']) - ord(row['Phase4-session'])


def join_allcols(rec, sep='_'):
    ''' dataframe apply function that simply joins the whole rows contents (should be strings),
        using sep as the separator '''
    return sep.join(rec)


def ID_in_x(rec, ck_set):
    ''' if row's ID in ck_set, return x '''

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


def first_session(df):
    ''' given a datframe with an (ID, session) index, return a version in which
        only the first sessions is kept for each '''

    out_df = df.copy()
    out_df['session'] = out_df.index.get_level_values('session')
    g = out_df.groupby(level=out_df.index.names[0])
    out_df = g.first()
    out_df.set_index('session', append=True, inplace=True)
    return out_df


def latest_session(in_df):
    ''' given a datframe with an (ID, session) index, return a version in which
        only the latest sessions is kept for each '''

    out_df = in_df.copy()
    out_df['session'] = out_df.index.get_level_values('session')
    g = out_df.groupby(level=out_df.index.names[0])  # ID
    out_df = g.last()
    out_df.set_index('session', append=True, inplace=True)
    return out_df


def idealage_session(in_df, ideal_age=18, age_col='session_age'):
    ''' given a datframe with an (ID, session) index, return a version in which
        the session with age_col nearest to a given age is kept. '''

    best_sessions = []
    IDs = list(set(in_df.index.get_level_values('ID')))
    for ID in IDs:
        sages = in_df.loc[ID, age_col]
        best_match = (sages - ideal_age).abs().argmin()
        best_sessions.append((ID, best_match))
    return in_df.loc[best_sessions, :]


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


def get_restingcnts(df):
    ''' given dataframe indexed by ID/session, find paths of resting CNTs '''

    df_out = df.copy()

    for cnt_exp in ['eec', 'eeo']:
        df_out[cnt_exp + '_cnt_path'] = df_out.apply(find_exppath, axis=1, args=[cnt_exp, 'cnt'])

    return df_out


def get_bestsession(df, datediff_col='date_diff_session'):
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


# groupby functions


def groupby_followup(df):
    ''' given ID+followup-indexed dataframe, create a groupby object, grouping by followup '''

    df['followup'] = df.index.get_level_values('followup')
    g = df.groupby(level=df.index.names[0])  # ID
    df.drop('followup', axis=1, inplace=True)
    return g


def groupby_session(df):
    ''' given ID+session-indexed dataframe, create a groupby object, grouping by session '''

    df['session'] = df.index.get_level_values('session')
    g = df.groupby(level=df.index.names[0])  # ID
    df.drop('session', axis=1, inplace=True)
    return g


# recently added

def convert_fupname(v):
    ''' given a COGA-style followup designation (e.g. found in core phenotype columns such as 'cor_aldp_first_whint'
        convert it to the HBNL database convention (e.g. p4f0) '''

    try:
        phase_str = v[1]
    except TypeError:
        return np.nan

    if phase_str == '4':
        try:
            followup_str = v[3]
        except IndexError:
            followup_str = '0'
        out_fupname = 'p' + phase_str + 'f' + followup_str
    else:
        out_fupname = 'p' + phase_str

    return out_fupname


def add_ages(ID_df, fup_df, col):
    ''' given an ID-indexed dataframe, a corresponding ID-followup-indexed dataframe containing an 'age' column,
        and the name of a column containing an HBNL-style followup designation that was previously converted
        (with the suffix _conv), add the age at that followup to the ID dataframe '''

    use_series = ID_df[col + '_conv']
    indices = []
    for ID, fup in zip(use_series.index, use_series.values):
        try:
            fup + ' '
            indices.append((ID, fup))
        except TypeError:
            pass
    ages = fup_df.loc[indices, 'age']
    ID_df[col + '_age'] = ages.reset_index('followup', drop=True)


def td_to_years(v):
    ''' given a timedelta, convert it to years in a slightly inaccurate fashion '''

    try:
        return v.days / 365.25
    except AttributeError:
        return np.nan


def daily_average(drink_df_al):
    ''' given a dataframe containing only the COGA drinking columns for individual days of the week,
        find out the daily average on drinking days '''

    drink_df_al_na = drink_df_al.copy()

    for day in range(1, 8):
        day_cols = [col for col in drink_df_al_na.columns if col[-1] == str(day)]
        print(day, day_cols)
        drink_df_al_na['day' + str(day) + '_sum'] = drink_df_al_na[day_cols].sum(axis=1)

    drink_df_al_na[(drink_df_al_na == 0)] = np.nan

    sum_cols = ['day' + str(n) + '_sum' for n in range(1, 8)]
    print(sum_cols)
    drink_df_al_na['da'] = drink_df_al_na[sum_cols].mean(axis=1)

    return drink_df_al_na['da']


def writecsv_date(df, base_path, suffix):
    ''' given dataframe df, a base_path (including directories and any file prefix), and a suffix,
        write the dataframe to a CSV that has a date attached to it '''

    today = datetime.now().strftime('%m-%d-%Y')
    output_str = '_'.join([base_path, today, suffix])
    output_path = output_str + '.csv'
    df.to_csv(output_path, float_format='%.3f')

