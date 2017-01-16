''' matching COGA followups to HBNL sessions '''

from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from statsmodels.robust.scale import huber

from .organization import Mdb
from .compilation import buildframe_fromdocs

p123_master_path = '/processed_data/zork/zork-phase123/subject/master/master.sas7bdat.csv'
p4_master_path = '/processed_data/zork/zork-phase4-72/subject/master/master4_30nov2016.sas7bdat.csv'


p1_cols = ['SSAGA_DT', 'CSAGA_DT', 'CSAGC_DT', 'FHAM_DT', 'TPQ_DT',
               'ZUCK_DT', 'ERP_DT',]
p2_cols = ['SAGA2_DT', 'CSGA2_DT', 'CSGC2_DT', 'ERP2_DT', 'FHAM2_DT',
          'AEQ_DT', 'AEQA_DT', 'QSCL_DT', 'DAILY_DT', 'NEO_DT',
          'SRE_DT', 'SSSC_DT']
p3_cols = ['SAGA3_DT', 'CSGA3_DT', 'CSGC3_DT', 'ERP3_DT', 'FHAM3_DT',
          'AEQ3_DT', 'AEQA3_DT', 'QSCL3_DT', 'DLY3_DT', 'NEO3_DT',
          'SRE3_DT', 'SSSC3_DT']
p4_col_prefixes = ['aeqg', 'aeqy', 'crv', 'cssaga', 'dp', 'hass',
                  'neo', 'ssaga', 'sssc', 'ssv']


def preparefupdfs_forbuild():

    allphase_master_means, pcols = get_allphasemastermeans_pcols()

    ID_session2fup = create_IDs2f(allphase_master_means)
    ID_fup2session = invert_innerdicts(ID_session2fup)
    ID_fup2session_df = IDmap2df(ID_fup2session, inner_keys='followup')

    phase_dfs = make_fupdfs(allphase_master_means, pcols, ID_fup2session_df)

    return phase_dfs


def make_fupdfs(allphase_master_means, pcols, ID_fup2session_df):
    ''' given the allphase_master_means df and the phase columns dict,
        return a dict of dataframes that can be converted to records for the followups collection '''

    phase_dfs = {}
    for phase, phase_cols in pcols.items():
        meandate_col = str(phase) + '_meandate'

        phase_df = allphase_master_means[phase_cols + [meandate_col]]
        phase_df = phase_df.join(ID_fup2session_df[phase])

        phase_df.dropna(axis=0, how='all', inplace=True)
        phase_df.dropna(axis=1, how='all', inplace=True)

        phase_df.rename(columns={meandate_col: 'date', phase: 'session'}, inplace=True)
        phase_df['followup'] = phase
        phase_dfs[phase] = phase_df

    return phase_dfs

def IDmap2df(IDmap, inner_keys):
    ''' given a dict in which the keys are IDs and the values are mappings between followups and sessions,
        create a dataframe indexed by ID with the inner keys as columns and the inner values as values '''

    IDmap_df = pd.DataFrame.from_dict(IDmap, orient='index')
    IDmap_df.index.name = 'ID'
    IDmap_df.columns.name = inner_keys

    return IDmap_df

def create_IDs2f(allphase_master_means):
    ''' given the allphase_master_means df, create a dict of dicts. the outer keys are IDs.
        the inner keys are session letters. the inner values are followup designations '''

    sdate_df = get_sessiondatedf(allphase_master_means)
    sdate_df_fupmeans = add_datediffcols(sdate_df, allphase_master_means)
    ID_session2fup = build_IDsessionfup_map(sdate_df_fupmeans)

    return ID_session2fup


def get_allphasemastermeans_pcols():

    p123m = import_mastercsv(p123_master_path)
    p4m = import_mastercsv(p4_master_path)
    allphase_master = p123m.join(p4m, how='outer', rsuffix='p4')

    pcols = define_phasedatecols(p4m)

    allphase_master_means = make_meancols(allphase_master, pcols)

    return allphase_master_means, pcols


def invert_innerdicts(d):
    ''' given a dict of dicts, return a version in which the inner mapping has been inverted '''

    new_d = {}
    for outer_key, inner_dict in d.items():
        new_d[outer_key] = {v:k for k,v in inner_dict.items()}

    return new_d


def build_allmasterdf():

    p123m = import_mastercsv(p123_master_path)
    p4m = import_mastercsv(p4_master_path)
    allphase_master = p123m.join(p4m, how='outer', rsuffix='p4')
    
    return allphase_master


def build_IDsessionfup_map(sdate_df_fupmeans):
    ''' given an ID/session-indexed sessions dataframe with followup-session date difference info,
        map session letters to followups for each ID, returning a dict of dicts '''

    datediff_cols = [col for col in sdate_df_fupmeans.columns if 'diff' in col]

    ID_session2fup = defaultdict(dict)

    ID_index = sdate_df_fupmeans.index.get_level_values('ID')
    for ID in ID_index:
        ID_df = sdate_df_fupmeans.ix[ID_index == ID, datediff_cols].dropna(axis=1, how='all').dropna(axis=0, how='all')
        for diff_col in ID_df.columns:
            best_session = ID_df[diff_col].argmin()[1]
            the_fup = extractfup_fromcolname(diff_col)
            ID_session2fup[ID][best_session] = the_fup

    return ID_session2fup

def add_datediffcols(sdate_df, allphase_master_means):
    ''' given a df with a session date column and a master dataframe with all mean phase dates,
        add columns indicating their differences '''

    datemean_cols = [col for col in allphase_master_means if '_meandate' in col]
    sdate_df_fupmeans = sdate_df.join(allphase_master_means[datemean_cols])
    for col in datemean_cols:
        datediff_col = col + '_diff'
        sdate_df_fupmeans[datediff_col] = (sdate_df_fupmeans['session_date'] - sdate_df_fupmeans[col]).abs()

    return sdate_df_fupmeans


def get_sessiondatedf(df):
    ''' given a df indexed by ID (minimally), get a corresponding session dataframe
        indexed by both ID and session, with the session date info '''

    IDs = df.index.get_level_values('ID').tolist()
    session_query = {'ID': {'$in': IDs}}
    session_fields = ['ID', 'session', 'date']
    session_proj = {k:1 for k in session_fields}
    session_proj['_id'] = 0
    session_docs = Mdb['sessions'].find(session_query, session_proj)
    sdate_df = buildframe_fromdocs(session_docs)
    sdate_df.rename(columns={'date':'session_date'}, inplace=True)

    return sdate_df

def import_mastercsv(path):
    ''' given path to a master CSV, import it and carefully set its ID index '''

    df = pd.read_csv(path)
    df['ID'] = df['IND_ID'].apply(int).apply(str)
    df.set_index('ID', inplace=True)
    return df

def make_meancols(allphase_master, pcols):
    ''' given a master dataframe with all phases and a dict mapping followup designations to corresponding date cols,
        add columns that indicate the mean date of each followup '''

    print('calculating mean phase dates')

    allphase_master_means = allphase_master.copy()

    for fup, cols in pcols.items():
        print(fup)
        fup_meandate_colname = str(fup) + '_meandate'
        calc_df = prepare_datedf(allphase_master_means[cols])
        calc_df[fup_meandate_colname] = calc_df.apply(robust_datemean, axis=1)
        calc_df[fup_meandate_colname] = pd.to_datetime(calc_df[fup_meandate_colname])
        allphase_master_means = allphase_master_means.join(calc_df[fup_meandate_colname])

    return allphase_master_means

def define_phasedatecols(p4master_df):
    ''' given the phase4 master dataframe,
        define the mapping between phases and date columns found in the COGA master files '''

    pcols = {}
    pcols['p1'] = p1_cols
    pcols['p2'] = p2_cols
    pcols['p3'] = p3_cols
    for fup in range(0, 7):
        potential_cols = [pfx+'_dtT'+str(fup+1) for pfx in p4_col_prefixes]
        pcols[fup] = [col for col in potential_cols if col in p4master_df.columns]
    
    return pcols

def my_strptime(v, dateform='%Y-%m-%d'):
    ''' applymap function to convert all dataframe elements based on a date format '''

    try:
        return datetime.strptime(v, dateform)
    except:
        return np.nan

def prepare_datedf(df):
    ''' prepare a date dataframe to be used for finding date means '''

    return df.dropna(how='all').applymap(my_strptime)

def robust_datemean(row):
    ''' row apply function that finds the robust mean (if possible) among all passed values, which should be dates.
        if the robust mean cannot be found, the regular mean is used. '''

    row_na = row.dropna()
    try:
        row_na_dates = row_na.astype(np.datetime64)
    except TypeError: # already datetime
        # print('could not convert', row_na)
        row_na_dates = row_na
    date_ints = row_na_dates.values.tolist()
    try:
        rm = huber(date_ints)[0].item()  # attempt to use huber robust mean
    except ValueError:
        rm = int(np.mean(reject_outliers(np.array(date_ints))))  # use mean after rejecting outliers
        # rm = int(np.mean(date_ints))
    return rm
        # return int(np.mean(date_ints))

def reject_outliers(data, m = 2.5):
    ''' return version of data only containing points inside arbitrary scaled absolute distance from the median '''
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s < m]

def extractfup_fromcolname(s):
    ''' given a column name that starts with a followup designation, extract the followup string and handle it '''

    fup_string = s.split('_')[0]
    try:
        return int(fup_string)
    except ValueError:
        return fup_string

