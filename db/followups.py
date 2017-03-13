''' extracting information about COGA followups from their master files '''

from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from .knowledge.questionnaires import (max_fups, p123_master_path, p4_master_path,
                                       p1_cols, p2_cols, p3_cols, p4_col_prefixes)
from .utils.math import robust_datemean


def convert_date_applymap(v):
    ''' version of convert date with a set dateform, for applymap below '''

    try:
        return datetime.strptime(v, '%Y-%m-%d')
    except:
        return np.nan


def prepare_datedf(df):
    ''' prepare a date dataframe to be used for finding date means '''

    return df.dropna(how='all').applymap(convert_date_applymap)


def import_mastercsv(path):
    ''' given path to a master CSV, import it and carefully set its ID index '''

    df = pd.read_csv(path)
    df['ID'] = df['IND_ID'].apply(int).apply(str)
    df.set_index('ID', inplace=True)
    return df


def preparefupdfs_forbuild():
    ''' main function, used by build. returns a dictionary
        that maps from phases to followup dataframes '''

    allphase_master_means, pcols = get_allphasemastermeans_pcols()

    phase_dfs = make_fupdfs(allphase_master_means, pcols)

    return phase_dfs


def make_fupdfs(allphase_master_means, pcols):
    ''' given the allphase_master_means df and the phase columns dict,
        return a dict of dataframes that can be converted to records for the followups collection '''

    phase_dfs = {}
    for phase, phase_cols in pcols.items():
        meandate_col = str(phase) + '_meandate'

        phase_df = allphase_master_means[phase_cols + [meandate_col]]
        phase_df.dropna(axis=0, how='all', inplace=True)
        phase_df.dropna(axis=1, how='all', inplace=True)

        phase_df.rename(columns={meandate_col: 'date'}, inplace=True)
        phase_df['followup'] = phase
        phase_dfs[phase] = phase_df

    return phase_dfs


def get_allphasemastermeans_pcols():
    ''' using the source master files, return a dataframe with mean dates from each phase,
        as well as the list of column names '''

    p123m = import_mastercsv(p123_master_path)
    p4m = import_mastercsv(p4_master_path)
    allphase_master = p123m.join(p4m, how='outer', rsuffix='p4')

    pcols = define_phasedatecols(p4m)

    allphase_master_means = make_meancols(allphase_master, pcols)

    return allphase_master_means, pcols


def build_allmasterdf():
    ''' combines all phase master dataframes into one dataframe '''

    p123m = import_mastercsv(p123_master_path)
    p4m = import_mastercsv(p4_master_path)
    allphase_master = p123m.join(p4m, how='outer', rsuffix='p4')

    return allphase_master


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

    pcols = dict()
    pcols['p1'] = p1_cols
    pcols['p2'] = p2_cols
    pcols['p3'] = p3_cols
    for fup in range(0, max_fups + 1):
        p4fup = 'p4f' + str(fup)
        potential_cols = [pfx + '_dtT' + str(fup + 1) for pfx in p4_col_prefixes]
        pcols[p4fup] = [col for col in potential_cols if col in p4master_df.columns]

    return pcols


def extractfup_fromcolname(s):
    ''' given a column name that starts with a followup designation, extract the followup string and handle it '''

    fup_string = s.split('_')[0]
    try:
        return int(fup_string)
    except ValueError:
        return fup_string
