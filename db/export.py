''' exporting some contents of MongoDB to text '''

from datetime import datetime

import numpy as np
import pandas as pd

import db.database as D
from .compilation import buildframe_fromdocs
from .file_handling import Neuropsych_XML
from .utils.compilation import get_bestsession, writecsv_date

# neuropsych

npsych_basepath = '/processed_data/neuropsych/neuropsych_all'
fhd_basepath = '/processed_data/fhd/fhd'

def export_a_collection(df,basepath, suffix=''):
        path = writecsv_date(df, basepath, midfix=D.Mdb.name, suffix=suffix )
        print('saved to', path )

def st_int(v):
    if pd.isnull(v):
        return ''
    else:
        return str(int(v))

def convert_int_cols_for_export(df):
    dfw = df.copy()
    for col in df.columns:
        intconv = False
        if df[col].dtype in ['float64','float']:
            vals = df[col].tolist()
            intcks = [v-np.round(v) for v in vals if not pd.isnull(v)]
            if sum(intcks) == 0.0:
                dfw[col] = dfw[col].apply(st_int)
    return dfw

def neuropsych(do_export=True,COGA=False):
    global npsych_basepath
    docs = D.Mdb['neuropsych'].find()
    npsych_df = buildframe_fromdocs(docs, inds=['ID', 'np_followup'])
    npsych_df = get_bestsession(npsych_df)
    npsych_df_IDdate = npsych_df.reset_index().set_index(['ID', 'testdate'])
    npsych_df_IDbs = npsych_df.dropna(subset=['session_best']).reset_index().set_index(['ID', 'session_best'])

    if npsych_df.index.has_duplicates:
        print('warning: there are duplicated ID/np_followup combinations')
    if npsych_df_IDdate.index.has_duplicates:
        print('warning: there are duplicated ID/testdate combinations')
    if npsych_df_IDbs.index.has_duplicates:
        print('warning: there are duplicated ID/bestsession combinations')

    # coding gender and handedness
    npsych_df['gender'] = npsych_df['gender'].apply(npsych_code_gender)
    npsych_df['hand'] = npsych_df['hand'].apply(npsych_code_hand)

    # setting non-phase-4 followup values to be missing
    nonp4_bool = npsych_df['followup'].isin(['p1', 'p2', 'p3'])
    print(nonp4_bool.sum(), 'non-phase4 followup values found, setting to missing')
    npsych_df.ix[nonp4_bool, 'followup'] = np.nan

    # defining and re-ordering the columns to export
    export_cols = Neuropsych_XML.cols.copy()
    export_cols.remove('id')
    export_cols.remove('sessioncode')
    export_cols.remove('testdate')
    export_cols = ['date','session_best', 'session', 'date_diff_session', 'followup'] + \
                  export_cols + ['np_session', 'site' ]
    for n_ring in ['3b', '4b', '5b', 'tt']:
        last_pos = export_cols.index('tolt_' + n_ring + '_atrti')
        otr_pos = export_cols.index('tolt_' + n_ring + '_otr')
        export_cols.insert(last_pos + 1, export_cols.pop(otr_pos))

    #using lowercase 'id' from here on to satisfy excel import behavior
    npsych_df_export = npsych_df[export_cols]
    npsych_df_export.reset_index(inplace=True)
    npsych_df_export.rename(columns={'ID':'id',
                                    'date':'np_date',
                                    'session_best': 'EEG_session_best',
                                     'session': 'EEG_session',
                                     'followup': 'COGA_followup',
                                     }, inplace=True)
    npsych_df_export.set_index(['id','np_session'],inplace=True)
    npsych_df_export.sort_index(inplace=True)

    if COGA == True:
        npsych_basepath = npsych_basepath.replace('_all','_COGA')

        ids = list(set(npsych_df_export.index.get_level_values('id').tolist()))
        npsych_df_export = npsych_df_export.iloc[\
            npsych_df_export.index.get_level_values('id').isin([i for i in ids if i[0] not in 'achpg']) ,: ]
        
        npsych_df_export = npsych_df_export.reset_index().set_index(['id','np_session'])

    if do_export:
        export_a_collection(convert_int_cols_for_export(npsych_df_export),
                         npsych_basepath)

    return npsych_df_export


def npsych_code_gender(v):
    ''' given neuropsych-style gender labels, return the agreed upon coding '''

    if v == 'Female':
        return 0
    elif v == 'Male':
        return 1
    else:
        return np.nan


def npsych_code_hand(v):
    ''' given neuropsych-style handedness labels, return the agreed upon coding '''

    if v == 'Right':
        return 1
    elif v == 'Left':
        return 0
    elif v == 'BOTH':
        return 2
    else:
        return np.nan


def fhd(do_export=True):
    fhd_cols = ['ID',
                'father_cor_alc_dep_dx',
                'father_cor_ald5dx',
                'parent_cor_alc_dep_dx',
                'parent_cor_ald5dx',
                'first_cor_alc_dep_dx',
                'first_cor_ald5dx',
                'parentCOGA_cor_alc_dep_dx',
                'parentCOGA_cor_ald5dx',
                'fhdratio_cor_alc_dep_dx',
                'fhdratio_cor_ald5dx',
                'fhdratio_cor_ald5sx_max_cnt_log',
                'fhdratio2_cor_alc_dep_dx',
                'fhdratio2_cor_ald5dx',
                'fhdratio2_cor_ald5sx_max_cnt_log',
                'nrels_cor_alc_dep_dx',
                'nrels_cor_ald5dx',
                'nrels_cor_ald5sx_max_cnt_log',
                'nrels2_cor_alc_dep_dx',
                'nrels2_cor_ald5dx',
                'nrels2_cor_ald5sx_max_cnt_log',
                ]
    fhd_proj = {c: 1 for c in fhd_cols}
    fhd_proj['_id'] = 0
    fhd_docs = D.Mdb['fhd'].find({}, fhd_proj)
    fhd_df = buildframe_fromdocs(fhd_docs, inds=['ID'])
    fhd_df = fhd_df[fhd_cols[1:]]

    if do_export:
        # today = datetime.now().strftime('%m-%d-%Y')
        # output_str = '_'.join([npsych_basepath, D.Mdb.name, today])
        # npsych_df_export.to_csv(output_str + '.csv')
        # print('saved to', output_str + '.csv')
        export_a_collection(fhd_df, fhd_basepath, suffix='all')
    else:
        return fhd_df
