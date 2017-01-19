''' exporting some contents of MongoDB to text '''

from datetime import datetime

import numpy as np

from .organization import Mdb
from .compilation import buildframe_fromdocs
from .file_handling import Neuropsych_XML

# neuropsych

npsych_basepath = '/processed_data/neuropsych/neuropsych_all'

def code_gender(v):
    if v == 'Female':
        return 0
    elif v == 'Male':
        return 1
    else:
        return np.nan

def code_hand(v):
    if v == 'Right':
        return 1
    elif v == 'Left':
        return 0
    elif v == 'BOTH':
        return 2
    else:
        return np.nan

def neuropsych():

    docs = Mdb['neuropsych'].find()
    npsych_df = buildframe_fromdocs(docs, inds=['ID', 'np_followup'])
    npsych_df_IDdate = npsych_df.reset_index().set_index(['ID', 'testdate'])

    if npsych_df.index.has_duplicates:
        print('warning: there are duplicated ID/np_followup combinations')
    if npsych_df_IDdate.index.has_duplicates:
        print('warning: there are duplicated ID/testdate combinations')

    npsych_df['gender'] = npsych_df['gender'].apply(code_gender)
    npsych_df['hand'] = npsych_df['hand'].apply(code_hand)

    export_cols = Neuropsych_XML.cols.copy()
    export_cols.remove('id')
    export_cols.remove('sessioncode')
    export_cols = ['followup', 'session',] + export_cols + ['np_session', 'site', 'filepath',]
    npsych_df_export = npsych_df[export_cols]

    npsych_df_export.rename(columns={'followup': 'COGA_followup',
                                     'session': 'EEG_session'}, inplace=True)
    npsych_df_export.sort_index(inplace=True)

    today = datetime.now().strftime('%m-%d-%Y')
    output_str = '_'.join([npsych_basepath, today])
    npsych_df_export.to_csv(output_str+'.csv')
