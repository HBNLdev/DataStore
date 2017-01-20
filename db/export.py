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

def neuropsych(do_export=True):

    docs = Mdb['neuropsych'].find()
    npsych_df = buildframe_fromdocs(docs, inds=['ID', 'np_followup'])
    npsych_df_IDdate = npsych_df.reset_index().set_index(['ID', 'testdate'])

    if npsych_df.index.has_duplicates:
        print('warning: there are duplicated ID/np_followup combinations')
    if npsych_df_IDdate.index.has_duplicates:
        print('warning: there are duplicated ID/testdate combinations')

    # coding gender and handedness
    npsych_df['gender'] = npsych_df['gender'].apply(code_gender)
    npsych_df['hand'] = npsych_df['hand'].apply(code_hand)

    # setting non-phase-4 followup values to be missing
    nonp4_bool = npsych_df['followup'].isin(['p1', 'p2', 'p3'])
    print(nonp4_bool.sum(), 'non-phase4 followup values found, setting to missing')
    npsych_df.ix[nonp4_bool, 'followup'] = np.nan

    # defining and re-ordering the columns to export
    export_cols = Neuropsych_XML.cols.copy()
    export_cols.remove('id')
    export_cols.remove('sessioncode')
    export_cols = ['session', 'followup'] + export_cols + ['np_session', 'site', 'filepath',]
    for n_ring in ['3b', '4b', '5b', 'tt']:
        last_pos = export_cols.index('atrti_' + n_ring)
        otr_pos = export_cols.index('otr_' + n_ring)
        export_cols.insert(last_pos+1, export_cols.pop(otr_pos))

    npsych_df_export = npsych_df[export_cols]
    npsych_df_export.rename(columns={'session': 'EEG_session',
                                     'followup': 'COGA_followup',
                                     }, inplace=True)
    npsych_df_export.sort_index(inplace=True)

    if do_export:
        today = datetime.now().strftime('%m-%d-%Y')
        output_str = '_'.join([npsych_basepath, today])
        npsych_df_export.to_csv(output_str+'.csv')
        print('saved to', output_str+'.csv')

    return npsych_df_export
