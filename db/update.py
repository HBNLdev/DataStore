'''update collections'''

from tqdm import tqdm

import numpy as np
import pandas as pd

from .organization import Mdb
from .compilation import buildframe_fromdocs

def subjects_from_followups():

    fup_query = {'session': {'$ne': np.nan}}
    fup_fields = ['ID', 'session', 'followup', 'date']
    fup_proj = {f:1 for f in fup_fields}
    fup_proj['_id'] = 0
    fup_docs = Mdb['followups'].find(fup_query, fup_proj)
    fup_df = buildframe_fromdocs(fup_docs, inds=['ID'])
    fup_IDs = list(set(fup_df.index.get_level_values('ID').tolist()))

    subject_query = {'ID': {'$in': fup_IDs}}
    subject_fields = ['ID']
    subject_proj = {f:1 for f in subject_fields}
    subject_docs = Mdb['subjects'].find(subject_query, subject_proj)
    subject_df = buildframe_fromdocs(subject_docs, inds=['ID'])

    comb_df = subject_df.join(fup_df)

    for ID, row in tqdm(comb_df.iterrows()):
        session = row['session']
        fup_field = session + '-fup'
        date_field = session + '-fdate'
        Mdb['subjects'].update_one({'_id': row['_id']},
                                     {'$set': {fup_field: row['followup'],
                                               date_field: row['date']}})


def sessions_from_followups():

    fup_query = {'session': {'$ne': np.nan}}
    fup_fields = ['ID', 'session', 'followup', 'date']
    fup_proj = {f:1 for f in fup_fields}
    fup_proj['_id'] = 0
    fup_docs = Mdb['followups'].find(fup_query, fup_proj)
    fupsession_df = buildframe_fromdocs(fup_docs, inds=['ID', 'session'])
    fup_IDs = list(set(fupsession_df.index.get_level_values('ID').tolist()))

    session_query = {'ID': {'$in': fup_IDs}}
    session_fields = ['ID', 'session']
    session_proj = {f:1 for f in session_fields}
    session_docs = Mdb['sessions'].find(session_query, session_proj)
    session_df = buildframe_fromdocs(session_docs, inds=['ID', 'session'])  

    combsession_df = session_df.join(fupsession_df)
    combsession_df.dropna(subset=['date'], inplace=True)

    for ID, row in tqdm(combsession_df.iterrows()):
        fup_field = 'followup'
        date_field = 'followup-date'
        Mdb['sessions'].update_one({'_id': row['_id']},
                                     {'$set': {fup_field: row['followup'],
                                               date_field: row['date']}})    


def match_fups_sessions_flex(coll, nonID_col, nonID_val, date_col='date'):
    print('matching', coll, 'IDs to sessions and followups for',
        nonID_col, nonID_val, 'using', date_col)

    match_collection = Mdb[coll]
    match_query = {nonID_col: nonID_val}
    match_proj = {'_id': 1, 'ID': 1, date_col: 1,}
    match_docs = match_collection.find(match_query, match_proj)
    match_df = buildframe_fromdocs(match_docs, inds=['ID'])
    if match_df.empty:
        print('no matchionnaire docs of this kind found')
        return
    if date_col not in match_df.columns:
        print('no date info available for this matchionnaire-fup combination')
        return
    match_df.rename(columns={date_col: 'match_date', '_id': 'match__id'}, inplace=True)
    IDs = match_df.index.tolist()

    session_proj = {'_id': 1, 'ID': 1, 'session': 1, 'date': 1, 'followup': 1}
    session_docs = Mdb['sessions'].find({'ID': {'$in': IDs}}, session_proj)
    session_df = buildframe_fromdocs(session_docs, inds=['ID'])
    if session_df.empty:
        print('no session docs of this kind found')
        return
    session_df.rename(columns={'date': 'session_date', '_id': 'session__id'}, inplace=True)

    resolve_df = session_df.join(match_df)
    resolve_df['date_diff'] = (resolve_df['session_date'] - resolve_df['match_date']).abs()
    resolve_df.set_index('session', append=True, inplace=True)
    resolve_df.set_index('followup', append=True, inplace=True)

    ID_session_map = dict()
    ID_followup_map = dict()
    for ID in IDs:
        ID_df = resolve_df.ix[resolve_df.index.get_level_values('ID') == ID, :]
        if ID_df.empty:
            continue
        try:
            best_index = ID_df['date_diff'].argmin()
            best_session = best_index[1]
            best_followup = best_index[2]
        except TypeError:  # trying to subscript a nan
            best_session = None
            best_followup = None
        ID_session_map[ID] = best_session
        ID_followup_map[ID] = best_followup

    ID_session_series = pd.Series(ID_session_map, name='nearest_session')
    ID_session_series.index.name = 'ID'

    ID_followup_series = pd.Series(ID_followup_map, name='nearest_followup')
    ID_followup_series.index.name = 'ID'

    match_df_forupdate = match_df.join(ID_session_series)
    match_df_forupdate = match_df_forupdate.join(ID_followup_series)

    for ind, qrow in tqdm(match_df_forupdate.iterrows()):
        match_collection.update_one({'_id': qrow['match__id']},
                                    {'$set': {'session': qrow['nearest_session'],
                                              'followup': qrow['nearest_followup']}})