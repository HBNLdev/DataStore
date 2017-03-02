''' performing updates on documents '''

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

def neuropsych_from_sfups():

    max_fups = max(Mdb['neuropsych'].distinct('np_followup'))
    for fup in range(max_fups + 1):
        # match sessions
        match_fups_sessions_flex('neuropsych', assessment_col='np_followup', assessment_val=fup,
                                 sfup_coll='sessions', date_col='testdate')
        # match followups
        match_fups_sessions_flex('neuropsych', assessment_col='np_followup', assessment_val=fup,
                                 sfup_coll='followups', date_col='testdate')



def clear_field(coll, field):
    ''' clear collection coll of field (delete the key-value pair from all docs) '''

    query = {field: {'$exists': True}}
    updater = {'$unset': {field: ''}}

    Mdb[coll].update_many(query, updater)


def match_fups_sessions_flex(match_coll, assessment_col, assessment_val,
                             sfup_coll='sessions', date_col='date'):
    print('matching', match_coll, 'assessments to', sfup_coll, 'for',
          assessment_col, assessment_val, 'using', date_col)

    match_collection = Mdb[match_coll]
    match_query = {assessment_col: assessment_val}
    match_proj = {'_id': 1, 'ID': 1, date_col: 1,}
    match_docs = match_collection.find(match_query, match_proj)
    match_df = buildframe_fromdocs(match_docs, inds=['ID'])
    if match_df.empty:
        print('no assessment docs of this kind found')
        return
    if date_col not in match_df.columns:
        print('no date info available for these assessment docs')
        return
    match_df.rename(columns={date_col: 'match_date', '_id': 'match__id'}, inplace=True)
    IDs = match_df.index.tolist()

    sfup_proj = {'_id': 1, 'ID': 1, 'session': 1, 'date': 1, 'followup': 1}
    sfup_docs = Mdb[sfup_coll].find({'ID': {'$in': IDs}}, sfup_proj)
    sfup_df = buildframe_fromdocs(sfup_docs, inds=['ID'])
    if sfup_df.empty:
        print('no session docs of this kind found')
        return
    sfup_df.rename(columns={'date': 'sfup_date', '_id': 'session__id'}, inplace=True)

    resolve_df = sfup_df.join(match_df)
    resolve_df['date_diff'] = (resolve_df['sfup_date'] - resolve_df['match_date']).abs()
    sfup_index = sfup_coll[:-1]  # i.e. session for sessions, followup for followup
    resolve_df.set_index(sfup_index, append=True, inplace=True)

    ID_sfup_map = dict()
    ID_datediff_map = dict()
    for ID in IDs:
        ID_df = resolve_df.ix[resolve_df.index.get_level_values('ID') == ID, :]
        if ID_df.empty:
            continue
        try:
            best_index = ID_df['date_diff'].argmin()
            best_sfup = best_index[1]
            best_datediff = ID_df.loc[best_index, 'date_diff']
        except TypeError:  # trying to subscript a nan
            best_sfup = None
            best_datediff = None
        ID_sfup_map[ID] = best_sfup
        ID_datediff_map[ID] = best_datediff

    ID_sfup_series = pd.Series(ID_sfup_map, name='nearest_sfup')
    ID_sfup_series.index.name = 'ID'
    ID_datediff_series = pd.Series(ID_datediff_map, name='date_diff')
    ID_datediff_series.index.name = 'ID'

    match_df_forupdate = match_df.join(ID_sfup_series)
    match_df_forupdate = match_df_forupdate.join(ID_datediff_series)

    for ind, qrow in tqdm(match_df_forupdate.iterrows()):
        try:
            datediff_days = qrow['date_diff'].days
        except AttributeError:
            datediff_days = None
        match_collection.update_one({'_id': qrow['match__id']},
                                    {'$set': {sfup_index: qrow['nearest_sfup'],
                                              sfup_index+'_datediff': datediff_days}
                                     })


# graveyard

# def internalizing_from_ssaga():
#     ''' adds date and session fields to internalizing followup docs, using the ssaga collection '''

#     int_query = {}
#     int_proj = {'_id': 1}
#     int_docs = Mdb['internalizing'].find(int_query, int_proj)
#     int_df4 = buildframe_fromdocs(int_docs, inds=['ID', 'questname', 'followup'])

#     IDs = list(set(int_df4.index.get_level_values('ID')))

#     ssaga_query = {'questname': {'$in': ['ssaga', 'cssaga']}, 'ID': {'$in': IDs}}
#     ssaga_proj = {'ID': 1, 'session': 1, 'followup': 1, 'questname': 1, 'date': 1, '_id': 0}

#     ssaga_docs = Mdb['ssaga'].find(ssaga_query, ssaga_proj)
#     ssaga_df = buildframe_fromdocs(ssaga_docs, inds=['ID', 'questname', 'followup'])

#     comb_df = int_df4.join(ssaga_df, lsuffix='_int')
    
#     for ID, row in tqdm(comb_df.iterrows()):
#         Mdb['internalizing'].update_one({'_id': row['_id']},
#             {'$set': {'session': row['session'],
#                       'date': row['date'],} ,} )