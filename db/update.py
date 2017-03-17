''' performing updates on documents '''

from tqdm import tqdm

from .assessment_matching import match_assessments
from .compilation import buildframe_fromdocs
from .knowledge.questionnaires import max_fups
from .organization import Mdb


# utils


def clear_field(coll, field):
    ''' clear collection coll of field (delete the key-value pair from all docs) '''

    query = {field: {'$exists': True}}
    updater = {'$unset': {field: ''}}

    Mdb[coll].update_many(query, updater)


# main functions


def followups_from_sessions():
    p123_fups = ['p1', 'p2', 'p3', ]
    p4_fups = ['p4f' + str(f) for f in range(max_fups + 1)]
    fups = p123_fups + p4_fups
    for fup in fups:
        match_assessments('followups', to_coll='sessions',
                          fup_field='followup', fup_val=fup,
                          match_datefield='date')


def sessions_from_followups():
    fups = list(map(chr, range(97, 100 + max_fups)))
    for fup in fups:
        match_assessments('sessions', to_coll='followups',
                          fup_field='session', fup_val=fup,
                          match_datefield='date')


def subjects_from_followups():
    fup_query = {'session': {'$exists': True}}
    fup_fields = ['ID', 'session', 'followup', 'date']
    fup_proj = {f: 1 for f in fup_fields}
    fup_proj['_id'] = 0
    fup_docs = Mdb['followups'].find(fup_query, fup_proj)
    fup_df = buildframe_fromdocs(fup_docs, inds=['ID'])
    fup_IDs = list(set(fup_df.index.get_level_values('ID').tolist()))

    subject_query = {'ID': {'$in': fup_IDs}}
    subject_fields = ['ID']
    subject_proj = {f: 1 for f in subject_fields}
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


def ssaga_from_sessions():
    ssaga_subcolls = ['ssaga', 'cssaga', 'pssaga', 'dx_ssaga', 'dx_cssaga', 'dx_pssaga']
    p4_fups = ['p4f' + str(f) for f in range(max_fups + 1)]
    ssaga_fups = ['p1', 'p2', 'p3', ] + p4_fups
    for ssaga_subcoll in ssaga_subcolls:
        for fup in ssaga_fups:
            subcoll_query = {'questname': ssaga_subcoll}
            match_assessments('ssaga', to_coll='sessions',
                              fup_field='followup', fup_val=fup,
                              match_datefield='date',
                              add_match_query=subcoll_query)


def questionnaires_from_sessions():
    quest_subcolls = ['dependence', 'craving', 'sre', 'daily', 'neo', 'sensation', 'aeq', 'bis', 'achenbach']
    p4_fups = ['p4f' + str(f) for f in range(max_fups + 1)]
    quest_fups = ['p2', 'p3', ] + p4_fups
    for quest_subcoll in quest_subcolls:
        for fup in quest_fups:
            subcoll_query = {'questname': quest_subcoll}
            match_assessments('questionnaires', to_coll='sessions',
                              fup_field='followup', fup_val=fup,
                              match_datefield='date',
                              add_match_query=subcoll_query)


def neuropsych_from_sfups():
    max_npsych_fups = max(Mdb['neuropsych'].distinct('np_followup'))
    for fup in range(max_npsych_fups + 1):
        # match sessions
        match_assessments('neuropsych', to_coll='sessions',
                          fup_field='np_followup', fup_val=fup,
                          match_datefield='date')
        # match followups
        match_assessments('neuropsych', to_coll='followups',
                          fup_field='np_followup', fup_val=fup,
                          match_datefield='date')


def internalizing_from_ssaga():
    ''' adds date and session fields to internalizing followup docs, using the ssaga collection '''

    int_query = {}
    int_proj = {'ID': 1, 'questname': 1, 'followup': 1, '_id': 1}
    int_docs = Mdb['internalizing'].find(int_query, int_proj)
    int_df = buildframe_fromdocs(int_docs, inds=['ID', 'questname', 'followup'])

    IDs = list(set(int_df.index.get_level_values('ID')))

    ssaga_query = {'questname': {'$in': ['ssaga', 'cssaga']}, 'ID': {'$in': IDs}, 'session': {'$exists': True}}
    ssaga_proj = {'ID': 1, 'session': 1, 'date_diff_session': 1, 'followup': 1, 'questname': 1, 'date': 1, '_id': 0}

    ssaga_docs = Mdb['ssaga'].find(ssaga_query, ssaga_proj)
    ssaga_df = buildframe_fromdocs(ssaga_docs, inds=['ID', 'questname', 'followup'])

    comb_df = int_df.join(ssaga_df).dropna(subset=['date'])
    print(int_df.shape[0] - comb_df.shape[0], 'internalizing docs could not be matched')

    for ID, row in tqdm(comb_df.iterrows()):
        Mdb['internalizing'].update_one({'_id': row['_id']},
                                        {'$set': {'session': row['session'],
                                                  'date_diff_session': row['date_diff_session'],
                                                  'date': row['date'], }
                                         })
