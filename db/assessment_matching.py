import pandas as pd
from tqdm import tqdm

from .compilation import buildframe_fromdocs
from .organization import Mdb


def match_assessments(match_coll, to_coll,
                      fup_field, fup_val, match_datefield='date',
                      add_match_query=None):
    ''' modify match_coll docs to reflect their optimal date-matching with to_coll.

        match_coll is a collection with multiple assessments
        to_coll is the assessment-tracking collection to match dates with (e.g. sessions or followups)

        fup_field is the name of the field labelling the distinct multiple assessments.
        fup_val is the current assessment to match.

        NOTE: each ID MUST have only one record with assessment_col of assessment_val.

        match_datefield is the date column for the match_coll
        add_match_query should be used when e.g. there are subcollections '''

    print('matching', match_coll, 'assessments to', to_coll, 'for',
          fup_field, fup_val, 'using', match_datefield)

    match_collection = Mdb[match_coll]
    match_query = {fup_field: fup_val}
    if add_match_query:
        print('with the additional query of', add_match_query)
        match_query.update(add_match_query)
    match_proj = {'_id': 1, 'ID': 1, match_datefield: 1, }
    match_docs = match_collection.find(match_query, match_proj)
    match_df = buildframe_fromdocs(match_docs, inds=['ID'])
    if match_df.empty:
        print('no assessment docs of this kind found')
        return
    if match_datefield not in match_df.columns:
        print('no date info available for these assessment docs')
        return
    match_df.rename(columns={match_datefield: 'match_date', '_id': 'match__id'}, inplace=True)
    IDs = match_df.index.tolist()

    sfup_query = {'ID': {'$in': IDs}}
    sfup_proj = {'_id': 1, 'ID': 1, 'session': 1, 'date': 1, 'followup': 1}
    sfup_docs = Mdb[to_coll].find(sfup_query, sfup_proj)
    sfup_df = buildframe_fromdocs(sfup_docs, inds=['ID'])
    if sfup_df.empty:
        print('no session docs of this kind found')
        return
    sfup_df.rename(columns={'date': 'sfup_date', '_id': 'session__id'}, inplace=True)

    resolve_df = sfup_df.join(match_df)
    resolve_df['date_diff'] = resolve_df['sfup_date'] - resolve_df['match_date']
    resolve_df['date_diff_abs'] = resolve_df['date_diff'].abs()
    sfup_index = to_coll[:-1]  # i.e. session for sessions, followup for followups
    resolve_df.set_index(sfup_index, append=True, inplace=True)

    ID_info_lst = []

    ID_index = resolve_df.index.get_level_values('ID')
    te_dummy = 0
    ae_dummy = 0
    for ID in IDs:

        ID_df = resolve_df.ix[ID_index == ID, :]

        if ID_df.empty:
            continue

        smallest_ind = ID_df['date_diff_abs'].argmin()
        try:
            best_sfup = smallest_ind[1]
            best_diff = ID_df.loc[smallest_ind, 'date_diff'].days
            best_date = ID_df.loc[smallest_ind, 'sfup_date']
        except TypeError:
            te_dummy += 1  # the best sfup is a nan because there was no minimum date difference.
            # this situation occurs when no date information in the match_coll was found for
            continue  # an individual who was found in the to_coll
        except AttributeError:
            ae_dummy += 1
            continue  # ???

        info_dict = {'ID': ID,
                     'nearest_sfup': best_sfup,
                     'date_diff_sfup': best_diff,
                     'date_sfup': best_date}
        ID_info_lst.append(info_dict)

    # print(te_dummy, 'type errors occurred')
    # print(ae_dummy, 'attribute errors occurred')

    ID_info_df = pd.DataFrame.from_records(ID_info_lst).set_index('ID')
    ID_info_forupdate = ID_info_df.join(match_df)

    for ind, irow in tqdm(ID_info_forupdate.iterrows()):
        match_collection.update_one({'_id': irow['match__id']},
                                    {'$set': {sfup_index: irow['nearest_sfup'],
                                              'date_diff_' + sfup_index: irow['date_diff_sfup'],
                                              'date_' + sfup_index: irow['date_sfup']}
                                     })
