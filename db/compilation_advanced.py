''' advanced compilation functionality '''

import numpy as np
import pandas as pd

from .compilation import prepare_joindata, get_colldocs, prepare_indices
from .organization import flatten_dict
from .quest_import import map_ph4, map_ph4_ssaga, build_inputdict

map_allothers = {'neuropsych': {'date_lbl': 'testdate'},
                 'EEGbehavior': {'date_lbl': 'date'}}

# utility functions


def get_kdict(collection, subcoll):
    ''' given a collection and subcollection, return the knowledge dict '''

    if collection is 'questionnaires':
        kmap = map_ph4
    elif collection is 'ssaga':
        kmap = map_ph4_ssaga
    elif collection in map_allothers:
        kmap = map_allothers
    else:
        print('collection not supported')
        kmap = None
        return
    i = build_inputdict(subcoll, kmap)
    return i


def handle_dupes(join_df_sdate, new_datecol, fup_col, session_datecol):
    ''' given a dataframe with a session-date column, a new info date column,
        and a follow-up column, return a version with duplicates removed '''

    def determine_session(row, df, datediff_col, target_col):
        ''' row-apply function. given a dataframe containing duplicates,
            a date difference column, and a target column (e.g. follow-up),
            determine the nearest to use '''

        ID, session = row.name
        ID_df = df.loc[ID].reset_index()
        row_ind = ID_df[datediff_col].argmin()
        correct_fup = ID_df.loc[row_ind, target_col]
        return correct_fup

    # find duplicate indices
    dupe_inds = join_df_sdate[join_df_sdate.index.duplicated()]. \
        index.values
    print('{} duplicates found'.format(len(dupe_inds)))

    # generate date_diff dataframe
    join_df_sdate['date_diff'] = join_df_sdate[new_datecol] - \
                                 join_df_sdate[session_datecol]
    join_df_sdate['date_diff'] = join_df_sdate['date_diff'].abs()
    dupe_df = join_df_sdate.loc[dupe_inds,
                        [session_datecol, new_datecol, fup_col, 'date_diff']]

    # apply custom function to determine the correct fup
    dupe_df['correct_fup'] = dupe_df.apply(determine_session,
                               axis=1, args=[dupe_df, 'date_diff', fup_col])

    # get incorrect fup indices and add fup as an index
    dupe_df_bad = dupe_df[dupe_df['correct_fup'] != dupe_df[fup_col]]
    join_df_sdate_fups = join_df_sdate.set_index(fup_col, append=True)
    dupe_df_bad_fupind = dupe_df_bad.set_index(fup_col, append=True)
    dupe_df_badinds = dupe_df_bad_fupind.index

    # drop them from the version of the df with the fup as an index
    join_df_sdate_goodfups = join_df_sdate_fups.drop(dupe_df_badinds)
    join_df_sdate_goodfups.reset_index(fup_col, inplace=True)

    # drop the session date information back out
    join_df_nodupes = join_df_sdate_goodfups.drop(
        ['session_date', 'date_diff'], axis=1)

    return join_df_nodupes


def time_proximal_fill_fast(comp_dfj, new_datecol, fup_col, joined_cols,
                                min_age=-np.inf, max_age=np.inf):
    ''' given: 1) a compilation df that has had new info columns joined to it,
        2) the date column name from that new info,
        3) the follow-up column from that new info, and
        4) a list of all new joined columns -->
        return a dataframe in which subjects' info from followups are
        filled into the sessions that are nearest in time '''

    # maps ID-session tuples that lack follow-up information to the
    # ID-session tuple that is nearest in time
    uID_tup_dict = dict()
    iter_df = comp_dfj.copy()
    iter_df['date_diff'] = (iter_df[new_datecol] - iter_df['date']).abs()
    IDs = list(set(iter_df.index.get_level_values('ID')))
    needed_cols = [new_datecol, fup_col, 'date', 'age']
    for ID in IDs:
        ID_df = iter_df.ix[iter_df.index.get_level_values('ID') == ID,
                            needed_cols]
        if ID_df.shape[0] == 1:
            continue
        fups = ID_df[~ID_df[fup_col].isnull()]
        if fups.shape[0] < 1:
            continue
        for id_i, id_r in ID_df.iterrows():
            # if (isinstance(id_r[fup_col], pd.tslib.NaTType)) and (id_r['age'] > min_age) and (id_r['age'] < max_age):
            if (np.isnan(id_r[fup_col])) and (id_r['age'] > min_age) and (id_r['age'] < max_age):
                date_diffs = (fups.loc[:, new_datecol] -
                              id_r['date']).abs()
                best_ind = date_diffs.argmin()
                uID_tup_dict[id_i] = best_ind

    new_index = []
    for uID in iter_df.index:
        try:
            new_index.append(uID_tup_dict[uID])
        except:
            new_index.append(uID)
    new_index = pd.Index(new_index)
    new_df_joincols = iter_df.loc[new_index, joined_cols]
    new_df_nonjoincols = iter_df.drop(joined_cols, axis=1)
    new_df_nonjoincols.index = new_index
    new_df = pd.concat([new_df_nonjoincols, new_df_joincols], axis=1)
    new_df.index = iter_df.index
    # new_df.rename({'date_diff':})
    new_colorder = [col for col in new_df.columns if col!='date_diff']+\
                                                            ['date_diff']
    new_df = new_df[new_colorder]
    new_df.rename(columns={'date_diff': fup_col[:3]+'_date_diff'},
                  inplace=True)

    return new_df

# main function (for now)


def careful_join(comp_df, collection, subcoll=None, do_fill=False,
                 min_age=-np.inf, max_age=np.inf,
                 session_datecol_in='date'):
    ''' given a compilation dataframe indexed by ID/session (comp_df) with an age column,
        and with a session date column named by session_datecol_in,
        join a collection / subcollection while handling:
        1.) duplicate ID-followup combinations,
        2.) multiple follow-ups assigned to the same session.
        if desired, fill info from followups to nearest session in time '''

    # handle the dx subcolls
    if subcoll:
        if 'dx_' in subcoll:
            subcoll_safe = subcoll[3:]
        else:
            subcoll_safe = subcoll
    else:
        subcoll_safe = None

    # parse the target name to get knowledge about it
    i = get_kdict(collection, subcoll_safe)
    if isinstance(i['date_lbl'], list):
        old_datecol = subcoll_safe[:3] + '_' + '-'.join(i['date_lbl'])
    else:
        old_datecol = subcoll_safe[:3] + '_' + i['date_lbl']
    new_datecol = subcoll_safe[:3] + '_date'
    fup_col = subcoll_safe[:3] + '_followup'

    # prepare data to join, rename the date column, and
    # drop rows which have the same ID and followup number
    # (should be very few of these, but they are erroneous)
    join_df = prepare_joindata(comp_df, collection, subcoll,
                                left_join_inds=['ID', 'session'])
    join_df.rename(columns={old_datecol: new_datecol}, inplace=True)
    join_df = join_df.reset_index().drop_duplicates(['ID', fup_col]).\
                                    set_index(['ID', 'session'])
    join_df.sort_index(inplace=True)

    # join in session date info from a sessions-collection-based df
    join_df_sdate = join_df.join(comp_df[session_datecol_in])
    session_datecol = 'session_date'
    join_df_sdate.rename(columns={session_datecol_in: session_datecol,
                                  old_datecol: new_datecol}, inplace=True)

    # handle follow-ups which were assigned to the same session letter
    if join_df_sdate.index.has_duplicates:
        join_df_nodupes = handle_dupes(join_df_sdate, new_datecol, fup_col,
                                                            session_datecol)
        comp_dfj = comp_df.join(join_df_nodupes)
    else:
        comp_dfj = comp_df.join(join_df_sdate)

    # fill sessions in a "nearest in time" manner within IDs
    # assess info coverage before and after
    joined_cols = [col for col in comp_dfj.columns if subcoll_safe[:3] + '_' in col]
    if do_fill:
        prefill_cvg = comp_dfj[joined_cols].count() / comp_dfj.shape[0]
        comp_dfj_out = time_proximal_fill_fast(comp_dfj, new_datecol, fup_col,
                                                joined_cols, min_age=min_age, max_age=max_age)
        postfill_cvg = comp_dfj_out[joined_cols].count() / comp_dfj_out.shape[0]
        fill_factor = postfill_cvg / prefill_cvg
        print('coverage after filling is up to {:.1f} times higher'.format(
                                                                fill_factor.max()))
    else:
        comp_dfj_out = comp_dfj

    comp_dfj_out[joined_cols].dropna(axis=0, how='all')
    return comp_dfj_out


def careful_join_ssaga(comp_df, do_fill=False, min_age=-np.inf, max_age=np.inf,
                 session_datecol_in='date'):
    ''' given a compilation dataframe indexed by ID/session (comp_df) with an age column,
        and with a session date column named by session_datecol_in,
        carefully join the SSAGA and CSSAGA info '''

    coll = 'ssaga'
    collection = 'ssaga'
    subcoll_safe = 'ssaga'
    prefix = 'ssa_'

    # parse the target name to get knowledge about it
    i = get_kdict(collection, subcoll_safe)
    if isinstance(i['date_lbl'], list):
        old_datecol = subcoll_safe[:3] + '_' + '-'.join(i['date_lbl'])
    else:
        old_datecol = subcoll_safe[:3] + '_' + i['date_lbl']
    new_datecol = subcoll_safe[:3] + '_date'
    fup_col = subcoll_safe[:3] + '_followup'

    id_field = 'ID'
    add_proj = {}
    flatten = True
    left_join_inds = ['ID', 'session']
    query = {id_field: {'$in': list(
        comp_df.index.get_level_values(id_field))}}
    ssaga_docs = get_colldocs(coll, 'dx_ssaga', query, add_proj)
    cssaga_docs = get_colldocs(coll, 'dx_cssaga', query, add_proj)
    docs = list(ssaga_docs) + list(cssaga_docs)
    del ssaga_docs
    del cssaga_docs

    if flatten:
        recs =[flatten_dict(r) for r in list(docs)]
    else:
        recs = docs

    join_df = pd.DataFrame.from_records(recs)
    join_df['ID'] = join_df[id_field]

    prepare_indices(join_df, left_join_inds)
    join_df.columns = [prefix + c for c in join_df.columns]
    join_df.dropna(axis=1, how='all', inplace=True) # drop empty columns
    join_df.sort_index(inplace=True)                # sort

    join_df.rename(columns={old_datecol: new_datecol}, inplace=True)
    join_df = join_df.reset_index().drop_duplicates(['ID', fup_col]).\
                                    set_index(['ID', 'session'])
    join_df.sort_index(inplace=True)

    # join in session date info from a sessions-collection-based df
    join_df_sdate = join_df.join(comp_df[session_datecol_in])
    session_datecol = 'session_date'
    join_df_sdate.rename(columns={session_datecol_in: session_datecol,
                                  old_datecol: new_datecol}, inplace=True)

    # handle follow-ups which were assigned to the same session letter
    if join_df_sdate.index.has_duplicates:
        join_df_nodupes = handle_dupes(join_df_sdate, new_datecol, fup_col,
                                                            session_datecol)
        comp_dfj = comp_df.join(join_df_nodupes)
    else:
        comp_dfj = comp_df.join(join_df_sdate)

    # fill sessions in a "nearest in time" manner within IDs
    # assess info coverage before and after
    joined_cols = [col for col in comp_dfj.columns if subcoll_safe[:3] + '_' in col]
    if do_fill:
        prefill_cvg = comp_dfj[joined_cols].count() / comp_dfj.shape[0]
        comp_dfj_out = time_proximal_fill_fast(comp_dfj, new_datecol, fup_col,
                                                joined_cols, min_age=min_age, max_age=max_age)
        postfill_cvg = comp_dfj_out[joined_cols].count() / comp_dfj_out.shape[0]
        fill_factor = postfill_cvg / prefill_cvg
        print('coverage after filling is up to {:.1f} times higher'.format(
                                                                fill_factor.max()))
    else:
        comp_dfj_out = comp_dfj

    comp_dfj_out[joined_cols].dropna(axis=0, how='all')
    return comp_dfj_out