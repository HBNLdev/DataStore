''' advanced compilation functionality '''

import numpy as np
import pandas as pd

from .compilation import prepare_joindata, get_colldocs, prepare_indices
from .utils.records import flatten_dict
from .knowledge.questionnaires import max_fups

map_allothers = {'neuropsych': {'date_lbl': 'testdate'},
                 'EEGbehavior': {'date_lbl': 'date'}}


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
        try:
            correct_fup = ID_df.loc[row_ind, target_col]
            return correct_fup
        except TypeError:
            return np.nan

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
                            diff_thresh=True, days_thresh=1330,
                            min_age=-np.inf, max_age=np.inf):
    ''' given: 1) a compilation df that has had new info columns joined to it,
        2) the date column name from that new info,
        3) the follow-up column from that new info, and
        4) a list of all new joined columns -->
        return a dataframe in which subjects' info from followups are
        filled into the sessions that are nearest in time '''

    print('doing a time-proximal fill using', days_thresh, 'days as the threshold')

    # maps ID-session tuples that lack follow-up information to the
    # ID-session tuple that is nearest in time
    uID_tup_dict = dict()
    iter_df = comp_dfj.copy()

    ID_index = iter_df.index.get_level_values('ID')
    ID_set = set(ID_index)
    needed_cols = [new_datecol, fup_col, 'date', 'age']
    for ID in ID_set:
        ID_df = iter_df.ix[ID_index == ID, needed_cols]
        if ID_df.shape[0] == 1:
            continue
        fups = ID_df.ix[ID_df[fup_col].notnull(), :]
        if fups.shape[0] < 1:
            continue
        nofups_inagerange = ID_df.ix[(ID_df[fup_col].isnull()) & (ID_df['age'] > min_age) & (ID_df['age'] < max_age), :]
        if nofups_inagerange.shape[0] < 1:
            continue
        for id_i, id_r in nofups_inagerange.iterrows():
            date_diffs = (fups[new_datecol] - id_r['date']).abs()
            best_diff = date_diffs.min().days
            if best_diff < days_thresh:
                best_ind = date_diffs.argmin()
                uID_tup_dict[id_i] = best_ind

    new_index = []
    for uID in iter_df.index:
        try:
            new_index.append(uID_tup_dict[uID])
        except KeyError:
            new_index.append(uID)
    new_index = pd.MultiIndex.from_tuples(new_index, names=['ID', 'session'])
    new_df_joincols = iter_df.loc[new_index, joined_cols]
    new_df_nonjoincols = iter_df.drop(joined_cols, axis=1)
    new_df_nonjoincols.index = new_index

    new_df = pd.concat([new_df_nonjoincols, new_df_joincols], axis=1)
    new_df.index = iter_df.index

    new_filldiff_col = fup_col[:3] + '_date_diff_fill'
    new_df[new_filldiff_col] = (new_df[new_datecol] - new_df['date']).abs()

    return new_df


# main function (for now)

default_fups = ['p2', 'p3'] + ['p4f'+str(f) for f in max_fups]
default_proj = {'_id': 0, 'ID': 1, 'session': 1, 'followup': 1, 'date': 1,
                        'date_diff_session': 1, 'date_diff_followup': 1}


def careful_join(comp_df, collection, subcoll=None,
                 add_query={}, add_proj={}, followups=None,
                 left_datecolumn='date', join_thresh=600,
                 do_fill=False, days_thresh=1330, min_age=-np.inf, max_age=np.inf):
    ''' given a compilation dataframe indexed by ID and some assessment column (e.g. session or followup)
        and with a session date column named by left_datecolumn,
        join a collection / subcollection while handling:
        1.) duplicate ID-followup combinations,
        2.) multiple follow-ups assigned to the same session.
        if desired, fill info from followups to nearest session in time '''

    if not followups:
        fups = default_fups
    else:
        fups = followups

    # handle the dx subcolls
    if subcoll:
        if 'dx_' in subcoll:
            subcoll_safe_prefix = subcoll[3:6]
        else:
            subcoll_safe_prefix = subcoll[:3]
    else:
        subcoll_safe_prefix = collection[:3]

    print('joining', subcoll_safe_prefix)

    new_datecol = subcoll_safe_prefix + '_date'
    fup_col = subcoll_safe_prefix + '_followup'

    # prepare data to join, rename the date column, and
    # drop rows which have the same ID and followup number
    # (should be very few of these, but they are erroneous)
    query = {'followup': {'$in': fups}, 'date_diff_session': {'$lt': join_thresh, '$gt': -join_thresh, }}
    if add_query:
        query.update(add_query)

    proj = {}
    if add_proj:
        proj.update(default_proj)
        proj.update(add_proj)

    join_df = prepare_joindata(comp_df, collection, subcoll,
                               add_query=query, add_proj=proj,
                               left_join_inds=['ID', 'session'])
    n_docs = join_df.shape[0]
    print(n_docs, 'docs found for requested IDs and followups')

    # drop rows with missing session letters
    # join_df_nomaroonedsessions = join_df.reset_index().dropna(subset=['session'])
    # n_docs_nomaroonedsessions = join_df_nomaroonedsessions.shape[0]
    # print(n_docs - n_docs_nomaroonedsessions, 'docs dropped for lacking a session association')

    # drop ID+followup duplicates
    join_df_noIDfupdupes = join_df.reset_index().drop_duplicates(['ID', fup_col])
    n_docs_noIDfupdupes = join_df_noIDfupdupes.shape[0]
    print(n_docs - n_docs_noIDfupdupes, 'docs dropped for being erroneous ID+fup dupes')

    # join in session date info from a sessions-collection-based df
    join_df_sdate = join_df_noIDfupdupes.join(comp_df[left_datecolumn])
    session_datecol = 'session_date'
    join_df_sdate.rename(columns={left_datecolumn: session_datecol}, inplace=True)

    # handle follow-ups which were assigned to the same session letter
    if join_df_sdate.index.has_duplicates:
        join_df_nodupes = handle_dupes(join_df_sdate, new_datecol, fup_col,
                                       session_datecol)
        comp_dfj = comp_df.join(join_df_nodupes)
    else:
        try:
            comp_dfj = comp_df.join(join_df_sdate)
        except ValueError:
            comp_dfj = comp_df.join(join_df_sdate, rsuffix=subcoll_safe_prefix)

    # fill sessions in a "nearest in time" manner within IDs
    # assess info coverage before and after
    joined_cols = [col for col in comp_dfj.columns if subcoll_safe_prefix + '_' in col]
    if do_fill:
        prefill_cvg = comp_dfj[joined_cols].count() / comp_dfj.shape[0]
        comp_dfj_out = time_proximal_fill_fast(comp_dfj, new_datecol, fup_col, joined_cols,
                                               days_thresh=days_thresh, min_age=min_age, max_age=max_age)
        postfill_cvg = comp_dfj_out[joined_cols].count() / comp_dfj_out.shape[0]
        fill_factor = postfill_cvg / prefill_cvg
        print('coverage after filling is up to {:.1f} times higher'.format(
            fill_factor.max()))
    else:
        comp_dfj_out = comp_dfj

    comp_dfj_out[joined_cols].dropna(axis=0, how='all')

    print('new shape is', comp_dfj_out.shape)
    print('------------')

    return comp_dfj_out


def careful_join_ssaga(comp_df, raw=False, do_fill=False, days_thresh=1330,
                       min_age=-np.inf, max_age=np.inf,
                       session_datecol_in='date',
                       add_query={}, add_proj={}, followups=None):
    ''' given a compilation dataframe indexed by ID/session (comp_df) with an age column,
        and with a session date column named by session_datecol_in,
        carefully join the SSAGA and CSSAGA info '''

    coll = 'ssaga'
    prefix = 'ssa_'

    if raw:
        ssaga_subcoll = 'ssaga'
        cssaga_subcoll = 'cssaga'
    else:
        ssaga_subcoll = 'dx_ssaga'
        cssaga_subcoll = 'dx_cssaga'

    if not followups:
        fups = ['p2', 'p3', 0, 1, 2, 3, 4, 5, 6]
    else:
        fups = followups

    print('joining ssaga and cssaga simultaneously')

    # parse the target name to get knowledge about it
    new_datecol = 'ssa_date'
    fup_col = 'ssa_followup'

    id_field = 'ID'
    flatten = True
    left_join_inds = ['ID', 'session']

    query = {id_field: {'$in': list(comp_df.index.get_level_values(id_field))},
             'followup': {'$in': fups}}
    if add_query:
        query.update(add_query)

    proj = {}
    if add_proj:
        default_proj = {'_id': 0, 'ID': 1, 'session': 1, 'followup': 1, 'date': 1, 'date_diff_associate': 1}
        proj.update(default_proj)
        proj.update(add_proj)

    ssaga_docs = get_colldocs(coll, ssaga_subcoll, query, proj)
    cssaga_docs = get_colldocs(coll, cssaga_subcoll, query, proj)
    docs = list(ssaga_docs) + list(cssaga_docs)
    del ssaga_docs
    del cssaga_docs

    if flatten:
        recs = [flatten_dict(r) for r in list(docs)]
    else:
        recs = docs

    join_df = pd.DataFrame.from_records(recs)
    join_df['ID'] = join_df[id_field]

    prepare_indices(join_df, left_join_inds)
    # print('flag1')
    # index_diagnostics(join_df)
    join_df.columns = [prefix + c for c in join_df.columns]
    join_df.dropna(axis=1, how='all', inplace=True)  # drop empty columns
    n_docs = join_df.shape[0]

    # drop rows with missing session letters
    join_df_nomaroonedsessions = join_df.reset_index().dropna(subset=['session'])
    n_docs_nomaroonedsessions = join_df_nomaroonedsessions.shape[0]
    print(n_docs - n_docs_nomaroonedsessions, 'docs dropped for lacking a session association')

    # drop ID+followup duplicates
    join_df_noIDfupdupes = join_df_nomaroonedsessions.drop_duplicates(['ID', fup_col]).set_index(['ID', 'session'])
    n_docs_noIDfupdupes = join_df_noIDfupdupes.shape[0]
    print(n_docs_nomaroonedsessions - n_docs_noIDfupdupes, 'docs dropped for being erroneous ID+fup dupes')

    # join in session date info from a sessions-collection-based df
    join_df_sdate = join_df_noIDfupdupes.join(comp_df[session_datecol_in])
    # print('flag3')
    # index_diagnostics(join_df_sdate)
    session_datecol = 'session_date'
    join_df_sdate.rename(columns={session_datecol_in: session_datecol}, inplace=True)

    # handle follow-ups which were assigned to the same session letter
    if join_df_sdate.index.has_duplicates:
        join_df_nodupes = handle_dupes(join_df_sdate, new_datecol, fup_col,
                                       session_datecol)
        # print('flag4')
        # index_diagnostics(join_df_nodupes)
        comp_dfj = comp_df.join(join_df_nodupes)
    else:
        comp_dfj = comp_df.join(join_df_sdate)

    # fill sessions in a "nearest in time" manner within IDs
    # assess info coverage before and after
    joined_cols = [col for col in comp_dfj.columns if prefix in col]
    if do_fill:
        prefill_cvg = comp_dfj[joined_cols].count() / comp_dfj.shape[0]
        comp_dfj_out = time_proximal_fill_fast(comp_dfj, new_datecol, fup_col, joined_cols,
                                               days_thresh=days_thresh, min_age=min_age, max_age=max_age)
        postfill_cvg = comp_dfj_out[joined_cols].count() / comp_dfj_out.shape[0]
        fill_factor = postfill_cvg / prefill_cvg
        print('coverage after filling is up to {:.1f} times higher'.format(fill_factor.max()))
    else:
        comp_dfj_out = comp_dfj

    comp_dfj_out[joined_cols].dropna(axis=0, how='all')

    print('new shape is', comp_dfj_out.shape)

    return comp_dfj_out
