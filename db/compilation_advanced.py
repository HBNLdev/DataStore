''' advanced compilation functionality '''

import numpy as np
import pandas as pd

from db import eromat as EM
from .compilation import prepare_joindata, get_colldocs, prepare_indices
from .knowledge.questionnaires import max_fups
from .utils.records import flatten_dict


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
    print(len(dupe_inds), 'uID duplicates found for requested IDs and assessments, keeping only nearest matching')

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
                            days_thresh=1330, min_age=-np.inf, max_age=np.inf):
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

    ID_index = comp_dfj.index.get_level_values('ID')
    ID_set = set(ID_index)
    needed_cols = [new_datecol, fup_col, 'date', 'age']
    for ID in ID_set:
        ID_df = comp_dfj.ix[ID_index == ID, needed_cols]
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
    for uID in comp_dfj.index:
        try:
            new_index.append(uID_tup_dict[uID])
        except KeyError:
            new_index.append(uID)
    new_index = pd.MultiIndex.from_tuples(new_index, names=['ID', 'session'])
    new_df_joincols = comp_dfj.loc[new_index, joined_cols]
    new_df_nonjoincols = comp_dfj.drop(joined_cols, axis=1)
    new_df_nonjoincols.index = new_index

    new_df = pd.concat([new_df_nonjoincols, new_df_joincols], axis=1)
    new_df.index = comp_dfj.index

    fill_uIDs = list(uID_tup_dict.keys())
    print(len(fill_uIDs), 'uIDs were filled')

    wasfilled_col = fup_col[:3] + '_wasfilled'
    filldiff_col = fup_col[:3] + '_date_diff_fill'
    new_df[wasfilled_col] = np.nan

    new_df.loc[fill_uIDs, wasfilled_col] = 'x'
    new_df[filldiff_col] = np.nan
    new_df.loc[fill_uIDs, filldiff_col] = (new_df.loc[fill_uIDs, new_datecol] - new_df.loc[fill_uIDs, 'date']).abs()

    return new_df


# main function (for now)

default_fups = ['p2', 'p3'] + ['p4f' + str(f) for f in range(max_fups + 1)]
default_proj = {'_id': 0, 'ID': 1, 'session': 1, 'followup': 1, 'date': 1,
                'date_diff_session': 1, 'date_diff_followup': 1}


def careful_join(comp_df, collection, subcoll=None,
                 add_query={}, add_proj={}, followups=None,
                 left_datecolumn='date', join_thresh=600,
                 do_fill=False, fill_thresh=1330, min_age=-np.inf, max_age=np.inf):
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

    # drop ID+followup duplicates
    join_df_noIDfupdupes = join_df.reset_index().drop_duplicates(['ID', fup_col]).set_index(['ID', 'session'])
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
    joined_cols = [col for col in comp_dfj.columns if col[:4] == subcoll_safe_prefix + '_']
    if do_fill:
        prefill_cvg = comp_dfj[joined_cols].count() / comp_dfj.shape[0]
        comp_dfj_out = time_proximal_fill_fast(comp_dfj, new_datecol, fup_col, joined_cols,
                                               days_thresh=fill_thresh, min_age=min_age, max_age=max_age)
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


def careful_join_ssaga(comp_df, raw=False,
                       add_query={}, add_proj={}, followups=None,
                       left_datecolumn='date', join_thresh=600,
                       do_fill=False, fill_thresh=1330, min_age=-np.inf, max_age=np.inf):
    ''' given a compilation dataframe indexed by ID/session (comp_df) with an age column,
        and with a session date column named by session_datecol_in,
        carefully join the SSAGA and CSSAGA info '''

    coll = 'ssaga'
    subcoll_safe_prefix = 'ssa'

    if raw:
        ssaga_subcoll = 'ssaga'
        cssaga_subcoll = 'cssaga'
    else:
        ssaga_subcoll = 'dx_ssaga'
        cssaga_subcoll = 'dx_cssaga'

    if not followups:
        fups = ['p1'] + default_fups
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
             'followup': {'$in': fups},
             'date_diff_session': {'$lt': join_thresh, '$gt': -join_thresh, }}
    if add_query:
        query.update(add_query)

    proj = {}
    if add_proj:
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

    prepare_indices(join_df, left_join_inds)

    join_df.columns = [subcoll_safe_prefix + '_' + c for c in join_df.columns]
    join_df.dropna(axis=1, how='all', inplace=True)  # drop empty columns
    n_docs = join_df.shape[0]

    print(n_docs, 'docs found for requested IDs and followups')

    # drop ID+followup duplicates
    join_df_noIDfupdupes = join_df.reset_index().drop_duplicates(['ID', fup_col]).set_index(['ID', 'session'])
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
        comp_dfj = comp_df.join(join_df_sdate)

    # fill sessions in a "nearest in time" manner within IDs
    # assess info coverage before and after
    joined_cols = [col for col in comp_dfj.columns if col[:4] == subcoll_safe_prefix + '_']
    if do_fill:
        prefill_cvg = comp_dfj[joined_cols].count() / comp_dfj.shape[0]
        comp_dfj_out = time_proximal_fill_fast(comp_dfj, new_datecol, fup_col, joined_cols,
                                               days_thresh=fill_thresh, min_age=min_age, max_age=max_age)
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

# utility functions
def compile_ERO(baseDF,proc_types,pwr_types,exs_cases,elecs,ex_tf_wins):
    compD = {}
    compDF = baseDF.reset_index().set_index(['ID','session'])
    start_cols = compDF.columns.copy()
    for typ in proc_types:
        colsB = compDF.columns
        compDF = EM.add_eropaths( compDF, typ, exp_cases=exs_cases, power_types=pwr_types)
        new_cols = set(compDF.columns).difference(colsB)
        new_path_cols = [ c for c in new_cols if typ in c]
        for col in new_path_cols:
            exp = [ ex for ex in exs_cases.keys() if ex in col][0]
            if col not in compD:
                try:
                    stack = EM.EROStack( compDF[col].tolist() )
                    compD[col] = stack.tfmean_multiwin_chans( ex_tf_wins[exp], elecs )
                except Exception as err:
                    print(err)
                    
        return compD, compDF
    
def clean_col(col):
    if col[1] == '':
        return col[0]
    else: return col    
    
def cols_for_join(DF):
    names = DF.index.names
    DFw = DF.reset_index().set_index(['ID','session'])
    DFw.columns = [clean_col(col) for col in DFw.columns.tolist()]
    #print(list(DFw.columns))\
    suffix = ''
    for nm in names:
        if nm not in ['ID','session']:
            val = DFw[nm].unique()
            if len(val) > 1:
                print(nm+' has multiple values')
                return
            suffix += '_'+val[0]
            DFw.drop(nm,axis=1,inplace=True)
            
    DFw.columns = [ '_'.join(c)+suffix for c in DFw.columns.tolist() ]            
    #DFw.columns = ['_'.join(c) if isinstance(c,tuple) else c for c in DFw.columns]
    return DFw                  
    
def join_compiled_ERO(DF,comps):
    jDF = DF.copy().reset_index()
    for comp in comps.values():
        jDF = jDF.join(cols_for_join(comp),on=['ID','session'])
        
    return jDF.set_index(['ID','session'])