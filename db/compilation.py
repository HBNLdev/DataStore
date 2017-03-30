''' compilation tools for the HBNL '''

import pprint

import numpy as np
import pandas as pd

from .organization import Mdb
from .utils.records import flatten_dict

pp = pprint.PrettyPrinter(indent=4)

subjects_queries = {'AAfamGWAS': {'AAfamGWAS': 'x'},
                    'AAfamGWASfam': {'AAfamGWAS': 'f'},
                    'a-subjects': {'POP': 'A'},
                    'BrainDysfunction': {'POP': {'$in': ['A', 'C', 'H', 'P']},
                                         'EEG': 'x'},
                    'BrainDysfunction-AC': {'POP': {'$in': ['A', 'C']},
                                            'EEG': 'x'},
                    'ccGWAS': {'ccGWAS': {'$ne': np.nan}},
                    'COGA': {'POP': {'$in': ['COGA', 'COGA-Ctl', 'IRPG', 'IRPG-Ctl']}, 'EEG': 'x'},
                    'COGA11k': {'COGA11k-fam': {'$ne': np.nan}},
                    'COGA4500': {'4500': 'x'},
                    'c-subjects': {'POP': 'C'},
                    'EAfamGWAS': {'EAfamGWAS': 'x'},
                    'EAfamGWASfam': {'EAfamGWAS': 'f'},
                    'ExomeSeq': {'ExomeSeq': 'x'},
                    'fMRI-NKI-bd1': {'fMRI': {'$in': ['1a', '1b']}},
                    # 'fMRI-NKI-bd2':{'fMRI':{'$in':['2a','2b']}},
                    # 'fMRI-NYU-hr':{'fMRI':{'$in':['3a','3b']}},
                    'h-subjects': {'POP': 'H'},
                    'HighRisk': {'POP': {'$in': ['COGA', 'COGA-Ctl', 'IRPG',
                                                 'IRPG-Ctl']},
                                 'site': 'suny', 'EEG': 'x'},
                    'HighRiskFam': {'POP': {'$in': ['COGA', 'COGA-Ctl', 'IRPG',
                                                    'IRPG-Ctl']},
                                    'site': 'suny'},
                    'PhaseIV': {'Phase4-session':
                                    {'$in': ['a', 'b', 'c', 'd']}},
                    'p-subjects': {'POP': 'P'},
                    'smokeScreen': {'SmS': {'$ne': np.nan}},
                    'bigsmokeScreen': {'$or': [{'SmS': {'$ne': np.nan}},
                                               {'AAfamGWAS': 'x'},
                                               {'EAfamGWAS': 'x'}]},
                    'wave12': {'Wave12': 'x'},
                    }

subjects_sparser_sub = \
    ['famID', 'mID', 'fID', 'DNA', 'rel2pro', 'famtype', 'POP',
     'DOB', 'twin', 'EEG', 'System', 'Wave12', 'Wave12-fam',
     'fMRI subject', 'Wave3', 'Phase4-testdate',
     'Phase4-age', '4500', 'ccGWAS', 'AAfamGWAS', 'ExomeSeq',
     'EAfamGWAS', 'EAfamGWAS-fam', 'wave12-race', '4500-race',
     'ccGWAS-race', 'core-race', 'COGA11k-fam', 'COGA11k-race',
     'COGA11k-fam-race', 'ruID', 'genoID', 'SmS', 'CA/CO',
     'a-session', 'b-session', 'c-session', 'd-session',
     'e-session', 'f-session', 'g-session', 'h-session',
     'i-session', 'j-session', 'k-session',
     'a-raw', 'b-raw', 'c-raw', 'd-raw', 'e-raw', 'f-raw', 'g-raw',
     'h-raw', 'i-raw', 'j-raw', 'k-raw', 'missing-EEG' 'remarks']

subjects_sparser_add = \
    ['ID', 'sex', 'handedness', 'Self-reported-race', 'alc_dep_dx',
     'alc_dep_ons', 'a-age', 'b-age', 'c-age', 'd-age', 'e-age',
     'f-age', 'g-age', 'h-age', 'i-age', 'j-age', 'k-age',
     'famID', 'mID', 'fID', 'POP', 'alc_dep_dx_f', 'alc_dep_dx_m', 'twin', 'Phase4-session']

session_sadd = [field for field in subjects_sparser_add if 'age' not in field]
session_sadd.extend(['session', 'followup', 'age', 'date'])

subcoll_fnames = {'questionnaires': 'questname',
                  'ssaga': 'questname',
                  }

quest_sparser_sub = {'achenbach': ['af_', 'bp_']}

sparse_submaps = {'questionnaires': quest_sparser_sub,
                  'subjects': subjects_sparser_sub,
                  }

sparse_addmaps = {'subjects': subjects_sparser_add,
                  'sessions': session_sadd,
                  }

default_ERPfields = {'ID': 1, 'session': 1, '_id': 0}
default_EROfields = {'ID': 1, 'session': 1, 'uID': 1, '_id': 0}


def populate_subcolldict():
    ''' make dict whose keys are collections and values are lists of
        subcollections (if they exist) '''
    subcoll_dict = {coll: Mdb[coll].distinct(subcoll_fnames[coll])
    if coll in subcoll_fnames.keys() else None
                    for coll in Mdb.collection_names()}
    return subcoll_dict


subcoll_dict = populate_subcolldict()


def display_samples():
    ''' display available subject / session samples '''
    lst = sorted([(k, get_subjectdocs(k).count()) for k in subjects_queries])
    pp.pprint(lst)


def display_dbcontents():
    ''' display the contents of the database '''
    pp.pprint(subcoll_dict)


def get_subjectdocs(sample, sparsify=False):
    ''' given sample, prepare matching docs from the subjects collection '''
    if sample not in subjects_queries.keys():
        print('sample incorrectly specified, the below are valid')
        pp.pprint(sorted(list(subjects_queries.keys())))
        return
    else:
        if sparsify:
            proj = format_sparseproj('subjects')
            proj.update({'_id': 0})
            docs = Mdb['subjects'].find(subjects_queries[sample], proj)
        else:
            docs = Mdb['subjects'].find(subjects_queries[sample])
        return docs


def get_sessiondocs(sample, followups=None, sparsify=False):
    ''' given sample, prepare matching docs from the sessions collection '''
    if sample not in subjects_queries.keys():
        print('sample incorrectly specified, the below are valid')
        pp.pprint(sorted(list(subjects_queries.keys())))
        return
    else:
        query = subjects_queries[sample]
        if followups:
            if len(followups) > 0:
                query.update({'followup': {'$in': followups}})
            else:
                query.update({'followup': followups[0]})
        if sparsify:
            proj = format_sparseproj('sessions')
            proj.update({'_id': 0})
            docs = Mdb['sessions'].find(subjects_queries[sample], proj)
        else:
            docs = Mdb['sessions'].find(subjects_queries[sample])
        return docs


def check_collinputs(coll, subcoll=None, mode='program'):
    ''' verify collection / sub-collection is in DB '''
    result = True
    if coll not in subcoll_dict.keys():
        result = False
        if mode == 'interactive':
            print('collection incorrectly specified, the below are valid')
            print(', '.join(subcoll_dict.keys()))

    if subcoll is not None and subcoll not in subcoll_dict[coll]:
        result = False
        if mode == 'interactive':
            print(
                '{0} not found in {1}, below are valid'.format(subcoll, coll))
            print(', '.join(subcoll_dict[coll]))

    return result


def display_collcontents(coll, subcoll=None):
    ''' display contents of a collection '''
    ck_res = check_collinputs(coll, subcoll, mode='interactive')
    if not ck_res:
        return
    if subcoll is None:
        doc = Mdb[coll].find_one()
    else:
        doc = Mdb[coll].find_one({subcoll_fnames[coll]: subcoll})
    pp.pprint(sorted(list(doc.keys())))
    # pp.pprint(sorted(list(unflatten_dict(doc).keys())))


def get_colldocs(coll, subcoll=None, add_query={}, add_proj={}):
    ''' get documents from a collection/sub-collection '''
    ck_res = check_collinputs(coll, subcoll, mode='interactive')
    if not ck_res:
        return
    query = {}
    proj = {}
    if subcoll:
        query.update({subcoll_fnames[coll]: subcoll})
    query.update(add_query)
    proj.update(add_proj)
    if proj:
        docs = Mdb[coll].find(query, proj)
    else:
        docs = Mdb[coll].find(query)
    return docs


def buildframe_fromdocs(docs, inds=['ID', 'session']):
    ''' build a dataframe from a list of docs '''
    df = pd.DataFrame.from_records(
        [flatten_dict(d) for d in list(docs)])
    for ind in inds:
        if ind in df.columns:
            if df.index.is_integer():
                df.set_index(ind, inplace=True)
            else:
                df.set_index(ind, append=True, inplace=True)

    df.dropna(axis=1, how='all', inplace=True)  # drop empty columns
    df.sort_index(inplace=True)  # sort
    return df


def format_sparseproj(coll, subcoll=None):
    ''' format a projection to sparsify a collection / subcollection '''
    if coll in sparse_addmaps.keys():
        proj = {field: 1 for field in sparse_addmaps[coll]}
    else:
        proj = {}
    return proj


def format_ERPprojection(conds_peaks, chans, measures=['amp', 'lat']):
    ''' format a projection to retrieve specific ERP peak information '''
    proj = default_ERPfields.copy()
    proj.update({'.'.join(['data', cp, chan, m]): 1
                 for cp in conds_peaks for chan in chans for m in measures})
    return proj


def format_ERPprojection_tups(cond_peak_chans_lst, measures=['amp', 'lat']):
    ''' format a projection to retrieve specific ERP peak information '''
    proj = default_ERPfields.copy()

    for cond_peak_chans in cond_peak_chans_lst:
        cond, peak, chans = cond_peak_chans
        for chan in chans:
            for measure in measures:
                key = 'data.' + cond + '_' + peak + '.' + chan + '.' + measure
                proj[key] = 1

    return proj


def format_EROprojection(conds, freqs, times, chans,
                         measures=['evo', 'tot'], calc_types=['v60-all']):
    ''' format a projection to retrieve specific ERO information '''

    freqs = [[str(float(lim)).replace('.', 'p') for lim in lims]
             for lims in freqs]
    times = [[str(int(lim)) for lim in lims] for lims in times]

    proj = default_EROfields.copy()
    proj.update({'.'.join(['data', c, m, cond, f[0], f[1], t[0], t[1], chan]): 1
                 for c in calc_types for m in measures for cond in conds
                 for f in freqs for t in times for chan in chans})
    return proj


def prepare_joindata(keyDF, coll, subcoll=None, add_query={}, add_proj={},
                     left_join_inds=['ID'], right_join_inds=['ID'],
                     id_field='ID', flatten=True, prefix=None):
    ''' given a "key" dataframe and target collection,
        prepare a dataframe of corresponding info '''

    if not prefix and prefix != '':
        if subcoll is not None:
            prefix = subcoll[:3] + '_'
            if prefix == 'dx__':  # catch the dx subcollections
                prefix = subcoll[3:6] + '_'
        else:
            prefix = coll[:3] + '_'

    query = {id_field: {'$in': list(
        keyDF.index.get_level_values(right_join_inds[0]))}}
    query.update(add_query)
    proj = {}
    if add_proj:
        proj.update(add_proj)
        for ind in right_join_inds:
            proj.update({ind: 1})
    proj.update({'_id': 0})
    docs = get_colldocs(coll, subcoll, query, proj)

    if flatten:
        recs = [flatten_dict(r) for r in list(docs)]
    else:
        recs = [r for r in list(docs)]

    newDF = pd.DataFrame.from_records(recs)
    newDF['ID'] = newDF[id_field]  # should be more general

    prepare_indices(newDF, left_join_inds)
    newDF.columns = [prefix + c for c in newDF.columns]
    newDF.dropna(axis=1, how='all', inplace=True)  # drop empty columns
    newDF.sort_index(inplace=True)  # sort

    return newDF


def join_collection(keyDF_in, coll, subcoll=None, add_query={}, add_proj={},
                    left_join_inds=['ID'], right_join_inds=['ID'],
                    id_field='ID', flatten=True, prefix=None,
                    drop_empty=True, how='left'):
    ''' given a "key" dataframe and target collection,
        join corresponding info '''

    keyDF = keyDF_in.copy()

    newDF = prepare_joindata(keyDF_in, coll, subcoll, add_query, add_proj,
                             left_join_inds, right_join_inds,
                             id_field, flatten, prefix)

    prepare_indices(keyDF, right_join_inds)

    jDF = keyDF.join(newDF, how=how)

    if drop_empty:  # remove duplicate & empty rows, empty columns
        # jDF.drop_duplicates(inplace=True)
        jDF.dropna(axis=0, how='all', inplace=True)
        jDF.dropna(axis=1, how='all', inplace=True)

    return jDF


def join_ssaga(keyDF_in, raw=False, add_query={}, add_proj={},
                left_join_inds=['ID'], right_join_inds=['ID'],
                id_field='ID', flatten=True, prefix=None,
                drop_empty=True, how='left'):
    ''' given a "key" dataframe and target collection,
        join corresponding info '''

    keyDF = keyDF_in.copy()

    newDF = prepare_ssagadata(keyDF_in, raw, add_query, add_proj,
                             left_join_inds, right_join_inds,
                             id_field, flatten, prefix)

    prepare_indices(keyDF, right_join_inds)

    jDF = keyDF.join(newDF, how=how)

    if drop_empty:  # remove duplicate & empty rows, empty columns
        # jDF.drop_duplicates(inplace=True)
        jDF.dropna(axis=0, how='all', inplace=True)
        jDF.dropna(axis=1, how='all', inplace=True)

    return jDF
    

def prepare_ssagadata(keyDF, raw, add_query, add_proj,
                         left_join_inds, right_join_inds,
                         id_field, flatten, prefix):
    
    if raw:
        ssaga_subcoll = 'ssaga'
        cssaga_subcoll = 'cssaga'
    else:
        ssaga_subcoll = 'dx_ssaga'
        cssaga_subcoll = 'dx_cssaga'

    if not prefix and prefix != '':
        prefix = 'ssa_'

    query = {id_field: {'$in': list(
        keyDF.index.get_level_values(right_join_inds[0]))}}
    query.update(add_query)
    proj = {}
    if add_proj:
        proj.update(add_proj)
        for ind in right_join_inds:
            proj.update({ind: 1})
    proj.update({'_id': 0})

    ssaga_docs = get_colldocs('ssaga', ssaga_subcoll, query, proj)
    cssaga_docs = get_colldocs('ssaga', cssaga_subcoll, query, proj)
    docs = list(ssaga_docs) + list(cssaga_docs)
    del ssaga_docs
    del cssaga_docs

    if flatten:
        recs = [flatten_dict(r) for r in list(docs)]
    else:
        recs = [r for r in list(docs)]

    newDF = pd.DataFrame.from_records(recs)
    newDF['ID'] = newDF[id_field]  # should be more general

    prepare_indices(newDF, left_join_inds)
    newDF.columns = [prefix + c for c in newDF.columns]
    newDF.dropna(axis=1, how='all', inplace=True)  # drop empty columns
    newDF.sort_index(inplace=True)  # sort

    return newDF

def prepare_indices(df, join_inds):
    ''' make certain a dataframe has certain indices '''
    for ji in join_inds:
        if ji not in df.index.names:
            if pd.isnull(df[ji]).values.any():
                df[ji] = df[ji].apply(fix_indexcol)
            do_append = df.index.name != None
            df.set_index(ji, append=do_append, inplace=True)  # inplace right?


def fix_indexcol(s):
    ''' convert nans in a column in preparation for creating an index '''
    if s is np.NaN:  # does this cover all cases?
        return 'x'
    else:
        return s


def get_cnth1s(df):
    ''' given a dataframe indexed by ID and sessions, use the cnth1s collection to retrieve
        the paths to cnth1s for all sessions present '''

    coll = 'cnth1s'
    exps = ['vp3', 'cpt', 'ern', 'err', 'ant', 'aod', 'ans', 'stp', 'gng']

    df_out = df.copy()
    for exp in exps:
        query = {'experiment': exp}
        matching_docs = Mdb[coll].find(query)
        eegDF = buildframe_fromdocs(matching_docs)
        eegDF[exp + '_cnth1_path'] = eegDF['filepath']
        df_out = df_out.join(eegDF[exp + '_cnth1_path'])

    return df_out


def get_famdf(df):
    ''' given df with famID column, get the corresponding family dataframe
        using the allrels collection '''

    famIDs = list(set(df['famID'].values.tolist()))
    docs = Mdb['allrels'].find({'famID': {'$in': famIDs}})
    fam_df_allrels = buildframe_fromdocs(docs)

    return fam_df_allrels


def get_sessiondatedf(df):
    ''' given a df indexed by ID (minimally), get a corresponding session dataframe
        indexed by both ID and session, with the session date info '''

    IDs = df.index.get_level_values('ID').tolist()
    session_query = {'ID': {'$in': IDs}}
    session_fields = ['ID', 'session', 'date']
    session_proj = {k: 1 for k in session_fields}
    session_proj['_id'] = 0
    session_docs = Mdb['sessions'].find(session_query, session_proj)
    sdate_df = buildframe_fromdocs(session_docs)
    sdate_df.rename(columns={'date': 'session_date'}, inplace=True)

    return sdate_df
