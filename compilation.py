''' Compilation tools for HBNL
'''

import numpy as np
import pandas as pd
import organization as O
import quest_import as qi
import master_info as mi
import pprint
pp = pprint.PrettyPrinter(indent=4)

subjects_queries = {'AAfamGWAS': {'AAfamGWAS': 'x'},
                    'AAfamGWASfam': {'AAfamGWAS': 'f'},
                    'a-subjects': {'POP': 'A'},
                    'BrainDysfunction': {'POP': {'$in': ['A', 'C', 'H', 'P']},
                                         'EEG': 'x'},
                    'BrainDysfunction-AC': {'POP': {'$in': ['A', 'C']},
                                         'EEG': 'x'},
                    'ccGWAS': {'ccGWAS': {'$ne': np.nan}},
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
                    'wave12': {'Wave12': 'x'},
                    }

subcoll_fnames = {'questionnaires': 'questname',
                  'ssaga': 'questname',
                  'neuropsych': 'testname',
                  }

sparse_submaps = {'questionnaires': qi.sparser_sub,
                  'subjects': mi.sparser_sub,
                  }

sparse_addmaps = {'subjects': mi.sparser_add,
                  'sessions': mi.session_sadd,
                  }

default_ERPfields = {'ID': 1, 'session': 1, 'version': 1, '_id': 0}
default_EROfields = {'ID': 1, 'session': 1, 'uID': 1, '_id': 0}


def populate_subcolldict():
    subcoll_dict = {coll: O.Mdb[coll].distinct(subcoll_fnames[coll])
                    if coll in subcoll_fnames.keys() else None
                    for coll in O.Mdb.collection_names()}
    return subcoll_dict

subcoll_dict = populate_subcolldict()


def display_samples():
    lst = sorted([(k, get_subjectdocs(k).count()) for k in subjects_queries])
    pp.pprint(lst)


def display_dbcontents():
    pp.pprint(subcoll_dict)


def get_subjectdocs(sample, sparsify=False):
    if sample not in subjects_queries.keys():
        print('sample incorrectly specified, the below are valid')
        pp.pprint(sorted(list(subjects_queries.keys())))
        return
    else:
        if sparsify:
            proj = format_sparseproj('subjects')
            proj.update({'_id': 0})
            docs = O.Mdb['subjects'].find(subjects_queries[sample], proj)
        else:    
            docs = O.Mdb['subjects'].find(subjects_queries[sample])
        return docs


def get_sessiondocs(sample, followups=None, sparsify=False):
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
            docs = O.Mdb['sessions'].find(subjects_queries[sample], proj)
        else:    
            docs = O.Mdb['sessions'].find(subjects_queries[sample])
        return docs


def check_collinputs(coll, subcoll=None, mode='program'):
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
    ck_res = check_collinputs(coll, subcoll, mode='interactive')
    if not ck_res:
        return
    if subcoll is None:
        doc = O.Mdb[coll].find_one()
    else:
        doc = O.Mdb[coll].find_one({subcoll_fnames[coll]: subcoll})
    pp.pprint(sorted(list(doc.keys())))
    # pp.pprint(sorted(list(O.unflatten_dict(doc).keys())))


def get_colldocs(coll, subcoll=None, add_query={}, add_proj={}):
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
        docs = O.Mdb[coll].find(query, proj)
    else:
        docs = O.Mdb[coll].find(query)
    return docs


def buildframe_fromdocs(docs, inds=['ID','session']):
    df = pd.DataFrame.from_records(
        [O.flatten_dict(d) for d in list(docs)])
    for ind in inds:
        if ind in df.columns:
            if df.index.is_integer():
                df.set_index(ind, inplace=True)
            else:
                df.set_index(ind, append=True, inplace=True)
    # drop empty columns
    df.dropna(axis=1, how='all', inplace=True)
    return df


def format_sparseproj(coll, subcoll=None):
    if coll in sparse_addmaps.keys():
        proj = {field: 1 for field in sparse_addmaps[coll]}
    else:
        proj = {}
    return proj


def format_ERPprojection(conds_peaks, chans, measures=['amp', 'lat']):
    proj = default_ERPfields.copy()
    proj.update({'.'.join([cp, chan, m]): 1
                 for cp in conds_peaks for chan in chans for m in measures})
    return proj


def format_EROprojection(conds, freqs, times, chans,
        measures=['evo', 'tot'], calc_types=['v60-all']):

    freqs = [[str(float(lim)).replace('.', 'p') for lim in lims]
             for lims in freqs]
    times = [[str(int(lim)) for lim in lims] for lims in times]

    proj = default_EROfields.copy()
    proj.update({'.'.join(['data', c, m, cond, f[0], f[1], t[0], t[1], chan]): 1
                 for c in calc_types for m in measures for cond in conds
                 for f in freqs for t in times for chan in chans})
    return proj


def join_collection(keyDF_in, coll, subcoll=None, add_query={}, add_proj={},
                    left_join_inds=['ID'], right_join_inds=['ID'], 
                    id_field='ID', drop_empty=True,
                    how='left', flatten=True, prefix=None):
    if not prefix and prefix != '':
        if subcoll is not None:
            prefix = subcoll[:3] + '_'
        else:
            prefix = coll[:3] + '_'

    keyDF = keyDF_in.copy()

    query = {id_field: {'$in': list(
        keyDF.index.get_level_values(right_join_inds[0]))}}
    query.update(add_query)
    docs = get_colldocs(coll, subcoll, query, add_proj)

    if flatten:
        recs =[O.flatten_dict(r) for r in list(docs)]
    else:
        recs = [r for r in list(docs)]

    newDF = pd.DataFrame.from_records(recs)
    newDF['ID'] = newDF[id_field]  # should be more general

    prepare_indices(newDF, left_join_inds)
    newDF.columns = [prefix + c for c in newDF.columns]

    prepare_indices(keyDF, right_join_inds)
    jDF = keyDF.join(newDF, how=how)

    if drop_empty:  # remove duplicate & empty rows, empty columns
        # jDF.drop_duplicates(inplace=True)
        jDF.dropna(axis=0, how='all', inplace=True)
        jDF.dropna(axis=1, how='all', inplace=True)

    return jDF


def prepare_indices(df, join_inds):
    for ji in join_inds:
        if ji not in df.index.names:
            if pd.isnull(df[ji]).values.any():
                df[ji] = df[ji].apply(fix_indexcol)
            do_append = df.index.name != None
            df.set_index(ji, append=do_append, inplace=True)  # inplace right?


def fix_indexcol(s):
    if s is np.NaN:  # does this cover all cases?
        return 'x'
    else:
        return s
