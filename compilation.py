''' Compilation tools for HBNL
'''

import itertools
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
                    'HighRisk': {'POP': {'$in': ['COGA', 'COGA-Ctl']},
                                 'site': 'suny', 'EEG': 'x'},
                    'PhaseIV': {'Phase4-session':
                                {'$in': ['a', 'b', 'c', 'd']}},
                    'p-subjects': {'POP': 'P'},
                    'smokeScreen': {'SmS': {'$ne': np.nan}},
                    'wave12': {'Wave12': 'x'},
                    }

subcoll_fnames = {'questionnaires': 'questname',
                  'neuropsych': 'testname',
                  }

sparse_submaps = {'questionnaires': qi.sparser_sub,
                  'subjects': mi.sparser_sub,
                  }

sparse_addmaps = {'subjects': mi.sparser_add,
                  }


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


def get_subjectdocs(sample):
    if sample not in subjects_queries.keys():
        print('sample incorrectly specified, the below are valid')
        # print(', '.join(sorted(list(subjects_queries.keys()))))
        pp.pprint(sorted(list(subjects_queries.keys())))
        return
    else:
        docs = O.Mdb['subjects'].find(subjects_queries[sample])
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


def get_colldocs(coll, subcoll=None, add_query={}):
    ck_res = check_collinputs(coll, subcoll, mode='interactive')
    if not ck_res:
        return
    query = {}
    if subcoll is not None:
        query.update({subcoll_fnames[coll]: subcoll})
    query.update(add_query)
    docs = O.Mdb[coll].find(query)
    return docs


def buildframe_fromdocs(docs):
    df = pd.DataFrame.from_records(
        [O.flatten_dict(d) for d in list(docs)])
    if 'ID' in df.columns:
        df.set_index(['ID'], inplace=True)
    return df


def buildframe_fromERPdocs(docs, conds_peaks, chans, measures):
    df = pd.DataFrame.from_records(
        [O.flatten_dict(d) for d in list(docs)])

    keep_list = set('_'.join(tup) for tup in \
        list(itertools.product(conds_peaks, chans, measures)))
    all_list = set(col for col in df.columns if 'amp' in col or 'lat' in col)

    drop_list = all_list - keep_list
    df.drop(drop_list, axis=1, inplace=True)
    if 'ID' in df.columns:
        df.set_index(['ID'], inplace=True)
    return df


def join_collection(keyDF_in, coll, subcoll=None, add_query={},
                    join_inds=['ID'], id_field='ID', sparsify=False,
                    drop_empty=True):
    if subcoll is not None:
        name = subcoll
    else:
        name = coll

    keyDF = keyDF_in.copy()

    # if len(keyDF.index.names) > 1:
    #     id_nind = 
    query = {id_field: {'$in': list(keyDF.index)}}  # should be more general
    query.update(add_query)
    docs = get_colldocs(coll, subcoll, query)

    newDF = pd.DataFrame.from_records(
        [O.flatten_dict(r) for r in list(docs)]) # BIG JUMP HERE
    newDF['ID'] = newDF[id_field]  # should be more general

    if sparsify:
        subsparsify_df(newDF, O.Mdb[coll].name, name)
    prepare_indices(newDF, join_inds)
    newDF.columns = [name[:3] + '_' + c for c in newDF.columns]

    
    prepare_indices(keyDF, join_inds)
    jDF = keyDF.join(newDF)

    if drop_empty:  # remove duplicate & empty rows, empty columns
        jDF.drop_duplicates(inplace=True)
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


def subsparsify_df(df, coll_name, subcoll_value=None):
    sdict = sparse_submaps[coll_name]
    if subcoll_value is None:
        skeys = sdict
    else:
        skeys = sdict[subcoll_value]
    columns_todrop = []
    for dfc in df.columns:
        for skey in skeys:
            if skey in dfc:
                columns_todrop.append(dfc)
    print('The following columns were dropped:')
    print(columns_todrop)
    df.drop(columns_todrop, axis=1, inplace=True)


def addsparsify_df(df, coll_name, subcoll_value=None):
    sdict = sparse_addmaps[coll_name]
    if subcoll_value is not None:
        skeys = sdict[subcoll_value]
    else:
        skeys = sdict
    columns_todrop = []
    for dfc in df.columns:
        if dfc not in skeys:
            columns_todrop.append(dfc)
    print('The following columns were dropped:')
    print(columns_todrop)
    df.drop(columns_todrop, axis=1, inplace=True)
