''' Compilation tools for HBNL
'''

import numpy as np
import pandas as pd
import organization as O
import quest_import as qi
import master_info as mi
import pprint
pp = pprint.PrettyPrinter(indent=4)

subjects_queries = { 'AAfamGWAS':{'AAfamGWAS':'x'},
                     'AAfamGWASfam':{'AAfamGWAS':'f'},
                     'a-subjects':{'POP':'A'},
                     'ccGWAS':{'ccGWAS':{'$ne':np.nan}},
                     'COGA11k':{'COGA11k-fam':{'$ne':np.nan}},
                     'COGA4500':{'4500':'x'},
                     'c-subjects':{'POP':'C'},
                     'EAfamGWAS':{'EAfamGWAS':'x'},
                     'EAfamGWASfam':{'EAfamGWAS':'f'},
                     'ExomeSeq':{'ExomeSeq':'x'},
                     'fMRI-NKI-bd1':{'fMRI':{'$in':['1a','1b']}},
                     'fMRI-NKI-bd2':{'fMRI':{'$in':['2a','2b']}},
                     'fMRI-NYU-hr':{'fMRI':{'$in':['3a','3b']}},
                     'h-subjects':{'POP':'H'},
                     'PhaseIV': {'Phase4-session':
                            {'$in':['a','b','c','d']}},
                     'p-subjects':{'POP':'P'},
                     'smokeScreen':{'SmS':{'$ne':np.nan}},
                     'wave12':{'Wave12':'x'},
                    }

subcoll_fnames = {'questionnaires': 'questname',
                  'neuropsych':'testname',
                  }
 
sparse_submaps = {'questionnaires': qi.sparser_sub,
                  'subjects': mi.sparser_sub,
                  }

sparse_addmaps = {'subjects': mi.sparser_add,
                  }

def populate_subcolldict():
    subcoll_dict = {coll:O.Mdb[coll].distinct(subcoll_fnames[coll])
        if coll in subcoll_fnames.keys() else None
        for coll in O.Mdb.collection_names() }
    return subcoll_dict

subcoll_dict = populate_subcolldict()

def display_dbcontents():
    pp.pprint(subcoll_dict)

def get_subjectdocs(sample):
    if sample not in subjects_queries.keys():
        print('sample incorrectly specified, the below are valid')
        print(', '.join(subjects_queries.keys()))
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
            print('{0} not found in {1}, below are valid'.format(subcoll,coll))
            print(', '.join(subcoll_dict[coll]))

    return result

def display_collcontents(coll, subcoll=None):
    ck_res = check_collinputs(coll, subcoll, mode='interactive')
    if not ck_res:
        return
    if subcoll is None:
        doc = O.Mdb[coll].find_one()
    else:
        doc = O.Mdb[coll].find_one({subcoll_fnames[coll]:subcoll})
    pp.pprint(sorted(list(doc.keys())))
    # pp.pprint(sorted(list(O.unflatten_dict(doc).keys())))

def get_colldocs(coll, subcoll=None, addquery={}):
    ck_res = check_collinputs(coll, subcoll, mode='interactive')
    if not ck_res:
        return
    query = {}
    if subcoll is not None:
        query.update({subcoll_fnames[coll]:subcoll})
    query.update(addquery)
    docs = O.Mdb[coll].find(query)
    return docs

def buildframe_fromdocs(docs):
    df = pd.DataFrame.from_records(
        [O.flatten_dict(d) for d in list(docs)])
    if 'ID' in df.columns:
        df.set_index(['ID'], inplace=True)
    return df

def join_collection(keyDF, coll, subcoll=None, add_query={},
    join_inds=['ID'], id_field='ID', sparsify=False,
    drop_empty=True):
    if subcoll is not None:
        name = subcoll
    else:
        name = coll

    query = {id_field: {'$in': list(keyDF.index)}} # should be more general
    query.update(add_query)
    docs = get_colldocs(coll, subcoll, query)

    newDF = pd.DataFrame.from_records(
        [O.flatten_dict(r) for r in list(docs)] )
    newDF['ID'] = newDF[id_field] # should be more general
    newDF.columns = [ name[:3]+'_'+c if c not in ['ID','session','followup']
                else c for c in newDF.columns ] # should be more general

    if sparsify:
        subsparsify_df(newDF, O.Mdb[coll].name, name)
    
    prepare_indices(keyDF, join_inds)
    prepare_indices(newDF, join_inds)

    jDF = keyDF.join(newDF)

    if drop_empty:
        drop_emptycols(jDF)

    return jDF

def prepare_indices(df, join_inds):
    for ji in join_inds:
        if ji not in df.index.names:
            if pd.isnull(df[ji]).values.any():
                df[ji] = df[ji].apply(fix_indexcol)
            do_append = df.index.name != None
            df.set_index(ji, append=do_append, inplace=True) # inplace right?

def fix_indexcol(s):
    if s is np.NaN: # does this cover all cases?
        return 'x'
    else:
        return s

def drop_emptycols(df):
    # check for empty columns and drop them
    dropped_lst = []
    for col in df.columns:
        if pd.isnull(df[col]).values.all():
            dropped_lst.append(col)
            df.drop(col, axis=1, inplace=True)
    print('the following empty columns were dropped:')
    print(dropped_lst)

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