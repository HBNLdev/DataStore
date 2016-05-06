''' Compilation tools for HBNL
'''

import numpy as np
import pandas as pd
import organization as O
import quest_import as qi
import master_info as mi
import pprint
pp = pprint.PrettyPrinter(indent=4)

subjects_queries = { 'AA GWAS subjects':{'AAfamGWAS':'x'},
                     'AA GWAS families':{'AAfamGWAS':'f'},
                     'Phase4 subjects': {'Phase4-session':
                            {'$in':['a','b','c','d']}},
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
    subcoll_dict = {}
    for coll in O.Mdb.collection_names():
        if coll in subcoll_fnames.keys():
            distinct_subcolls = O.Mdb[coll].distinct(subcoll_fnames[coll])
            subcoll_dict.update({coll:distinct_subcolls})
        else:
            subcoll_dict.update({coll:None})
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

def check_collinputs(coll, subcoll=None):
    if coll not in subcoll_dict.keys():
        print('collection incorrectly specified, the below are valid')
        print(', '.join(subcoll_dict.keys()))
        raise 
    if subcoll is not None and subcoll not in subcoll_dict[coll]:
        print('{0} not found in {1}, below are valid'.format(subcoll, coll))
        print(', '.join(subcoll_dict[coll]))
        raise

def display_collcontents(coll, subcoll=None):
    check_collinputs(coll, subcoll)
    if subcoll is None:
        doc = O.Mdb[coll].find_one()
    else:
        doc = O.Mdb[coll].find_one({subcoll_fnames[coll]:subcoll})
    pp.pprint(sorted(list(doc.keys())))

def get_colldocs(coll, subcoll=None, addquery={}):
    check_collinputs(coll, subcoll)
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
    join_inds=['ID'], id_field='ID', sparsify=False):
    if subcoll is not None:
        name = subcoll
    else:
        name = coll

    query = {id_field: {'$in': list(keyDF.index)}}
    query.update(add_query)
    docs = get_colldocs(coll, subcoll, query)

    newDF = pd.DataFrame.from_records(
        [O.flatten_dict(r) for r in list(docs)] )
    newDF['ID'] = newDF[id_field]
    newDF.columns = [ name[:3]+'_'+c if c not in ['ID','session','followup']
                else c for c in newDF.columns ]

    if sparsify:
        subsparsify_df(newDF, O.Mdb[coll].name, name)
    
    keyDF = prepare_indices(keyDF, join_inds)
    newDF = prepare_indices(newDF, join_inds)

    jDF = keyDF.join(newDF)
    return jDF

def prepare_indices(df, join_inds):
    for ji in join_inds:
        if ji not in df.index.names:
            if pd.isnull(df[ji]).values.any():
                df[ji] = df[ji].apply(fix_indexcol)
            do_append = not df.index.name is None
            df = df.set_index(ji, append=do_append)
    return df

def fix_indexcol(s):
    if s is np.NaN:
        return 'x'
    else:
        return s

def subsparsify_df(df, coll_name, subcoll_value=None):
    sdict = sparse_submaps[coll_name]
    if subcoll_value is not None:
        skeys = sdict[subcoll_value]
    else:
        skeys = sdict
    columns_todrop = []
    for dfc in df.columns:
        for skey in skeys:
            if skey in dfc:
                columns_todrop.append(dfc)
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
    df.drop(columns_todrop, axis=1, inplace=True)