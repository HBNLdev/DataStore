''' Compilation tools for HBNL
'''

import pandas as pd
import organization as O
import quest_import as qi
import master_info as mi

queries = { 'AA GWAS subjects':{'AAfamGWAS':'x'},
            'AA GWAS families':{'AAfamGWAS':'f'},
            'Phase4 subjects': {'Phase4-session':
                            {'$in':['a','b','c','d']}},
        }

sparse_submaps = {'questionnaires': qi.sparser_sub,
                  'subjects': mi.sparser_sub}

sparse_addmaps = {'subjects': mi.sparser_add}

def get_subjectdocs(sample):
    if sample not in queries.keys():
        print('sample incorrectly specified, the below are valid')
        print(queries.keys())
        return
    else:
        docs = O.Mdb['subjects'].find(queries[sample])
        return docs

def buildframe_fromdocs(docs):
    df = pd.DataFrame.from_records(list(docs))
    if 'ID' in df.columns:
        df.set_index(['ID'], inplace=True)
    return df

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

def join_collection(keyDF, coll, name, id_field='ID', join_inds=['ID'],
    add_query={}, sparsify=False):

    query = {id_field:{'$in':list(keyDF.index)}}
    query.update(add_query)

    dDF = pd.DataFrame.from_records([O.flatten_dict(r) for r in \
            list( coll.find(query) ) ] )

    dDF['ID'] = dDF[id_field]
    dDF.set_index(join_inds,inplace=True)
    dDF.columns = [ name[:3]+'_'+c for c in dDF.columns ]

    if sparsify:
        subsparsify_df(dDF, coll.name, name)

    jDF = keyDF.join(dDF)
    return jDF