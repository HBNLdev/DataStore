''' Compilation tools for HBNL
'''

import pandas as pd
import organization as O

queries = { 'AA GWAS subjects':{'AAfamGWAS':'x'},
            'AA GWAS families':{'AAfamGWAS':'f'},
            'Phase4 subjects': {'Phase4-session':
                            {'$in':['a','b','c','d']}},
        }

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

def join_collection(keyDF, coll, name, id_field='ID', join_inds=['ID'],
    add_query={}):

    query = {id_field:{'$in':list(keyDF.index)}}
    query.update(add_query)

    dDF = pd.DataFrame.from_records([O.flatten_dict(r) for r in \
            list( coll.find(query) ) ] )

    dDF['ID'] = dDF[id_field]
    dDF.set_index(join_inds,inplace=True)
    dDF.columns = [ name+'_'+c for c in dDF.columns ]

    jDF = keyDF.join(dDF)
    return jDF