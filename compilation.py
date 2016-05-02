''' Compilation tools for HBNL
'''

import pandas as pd

import organization as O

queries = { 'AA GWAS subjects':{'AAfamGWAS':'x'},
            'AA GWAS families':{'AAfamGWAS':'f'},
            'Phase4 subjects': {'Phase4-session':
                            {'$in':['a','b','c','d']}},
        }

def join_collection(compDF, collection, name, id_field, df_indices):
    dDF = pd.DataFrame.from_records([O.flatten_dict(r) for r in \
            list( collection.find({id_field:{'$in':list(compDF.index)},
                                  'questname':name}) ) ] )
    dDF['ID'] = dDF[id_field]
    dDF.set_index(df_indices,inplace=True)
    dDF.columns = [ name+'_'+c for c in dDF.columns ]
    jDF = compDF.join(dDF)
    return jDF