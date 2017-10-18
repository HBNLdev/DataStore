'''refining and describing'''

from db import database as D
from db.utils import text as tU
from db.knowledge import drinking as drK


import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity  
from datetime import datetime as dt
from numbers import Number
from collections import Counter

import sys


default_sniff_guide = {'category cutoff':20,
                      'numeric tolerance':0,
                      'date tolerance':0}

def find_fields(doc,guide):
 
    if not guide:
        skip_cols = ['_id']
        return [ k for k in doc.keys() if k not in skip_cols ]
    else:
        pass
    
def summarize_collection_meta(db,mdb,coll,coll_guide=None):
    ''' Scan collection and build description of fields.
    '''
    
    D.set_db(db)
    cc = D.Mdb[coll].find() #collection_cursor
    fields = set()
    [ fields.update( find_fields(doc,coll_guide) ) for doc in cc ]
    
    D.set_db(mdb)
    D.Mdb['summaries'].insert_one({'name':coll,
                                  'fields':list(fields)})


def sniff_field_type(vals,sniff_guide={}):
    ''' determines type as one of:
            - categorical
            - numeric
            - datetime
            - unknown
    '''
    sg = default_sniff_guide.copy()
    sg.update(sniff_guide)
    
    if len(vals) <= sg['category cutoff']:
        return 'categorical'
    else:
        num_ck_cnts = Counter([isinstance(v,Number) for v in vals])
        num_cnt = sorted([ (v,k) for k,v in num_ck_cnts.items() ])
        if num_cnt[-1][1]:
            if len(num_cnt) == 1 or num_cnt[-2][0]/num_cnt[-1][0] < sg['numeric tolerance']:
                return 'numeric'
        else:
            date_ck_cnts = Counter([isinstance(v,dt) for v in vals])
            date_cnt = sorted([ (v,k) for k,v in date_ck_cnts.items() ])
            if date_cnt[-1][1]:
                if len(date_cnt) == 1 or date_cnt[-2][0]/date_cnt[-1][0] < sg['date tolerance']:
                    return 'datetime'
        
    return 'unknown'

def try_con(conF,v):
    try:
        ck = conF(v)
        return (ck,True)
    except:
        return (None,False)

def field_ranges(db,mdb,coll,guide={},sniff_guide={}):
    D.set_db(mdb)
    summ = D.Mdb['summaries'].find_one({'name':coll})
    ranges = {}
    D.set_db(db)
    for fd in summ['fields']:
        skip_col = False; con_flag = False
        if 'skip patterns' in guide:
            for sp in guide['skip patterns']:
                if sp in fd:
                    skip_col = True
        if not skip_col:
            Dvals = D.Mdb[coll].distinct(fd)
            Ndocs = D.Mdb[coll].find().count() # should have query to skip descriptive docs
            if fd in guide:
                fgd = guide[fd]
                vtype = fgd['type']
                if 'converter' in fgd:
                    Dvals = [ fgd['converter'](v) for v in Dvals ]
                    con_flag = True
            else:
                vtype = sniff_field_type(Dvals, sniff_guide )

            if vtype in ['categorical','numeric','datetime']:
                raw_vals = [d[fd] for d in \
                            D.Mdb[coll].find({fd:{'$exists':True}},{fd:1}) ]
                rawV_Nnan = [ rv for rv in raw_vals if not pd.isnull(rv)]
                data_portion = len(rawV_Nnan)/len(raw_vals)
                if con_flag:
                    Cvals_ck = [ try_con(fgd['converter'],v) for v in raw_vals ]


            if vtype == 'categorical':
                try:
                    v_counts = sorted([ (v,k) for k,v in Counter(rawV_Nnan).items() ])[:20]
                    if all([ v.replace('.','').isdigit() for c,v in v_counts if type(v)==str ]):
                        converter = int
                        if any([ ( type(v[1]) == str and '.' in v[1] ) or isinstance(v[1],Number)\
                                    for v in v_counts ]):
                            converter = float
                        print('num cat conv:',converter)
                        ranges[fd] = { 'type':'num cat', 
                                'value counts':sorted([ ( converter(v),k ) for k,v in v_counts]),
                                'data portion':data_portion }
                    else:
                        print('cat')
                        ranges[fd] = {'type':'categorical',
                                'value counts':[ (v,k) for k,v in v_counts],
                                'data portion':data_portion}                    
                except:
                    ranges[fd] = {'skipped':'categorical error'}
                    #print( fd, set([type(v) for v in raw_vals]), set(raw_vals) )

            elif vtype == 'numeric':
                pres_vals = [v for v in rawV_Nnan if v]
                if not con_flag:
                    Cvals_ck = [ try_con(float,v) for v in Dvals ]
                Cvals = [ cc[0] for cc in Cvals_ck if cc[1] ]
                try:
                    mn = np.mean(Cvals)
                    med = np.median(Cvals)
                    std = np.std(Cvals)
                    low = min(Cvals)
                    hi = max(Cvals)
                    ranges[fd] = {'type':'numeric',
                            'min':low,
                             'max':hi,
                             'mean':mn,
                             'median':med,
                            'std':std,
                            'portion':len(Cvals)/Ndocs,
                              #add bad bad vals with trace
                            'Nmin':(low - mn)/std,
                            'Nmax':(hi - mn)/std,
                            'Nmedian':(med - mn)/std,
                            'data portion':data_portion,
                             }
                    for pct in [1,5,25,75,95,99]:
                        pctV = np.percentile(Cvals,pct)
                        spct = str(pct)
                        ranges[fd]['p'+spct] = pctV
                        ranges[fd]['Np'+spct] = (pctV - mn)/std

                    kernel = KernelDensity(kernel='gaussian',bandwidth=4).fit(np.array(Cvals)[:,np.newaxis])
                    pdf_xs = np.linspace(ranges[fd]['p1'],ranges[fd]['p99'],150)[:,np.newaxis]
                    pdf_vals = [np.exp(v) for v in kernel.score_samples(pdf_xs)]
                    ranges[fd]['pdf_x'] = list( np.squeeze(pdf_xs) )
                    ranges[fd]['Npdf_x'] = list( np.squeeze((pdf_xs-mn)/std) )
                    ranges[fd]['pdf'] = list( np.squeeze(pdf_vals) )
                except:
                    D.set_db(mdb)
                    D.Mdb['errors'].insert_one({'collection':coll,
                                           'field':fd,
                                           #'data':list(set(Cvals)),
                                           'error':str(sys.exc_info()[1])
                                               })
                    D.set_db(db)

            elif vtype == 'datetime':

                if not con_flag:
                    Cvals_ck = [ try_con(float,v) for v in Dvals ]
                Cvals = [ cc[0] for cc in Cvals_ck if cc[1] ]
                if len(Cvals) == 0:
                    Cvals = [-1]
                try:
                    ranges[fd] = {'type':'datetime',
                            'min':min(Cvals),
                            'max':max(Cvals),
                            'std':np.std(Cvals),
                             'portion':len(Cvals)/Ndocs,
                            'data portion':data_portion, }
                except:
                    D.set_db(mdb)
                    D.Mdb['errors'].insert_one({'collection':coll,
                                           'field':fd,
                                           #'data':str(list(set(Cvals))),
                                           'error':str(sys.exc_info()[1])
                                               } )
                    D.set_db(db)
            elif vtype == 'unknown':   
                ranges[fd] = {'type':'unknown',
                                  'values subset':Dvals[:10] }
            else:
                ranges[fd] = {'type':vtype,
                             'count':len(Dvals)}
        else:
            ranges[fd] = {'skipped':'guide'}
    D.set_db(mdb)
    ranges['collection name'] = coll
    D.Mdb['ranges'].insert_one(ranges)