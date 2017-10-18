from db import database as D
import db.compilation as C
import pandas as pd


#default projections
default_admin_nested = ['filepath', 'run', 'version']
default_admin_proj = {'family' : 1, 'insert_time': 1, 'site': 1,
                      'subject': 1, 'system': 1, 'update_time': 1
                     }
default_EEG = {'ID': 1, 'session': 1, '_id': 0}


#knowledge 
EEGbehavior_dict = {'vp3' : ['T', 'NT', 'N'], 
                    'cpt' : ['C', 'CN', 'DD', 'N', 'NG'],
                    'ern' : ['CH', 'CH0', 'CH1', 'MD', 'ML', 'MW', 'N10', 'N50', 'NR', 'P10', 'P50'],
                    'ant' : ['A','J', 'P', 'W'],
                    'aod' : ['NT', 'T'],
                    'stp' : ['C', 'C1', 'C8', 'IN', 'IN1', 'IN8'],
                    'gng' : ['D1', 'D2', 'G', 'NG', 'X1', 'X2']}

eegbehavior_measures = ['accwithlate', 'medianwithlate', 'accwithresp', 
                        'accwithrespwitlate', 'medianrtwithlate', 'medianrt', 'acc']



def format_EEGbehavior_projection(experiment, cases, admin, 
                                  measures= ['accwithlate', 'medianwithlate', 
                                             'accwithresp', 'accwithrespwitlate',
                                             'medianrtwithlate', 'medianrt', 'acc']):
    
    ''' format a projection to retrieve specific EEG Behavior information '''
    
    proj = default_EEG.copy()

    if admin:
        proj.update({'.'.join([experiment, admin]) :1
                     for admin in default_admin_nested})
        proj.update(default_admin_proj)
        return proj
    
    if not admin: 
        proj.update({'.'.join([experiment, case, m]): 1
                    for case in cases for m in measures})
        return proj


def parse_EEGbehavior_args(exp, cond=None, measures=None, 
                           exp_cases_dict = None,
                           admin=None):
    
    '''parses arguments for EEGbehaviorDF'''
    
    if not cond:
        cond = EEGbehavior_dict[exp]   
        
    if not measures:
        measures = eegbehavior_measures.copy()
        
    proj = format_EEGbehavior_projection(exp, cond, admin, measures=measures)
    
    return proj 


def EEGbehaviorDF(uIDs, 
                  exp=None, cond=None,  measures=None,
                  admin=False, flatten=False):
    '''
       Example args: tup(uIDs)
                     str(exp)
                     list(cond)
                     list(measures)
                     flatten & admin = boolean
    '''
    
    query = {'uID': {'$in': uIDs}}
    
    if exp: 
        proj = parse_EEGbehavior_args(exp, cond=cond, admin=admin, exp_cases_dict=EEGbehavior_dict, measures = measures)
        add_query = {exp: {'$exists': True}}
        proj.update(proj)
        query.update(add_query)
        
    docs = D.Mdb['EEGbehavior'].find(query, proj)

    df = C.buildframe_fromdocs(docs, inds=['ID', 'session', 'experiment'])
    
    if flatten:
        df.columns = pd.MultiIndex.from_tuples([tuple(name.split('_')) for name in df.columns])
        df1 = df.reset_index()
        df1.set_index(keys = ['ID', 'session'])
        return df1
    else:
        return df