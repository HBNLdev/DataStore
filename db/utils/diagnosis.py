'''Diagnosis module.
Diagnoses are made using a framework that collects information from ssaga questionnaires.
The variables are designated (in lowercase) in utility functions that calculate individual
criteria. A dictionary of (index,full name,shortname):function key:value pairs is taken
as input to setup the framework, and each criteria function will then be applied with
appropraite input variables using introspection and "keyfinder" lookuup dictionaries
created in the module.
'''
from db import database as D
from db import compilation as C
from db.utils.records import get_SSAGA_doc_var

import inspect
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from functools import reduce

#Criteria functions for DSM5 alcoholism
def tolerance5(al9d,al9i):
    if al9d==5 or al9i==5:
            return 5
    elif np.isnan(al9d) or np.isnan(al9d):
            return 9
    else: return 1 
def withdrawal5(al37i,al38,al38c,al39,al39c):
    meas = [al37i,al38,al38c,al39,al39c]
    cnt5 = sum([1 for m in meas if m==5])
    cnt9 = sum([1 for m in meas if np.isnan(m)])
    if cnt5 > 0:
        return 5
    elif cnt9 > 0:
        return 9
    else: return 1
def more5(al12c,al13b):
    meas = [al12c,al13b]
    cnt5 = sum([1 for m in meas if m==5])
    cnt9 = sum([1 for m in meas if np.isnan(m)])
    if cnt5 > 0:
        return 5
    elif cnt9 > 0:
        return 9
    else: return 1
def effort5(al10,al10d,al10d1):
    cnt5 = 0; cnt9 = 0
    if al10 == 5: 
        cnt5+=1
    elif np.isnan(al10):
        cnt9+=1
    if al10d>=3 or al10d1==5:
        cnt5+=1
    elif np.isnan(al10d1):
        cnt9+=1
    if cnt5>0:
        return 5
    elif cnt9>0:
        return 9
    else: return 1

def time5(al15a):
    if al15a==5:
        return 5
    elif np.isnan(al15a):
        return 9
    else: return 1
def distract5(al14b):
    if al14b==5:
        return 5
    elif np.isnan(al14b):
        return 9
    else: return 1
def pproblem5(al31b,al32,al33a):
    meas = [al31b,al32,al33a]
    cnt5 = sum([1 for m in meas if m==5])
    cnt9 = sum([1 for m in meas if np.isnan(m)])
    if cnt5 > 0:
        return 5
    elif cnt9 > 0:
        return 9
    else: return 1
def sproblem5(al26c,al27b):
    if al26c==5 or al27b==5:
        return 5
    elif np.isnan(al26c) or np.isnan(al27b):
        return 9
    else: return 1
def obligations5(al16d,al25b):
    if al16d==5 or al25b==5:
        return 5
    elif np.isnan(al16d) or np.isnan(al25b):
        return 9
    else: return 1
def hazardous5(al21c,al22c,al24c,al29c):
    meas = [al21c,al22c,al24c,al29c]
    cnt5 = sum([1 for m in meas if m==5])
    cnt9 = sum([1 for m in meas if np.isnan(m)])
    if cnt5 > 0:
        return 5
    elif cnt9 > 0:
        return 9
    else: return 1
def craving5(al19):
    if al19==5:
        return 5
    elif np.isnan(al19):
        return 9
    else: return 1


def DSM5comp(criteriaD):
'''Compilation function for DSM5 alcoholism
'''    
    sList = [nls[1] for nls in sorted([k for k,v in criteriaD.items() if v==5]) ]
    count = len(sList)
    dx = False
    if count > 1: dx = True
    severity = np.nan
    if dx:
        severity = 'mild'
        if count > 3: severity = 'moderate'
        if count > 5: severity = 'severe'

    dd = {'dx':dx,
        'severity':severity,
        'count':count,
        'list':sList
        }
    return dd

#criteria are (DSM5 index, full name, short name)
DSM5_criteria = {(10,'tolerance','tolerance'):tolerance5,
                (11,'withdrawal','withdrawal'):withdrawal5,
                (1,'excess_consumption','more'):more5,
                (2,'recurrent_unsuccessful_stop_efforts','stop_efforts'):effort5,
                (3,'OUR_time','time'):time5,
                (7,'reduced_activities','distract'):distract5,
                (9,'continued_despite_health_problem','phys_problems'):pproblem5,
                (6,'social_problems','social_problems'):sproblem5,
                (5,'obligations_interference','obligations'):obligations5,
                (8,'recurrent_hazardous_drinking','hazardous'):hazardous5,
                (4,'craving','craving'):craving5,
                }

#Need to define a generalized categorization scheme
#DSM5_severity = 

#Assembly of ssaga keyfinder dictionarres for harmonizing across different versions
db_name = D.Mdb.name
meta_name = db_name+'meta'
D.set_db(meta_name)
all_SSAGA_vars = D.Mdb['summaries'].find_one({'name':'ssaga'})['fields']
lu_SSAGA_vars = [(f.lower().replace('_',''),f) for f in all_SSAGA_vars]
lu_counts = Counter([ls[0] for ls in lu_SSAGA_vars])
multi_fields = [f for f,c in lu_counts.items() if c>1]
SSAGA_multi_kfinder = {f:[lf[1] for lf in lu_SSAGA_vars if lf[0]==f] for f in multi_fields}

all_SSAGA_lower_lk = { f.lower():f for f in all_SSAGA_vars }
und_vars = [ v for v in all_SSAGA_vars if '_' in v ]
non_und_vars = set(all_SSAGA_vars).difference( und_vars )
SSAGA_kfinder = { k.lower().replace('_',''):k for k in und_vars\
                        if k.lower().replace('_','') not in multi_fields }
SSAGA_kfinder.update( {k.lower():k for k in non_und_vars if k not in multi_fields\
                            and 'ID' not in k} )
SSAGA_kfind_rev = { v:k for k,v in SSAGA_kfinder.items() }
D.set_db(db_name)

def get_criteria_DB(doc,prop_list):
    '''maps upper cased, '_' separated fields from database to lower cased items here
    '''
    kfinder = { k.lower().replace('_',''):k for k in doc.keys() }
    assD = { prop:doc[kfinder[prop]] if prop in kfinder else np.nan\
                for prop in prop_list  }
    return assD

def make_diagnosis(framework,measures):

    F = framework
    M = measures

    values,diag = F.evaluate_criteria()

    Dout = {F.name+'_'+d for d,v in diag.items()}
    return Dout

class diagnosticFramework:
'''Generalized framework for diagnoses based on ssaga questionnaires.
    Initialization takes a name for the diagnosis, criteria dictionary mapping 
    (index,name,shortname) to criteria functions and a compilation function 
    that accepts the criteria dictionary.
'''
    def __init__(s,name,criteriaD,compFun):
        s.name=name
        s.criteria = criteriaD
        s.compFun = compFun

        s.setup_criteria()

    def setup_criteria(s):
        '''sets up guide for calculating criteria and returns list of all inputs needed
        '''
        s.all_inputs = []
        s.function_guide = {}
        for crit,fun in s.criteria.items():
            inputs = inspect.getargspec(fun)[:][0]
            s.all_inputs.extend(inputs)
            s.function_guide[crit] = (fun,inputs)

        return s.all_inputs

    def diagnosis_for_fIDs(s,fID_list):

        diagDF = items_for_diagnosis(fID_list,'ssaga',s.all_inputs)
        diagDF.fillna(np.nan)
        print(diagDF.columns)
        for crit,fun in DSM5_criteria.items():
            var = inspect.getargspec(fun)[:][0]
            print(crit,var)
            diagDF[('criteria',crit[1])] = diagDF.apply(lambda r: fun(*[r[v] for v in var]),axis=1)

        ddf = diagDF.set_index('fID')[ [c for c in diagDF.columns if 'criteria' in c] ]
        ddf[('diagnosis','list')] = ddf.apply(lambda r: [c[1] for c in ddf.columns if r[c]==5], axis=1 )
        ddf[('diagnosis','count')] = ddf[('diagnosis','list')].apply(len) #ddf.apply(lambda r: sum([r[c] for c in ddf.columns]) axis=1)
        ddf[('diagnosis','dx')] = ddf[('diagnosis','count')].apply(lambda c: c > 1)

        def severity(count):
            if count > 5:return 'severe'
            elif count > 3: return 'moderate'
            elif count > 1: return 'mild'
            else: return None
        ddf[('diagnosis','severity')] = ddf[('diagnosis','count')].apply(severity)
        
        return ddf

    def evaluate_criteria(s,inputsD):
        inputs = assVars = {k:np.nan if v==None else v for k,v in inputsD.items()}
        vals = {}
        for crit,fun_ins in s.function_guide.items():
            vals[crit] = fun_ins[0](*[ inputs[var] for var in fun_ins[1] ])

        comp = s.compFun(vals)
        return vals,comp

def items_for_diagnosis(fIDs,collection,fields):
'''Utility to retrieve values from ssaga documents.
'''
    fields = list(set(fields))
    proj = {'ID':1,'followup':1,'fID':1}
    multi_fields = [f for f in fields if f in SSAGA_multi_kfinder]
    print('multi fields',multi_fields)
    all_multi = reduce(lambda a,b: a+b, [ SSAGA_multi_kfinder[mf] for mf in multi_fields ] )
    proj.update( {f:1 for f in all_multi} )
    proj.update({SSAGA_kfinder[f]:1 for f in fields if f not in multi_fields})
     
    print(proj)
    itemsDF = C.buildframe_fromdocs( D.Mdb[collection].find(
                    {'fID':{'$in':fIDs},'questname':{'$in':['ssaga','cssaga']}},proj),
                    inds=['ID','followup'] )
    print( itemsDF.columns )
    for mf in multi_fields:
        SSfields = SSAGA_multi_kfinder[mf]
        def mergeCols(row):
            vals = [row[sf] for sf in SSfields if sf in row]
            if len(vals) > 0:
                return vals[0]
            return np.nan
        itemsDF[mf] = itemsDF.apply(mergeCols,axis=1)
        itemsDF.drop([f for f in SSfields if f in itemsDF.columns],axis=1,inplace=True)
    print(itemsDF.columns)
    return itemsDF.reset_index().rename(columns=SSAGA_kfind_rev)



def DSM5_for_fups(fIDs):
'''Wrapper function for DSM5 alcoholism.
'''
    DSM5fr = diagnosticFramework('DSM5',DSM5_criteria,DSM5comp)

    recs = []
    it_recs = []
    assDs = []
    for fID in tqdm(fIDs):
        ssagaD = D.Mdb['ssaga'].find_one({'fID':fID,'questname':{'$in':['ssaga','cssaga']}})
        if ssagaD:
            assD = { it:get_SSAGA_doc_var(it,ssagaD,SSAGA_kfinder) for it in DSM5fr.all_inputs }
            assDs.append(assD)
            crit,diag = DSM5fr.evaluate_criteria(assD)
            rec = {'fID':fID}
            it_rec = rec.copy()
            rec.update({('diagnosis',k):v for k,v in diag.items()})
            rec.update({('criteria',k[1]):v for k,v in crit.items()})
            recs.append(rec)
        
        it_rec.update(assD)
        it_recs.append(it_rec)
    dxDF = pd.DataFrame.from_records(recs).set_index( 'fID' )
    itDF = pd.DataFrame.from_records(it_recs).set_index('fID')
    
    dxDF = dxDF[ sorted(dxDF.columns.tolist(),reverse=True) ]
    dxDF.columns = pd.MultiIndex.from_tuples(dxDF.columns)

    return dxDF,itDF


# class criterion:
#   '''formula is specified by (default_value,(value,(ors)),(value,[ands]))
#   '''
#   def __init__(s,desc,formula):

#       s.formula = formula     

#   def calc(s)

# class measures(dict):

#   def __init__(s,data):
#       s.__


# class dagnosis():

#       def __init__(s):