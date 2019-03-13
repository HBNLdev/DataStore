'''Diagnosis module
'''
import inspect
import numpy as np

def tolerance5(al9d,al9i):
    if al9d==5 or al9i==5:
            return 5
    elif (al9d==9 or np.isnan(al9d)) or (al9d==9 or np.isnan(al9d)):
            return 9
    else: return 1 
def withdrawal5(al37i,al38,al38c,al39,al39c):
    meas = [al37i,al38,al38c,al39,al39c]
    cnt5 = sum([1 for m in meas if m==5])
    cnt9 = sum([1 for m in meas if m==9 or np.isnan(m)])
    if cnt5 > 0:
        return 5
    elif cnt9 > 0:
        return 9
    else: return 1
def more5(al12c,al13b):
    meas = [al12c,al13b]
    cnt5 = sum([1 for m in meas if m==5])
    cnt9 = sum([1 for m in meas if m==9 or np.isnan(m)])
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
    elif al10d1==9 or np.isnan(al10d1):
        cnt9+=1
    if cnt5>0:
        return 5
    elif cnt9>0:
        return 9
    else: return 1

def time5(al15a):
    if al15a==5:
        return 5
    elif al15a==9 or np.isnan(al15a):
        return 9
    else: return 1
def distract5(al14b):
    if al14b==5:
        return 5
    elif al14b==9 or np.isnan(al14b):
        return 9
    else: return 1
def pproblem5(al31b,al32,al33a):
    meas = [al31b,al32,al33a]
    cnt5 = sum([1 for m in meas if m==5])
    cnt9 = sum([1 for m in meas if m==9 or np.isnan(m)])
    if cnt5 > 0:
        return 5
    elif cnt9 > 0:
        return 9
    else: return 1
def sproblem5(al26c,al27b):
    if al26c==5 or al27b==5:
        return 5
    elif (al26c==9 or np.isnan(al26c)) or (al27b==9 or np.isnan(al27b)):
        return 9
    else: return 1
def obligations5(al16d,al25b):
    if al16d==5 or al25b==5:
        return 5
    elif (al16d==9 or np.isnan(al16d)) or (al25b==9 or np.isnan(al25b)):
        return 9
    else: return 1
def hazardous5(al21c,al22c,al24c,al29c):
    meas = [al21c,al22c,al24c,al29c]
    cnt5 = sum([1 for m in meas if m==5])
    cnt9 = sum([1 for m in meas if m==9 or np.isnan(m)])
    if cnt5 > 0:
        return 5
    elif cnt9 > 0:
        return 9
    else: return 1
def craving5(al19):
    if al19==5:
        return 5
    elif al19==9 or np.isnan(al19):
        return 9
    else: return 1
def DSM5comp(criteriaD):
    
    sList = [k for k,v in criteriaD.items() if v==5]
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

DSM5_criteria = {'tolerance':tolerance5,
                'withdrawal':withdrawal5,
                'more':more5,
                'effort':effort5,
                'time':time5,
                'distract':distract5,
                'phys_problem':pproblem5,
                'social_problem':sproblem5,
                'obligations':obligations5,
                'hazardous':hazardous5,
                'craving':craving5,
                }

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

    def evaluate_criteria(s,inputsD):
        inputs = assVars = {k:np.nan if v==None else v for k,v in inputsD.items()}
        vals = {}
        for crit,fun_ins in s.function_guide.items():
            vals[crit] = fun_ins[0](*[ inputs[var] for var in fun_ins[1] ])

        comp = s.compFun(vals)
        return vals,comp

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