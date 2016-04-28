''' Organization for HBNL
'''

import datetime
import pymongo
import pandas as pd
import numpy as np


MongoConn = pymongo.MongoClient('/tmp/mongodb-27017.sock')
Mdb = MongoConn['COGAm']

def flatten_dict(D,prefix=''):
    if len(prefix) > 0:
        prefix+='_'
    flat = {}
    for k,v in D.items():
        if type(v) == dict:
            F = flatten_dict(v,prefix+k)
            flat.update(F)
        else:
            flat[prefix+k] = v
    return flat

def unflatten_dict(dictionary,delimiter='_',skipkeys=['_id','test_type']):
	# use with caution, specify keys to skip
    resultDict = dict()
    for key, value in dictionary.items():
        if key in skipkeys:
            d = resultDict
            d[key] = value
        else:
            parts = key.split(delimiter)
            d = resultDict
            for part in parts[:-1]:
                if part not in d:
                    d[part] = dict()
                d = d[part]
            d[parts[-1]] = value
    return resultDict

def remove_NaTs(rec):
	for k,v in rec.items():
		typ = type(v)
		if typ != dict and typ == pd.tslib.NaTType:
			rec[k] = None

class MongoBacked:
	
	def store(s):
		s.data['insert_time']=datetime.datetime.now()
		Mdb[s.collection].insert(s.data)

	def storeNaTsafe(s):
		s.data['insert_time']=datetime.datetime.now()
		remove_NaTs(s.data)
		Mdb[s.collection].insert(s.data)		

class Acquisition(MongoBacked):
	
	def_info = {'site':None,
				'date':datetime.datetime(1901,1,1),
				'subject':'t0001001'
				}
	collection = 'Acquisition'
	
	part_name = 'part'
	repeat_name = 'repeat'
	
	
	def __init__(s,info={}):
		I = Acquisition.def_info.copy()
		I.update(info)
		s.site = I['site']
		s.date = I['date']
		s.subject = I['subject']
		s.data = I
		
class Neuropsych(Acquisition):
	
	def_info = { 'technique':'cognitive test' }
	
	part_name = 'experiment'
	repeat_name = 'session'
	
	def __init__(s,test_type,info = {}):
		s.collection = test_type
		I = Neuropsych.def_info.copy()
		I.update(info)
		Acquisition.__init__(s,I)
		
				
class Electrophysiology(Acquisition):
	
	def_info = {'technique':'EEG',
				'system':'unknown'}

	collection = 'EEG'
	
	part_name = 'experiment'
	repeat_name = 'session'
	
	def __init__(s, info={}):
		I = s.def_info.copy()
		I.update(info)
		Acquisition.__init__(s,I)

class ElectrophysiologyCalculations(Acquisition):

	def_info = {'technique':'EEG'}

	collection = 'EEGresults'

	part_name = 'experiment'
	repeat_name = 'session'

	def __init__(s,info={}):
		I = s.def_info.copy()
		I.update(info)
		Acquisition.__init__(s,I)		

	
class SSAGA(Acquisition):
	
	def_info = {'technique':'interview'}
	collection = 'SSAGA'
	
	part_name = 'question'
	repeat_name = 'followup'

	def __init__(s, info={}):
		I = s.def_info.copy()
		I.update(info)
		Acquisition.__init__(s,I)

class Questionnaire(Acquisition):

	def_info = {'technique':'questionnaire'}
	collection = 'questionnaires'

	part_name = 'question'
	repeat_name = 'followup'

	def __init__(s,test_type,info={}):

		I = s.def_info.copy()
		I.update( {'test_type':test_type} )
		I.update(info)
		Acquisition.__init__(s,I)

class Subject(MongoBacked):
	
	def_info = { 'DOB':datetime.datetime(1900,1,1),
				'sex':'f',
				'ID':'d0001001'
				}
	collection = 'subjects'
	
	def __init__(s,info={}):
		
		I = s.def_info.copy()
		I.update(info)
		I['_ID'] = I['ID']
		
		s.DOB = I['DOB']
		s.sex = I['sex']
		s.studies = []
		s.acquisitions = []
		s.data = I

class Study:
	
	def __init__(s):
		s.experiments = []
		
		
class Experiment:
	
	def __init__(s):
		s.processing_steps = []
		
		
class ProcessingStep:
	
	def __init__(s,inputs,outputs,fxn):
		for inp in inputs:
			s.input_description = DataDescription(inp)
		for oup in outputs:
			s.output = DataDescription(oup)
		
		s.function = fxn
		

