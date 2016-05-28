''' Organization for HBNL
'''

import datetime
import pymongo
import pandas as pd
import file_handling as FH

MongoConn = pymongo.MongoClient('/tmp/mongodb-27017.sock')
Mdb = MongoConn['COGAt']


def flatten_dict(D, prefix=''):
    if len(prefix) > 0:
        prefix += '_'
    flat = {}
    for k, v in D.items():
        if type(v) == dict:
            F = flatten_dict(v, prefix + k)
            flat.update(F)
        else:
            flat[prefix + k] = v
    return flat


def unflatten_dict(D, delimiter='_', skipkeys=set()):
    # use with caution, specify keys to skip
    default_skipkeys = {'_id', 'test_type', 'ind_id', 'insert_time', 'form_id'}
    skipkeys.update(default_skipkeys)
    resultDict = dict()
    for key, value in D.items():
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

# if type(d[part]) is not float: # should be more general check


def remove_NaTs(rec):
    for k, v in rec.items():
        typ = type(v)
        if typ != dict and typ == pd.tslib.NaTType:
            rec[k] = None


class MongoBacked:
    def store(s):
        s.data['insert_time'] = datetime.datetime.now()
        Mdb[s.collection].insert_one(s.data)

    def store_track(s):
        s.data['insert_time'] = datetime.datetime.now()
        insert = Mdb[s.collection].insert_one(s.data)
        return insert

    def storeNaTsafe(s):
        s.data['insert_time'] = datetime.datetime.now()
        remove_NaTs(s.data)
        Mdb[s.collection].insert_one(s.data)

    def update(s):
        s.data['update_time'] = datetime.datetime.now()
        Mdb[s.collection].update_one(s.update_query, {'$set': s.data})


class SourceInfo(MongoBacked):

    def __init__(s, coll, source_set, subcoll=None):

        s.collection = coll
        s.data = {'_source': source_set, '_subcoll': subcoll}
        s.update_query = {'_source': {'$exists': True}}


class Acquisition(MongoBacked):
    def_info = {'site': None,
                'date': datetime.datetime(1901, 1, 1),
                'subject': 't0001001'
                }
    collection = 'Acquisition'

    part_name = 'part'
    repeat_name = 'repeat'

    def __init__(s, info={}):
        I = Acquisition.def_info.copy()
        I.update(info)
        s.site = I['site']
        s.date = I['date']
        s.subject = I['subject']
        s.data = I


class EROcsv(MongoBacked):

    collection = 'EROcsv'

    desc_fields = ['power type', 'experiment', 'case', 'frequency min', 'frequency max',
                   'time min', 'time max', 'parameters', 'file date', 'mod date']

    def __init__(s, filepath, info):
        s.filepath = filepath
        s.data = info
        s.data['filepath'] = filepath
        s.data['unknown'] = list(s.data['unknown'])


class EROpheno(Acquisition):
    def_info = {'technique': 'EEG',
                'system': 'unknown'}
    collection = 'EROpheno'

    def __init__(s, data, data_file_id):
        s.data = data
        s.data_file_link = data_file_id

        s.subject = data['ID']
        s.session = data['session']
        s.experiment = data['experiment']

    def store(s):
        Sdata = s.data.copy()

        doc_query = {fd: Sdata.pop(fd)
                     for fd in ['ID', 'session', 'experiment']}
        desc_fields = ['power type', 'case', 'frequency min', 'frequency max',
                       'time min', 'time max']
        # converting description fields to strings and replacing '.' with 'p' to
        # avoid conflict with mongo nesting syntax
        data_desc = dd = {fd: str(Sdata.pop(fd)).replace('.', 'p')
                          for fd in desc_fields}

        doc_lookup = Mdb[s.collection].find(doc_query)
        dataD = {'EROcsv_link': s.data_file_link, 'data': Sdata}
        if doc_lookup.count() == 0:
            doc = doc_query
            doc[dd['power type']] = {dd['case']: {
                dd['frequency min']: {
                    dd['frequency max']: {
                        dd['time min']: {
                            dd['time max']: dataD}}}}
            }
            doc['insert time'] = datetime.datetime.now()

            Mdb[s.collection].insert_one(doc)
        else:
            update_str = '.'.join([dd[fd] for fd in desc_fields])
            Mdb[s.collection].update({'_id': doc_lookup[0]['_id']},
                                     {'$set': {update_str: dataD,
                                               'update time': datetime.datetime.now()}})


class Neuropsych(Acquisition):
    def_info = {'technique': 'cognitive test'}
    collection = 'neuropsych'

    part_name = 'experiment'
    repeat_name = 'session'

    def __init__(s, testname, info={}):
        I = s.def_info.copy()
        I.update({'testname': testname})
        I.update(info)
        Acquisition.__init__(s, I)


class ERPPeak(Acquisition):
    def_info = {'technique': 'EEG',
                'system': 'unknown'}

    collection = 'ERPpeaks'

    part_name = 'experiment'
    repeat_name = 'session'

    def __init__(s, info={}):
        I = s.def_info.copy()
        I.update(info)
        Acquisition.__init__(s, I)


class ERPData(Acquisition):
    def_info = {'technique': 'EEG',
                'system': 'unknown'}

    collection = 'ERPdata'

    part_name = 'experiment'
    repeat_name = 'session'

    def __init__(s, info={}):
        I = s.def_info.copy()
        I.update(info)
        Acquisition.__init__(s, I)


class EEGData(Acquisition):
    def_info = {'technique': 'EEG',
                'system': 'unknown'}

    collection = 'EEGdata'

    part_name = 'experiment'
    repeat_name = 'session'

    def __init__(s, info={}):
        I = s.def_info.copy()
        I.update(info)
        Acquisition.__init__(s, I)


class ElectrophysiologyCalculations(Acquisition):
    def_info = {'technique': 'EEG'}

    collection = 'EEGresults'

    part_name = 'experiment'
    repeat_name = 'session'

    def __init__(s, info={}):
        I = s.def_info.copy()
        I.update(info)
        Acquisition.__init__(s, I)


class EEGBehavior(Acquisition):
    def_info = {'technique': 'EEG Tasks'}
    collection = 'EEGbehavior'

    part_name = 'experiment'
    repeat_name = 'session'

    def __init__(s, info={}):
        I = s.def_info.copy()
        I.update(info)
        Acquisition.__init__(s, I)


class SSAGA(Acquisition):
    def_info = {'technique': 'interview'}
    collection = 'SSAGA'

    part_name = 'question'
    repeat_name = 'followup'

    def __init__(s, info={}):
        I = s.def_info.copy()
        I.update(info)
        Acquisition.__init__(s, I)


class Core(Acquisition):
    def_info = {'technique': 'interview'}
    collection = 'core'

    part_name = 'question'
    repeat_name = 'followup'

    def __init__(s, info={}):
        I = s.def_info.copy()
        I.update(info)
        Acquisition.__init__(s, I)


class Questionnaire(Acquisition):
    def_info = {'technique': 'questionnaire'}
    collection = 'questionnaires'

    part_name = 'question'
    repeat_name = 'followup'

    def __init__(s, questname, followup, info={}):
        I = s.def_info.copy()
        I.update({'questname': questname})
        I.update({'followup': followup})
        I.update(info)
        Acquisition.__init__(s, I)


class Subject(MongoBacked):
    def_info = {'DOB': datetime.datetime(1900, 1, 1),
                'sex': 'f',
                'ID': 'd0001001'
                }
    collection = 'subjects'

    def __init__(s, info={}):
        I = s.def_info.copy()
        I.update(info)
        I['_ID'] = I['ID']

        s.DOB = I['DOB']
        s.sex = I['sex']
        s.studies = []
        s.acquisitions = []
        s.data = I


class Session(MongoBacked):
    def_info = {'DOB': datetime.datetime(1900, 1, 1),
                'sex': 'f',
                'ID': 'd0001001'
                }
    collection = 'sessions'

    def __init__(s, info={}):
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
    def __init__(s, inputs, outputs, fxn):
        for inp in inputs:
            s.input_description = DataDescription(inp)
        for oup in outputs:
            s.output = DataDescription(oup)

        s.function = fxn
