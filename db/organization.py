''' tools for dealing with and objects for representing record-style dicts.
    does storage, comparison, and updating into MongoDB database collections '''

import datetime

import pymongo

from .utils.records import remove_NaTs, unflatten_dict

MongoConn = pymongo.MongoClient('/tmp/mongodb-27017.sock')


class MongoBacked:
    ''' base class representing record obj to be stored, compared, or updated
        into a MongoDB collection. classes that inherit specify type
        of info and target collection '''

    Mdb = MongoConn['COGA']
    def_info = {}

    def __init__(s, info={}):
        I = s.def_info.copy()
        I.update(info)
        s.data = I

    def store(s):
        ''' store the record info (in data attr) into the target collection '''
        s.data['insert_time'] = datetime.datetime.now()
        s.Mdb[s.collection].insert_one(s.data)

    def store_track(s):
        ''' same as above but return the insert (for diagnostics) '''
        s.data['insert_time'] = datetime.datetime.now()
        insert = s.Mdb[s.collection].insert_one(s.data)
        return insert

    def storeNaTsafe(s):
        ''' store, removing NaTs from record (incompatible with mongo) '''
        s.data['insert_time'] = datetime.datetime.now()
        remove_NaTs(s.data)
        s.Mdb[s.collection].insert_one(s.data)

    def compare(s, field='uID'):
        ''' based on a key field (usually a uniquely identifying per-record
            string), determine if a record of that type already exists
            in the collection, and inform with boolean attr called new.
            if new is False, the _id of the existing record is kept. '''
        c = s.Mdb[s.collection].find({field: s.data[field]})
        if c.count() == 0:
            s.new = True
        else:
            s.new = False
            s.doc = next(c)
            s._id = s.doc['_id']
            s.update_query = {'_id': s._id}

    def update(s):
        ''' update an existing record '''
        if 'insert_time' in s.data:
            del s.data['insert_time']
        s.data['update_time'] = datetime.datetime.now()
        s.Mdb[s.collection].update_one(s.update_query, {'$set': s.data})


# expose Mdb for use at module level
Mdb = MongoBacked.Mdb


class SourceInfo(MongoBacked):
    '''  a record containing info about the data source
        for an obj or a collection build operation '''

    def __init__(s, coll, source_set, subcoll=None):
        s.collection = coll
        s.data = {'_source': source_set, '_subcoll': subcoll}
        s.update_query = {'_source': {'$exists': True}}


# specific record type classes

class Neuropsych(MongoBacked):
    ''' results of neuropsychological tests '''

    collection = 'neuropsych'


class RawEEGData(MongoBacked):
    ''' *.cnt file containing raw continuous EEG data '''

    collection = 'raw_eegdata'


class EEGData(MongoBacked):
    ''' *.cnt.h1 file containing continuous EEG data '''

    collection = 'cnth1s'


class ERPData(MongoBacked):
    ''' *.avg.h1 file containing ERP data '''

    collection = 'avgh1s'


class ERPPeak(MongoBacked):
    ''' time-regional maxima in event-related potential waveforms '''

    collection = 'ERPpeaks'


class RestingPower(MongoBacked):
    ''' resting state power estimates '''

    collection = 'resting_power'


class STransformInverseMats(MongoBacked):
    ''' *.mat file containing inverse S-tranformed ERO power data '''

    collection = 'STinverseMats'


class EEGBehavior(MongoBacked):
    ''' behavioral info (accuracy, reaction time) from EEG experiments '''

    collection = 'EEGbehavior'


class Core(MongoBacked):
    ''' substance abuse info distilled from SSAGA questionnaires '''

    collection = 'core'


class Internalizing(MongoBacked):
    ''' John Kramer's internalizing phenotype info '''

    collection = 'internalizing'


class Externalizing(MongoBacked):
    ''' externalizing phenotype info '''

    collection = 'externalizing'


class FHAM(MongoBacked):
    ''' family history assessment module '''

    collection = 'fham'


class AllRels(MongoBacked):
    ''' all relatives file '''

    collection = 'allrels'


class Questionnaire(MongoBacked):
    ''' questionnaire info '''

    collection = 'questionnaires'

    def __init__(s, questname, followup, session=None, info={}):
        I = s.def_info.copy()
        I.update({'questname': questname})
        I.update({'followup': followup})
        if session:
            I.update({'session': session})
        I.update(info)
        s.data = I


class SSAGA(Questionnaire):
    ''' SSAGA questionnaire info '''

    collection = 'ssaga'


class Subject(MongoBacked):
    ''' subject info, including demographics and sample membership '''

    collection = 'subjects'

    def __init__(s, info={}):
        I = s.def_info.copy()
        I.update(info)
        I['_ID'] = I['ID']
        s.data = I


class Session(MongoBacked):
    ''' HBNL (EEG) session info, including date and followup # '''

    collection = 'sessions'

    def __init__(s, info={}):
        I = s.def_info.copy()
        I.update(info)
        I['_ID'] = I['ID']
        s.data = I


class FollowUp(MongoBacked):
    ''' COGA follow-up info, including mean date and corresponding session letter '''

    collection = 'followups'

    def __init__(s, info={}):
        I = s.def_info.copy()
        I.update(info)
        I['_ID'] = I['ID']
        s.data = I


class EROcsv(MongoBacked):
    ''' CSV containing ERO power results '''

    collection = 'EROcsv'
    desc_fields = ['power type', 'experiment', 'case', 'frequency min',
                   'frequency max', 'time min', 'time max', 'parameters',
                   'file date', 'mod date']

    def __init__(s, filepath, info):
        s.data = info
        s.data['filepath'] = filepath
        s.data['unknown'] = list(s.data['unknown'])


class EROcsvresults(MongoBacked):
    ''' ERO power results inside a EROcsv '''

    collection = 'EROcsvresults'

    def __init__(s, data, data_file_id):
        s.data = data
        s.data_file_link = data_file_id

    def store_joined_bulk(s):
        ''' bulk_write list of records that have been formatted
            from joining many CSVs together '''
        adding_uids = [new_rec['uID'] for new_rec in s.data]
        doc_lookup = s.Mdb[s.collection].find(
            {'uID': {'$in': adding_uids}}, {'uID': 1})
        existing_mapper = {doc['uID']: doc['_id'] for doc in doc_lookup}

        bulk_lst = []
        for new_rec in s.data:
            if new_rec['uID'] in existing_mapper.keys():
                # add UpdateOne
                set_spec = unflatten_dict({k.replace('_', '.', 4): v
                                           for k, v in new_rec.items()})
                set_spec.update({'update time': datetime.datetime.now()})
                update_op = pymongo.operations.UpdateOne(
                    {'_id': existing_mapper[new_rec['uID']]},
                    {'$set': set_spec})
                bulk_lst.append(update_op)
            else:
                # add InsertOne
                add_doc = unflatten_dict(new_rec)
                add_doc['insert time'] = datetime.datetime.now()
                insert_op = pymongo.operations.InsertOne(add_doc)
                bulk_lst.append(insert_op)
        try:
            s.Mdb[s.collection].bulk_write(bulk_lst, ordered=False)
        except:
            print(s.data_file_link)
            print(bulk_lst)
            raise
