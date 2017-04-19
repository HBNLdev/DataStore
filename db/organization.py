''' interface to MongoDB database and class representing docs '''

from datetime import datetime

import pymongo

socket_path = '/tmp/mongodb-27017.sock'
use_db_name = 'HBNL1'

MongoConn = pymongo.MongoClient(socket_path)
Mdb = MongoConn[use_db_name]


def get_db(db_name):
    ''' return a Mongo DB object with the given name '''

    new_use_db_name = db_name
    new_Mdb = MongoConn[new_use_db_name]
    return new_Mdb


# DOC RELATED CODE

def remove_NaTs(rec):
    ''' given a record-style dict, convert all NaT vals to None '''
    for k, v in rec.items():
        typ = type(v)
        if typ != dict and typ == pd.tslib.NaTType:
            rec[k] = None


class MongoDoc(object):
    ''' base class representing record obj to be stored, compared, or updated
        into a MongoDB collection. classes that inherit specify type
        of info and target collection '''

    Mdb = MongoConn[use_db_name]

    def __init__(s, collection='', data=None):
        s.collection = collection
        s.data = data

    def store(s):
        ''' store the record info (in data attr) into the target collection '''
        s.data['insert_time'] = datetime.now()
        s.Mdb[s.collection].insert_one(s.data)

    def store_track(s):
        ''' same as above but return the insert (for diagnostics) '''
        s.data['insert_time'] = datetime.now()
        insert = s.Mdb[s.collection].insert_one(s.data)
        return insert

    def storeNaTsafe(s):
        ''' store, removing NaTs from record (incompatible with mongo) '''
        s.data['insert_time'] = datetime.now()
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
        s.data['update_time'] = datetime.now()
        s.Mdb[s.collection].update_one(s.update_query, {'$set': s.data})


class SourceInfo(MongoDoc):
    '''  a record containing info about the data source
        for an obj or a collection build operation '''

    def __init__(s, collection='', data=None, subcoll=None):
        s.collection = collection
        s.data = {'_source': data, '_subcoll': subcoll}
        s.update_query = {'_source': {'$exists': True}}


class Questionnaire(MongoDoc):
    ''' questionnaire info '''

    def __init__(s, collection, questname, followup_lbl, data=None):
        s.collection = collection
        s.data = data
        s.data.update({'questname': questname})
        s.data.update({'followup': followup_lbl})


class MongoSocket(object):
    def __init__(s, socket_address):
        s.socket_address = socket_address
        s.conn = pymongo.MongoClient(socket_address)

    def display_contents(s):
        pass


class MongoDB(object):
    def __init__(s, db_name):
        s.db_name = db_name
        s.db = MongoConn[db_name]

    def display_contents(s):
        print('{:>14} | {:>7}'.format('------------', '-------'))
        print('{:>14} | {:>7}'.format('collection', 'count'))
        print('{:>14} | {:>7}'.format('------------', '-------'))

        for cn in s.db.collection_names():
            c = Mdb[cn]
            c_count = c.count()

            line = '{:>14} | {:>7}'.format(cn, c_count)

            print(line)

    def display_indices(s):
        pass
