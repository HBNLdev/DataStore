''' interface to MongoDB database '''

from datetime import datetime

import pymongo
import pandas as pd

socket_path = '/tmp/mongodb-27017.sock'
socket_path2 = 'mongodb://192.168.1.4/'
MongoConn = pymongo.MongoClient(socket_path)
Mdb = None


def set_db(db_name,verbose=True):
    global Mdb
    Mdb = MongoConn[db_name]
    if verbose:
        print('you are now accessing the db named', Mdb.name)


use_db_name = 'HBNL8'  # the "default" DB
set_db(use_db_name)  # always set the DB to default on import


def get_db(db_name):
    ''' return a Mongo DB object with the given name '''

    new_use_db_name = db_name
    new_Mdb = MongoConn[new_use_db_name]
    return new_Mdb


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


class MongoSocket(object):
    def __init__(s, socket_address):
        s.socket_address = socket_address
        s.conn = pymongo.MongoClient(socket_address)

    def display_contents(s):
        pass



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

    def __init__(s, collection='', data=None):
        s.collection = collection
        s.data = data

    def store(s):
        ''' store the record info (in data attr) into the target collection '''
        s.data['insert_time'] = datetime.now()
        Mdb[s.collection].insert_one(s.data)

    def store_track(s):
        ''' same as above but return the insert (for diagnostics) '''
        s.data['insert_time'] = datetime.now()
        insert = Mdb[s.collection].insert_one(s.data)
        return insert

    def storeNaTsafe(s):
        ''' store, removing NaTs from record (incompatible with mongo) '''
        s.data['insert_time'] = datetime.now()
        remove_NaTs(s.data)
        Mdb[s.collection].insert_one(s.data)

    def compare(s, field='uID'):
        ''' based on a key field (usually a uniquely identifying per-record
            string), determine if a record of that type already exists
            in the collection, and inform with boolean attr called new.
            if new is False, the _id of the existing record is kept. '''
        c = Mdb[s.collection].find({field: s.data[field]})
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
        Mdb[s.collection].update_one(s.update_query, {'$set': s.data})


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
