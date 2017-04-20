''' interface to MongoDB database '''

import pymongo

socket_path = '/tmp/mongodb-27017.sock'
MongoConn = pymongo.MongoClient(socket_path)
Mdb = None


def set_db(db_name):
    global Mdb
    Mdb = MongoConn[db_name]
    print('you are now accessing the db named', Mdb.name)


use_db_name = 'HBNL1'  # the "default" DB
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



