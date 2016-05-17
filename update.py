'''update collections'''

import master_info as mi
import organization as O
import pandas as pd
from datetime import timedelta


def subjects():

    mi_mtime = mi.load_master()
    source_rec = O.Mdb['subjects'].find({'_source': {'$exists': True}})

    # compare source file names and date modified

    if mi.master_path == source_rec[0]['_source'][0] and \
            abs(mi_mtime - source_rec[0]['_source'][1])<timedelta(0.00001):
        print('up to date')
        return  # same path and older/same mdate, no update required

    else:  # new masterfile, do update

        old_ids = set(rec['ID'] for rec in O.Mdb['subjects'].find(
            {'ID': {'$exists': True}}))
        new_ids = set(mi.master['ID'].tolist())  # sets
        add_ids = new_ids - old_ids
        print('the following IDs are being added:')
        print(add_ids)

        addID_df = mi.master[mi.master['ID'].isin(add_ids)]
        for rec in addID_df.to_dict(orient='records'):
            sO = O.Subject(rec)
            sO.storeNaTsafe()
            # can do sessions here too

        sourceO = O.SourceInfo('subjects', [mi.master_path, mi_mtime])
        sourceO.update()


def sessions():

    mi_mtime = mi.load_master()
    source_rec = O.Mdb['sessions'].find({'_source': {'$exists': True}})

    if mi.master_path == source_rec[0]['_source'][0] and \
            abs(mi_mtime - source_rec[0]['_source'][1])<timedelta(0.00001):
        print('up to date')
        return

    else:

        old_uids = set((r['ID'], r['session']) for r in O.Mdb['sessions'].find(
            {'ID': {'$exists': True}, 'session': {'$exists': True}}))

        df_lst = []
        for char in 'abcdefghijk':
            sessionDF = mi.master[mi.master[char + '-run'].notnull()]
            for col in ['raw', 'date', 'age']:
                sessionDF[col] = sessionDF[char + '-' + col]
            sessionDF['session'] = char
            df_lst.append(sessionDF)
        allsessionDF = pd.concat(df_lst)

        newuidDF = allsessionDF[['ID', 'session']]
        new_uids = set(tuple(row) for row in newuidDF.values)
        add_uids = new_uids - old_uids
        print(add_uids)

        allsessionDF.set_index('session', append=True, inplace=True)
        adduidDF = allsessionDF[allsessionDF.index.isin(add_uids)]
        for rec in adduidDF.to_dict(orient='records'):
            so = O.Session(rec)
            so.storeNaTsafe()

        sourceO = O.SourceInfo('sessions', (mi.master_path, mi_mtime))
        sourceO.update()


def erp():
    pass


def neuropsych_xml():
    pass
