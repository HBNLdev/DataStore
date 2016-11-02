'''update collections'''

from datetime import timedelta

import pandas as pd

from .master_info import load_master, master_path
from .file_handling import identify_files, MT_File
from .organization import Mdb, Subject, SourceInfo, Session, ERPPeak


def subjects():

    master, mi_mtime = load_master()
    source_rec = Mdb['subjects'].find({'_source': {'$exists': True}})

    # compare source file names and date modified

    if master_path == source_rec[0]['_source'][0] and \
            abs(mi_mtime - source_rec[0]['_source'][1])<timedelta(0.00001):
        print('up to date')
        return  # same path and older/same mdate, no update required

    else:  # new masterfile, do update

        old_ids = set(rec['ID'] for rec in Mdb['subjects'].find(
            {'ID': {'$exists': True}}))
        new_ids = set(master['ID'].tolist())  # sets
        add_ids = new_ids - old_ids
        print('the following IDs are being added:')
        print(add_ids)

        addID_df = master[master['ID'].isin(add_ids)]
        for rec in addID_df.to_dict(orient='records'):
            sO = Subject(rec)
            sO.storeNaTsafe()
            # can do sessions here too

        sourceO = SourceInfo('subjects', [master_path, mi_mtime])
        sourceO.update()


def sessions():

    master, mi_mtime = load_master()
    source_rec = Mdb['sessions'].find({'_source': {'$exists': True}})

    if master_path == source_rec[0]['_source'][0] and \
            abs(mi_mtime - source_rec[0]['_source'][1])<timedelta(0.00001):
        print('up to date')
        return

    else:

        old_uids = set((r['ID'], r['session']) for r in Mdb['sessions'].find(
            {'ID': {'$exists': True}, 'session': {'$exists': True}}))

        df_lst = []
        for char in 'abcdefghijk':
            sessionDF = master[master[char + '-run'].notnull()]
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
            so = Session(rec)
            so.storeNaTsafe()

        sourceO = SourceInfo('sessions', (master_path, mi_mtime))
        sourceO.update()


def erp():
    mt_files, datemods = identify_files('/processed_data/mt-files/','*.mt')
    bad_files=['/processed_data/mt-files/ant/uconn/mc/an1a0072007.df.mt',
        ]

    source_rec = Mdb['ERP'].find({'_source': {'$exists': True}})[0]
    
    old_files = set(t[0] for t in source_rec['_source'])
    new_files = set(mt_files)
    add_files = new_files - old_files

    for fp in add_files:
        if fp in bad_files:
            continue
        mtO = MT_File(fp)
        mtO.parse_fileDB()
        erpO = ERPPeak( mtO.data )
        erpO.store()

    # sourceO = SourceInfo('ERP', list(add_uids))
    # sourceO.update()

def neuropsych_xml():
    pass
