''' represents Mongo collections (including build/update functions) as well as
    Mongo docs (including store/update/compare methods) '''

# note that if you are building the DB, you can set the name of the DB from database.py
# OR you can import db.database into the same namespace and use its set_db method

import os
from datetime import datetime
from glob import glob

import pandas as pd
import pymongo
from tqdm import tqdm

import db.database as D
from .assessment_matching import match_assessments
from .compilation import buildframe_fromdocs
from .file_handling import (MT_File, AVGH1_File, CNTH1_File, RestingDAT, Neuropsych_XML)
from .followups import preparefupdfs_forbuild
from .knowledge.questionnaires import (map_ph4, map_ph4_ssaga, map_ph123, map_ph123_ssaga,
                                       zork_p123_path, zork_p4_path,
                                       internalizing_dir, internalizing_file, externalizing_dir, externalizing_file,
                                       allrels_file, fham_file, max_fups)
from .master_info import load_master
from .quest_import import (import_questfolder_ph4, import_questfolder_ssaga_ph4, import_questfolder_ph123,
                           import_questfolder_ssaga_ph123, )
from .utils.compilation import (calc_followupcol, join_ufields, groupby_followup, ID_nan_strintfloat_COGA,
                                build_parentID, df_fromcsv)
from .utils.filename_parsing import parse_STinv_path, parse_cnt_path, parse_rd_path, parse_cnth1_path
from .utils.files import identify_files

# maps collection objects to their collection names
collection_names = {
    'Subjects': 'subjects',
    'Sessions': 'sessions',
    'Followups': 'followups',
    'ERPPeaks': 'ERPpeaks',
    'Neuropsych': 'neuropsych',
    'Questionnaires': 'questionnaires',
    'SSAGAs': 'ssaga',
    'Core': 'core',
    'Internalizing': 'internalizing',
    'Externalizing': 'externalizing',
    'AllRels': 'allrels',
    'FHAM': 'fham',
    'RawEEGData': 'raw_eegdata',
    'EEGData': 'cnth1s',
    'ERPData': 'avgh1s',
    'RestingPower': 'resting_power',
    'STInverseMats': 'STinverseMats',
    'EEGBehavior': 'EEGbehavior',
}

build_order = ['Subjects', 'Sessions', 'Followups', 'SSAGAs', 'Questionnaires', 'Internalizing',
               'Externalizing', 'Core', 'AllRels', 'FHAM',
               'ERPPeaks', 'Neuropsych', 'RestingPower',
               'RawEEGData', 'EEGData', 'ERPData',
               'STInverseMats',
               'EEGBehavior',
               ]

incremental_collections = {'ERPPeaks', 'EEGBehavior'}

restingpower_dir = '/processed_data/eeg/complete_result_09_16.d/results/'
restingpower_ns_file = 'ns_all_tests.dat'
restingpower_mc_fileA = 'mc_1st_test.dat'
restingpower_mc_fileB = 'mc_2nd_test.dat'

stinv_dir_base = '/processed_data/mat-files-v'


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
        D.Mdb[s.collection].insert_one(s.data)

    def store_track(s):
        ''' same as above but return the insert (for diagnostics) '''
        s.data['insert_time'] = datetime.now()
        insert = D.Mdb[s.collection].insert_one(s.data)
        return insert

    def storeNaTsafe(s):
        ''' store, removing NaTs from record (incompatible with mongo) '''
        s.data['insert_time'] = datetime.now()
        remove_NaTs(s.data)
        D.Mdb[s.collection].insert_one(s.data)

    def compare(s, field='uID'):
        ''' based on a key field (usually a uniquely identifying per-record
            string), determine if a record of that type already exists
            in the collection, and inform with boolean attr called new.
            if new is False, the _id of the existing record is kept. '''
        c = D.Mdb[s.collection].find({field: s.data[field]})
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
        D.Mdb[s.collection].update_one(s.update_query, {'$set': s.data})


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


# COLLECTION RELATED CODE


class MongoCollection(object):
    ''' parent class for a Mongo collection '''

    collection_name = ''

    def __init__(s):
        s.collection_name = collection_names[s.class_name()]

    @staticmethod
    def db_name():
        print(D.Mdb.name)

    def count(s):
        ''' return count of documents in collection '''
        return D.Mdb[s.collection_name].count()

    def drop(s):
        ''' drop collection, with a safety prompt '''

        if s.count() > 0:
            prompt = '{} contains {} docs, are you sure you want to drop? y/n: '.format(s.collection_name, s.count())
            actually = input(prompt)
            if actually == 'y':
                D.Mdb[s.collection_name].drop()
            else:
                print('drop aborted')
        else:
            print('collection already empty; nothing to drop')

    def clear_field(s, field):
        ''' clear collection coll of field (delete the key-value pair from all docs) '''

        query = {field: {'$exists': True}}
        updater = {'$unset': {field: ''}}

        D.Mdb[s.collection_name].update_many(query, updater)

    def describe(s):
        ''' print key pieces of info about collection '''
        # use count
        # use stats
        # use storageSize / totalSize / totalIndexSize
        pass

    def index(s):
        ''' if an index doesn't exist, create it. otherwise, reindex. '''
        # use createIndex
        # use dropIndex
        # use reIndex
        pass

    def class_name(s):
        return type(s).__name__


class Subjects(MongoCollection):
    ''' subject info, including demographics and sample membership '''

    def build(s):
        # fast
        master, master_mtime = load_master()
        for rec in tqdm(master.to_dict(orient='records')):
            so = MongoDoc(s.collection_name, rec)
            so.storeNaTsafe()

        D.Mdb[s.collection_name].create_index([('ID', pymongo.ASCENDING)])

    def update_from_followups(s):
        fup_query = {'session': {'$exists': True}}
        fup_fields = ['ID', 'session', 'followup', 'date']
        fup_proj = {f: 1 for f in fup_fields}
        fup_proj['_id'] = 0
        fup_docs = D.Mdb['followups'].find(fup_query, fup_proj)
        fup_df = buildframe_fromdocs(fup_docs, inds=['ID'])
        fup_IDs = list(set(fup_df.index.get_level_values('ID').tolist()))

        subject_query = {'ID': {'$in': fup_IDs}}
        subject_fields = ['ID']
        subject_proj = {f: 1 for f in subject_fields}
        subject_docs = D.Mdb[s.collection_name].find(subject_query, subject_proj)
        subject_df = buildframe_fromdocs(subject_docs, inds=['ID'])

        comb_df = subject_df.join(fup_df)

        for ID, row in tqdm(comb_df.iterrows()):
            session = row['session']
            fup_field = session + '-fup'
            date_field = session + '-fdate'
            D.Mdb[s.collection_name].update_one({'_id': row['_id']},
                                                {'$set': {fup_field: row['followup'],
                                                          date_field: row['date']}})

    def reset_update(s):

        session_letters = list(map(chr, range(97, 100 + max_fups)))
        for sletter in session_letters:
            fup_field = sletter + '-fup'
            date_field = sletter + '-fdate'
            s.clear_field(fup_field)
            s.clear_field(date_field)


class Sessions(MongoCollection):
    ''' HBNL (EEG) session info, including date and followup # '''

    def build(s):
        # fast
        master, master_mtime = load_master()
        run_letters = [col[0] for col in master.columns if col[-4:] == '-run']
        for char in run_letters:
            sessionDF = master[master[char + '-run'].notnull()]
            if sessionDF.empty:
                continue
            else:
                print(char)
                sessionDF.loc[:, 'session'] = char
                sessionDF.loc[:, 'followup'] = \
                    sessionDF.apply(calc_followupcol, axis=1)
                for col in ['raw', 'date', 'age']:
                    sessionDF.loc[:, col] = sessionDF[char + '-' + col]
                sessionDF.loc[:, 'uID'] = sessionDF.apply(join_ufields, axis=1)
                # drop unneeded columns ?
                # drop_cols = [col for col in sessionDF.columns if '-age' in col or
                #   '-date' in col or '-raw' in col or '-run' in col]
                # sessionDF.drop(drop_cols, axis=1, inplace=True)
                for rec in tqdm(sessionDF.to_dict(orient='records')):
                    so = MongoDoc(s.collection_name, rec)
                    so.storeNaTsafe()

        D.Mdb[s.collection_name].create_index([('ID', pymongo.ASCENDING)])

    def update_from_followups(s):
        session_letters = list(map(chr, range(97, 100 + max_fups)))
        for sletter in session_letters:
            match_assessments(s.collection_name, to_coll='followups',
                              fup_field='session', fup_val=sletter,
                              match_datefield='date')

    def reset_update(s):
        s.clear_field('followup')
        s.clear_field('date_followup')
        s.clear_field('date_diff_followup')


class Followups(MongoCollection):
    ''' COGA follow-up info, including mean date and corresponding session letter '''

    def build(s):

        fup_dfs = preparefupdfs_forbuild()
        for fup, df in fup_dfs.items():
            print(fup)
            for rec in tqdm(df.reset_index().to_dict(orient='records')):
                fupO = MongoDoc(s.collection_name, rec)
                fupO.storeNaTsafe()

        D.Mdb[s.collection_name].create_index([('ID', pymongo.ASCENDING)])

    def update_from_sessions(s):
        p123_fups = ['p1', 'p2', 'p3', ]
        p4_fups = ['p4f' + str(f) for f in range(max_fups + 1)]
        fups = p123_fups + p4_fups
        for fup in fups:
            match_assessments(s.collection_name, to_coll='sessions',
                              fup_field='followup', fup_val=fup,
                              match_datefield='date')

    def reset_update(s):
        s.clear_field('session')
        s.clear_field('date_session')
        s.clear_field('date_diff_session')


class ERPPeaks(MongoCollection):
    ''' time-regional extrema in event-related potential waveforms '''

    def build(s):

        mt_files, datemods = identify_files('/processed_data/mt-files/', '*.mt')
        add_dirs = ['ant_phase4__peaks_2014', 'ant_phase4_peaks_2015',
                    'ant_phase4_peaks_2016',
                    'vp3_peak_master',
                    'vp3_phase4__peaks_2015', 'vp3_phase4__peaks_2016',
                    'aod_phase4__peaks_2015', 'aod_phase4__peaks_2016',
                    'cpt_h1_peaks_may_2016',
                    # 'non_coga_vp3',
                    # 'aod_bis_18-25controls',
                    # 'nki_ppick', 'phase4_redo',
                    ]
        for subdir in add_dirs:
            mt_files2, datemods2 = identify_files(
                '/active_projects/HBNL/' + subdir + '/', '*.mt')
            mt_files.extend(mt_files2)
            datemods.extend(datemods2)
        bad_files = ['/processed_data/mt-files/ant/uconn/mc/an1a0072007.df.mt',
                     ]
        for fp in tqdm(mt_files):
            if '/waves/' in fp or fp in bad_files:
                continue

            mtO_ck = MT_File(fp)  # get file info

            erpO_ck = MongoDoc(s.collection_name, mtO_ck.data)
            erpO_ck.compare()  # populates s.new with bool

            if erpO_ck.new:  # "brand new" get general info

                mtO = MT_File(fp)
                try:
                    mtO.parse_fileDB(general_info=True)
                except:
                    print(fp, 'failed')
                    continue
                erpO = MongoDoc(s.collection_name, mtO.data)
                erpO.store()
            else:  # if not brand new, check if the experiment is already in the doc
                try:
                    erpO_ck.doc[mtO.file_info['experiment']]
                except KeyError:  # only update experiment info if not already in db
                    mtO = MT_File(fp)  # refresh the file obj
                    try:
                        mtO.parse_fileDB(general_info=False)
                    except:
                        print(fp, 'failed')
                        continue
                    mtO.parse_fileDB()
                    erpO = MongoDoc(s.collection_name, mtO.data)
                    erpO.compare()  # to get update query
                    erpO.update()


class Neuropsych(MongoCollection):
    ''' results of neuropsychological tests '''

    def build(s):
        # 10 minutes
        xml_files, datemods = identify_files('/raw_data/neuropsych/', '*.xml')
        for fp in tqdm(xml_files):
            xmlO = Neuropsych_XML(fp)
            xmlO.assure_quality()
            xmlO.data['date'] = xmlO.data['testdate']
            nsO = MongoDoc(s.collection_name, xmlO.data)
            nsO.store()

    def update_from_sfups(s):
        max_npsych_fups = max(D.Mdb[s.collection_name].distinct('np_followup'))
        for fup in range(max_npsych_fups + 1):
            # match sessions
            match_assessments(s.collection_name, to_coll='sessions',
                              fup_field='np_followup', fup_val=fup,
                              match_datefield='date')
            # match followups
            match_assessments(s.collection_name, to_coll='followups',
                              fup_field='np_followup', fup_val=fup,
                              match_datefield='date')

    def reset_update(s):
        s.clear_field('followup')
        s.clear_field('date_followup')
        s.clear_field('date_diff_followup')

        s.clear_field('session')
        s.clear_field('date_session')
        s.clear_field('date_diff_session')


class Questionnaires(MongoCollection):
    ''' questionnaire info '''

    def questionnaires_ph123(s):
        kmap = map_ph123
        path = zork_p123_path + 'session/'
        for qname in kmap.keys():
            print(qname)
            import_questfolder_ph123(qname, kmap, path, s.collection_name)

    def questionnaires_ph4(s):
        # takes  ~20 seconds per questionnaire
        # phase 4 non-SSAGA
        kmap = map_ph4.copy()
        del kmap['cal']
        path = zork_p4_path + 'session/'
        for qname in kmap.keys():
            print(qname)
            import_questfolder_ph4(qname, kmap, path, s.collection_name)

    def build(s):
        s.questionnaires_ph123()
        s.questionnaires_ph4()

    def update_from_sessions(s):
        quest_subcolls = ['dependence', 'craving', 'sre', 'daily', 'neo', 'sensation', 'aeq', 'bis', 'achenbach']
        p4_fups = ['p4f' + str(f) for f in range(max_fups + 1)]
        quest_fups = ['p2', 'p3', ] + p4_fups
        for quest_subcoll in quest_subcolls:
            for fup in quest_fups:
                subcoll_query = {'questname': quest_subcoll}
                match_assessments(s.collection_name, to_coll='sessions',
                                  fup_field='followup', fup_val=fup,
                                  match_datefield='date',
                                  add_match_query=subcoll_query)

    def reset_update(s):

        s.clear_field('session')
        s.clear_field('date_session')
        s.clear_field('date_diff_session')


class SSAGAs(MongoCollection):
    ''' SSAGA questionnaire info '''

    def ssaga_ph123(s):
        kmap = map_ph123_ssaga
        path = zork_p123_path + 'session/'
        for qname in kmap.keys():
            print(qname)
            import_questfolder_ssaga_ph123(qname, kmap, path, s.collection_name)

    def ssaga_ph4(s):
        # SSAGA
        kmap = map_ph4_ssaga.copy()
        path = zork_p4_path + 'session/'
        for qname in kmap.keys():
            print(qname)
            import_questfolder_ssaga_ph4(qname, kmap, path, s.collection_name)

    def build(s):
        s.ssaga_ph123()
        s.ssaga_ph4()

    def update_from_sessions(s):
        ssaga_subcolls = ['ssaga', 'cssaga', 'pssaga', 'dx_ssaga', 'dx_cssaga', 'dx_pssaga']
        p4_fups = ['p4f' + str(f) for f in range(max_fups + 1)]
        ssaga_fups = ['p1', 'p2', 'p3', ] + p4_fups
        for ssaga_subcoll in ssaga_subcolls:
            for fup in ssaga_fups:
                subcoll_query = {'questname': ssaga_subcoll}
                match_assessments(s.collection_name, to_coll='sessions',
                                  fup_field='followup', fup_val=fup,
                                  match_datefield='date',
                                  add_match_query=subcoll_query)

    def reset_update(s):

        s.clear_field('session')
        s.clear_field('date_session')
        s.clear_field('date_diff_session')


class Core(MongoCollection):
    ''' substance abuse info distilled from SSAGA questionnaires '''

    def build(s):
        # fast
        folder = zork_p4_path + 'subject/core/'
        csv_files = glob(folder + '*.csv')
        if len(csv_files) != 1:
            print(len(csv_files), 'csvs found, aborting')
        path = csv_files[0]
        df = df_fromcsv(path)
        for drec in tqdm(df.to_dict(orient='records')):
            ro = MongoDoc(s.collection_name, drec)
            ro.store()


class Internalizing(MongoCollection):
    ''' John Kramer's internalizing phenotype info '''

    def build(s):
        def convert_internalizing_columns(cols):
            col_tups = []

            for col in cols:
                pieces = col.split('_')
                info = '_'.join(pieces[:-2])
                fup = '_'.join(pieces[-2:])

                col_tups.append((info, fup))

            return pd.MultiIndex.from_tuples(col_tups, names=('', 'followup'))

        def convert_intfup(fup):

            if int(fup[1]) < 4:
                fup_rn = 'p' + fup[1]
            else:
                fup_rn = 'p' + fup[1] + fup[-2:]

            return fup_rn

        def convert_questname(v):
            if v[0] == 'c':
                return 'cssaga'
            elif v[0] == 's':
                return 'ssaga'

        int_path = internalizing_dir + internalizing_file

        int_df = pd.read_csv(int_path, na_values=' ')
        int_df['ID'] = int_df['ID'].apply(int).apply(str)
        int_df.set_index('ID', inplace=True)

        # convert columns into multiindex
        int_df2 = int_df.copy()
        int_df2.columns = convert_internalizing_columns(int_df2.columns)

        # stack followups and rename columns to be in DB convention
        int_df3 = int_df2.stack(-1).dropna(how='all').reset_index(). \
            rename(columns={'intvyr': 'date', 'intvsource': 'questname'})

        # convert followups and questnames to be in DB convention
        int_df4 = int_df3.copy()
        int_df4['followup'] = int_df4['followup'].apply(convert_intfup)
        int_df4['questname'] = int_df4['questname'].apply(convert_questname)
        int_df4.set_index(['ID', 'questname', 'followup'], inplace=True)
        int_df4.sort_index(inplace=True)

        # calculate maximum score across fups for each individual
        g = groupby_followup(int_df4)
        score_cols = [col for col in int_df4.columns if 'score' in col]
        int_df5 = int_df4.reset_index().set_index('ID')
        for col in score_cols:
            int_df5[col + '_fupmax'] = g[col].max()

        for drec in tqdm(int_df5.reset_index().to_dict(orient='records')):
            ro = MongoDoc(s.collection_name, drec)
            ro.store()

    def update_from_ssaga(s):

        int_query = {}
        int_proj = {'ID': 1, 'questname': 1, 'followup': 1, '_id': 1}
        int_docs = D.Mdb[s.collection_name].find(int_query, int_proj)
        int_df = buildframe_fromdocs(int_docs, inds=['ID', 'questname', 'followup'])

        IDs = list(set(int_df.index.get_level_values('ID')))

        ssaga_query = {'questname': {'$in': ['ssaga', 'cssaga']}, 'ID': {'$in': IDs}, 'session': {'$exists': True}}
        ssaga_proj = {'ID': 1, 'session': 1, 'date_diff_session': 1, 'followup': 1, 'questname': 1, 'date': 1, '_id': 0}

        ssaga_docs = D.Mdb[collection_names['SSAGAs']].find(ssaga_query, ssaga_proj)
        ssaga_df = buildframe_fromdocs(ssaga_docs, inds=['ID', 'questname', 'followup'])

        comb_df = int_df.join(ssaga_df).dropna(subset=['date'])
        print(int_df.shape[0] - comb_df.shape[0], 'internalizing docs could not be matched')

        for ID, row in tqdm(comb_df.iterrows()):
            D.Mdb['internalizing'].update_one({'_id': row['_id']},
                                              {'$set': {'session': row['session'],
                                                        'date_diff_session': row['date_diff_session'],
                                                        'date': row['date'], }
                                               })

    def reset_update(s):

        s.clear_field('session')
        s.clear_field('date_session')
        s.clear_field('date_diff_session')


class Externalizing(MongoCollection):
    ''' externalizing phenotype info '''

    def build(s):
        path = externalizing_dir + externalizing_file
        datemod = datetime.fromtimestamp(os.path.getmtime(path))
        df = df_fromcsv(path, 'IND_ID')
        for drec in tqdm(df.to_dict(orient='records')):
            ro = MongoDoc(s.collection_name, drec)
            ro.store()


class AllRels(MongoCollection):
    ''' all relatives file info '''

    def build(s):
        folder = zork_p4_path + 'subject/rels/'
        path = folder + allrels_file
        datemod = datetime.fromtimestamp(os.path.getmtime(path))

        import_convcols = ['IND_ID', 'FAM_ID', 'F_ID', 'M_ID']
        import_convdict = {col: ID_nan_strintfloat_COGA for col in import_convcols}
        rename_dict = {'FAM_ID': 'famID', 'IND_ID': 'ID', 'TWIN': 'twin', 'SEX': 'sex'}

        rel_df = pd.read_csv(path, converters=import_convdict, low_memory=False)
        rel_df = rel_df.rename(columns=rename_dict)
        rel_df['fID'] = rel_df[['famID', 'F_ID']].apply(build_parentID, axis=1, args=['famID', 'F_ID'])
        rel_df['mID'] = rel_df[['famID', 'M_ID']].apply(build_parentID, axis=1, args=['famID', 'M_ID'])

        for drec in tqdm(rel_df.to_dict(orient='records')):
            ro = MongoDoc(s.collection_name, drec)
            ro.store()


class FHAM(MongoCollection):
    ''' family history assessment module '''

    def build(s):
        folder = zork_p4_path + 'subject/fham/'
        path = folder + fham_file
        datemod = datetime.fromtimestamp(os.path.getmtime(path))
        df = df_fromcsv(path, 'IND_ID')
        for drec in tqdm(df.to_dict(orient='records')):
            ro = MongoDoc(s.collection_name, drec)
            ro.store()


class RawEEGData(MongoCollection):
    ''' *.cnt or *.rd file containing raw continuous EEG data '''

    def build(s):

        rd_start_dir = '/raw_data/masscomp/'
        rd_glob_expr = '*rd'
        rd_files, rd_datemods = identify_files(rd_start_dir, rd_glob_expr)
        for rd_path in tqdm(rd_files):
            try:
                if 'bad' in rd_path:
                    continue
                info = parse_rd_path(rd_path)
                raweegO = MongoDoc(s.collection_name, info)
                raweegO.store()
            except:
                print('problem with', rd_path)

        # cnts
        cnt_start_dir = '/raw_data/neuroscan/'
        cnt_glob_expr = '*cnt'
        cnt_files, cnt_datemods = identify_files(cnt_start_dir, cnt_glob_expr)
        for cnt_path in tqdm(cnt_files):
            try:
                info = parse_cnt_path(cnt_path)
                if not info['note']:
                    raweegO = MongoDoc(s.collection_name, info)
                    raweegO.store()
            except:
                print('problem with', cnt_path)


class EEGData(MongoCollection):
    ''' *.cnt.h1 file containing continuous EEG data '''

    def build(s):
        start_dir = '/processed_data/cnt-h1-files/'
        glob_expr = '*cnt.h1'
        cnth1_files, datemods = identify_files(start_dir, glob_expr)
        for f in tqdm(cnth1_files):
            data = parse_cnth1_path(f)
            if data['n_chans'] not in ['21', '32', '64']:
                print(f, 'had unexpected number of chans')
                continue
            eegO = MongoDoc(s.collection_name, data)
            eegO.store()


class ERPData(MongoCollection):
    ''' *.avg.h1 file containing ERP data '''

    def build(s):
        start_dir = '/processed_data/avg-h1-files/'
        glob_expr = '*avg.h1'
        avgh1_files, datemods = identify_files(start_dir, glob_expr)
        for f in tqdm(avgh1_files):
            fO = CNTH1_File(f)  # basically identical at this point
            fO.parse_fileDB()
            erpO = MongoDoc(s.collection_name, fO.data)
            erpO.store()


class RestingPower(MongoCollection):
    ''' resting state power estimates (calculated by David) '''

    def build(s):
        nsO = RestingDAT(restingpower_dir + restingpower_ns_file)
        nsO.ns_to_dataframe()
        rec_lst = nsO.file_df.to_dict(orient='records')

        mcO_A = RestingDAT(restingpower_dir + restingpower_mc_fileA)
        mcO_A.mc_to_dataframe(session='a')
        rec_lst.extend(mcO_A.file_df.to_dict(orient='records'))

        mcO_B = RestingDAT(restingpower_dir + restingpower_mc_fileB)
        mcO_B.mc_to_dataframe(session='b')
        rec_lst.extend(mcO_B.file_df.to_dict(orient='records'))

        for rec in tqdm(rec_lst):
            rpO = MongoDoc(s.collection_name, rec)
            rpO.store()


class STInverseMats(MongoCollection):
    ''' *.mat file containing inverse S-tranformed ERO power data '''

    def build(s, check_update=False, mat_files=None):

        if mat_files is None:
            start_fins = ['40', '60']
            glob_expr = '*st.mat'
            mat_files = []
            dates = []
            for fin in start_fins:
                f_mats, f_dates = identify_files(stinv_dir_base + fin, glob_expr)
                mat_files.extend(f_mats)
                dates.extend(f_dates)
        for f in tqdm(mat_files):
            infoD = parse_STinv_path(f)
            infoD['path'] = f
            matO = MongoDoc(s.collection_name, infoD)

            store = False
            if check_update:
                matO.compare(field='path')
                if matO.new:
                    store = True
            else:
                store = True

            if store:
                matO.store()


class EEGBehavior(MongoCollection):
    ''' behavioral info (accuracy, reaction time, etc.) from EEG experiments '''

    def build(s, files_dms=None):
        ''' unlike others, this build does an incremental "update".
                if used, files_dms should be a list of file/datemodifed tuples '''

        # ~8 hours total to parse all *.avg.h1's for behavior
        # files_dms = pickle.load( open(
        #    '/active_projects/mike/pickles/avgh1s_dates.p', 'rb')  )
        if not files_dms:
            start_dir = '/processed_data/avg-h1-files/'
            glob_expr = '*avg.h1'
            avgh1_files, datemods = identify_files(start_dir, glob_expr)
        else:
            avgh1_files, datemods = zip(*files_dms)

        for f in tqdm(avgh1_files):
            try:
                fO = AVGH1_File(f)  # get uID and file_info
                if fO.file_info['experiment'] == 'err':
                    continue

                # simply check if the ID-session-experiment already exists
                erpbeh_obj_ck = MongoDoc(s.collection_name, fO.data)
                erpbeh_obj_ck.compare()  # populates s.new with bool

                if erpbeh_obj_ck.new:  # "brand new", get general info
                    fO.parse_behav_forDB(general_info=True)
                    erpbeh_obj = MongoDoc(s.collection_name, fO.data)
                    erpbeh_obj.store()
                else:  # if not brand new, check if the experiment is already in the doc
                    try:
                        erpbeh_obj_ck.doc[fO.file_info['experiment']]
                    except KeyError:  # only update experiment info if not already in db
                        fO = AVGH1_File(f)  # refresh the file obj
                        fO.parse_behav_forDB()
                        erpbeh_obj = MongoDoc(s.collection_name, fO.data)
                        erpbeh_obj.compare()  # to get update query
                        erpbeh_obj.update()
            except:
                print(f, 'failed')
        # sourceO = SourceInfo('EEGbehavior', list(zip(avgh1_files, datemods)))
        # sourceO.store()
        inds = D.Mdb[s.collection_name].list_indexes()
        try:
            next(inds)  # returns the _id index
            next(inds)  # check if any other index exists
            D.Mdb[s.collection_name].reindex()  # if it does, just reindex
        except StopIteration:  # otherwise, create it
            D.Mdb[s.collection_name].create_index([('uID', pymongo.ASCENDING)])


