''' build collections '''

import os
from glob import glob
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
from tqdm import tqdm
import pymongo

from .master_info import load_master, master_path, ID_nan_strintfloat_COGA, build_parentID
from .quest_import import (map_ph4, map_ph4_ssaga, map_ph123, map_ph123_ssaga,
                           import_questfolder_ph4, import_questfolder_ssaga_ph4, import_questfolder_ph123,
                           import_questfolder_ssaga_ph123,
                           match_fups2sessions_fast_multi,
                           match_fups_sessions_generic, df_fromcsv)
from .organization import (Subject, SourceInfo, Session, FollowUp, ERPPeak, Neuropsych,
    Questionnaire, Core, Internalizing, Externalizing, FHAM, AllRels, RawEEGData, EEGData, ERPData, RestingPower,
    STransformInverseMats, EEGBehavior, SSAGA, Mdb, EROcsv, EROcsvresults)
from .file_handling import (identify_files,
                            parse_STinv_path, parse_cnt_path, parse_rd_path, parse_cnth1_path,
                            MT_File, CNTH1_File, AVGH1_File, RestingDAT,
                            Neuropsych_XML, TOLT_Summary_File, CBST_Summary_File,
                            ERO_CSV)
from .followups import preparefupdfs_forbuild


buildAssociations = { 
            'Core':'core',
            'EEGBehavior':'eeg_behavior',
            'EEGData':'raw_eegdata',
            'EROcsv':'ero_pheno_join_bulk',
            'EROcsvresults':['shared','EROcsv','ero_pheno_join_bulk'],
            'ERPData':'erp_data',
            'ERPPeak':'erp_peaks',
            'Externalizing':'externalizing',
            'FHAM':'fham',
            'Internalizing':'internalizing',
            'Neuropsych':['multiple','neuropsych_TOLT','neuropsych_CBST'],
            'Questionnaire':'questionnaires_ph4',#questionnaires_ph123
            'RawEEGData':'raw_eegdata',
            'SSAGA':'questionnaires_ssaga',
            'STransformInverseMats':'mat_st_inv_walk',
            'Session':'sessions',
            'Subject':'subjects'
            }


def builderEngine(which='all'):
    funcs = []
    for category,builder in buildAssociations.items():
        if which == 'all' or category in which:
            if type( builder ) == list:
                if builder[0] == 'multiple':
                    builder_list = builder[1:]
                elif builder[0] == 'shared':
                    builder_list = []
            else: builder_list = [builder]
            for bf_name in builder_list:
                build_func = eval( bf_name )

                funcs.append(build_func)
    return funcs


zork_path = '/processed_data/zork/zork-phase4-72/'

# utility functions

def calc_followupcol(row):
    ''' return the Phase 4 followup # '''
    if row['Phase4-session'] is np.nan or row['Phase4-session'] not in 'abcd':
        return np.nan
    else:
        return ord(row['session']) - ord(row['Phase4-session'])

def join_ufields(row):
    ''' join ID and session fields in a dataframe '''
    return '_'.join([row['ID'], row['session']])

def get_toc(target_dir, toc_str):
    ''' given dir containing toc files and string to be found in one,
        find the path of the most recently modified one matching the string '''
    pd_tocfiles = [f for f in glob(target_dir+'*.toc') if toc_str in f]
    pd_tocfiles.sort(key=os.path.getmtime)
    latest = pd_tocfiles[-1]
    return latest

def txt_tolines(path):
    ''' given path to text file, return its lines as list '''
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    return lines
    
def find_lines(lines, start, end):
    ''' find lines that match start and end exressions '''
    tmp_lines = [l for l in lines if l[:len(start)] == start \
                                 and l[-len(end):] == end]
    return tmp_lines

def verify_files(files):
    ''' given a list of paths, return the ones that exist '''
    existent_files = []
    for f in files:
        if os.path.isfile(f):
            existent_files.append(f)
        else:
            print(f + ' does not exist')
    return existent_files

def get_dates(files):
    ''' given a list of paths, return a matching list of modified times '''
    return [os.path.getmtime(f) for f in files]

# build functions
#   each builds a collection, usually named after the function

def subjects():
    # fast
    master, master_mtime = load_master()
    for rec in tqdm(master.to_dict(orient='records')):
        so = Subject(rec)
        so.storeNaTsafe()
    sourceO = SourceInfo(Subject.collection, (master_path, master_mtime))
    sourceO.store()
    Mdb[Subject.collection].create_index([('ID', pymongo.ASCENDING)])

def sessions():
    # fast
    master, master_mtime = load_master()
    for char in 'abcdefghijk':
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
                so = Session(rec)
                so.storeNaTsafe()
    sourceO = SourceInfo(Session.collection, (master_path, master_mtime))
    sourceO.store()
    Mdb[Session.collection].create_index([('ID', pymongo.ASCENDING)])

def followups():

    fup_dfs = preparefupdfs_forbuild()
    for fup, df in fup_dfs.items():
        print(fup)
        for rec in tqdm(df.reset_index().to_dict(orient='records')):
            fupO = FollowUp(rec)
            fupO.storeNaTsafe()

    Mdb[FollowUp.collection].create_index([('ID', pymongo.ASCENDING)])

def add_sessions_info():    
    pass

def erp_peaks():
    # 3 minutes
    # or 3 hours? depending on network traffic
    mt_files, datemods = identify_files('/processed_data/mt-files/', '*.mt')
    add_dirs = ['ant_phase4__peaks_2014', 'ant_phase4_peaks_2015',
                'ant_phase4_peaks_2016',
                'vp3_phase4__peaks_2015','vp3_phase4__peaks_2016',
                'aod_phase4__peaks_2015','aod_phase4__peaks_2016',
                'cpt_h1_peaks_may_2016',
                # 'non_coga_vp3',
                # 'aod_bis_18-25controls',
                # 'nki_ppick', 'phase4_redo',
                ]
    for subdir in add_dirs:
        mt_files2, datemods2 = identify_files(
            '/active_projects/HBNL/'+subdir+'/', '*.mt')
        mt_files.extend(mt_files2)
        datemods.extend(datemods2)
    bad_files = ['/processed_data/mt-files/ant/uconn/mc/an1a0072007.df.mt',
                 ]
    for fp in tqdm(mt_files):
        if '/waves/' in fp or fp in bad_files:
            continue
        mtO = MT_File(fp)
        mtO.parse_fileDB()
        erpO = ERPPeak(mtO.data)
        erpO.store()

def neuropsych_xmls():
    # 10 minutes
    xml_files, datemods = identify_files('/raw_data/neuropsych/', '*.xml')
    for fp in tqdm(xml_files):
        xmlO = Neuropsych_XML(fp)
        xmlO.assure_quality()
        # xmlO.data['date'] = xmlO.data['testdate']
        nsO = Neuropsych(xmlO.data)
        nsO.store()


def questionnaires_ph123():
    kmap = map_ph123
    path = '/processed_data/zork/zork-phase123/session/'
    followups = ['p2', 'p3']
    for qname in kmap.keys():
        print(qname)
        import_questfolder_ph123(qname, kmap, path)
        for fup in followups:
            match_fups_sessions_generic(Questionnaire.collection, fup, qname)

def questionnaires_ph123_ssaga():
    kmap = map_ph123_ssaga
    path = '/processed_data/zork/zork-phase123/session/'
    followups = ['p1', 'p2', 'p3']
    for qname in kmap.keys():
        print(qname)
        import_questfolder_ssaga_ph123(qname, kmap, path)
        for fup in followups:
            match_fups_sessions_generic(SSAGA.collection, fup, qname)
            match_fups_sessions_generic(SSAGA.collection, fup, 'dx_' + qname)

def questionnaires_ph4():
    # takes  ~20 seconds per questionnaire
    # phase 4 non-SSAGA
    kmap = map_ph4.copy()
    del kmap['cal']
    path = zork_path + 'session/'
    followups = list(range(7))
    for qname in kmap.keys():
        print(qname)
        import_questfolder_ph4(qname, kmap, path)
        for fup in followups:
            match_fups_sessions_generic(Questionnaire.collection, fup, qname)

def questionnaires_ph4_ssaga():
    ''' import all session-based questionnaire info related to SSAGA '''
    # SSAGA
    kmap = map_ph4_ssaga.copy()
    path = zork_path + 'session/'
    followups = list(range(7))
    for qname in kmap.keys():
        print(qname)
        import_questfolder_ssaga_ph4(qname, kmap, path)
        for fup in followups:
            match_fups_sessions_generic(SSAGA.collection, fup, qname)
            match_fups_sessions_generic(SSAGA.collection, fup, 'dx_' + qname)


def ssaga_all():
    ph123_path = '/processed_data/zork/zork-phase123/session/'
    ph4_path = zork_path + 'session/'
    match_followups = ['p2', 'p3', 0, 1, 2, 3, 4, 5, 6]
    for qname in map_ph123_ssaga.keys():
        print(qname)
        import_questfolder_ssaga_ph123(qname, map_ph123_ssaga, ph123_path)
    for qname in map_ph4_ssaga.keys():
        print(qname)
        import_questfolder_ssaga_ph4(qname, map_ph4_ssaga, ph4_path)
    match_fups2sessions_fast_multi(SSAGA.collection, followups=match_followups)

def core():
    # fast
    folder = zork_path + 'subject/core/'
    csv_files = glob(folder+'*.csv')
    if len(csv_files) != 1:
        print(len(csv_files), 'csvs found, aborting')
    path = csv_files[0]
    datemod = datetime.fromtimestamp(os.path.getmtime(path))
    df = df_fromcsv(path)
    for drec in tqdm(df.to_dict(orient='records')):
        ro = Core(drec)
        ro.store()
    sourceO = SourceInfo(Core.collection, (path, datemod))
    sourceO.store()

def internalizing():
    # fast
    folder = '/processed_data/zork/zork-phase4-69/subject/internalizing/'
    file = 'INT_Scale_JK_Scores_n11271.csv'
    path = folder + file
    datemod = datetime.fromtimestamp(os.path.getmtime(path))
    df = df_fromcsv(path, 'IND_ID')
    for drec in tqdm(df.to_dict(orient='records')):
        ro = Internalizing(drec)
        ro.store()
    sourceO = SourceInfo(Internalizing.collection, (path, datemod))
    sourceO.store()

def externalizing():
    # fast
    folder = '/processed_data/zork/zork-phase4-69/subject/vcuext/'
    file = 'vcu_ext_all_121112.sas7bdat.csv'
    path = folder + file
    datemod = datetime.fromtimestamp(os.path.getmtime(path))
    df = df_fromcsv(path, 'IND_ID')
    for drec in tqdm(df.to_dict(orient='records')):
        ro = Externalizing(drec)
        ro.store()
    sourceO = SourceInfo(Externalizing.collection, (path, datemod))
    sourceO.store()

def allrels():

    folder = zork_path + 'subject/rels/'
    file = 'allrels_30nov2016.sas7bdat.csv'
    path = folder + file
    datemod = datetime.fromtimestamp(os.path.getmtime(path))
    
    import_convcols = ['IND_ID', 'FAM_ID', 'F_ID', 'M_ID']
    import_convdict = {col:ID_nan_strintfloat_COGA for col in import_convcols}
    rename_dict = {'FAM_ID': 'famID', 'IND_ID': 'ID', 'TWIN': 'twin', 'SEX': 'sex'}

    rel_df = pd.read_csv(path, converters=import_convdict, low_memory=False)
    rel_df = rel_df.rename(columns=rename_dict)
    rel_df['fID'] = rel_df[['famID', 'F_ID']].apply(build_parentID, axis=1, args=['famID', 'F_ID'])
    rel_df['mID'] = rel_df[['famID', 'M_ID']].apply(build_parentID, axis=1, args=['famID', 'M_ID'])

    for drec in tqdm(rel_df.to_dict(orient='records')):
        ro = AllRels(drec)
        ro.store()
    sourceO = SourceInfo(AllRels.collection, (path, datemod))
    sourceO.store()

def fham():
    # fast
    folder = zork_path + 'subject/fham/'
    file = 'bigfham4.sas7bdat.csv'
    path = folder + file
    datemod = datetime.fromtimestamp(os.path.getmtime(path))
    df = df_fromcsv(path, 'IND_ID')
    for drec in tqdm(df.to_dict(orient='records')):
        ro = FHAM(drec)
        ro.store()
    sourceO = SourceInfo(FHAM.collection, (path, datemod))
    sourceO.store()

def raw_eegdata():
    # ~2 mins?

    # rds
    rd_start_dir = '/raw_data/masscomp/'
    rd_glob_expr = '*rd'
    rd_files, rd_datemods = identify_files(rd_start_dir, rd_glob_expr)
    for rd_path in tqdm(rd_files):
        try:
            if 'bad' in rd_path:
                continue
            info = parse_rd_path(rd_path)
            raweegO = RawEEGData(info)
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
                raweegO = RawEEGData(info)
                raweegO.store()
        except:
            print('problem with', cnt_path)

    # raw_files = rd_files + cnt_files
    # raw_datemods = rd_datemods + cnt_datemods
    # sourceO = SourceInfo(RawEEGData.collection,
    #                 list(zip(raw_files, raw_datemods)))
    # sourceO.store()
    # too large!

def eeg_data():
    # ~2 mins?
    start_dir = '/processed_data/cnt-h1-files/'
    glob_expr = '*cnt.h1'
    cnth1_files, datemods = identify_files(start_dir, glob_expr)
    for f in tqdm(cnth1_files):
        data = parse_cnth1_path(f)
        if data['n_chans'] not in ['21', '32', '64']:
            print(f, 'had unexpected number of chans')
            continue
        eegO = EEGData(data)
        eegO.store()
    sourceO = SourceInfo(EEGData.collection,
                    list(zip(cnth1_files, datemods)))
    sourceO.store()

def erp_data():
    # ~2 mins?
    start_dir = '/processed_data/avg-h1-files/'
    glob_expr = '*avg.h1'
    avgh1_files, datemods = identify_files(start_dir, glob_expr)
    for f in tqdm(avgh1_files):
        fO = CNTH1_File(f)  # basically identical at this point
        fO.parse_fileDB()
        erpO = ERPData(fO.data)
        erpO.store()
    sourceO = SourceInfo(ERPData.collection,
                    list(zip(avgh1_files, datemods)))
    sourceO.store()

def resting_power():
    # fast
    start_dir = '/processed_data/eeg/complete_result_09_16.d/results/'
    ns_file = 'ns_all_tests.dat'
    mc_fileA = 'mc_1st_test.dat'
    mc_fileB = 'mc_2nd_test.dat'

    nsO = RestingDAT(start_dir + ns_file)
    nsO.ns_to_dataframe()
    rec_lst = nsO.file_df.to_dict(orient='records')

    mcO_A = RestingDAT(start_dir + mc_fileA)
    mcO_A.mc_to_dataframe(session='a')
    rec_lst.extend(mcO_A.file_df.to_dict(orient='records'))
    
    mcO_B = RestingDAT(start_dir + mc_fileB)
    mcO_B.mc_to_dataframe(session='b')
    rec_lst.extend(mcO_B.file_df.to_dict(orient='records'))

    for rec in tqdm(rec_lst):
        rpO = RestingPower(rec)
        rpO.store()

    # sourceO = SourceInfo(RestingPower.collection, [])

def mat_st_inv_walk(check_update=False, mat_files=None):
    # can take a while depending on network traffic
    if mat_files is None:
        start_base = '/processed_data/mat-files-v'
        start_fins = ['40','60']
        glob_expr = '*st.mat'
        mat_files = []
        dates = []
        for fin in start_fins:
            f_mats, f_dates = identify_files(start_base+fin,glob_expr)
            mat_files.extend(f_mats)
            dates.extend(f_dates)
    for f in tqdm(mat_files):
        infoD = parse_STinv_path(f)
        infoD['path'] = f
        matO = STransformInverseMats(infoD)

        store = False
        if check_update:
            matO.compare(field='path')
            if matO.new:
                store = True
        else:
            store = True

        if store:
            matO.store()
    # Mdb[STransformInverseMats.collection].create_index([('id', pymongo.ASCENDING)])

def eeg_behavior(files_dms=None):
    ''' unlike others, this build does an "update".
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
            erpbeh_obj_ck = EEGBehavior(fO.data)
            erpbeh_obj_ck.compare()  # populates s.new with bool

            if erpbeh_obj_ck.new:  # "brand new", get general info
                fO.parse_behav_forDB(general_info=True)
                erpbeh_obj = EEGBehavior(fO.data)
                erpbeh_obj.store()
            else:  # if not brand new, check if the experiment is already in the doc
                try:
                    erpbeh_obj_ck.doc[fO.file_info['experiment']]
                except KeyError:  # only update experiment info if not already in db
                    fO = AVGH1_File(f)  # refresh the file obj
                    fO.parse_behav_forDB()
                    erpbeh_obj = EEGBehavior(fO.data)
                    erpbeh_obj.compare()  # to get update query
                    erpbeh_obj.update()
        except:
            print(f, 'failed')
    # sourceO = SourceInfo('EEGbehavior', list(zip(avgh1_files, datemods)))
    # sourceO.store()
    inds = Mdb[EEGBehavior.collection].list_indexes()
    try:
        next(inds) # returns the _id index
        next(inds) # check if any other index exists
        Mdb[EEGBehavior.collection].reindex() # if it does, just reindex
    except StopIteration: # otherwise, create it
        Mdb[EEGBehavior.collection].create_index([('uID', pymongo.ASCENDING)])
        
# not recommended / graveyard below

def mat_st_inv_toc():
    # can take a while depending on network traffic
    toc_dir = '/archive/backup/toc.d/'
    toc_str = 'processed_data'
    latest = get_toc(toc_dir, toc_str)

    lines = txt_tolines(latest)

    start = './mat-files-v'
    end = 'st.mat'
    tmp_lines = find_lines(lines, start, end)

    new_prefix = '/processed_data'
    files = [new_prefix + l[1:] for l in tmp_lines]
    mat_files = verify_files(files)
    # dates = get_dates(files)
    for f in tqdm(mat_files):
        infoD = parse_STinv_path(f)
        infoD['path'] = f
        infoD['prc_ver'] = f.split(os.path.sep)[2][-2]
        matO = STransformInverseMats(infoD)
        matO.store()

def neuropsych_TOLT():
    # 30 seconds
    tolt_files, datemods = identify_files('/raw_data/neuropsych/',
                                             '*TOLT*sum.txt')
    for fp in tqdm(tolt_files):
        toltO = TOLT_Summary_File(fp)
        nsO = Neuropsych('TOLT', toltO.data)
        nsO.store()
    sourceO = SourceInfo(Neuropsych.collection, list(
        zip(tolt_files, datemods)), 'TOLT')
    sourceO.store()

def neuropsych_CBST():
    # 30 seconds
    cbst_files, datemods = identify_files('/raw_data/neuropsych/',
                                             '*CBST*sum.txt')
    for fp in tqdm(cbst_files):
        cbstO = CBST_Summary_File(fp)
        nsO = Neuropsych('CBST', cbstO.data)
        nsO.store()
    sourceO = SourceInfo(Neuropsych.collection, list(
        zip(cbst_files, datemods)), 'CBST')
    sourceO.store()

def ero_pheno_join_bulk(csvs, start_ind=0):
    ''' build collection of ERO results from CSVs by joining all CSVs in one
        terminal subdirectory together, then bulk_writing their rows '''
    def split_field(s, ind, delim='_'):
        return s.split(delim)[ind]

    fp_dict = OrderedDict()
    for fp in csvs:
        subdir, file = os.path.split(fp)
        if subdir not in fp_dict.keys():
            fp_dict.update({subdir: []})
        fp_dict[subdir].append(file)

    try:
        for subdir, file_list in list(fp_dict.items())[start_ind:]:
            joinDF = pd.DataFrame()
            for filename in file_list:
                fpath = os.path.join(subdir, filename)
                csvfileO = ERO_CSV(fpath)
                file_info = csvfileO.data_for_file()  #filename parsing to here
                
                # '''
                if (file_info['site']=='washu' or \
                    file_info['site']=='suny') and \
                    file_info['experiment']=='vp3' and \
                    'threshold electrodes' in csvfileO.parameters and \
                    csvfileO.parameters['threshold electrodes']==9:
                    print(',', end='')
                    continue
                # '''

                eroFileQ = Mdb['EROcsv'].find({'filepath': fpath}, {'_id': 1})
                if eroFileQ.count() >= 1:
                    # print('Repeat for ' + fpath)
                    continue
                else:
                    csvorgO = EROcsv(fpath, file_info)
                    csvorgO.store_track()

                csvfileO.data_forjoin()  # here CSV actually gets read
                if csvfileO.data.empty:
                    print(fpath, 'was empty')
                    continue
                if joinDF.empty:
                    joinDF = csvfileO.data
                else:
                    # check if the columns already exist in the joinDF
                    new_cols = csvfileO.data.columns.difference(joinDF.columns)
                    if len(new_cols) > 0:
                        joinDF = joinDF.join(csvfileO.data, how='outer')
                # del csvfileO
                csvfileO = None

            if joinDF.empty:
                print('x', end='')
                continue
            joinDF.reset_index(inplace=True)
            joinDF['ID'] = joinDF['uID'].apply(split_field, args=[0])
            joinDF['session'] = joinDF['uID'].apply(split_field, args=[1])
            joinDF['experiment'] = joinDF['uID'].apply(split_field, args=[2])

            orgO = EROcsvresults(joinDF.to_dict(orient='records'), subdir)
            # del joinDF
            joinDF = None
            orgO.store_joined_bulk()
            # del orgO
            orgO = None
            print('.', end='')
    except:
        print(subdir)
        print(filename)
        raise
