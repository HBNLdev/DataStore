''' build collections '''

import os
from glob import glob
from datetime import datetime
from collections import OrderedDict
import numpy as np
import pandas as pd
from tqdm import tqdm

import master_info as mi
import quest_import as qi
import organization as O
import file_handling as FH

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
    master_mtime = mi.load_master()
    for rec in tqdm(mi.master.to_dict(orient='records')):
        so = O.Subject(rec)
        so.storeNaTsafe()
    sourceO = O.SourceInfo(O.Subject.collection, (mi.master_path, master_mtime))
    sourceO.store()


def sessions():
    # fast
    master_mtime = mi.load_master()
    for char in 'abcdefghijk':
        sessionDF = mi.master[mi.master[char + '-run'].notnull()]
        if sessionDF.empty:
            continue
        else:
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
                so = O.Session(rec)
                so.storeNaTsafe()
    sourceO = O.SourceInfo(O.Session.collection, (mi.master_path, master_mtime))
    sourceO.store()


def erp_peaks():
    # 3 minutes
    # or 3 hours? depending on network traffic
    mt_files, datemods = FH.identify_files('/processed_data/mt-files/', '*.mt')
    add_dirs = ['ant_phase4__peaks_2014', 'ant_phase4_peaks_2015',
                'ant_phase4_peaks_2016']
    for subdir in add_dirs:
        mt_files2, datemods2 = FH.identify_files(
            '/active_projects/HBNL/'+subdir+'/', '*.mt')
        mt_files.extend(mt_files2)
        datemods.extend(datemods2)
    bad_files = ['/processed_data/mt-files/ant/uconn/mc/an1a0072007.df.mt',
                 ]
    for fp in tqdm(mt_files):
        if fp in bad_files:
            continue
        mtO = FH.mt_file(fp)
        mtO.parse_fileDB()
        erpO = O.ERPPeak(mtO.data)
        erpO.store()
    sourceO = O.SourceInfo(O.ERPPeak.collection, list(zip(mt_files, datemods)))
    sourceO.store()


def neuropsych_xml():
    # 10 minutes
    xml_files, datemods = FH.identify_files('/raw_data/neuropsych/', '*.xml')
    for fp in tqdm(xml_files):
        xmlO = FH.neuropsych_xml(fp)
        nsO = O.Neuropsych('all', xmlO.data)
        nsO.store()
    sourceO = O.SourceInfo(O.Neuropsych.collection,
                            list(zip(xml_files, datemods)), 'all')
    sourceO.store()

def questionnaires_ph4():
    # takes  ~20 seconds per questionnaire
    # phase 4 non-SSAGA
    kmap = qi.map_ph4
    path = '/processed_data/zork/zork-phase4-69/session/'
    for qname in kmap.keys():
        print(qname)
        qi.import_questfolder(qname, kmap, path)
        qi.match_fups2sessions(qname, kmap, path, O.Questionnaire.collection)

def core():
    # fast
    folder = '/processed_data/zork/zork-phase4-69/subject/core/'
    file = 'core_pheno_20160822.sas7bdat.csv'
    path = folder + file
    datemod = datetime.fromtimestamp(os.path.getmtime(path))
    df = qi.df_fromcsv(path)
    for drec in tqdm(df.to_dict(orient='records')):
        ro = O.Core(drec)
        ro.store()
    sourceO = O.SourceInfo(O.Core.collection, (path, datemod))
    sourceO.store()

def internalizing():
    # fast
    folder = '/processed_data/zork/zork-phase4-69/subject/internalizing/'
    file = 'INT_Scale_JK_Scores_n11271.csv'
    path = folder + file
    datemod = datetime.fromtimestamp(os.path.getmtime(path))
    df = qi.df_fromcsv(path, 'IND_ID')
    for drec in tqdm(df.to_dict(orient='records')):
        ro = O.Internalizing(drec)
        ro.store()
    sourceO = O.SourceInfo(O.Internalizing.collection, (path, datemod))
    sourceO.store()

def externalizing():
    # fast
    folder = '/processed_data/zork/zork-phase4-69/subject/vcuext/'
    file = 'vcu_ext_all_121112.sas7bdat.csv'
    path = folder + file
    datemod = datetime.fromtimestamp(os.path.getmtime(path))
    df = qi.df_fromcsv(path, 'IND_ID')
    for drec in tqdm(df.to_dict(orient='records')):
        ro = O.Externalizing(drec)
        ro.store()
    sourceO = O.SourceInfo(O.Externalizing.collection, (path, datemod))
    sourceO.store()

def fham():
    # fast
    folder = '/processed_data/zork/zork-phase4-69/subject/fham4/'
    file = 'bigfham4.sas7bdat.csv'
    path = folder + file
    datemod = datetime.fromtimestamp(os.path.getmtime(path))
    df = qi.df_fromcsv(path, 'IND_ID')
    for drec in tqdm(df.to_dict(orient='records')):
        ro = O.FHAM(drec)
        ro.store()
    sourceO = O.SourceInfo(O.FHAM.collection, (path, datemod))
    sourceO.store()

def eeg_data():
    # ~2 mins?
    start_dir = '/processed_data/cnt-h1-files/'
    glob_expr = '*cnt.h1'
    cnth1_files, datemods = FH.identify_files(start_dir, glob_expr)
    for f in tqdm(cnth1_files):
        fO = FH.cnth1_file(f)
        fO.parse_fileDB()
        eegO = O.EEGData(fO.data)
        eegO.store()
    sourceO = O.SourceInfo(O.EEGData.collection,
                    list(zip(cnth1_files, datemods)))
    sourceO.store()

def erp_data():
    # ~2 mins?
    start_dir = '/processed_data/avg-h1-files/'
    glob_expr = '*avg.h1'
    avgh1_files, datemods = FH.identify_files(start_dir, glob_expr)
    for f in tqdm(avgh1_files):
        fO = FH.cnth1_file(f)  # basically identical at this point
        fO.parse_fileDB()
        eegO = O.ERPData(fO.data)
        eegO.store()
    sourceO = O.SourceInfo(O.ERPData.collection,
                    list(zip(avgh1_files, datemods)))
    sourceO.store()

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
        infoD = FH.parse_mt_name(f)
        infoD['path'] = f
        infoD['prc_ver'] = f.split(os.path.sep)[2][-2]
        matO = O.STransformInverseMats(infoD)
        matO.store()

def mat_st_inv_walk(check_update=False, mat_files=None):
    # can take a while depending on network traffic
    if mat_files is None:
        start_base = '/processed_data/mat-files-v'
        start_fins = ['40','60']
        glob_expr = '*st.mat'
        mat_files = []
        dates = []
        for fin in start_fins:
            f_mats, f_dates = FH.identify_files(start_base+fin,glob_expr)
            mat_files.extend(f_mats)
            dates.extend(f_dates)
    for f in tqdm(mat_files):
        infoD = FH.parse_mt_name(f)
        infoD['path'] = f
        infoD['prc_ver'] = f.split(os.path.sep)[2][-2]
        matO = O.STransformInverseMats(infoD)

        store = False
        if check_update:
            matO.compare(field='path')
            if matO.new:
                store = True
        else:
            store = True

        if store:
            matO.store()

def eeg_behavior(files_dms=None):
    # ~8 hours total to parse all *.avg.h1's for behavior
    # files_dms = pickle.load( open(
    #    '/active_projects/mike/pickles/avgh1s_dates.p', 'rb')  )
    if not files_dms:
        start_dir = '/processed_data/avg-h1-files/'
        glob_expr = '*avg.h1'
        avgh1_files, datemods = FH.identify_files(start_dir, glob_expr)
    else:
        avgh1_files, datemods = zip(*files_dms)

    for f in tqdm(avgh1_files):
        try:
            fO = FH.avgh1_file(f)
            if fO.file_info['experiment'] == 'err':
                continue # have corrupted trial info and will overwrite ern
            fO.parse_behav_forDB()
            erpbeh_obj = O.EEGBehavior(fO.data)
            erpbeh_obj.compare()
            if erpbeh_obj.new:
                erpbeh_obj.store()
            else:
                erpbeh_obj.update()
        except:
            print(f, 'failed')
    sourceO = O.SourceInfo('EEGbehavior', list(zip(avgh1_files, datemods)))
    sourceO.store()

# not recommended / graveyard below

def questionnaires_ssaga():
    ''' import all session-based questionnaire info related to SSAGA
        i recommend not using this unless absolutely necessary because
        most of this info is in the core phenotype file '''
    # SSAGA
    path = '/processed_data/zork/zork-phase4-69/session/'
    kmap = qi.map_ph4_ssaga
    for qname in kmap.keys():
        print(qname)
        qi.import_questfolder_ssaga(qname, kmap, path)
        qi.match_fups2sessions(qname, kmap, path, O.SSAGA.collection)

def questionnaires_ph123():
    ''' import all session-based questionnaire info from phase 4
        unused because difficult but should fix it later '''
    # takes  ~20 seconds per questionnaire
    # phase 1, 2, and 3 non-SSAGA
    kmap = qi.map_ph123
    path = '/processed_data/zork/zork-phase123/session/'
    for qname in kmap.keys():
        print(qname)
        qi.import_questfolder(qname, kmap, path)
        qi.match_fups2sessions(qname, kmap, path, O.Questionnaire.collection)

def neuropsych_TOLT():
    # 30 seconds
    tolt_files, datemods = FH.identify_files('/raw_data/neuropsych/',
                                             '*TOLT*sum.txt')
    for fp in tqdm(tolt_files):
        toltO = FH.tolt_summary_file(fp)
        nsO = O.Neuropsych('TOLT', toltO.data)
        nsO.store()
    sourceO = O.SourceInfo(O.Neuropsych.collection, list(
        zip(tolt_files, datemods)), 'TOLT')
    sourceO.store()


def neuropsych_CBST():
    # 30 seconds
    cbst_files, datemods = FH.identify_files('/raw_data/neuropsych/',
                                             '*CBST*sum.txt')
    for fp in tqdm(cbst_files):
        cbstO = FH.cbst_summary_file(fp)
        nsO = O.Neuropsych('CBST', cbstO.data)
        nsO.store()
    sourceO = O.SourceInfo(O.Neuropsych.collection, list(
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
                csvfileO = FH.ERO_csv(fpath)
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

                eroFileQ = O.Mdb['EROcsv'].find({'filepath': fpath}, {'_id': 1})
                if eroFileQ.count() >= 1:
                    # print('Repeat for ' + fpath)
                    continue
                else:
                    csvorgO = O.EROcsv(fpath, file_info)
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

            orgO = O.EROcsvresults(joinDF.to_dict(orient='records'), subdir)
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
