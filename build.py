'''build collections'''

import os
from datetime import datetime
from glob import glob
import numpy as np

import master_info as mi
import quest_import as qi
import organization as O
import file_handling as FH


def subjects():
    # fast
    master_mtime = mi.load_master()
    for rec in mi.master.to_dict(orient='records'):
        so = O.Subject(rec)
        so.storeNaTsafe()
    sourceO = O.SourceInfo('subjects', (mi.master_path, master_mtime))
    sourceO.store()


def sessions():
    # fast
    master_mtime = mi.load_master()
    for char in 'abcdefghijk':
        sessionDF = mi.master[mi.master[char + '-run'].notnull()]
        if sessionDF.empty:
            continue
        else:
            sessionDF['session'] = char
            sessionDF['followup'] = sessionDF.apply(calc_followupcol, axis=1)
            for col in ['raw', 'date', 'age']:
                sessionDF[col] = sessionDF[char + '-' + col]
            for rec in sessionDF.to_dict(orient='records'):
                so = O.Session(rec)
                so.storeNaTsafe()
    sourceO = O.SourceInfo('sessions', (mi.master_path, master_mtime))
    sourceO.store()


def calc_followupcol(row):
    if row['Phase4-session'] is np.nan or row['Phase4-session'] not in 'abcd':
        return np.nan
    else:
        return ord(row['session']) - ord(row['Phase4-session'])


def erp_peaks():
    # 3 minutes
    mt_files, datemods = FH.identify_files('/processed_data/mt-files/', '*.mt')
    bad_files = ['/processed_data/mt-files/ant/uconn/mc/an1a0072007.df.mt',
                 ]
    for fp in mt_files:
        if fp in bad_files:
            continue
        mtO = FH.mt_file(fp)
        mtO.parse_fileDB()
        erpO = O.ERPPeak(mtO.data)
        erpO.store()
    sourceO = O.SourceInfo('ERPpeaks', list(zip(mt_files, datemods)))
    sourceO.store()


def neuropsych_xml():
    # 10 minutes
    xml_files, datemods = FH.identify_files('/raw_data/neuropsych/', '*.xml')
    for fp in xml_files:
        xmlO = FH.neuropsych_xml(fp)
        nsO = O.Neuropsych('all', xmlO.data)
        nsO.store()
    sourceO = O.SourceInfo('neuropsych', list(zip(xml_files, datemods)), 'all')
    sourceO.store()


def neuropsych_TOLT():
    # 30 seconds
    tolt_files, datemods = FH.identify_files('/raw_data/neuropsych/',
                                             '*TOLT*sum.txt')
    for fp in tolt_files:
        toltO = FH.tolt_summary_file(fp)
        nsO = O.Neuropsych('TOLT', toltO.data)
        nsO.store()
    sourceO = O.SourceInfo('neuropsych', list(
        zip(tolt_files, datemods)), 'TOLT')
    sourceO.store()


def neuropsych_CBST():
    # 30 seconds
    cbst_files, datemods = FH.identify_files('/raw_data/neuropsych/',
                                             '*CBST*sum.txt')
    for fp in cbst_files:
        cbstO = FH.cbst_summary_file(fp)
        nsO = O.Neuropsych('CBST', cbstO.data)
        nsO.store()
    sourceO = O.SourceInfo('neuropsych', list(
        zip(cbst_files, datemods)), 'CBST')
    sourceO.store()


def eeg_behavior():
    # ~30 seconds per ~1000 subs
    path = '/active_projects/mike/ph4bl_tb'
    mats = glob(path + '/*.mat')
    datemods = []
    for mat in mats:
        datemods.append(datetime.fromtimestamp(os.path.getmtime(mat)))
        erpbeh_obj = FH.erpbeh_mat(mat)
        record_obj = O.EEGBehavior(erpbeh_obj.data)
        record_obj.store()
    sourceO = O.SourceInfo('EEGbehavior', list(zip(mats, datemods)))
    sourceO.store()


def core():
    # fast
    folder = '/active_projects/mike/zork-ph4-65-bl/subject/core/'
    file = 'core_pheno_20160411.sas7bdat.csv'
    path = folder + file
    datemod = datetime.fromtimestamp(os.path.getmtime(path))
    df = qi.df_fromcsv(path)
    for drec in df.to_dict(orient='records'):
        ro = O.Core(drec)
        ro.store()
    sourceO = O.SourceInfo('core', (path, datemod))
    sourceO.store()


def questionnaires():
    # takes  ~20 seconds per questionnaire
    for qname in qi.knowledge.keys():
        qi.import_questfolder(qname)
        qi.match_fups2sessions(qname)


def eeg_data():
    # ~2 mins?
    start_dir = '/processed_data/cnt-h1-files/'
    glob_expr = '*cnt.h1'
    cnth1_files, datemods = FH.identify_files(start_dir, glob_expr)
    for f in cnth1_files:
        fO = FH.cnth1_file(f)
        fO.parse_fileDB()
        eegO = O.EEGData(fO.data)
        eegO.store()
    sourceO = O.SourceInfo('EEGdata', list(zip(cnth1_files, datemods)))
    sourceO.store()


def erp_data():
    # ~2 mins?
    start_dir = '/processed_data/avg-h1-files/'
    glob_expr = '*avg.h1'
    avgh1_files, datemods = FH.identify_files(start_dir, glob_expr)
    for f in avgh1_files:
        fO = FH.cnth1_file(f)  # basically identical at this point
        fO.parse_fileDB()
        eegO = O.ERPData(fO.data)
        eegO.store()
    sourceO = O.SourceInfo('ERPdata', list(zip(avgh1_files, datemods)))
    sourceO.store()


def ero_pheno(files_dates=None,debug=False):
    if not files_dates:
        base_dir = '/processed_data'
        start_dirs = ['csv-files-v60']  # ,'csv-files-v40'
        glob_expr = '*.csv'
        skip_dir = 'ERO-results'
        for start_dir in start_dirs:
            path = os.path.join(base_dir, start_dir)
            all_eeg_csvs, datemods = FH.identify_files(path, glob_expr)
            site_eeg_csvs_dates = [fp_d for fp_d in zip(all_eeg_csvs, datemods)
                                   if skip_dir not in fp_d[0]]

    else:
        site_eeg_csvs_dates = files_dates

    if debug:
        D_loop_times = []
        D_sub_ses_times = []

    for fileP, date in site_eeg_csvs_dates:
        if debug:
            D_loop_start = datetime.now()
            D_loop_sub_ses_times = []

        fileO = FH.ERO_csv(fileP)
        file_info = fileO.data_for_file()  # all filename parsing to here

        eroFileQ = O.Mdb['EROcsv'].find({'filepath': fileP})
        if eroFileQ.count() > 1:
            print('Repeat for ' + fileP)
        if eroFileQ.count() > 0:
            fileO_id = eroFileQ.next()['_id']
        else:
            eroFileO = O.EROcsv(fileP, file_info)
            insert_info = eroFileO.store_track()
            fileO_id = insert_info.inserted_id

        for sub_ses in fileO.data_by_sub_ses():
            if debug:
                D_sub_ses_start = datetime.now()
            eroPhenoO = O.EROpheno(sub_ses, fileO_id)
            eroPhenoO.store()
            if debug:
                D_sub_ses_t = datetime.now() - D_sub_ses_start
                D_loop_sub_ses_times.append(D_sub_ses_t)
        if debug:
            D_sub_ses_times.append(D_loop_sub_ses_times)
            D_loop_times.append( datetime.now() - D_loop_start )


        print('.', end='')

    if debug:
        return {'loops':D_loop_times,
                'sub_ses':D_sub_ses_times}

    # return site_eeg_csvs_dates
