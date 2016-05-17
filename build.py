'''build collections'''

import os
from datetime import datetime
from glob import glob

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
        sessionDF = mi.master[mi.master[char+'-run'].notnull()]
        sessionDF['session'] = char
        for col in ['raw', 'date', 'age']:
            sessionDF[col] = sessionDF[char+'-'+col]
        for rec in sessionDF.to_dict(orient='records'):
            so = O.Session(rec)
            so.storeNaTsafe()
    sourceO = O.SourceInfo('sessions', (mi.master_path, master_mtime))
    sourceO.store()

def erp():
    # 3 minutes
    mt_files, datemods = FH.identify_files('/processed_data/mt-files/','*.mt')
    bad_files=['/processed_data/mt-files/ant/uconn/mc/an1a0072007.df.mt',
        ]
    for fp in mt_files:
        if fp in bad_files:
            continue
        mtO = FH.mt_file( fp )
        mtO.parse_fileDB()
        erpO = O.ERPPeak( mtO.data )
        erpO.store()
    sourceO = O.SourceInfo('ERP', list(zip(mt_files, datemods)))
    sourceO.store()

def neuropsych_xml():
    # 10 minutes
    xml_files, datemods = FH.identify_files('/raw_data/neuropsych/','*.xml')
    for fp in xml_files:
        xmlO = FH.neuropsych_xml( fp )
        nsO = O.Neuropsych('all', xmlO.data)
        nsO.store()
    sourceO = O.SourceInfo('neuropsych', list(zip(xml_files,datemods)), 'all')
    sourceO.store()

def neuropsych_TOLT():
    # 30 seconds
    tolt_files, datemods = FH.identify_files('/raw_data/neuropsych/',
                                             '*TOLT*sum.txt')
    for fp in tolt_files:
        toltO = FH.tolt_summary_file( fp )
        nsO = O.Neuropsych('TOLT', toltO.data)
        nsO.store()
    sourceO = O.SourceInfo('neuropsych', list(zip(tolt_files,datemods)), 'TOLT')
    sourceO.store()

def neuropsych_CBST():
    # 30 seconds
    cbst_files, datemods = FH.identify_files('/raw_data/neuropsych/',
                                             '*CBST*sum.txt')
    for fp in cbst_files:
        cbstO = FH.cbst_summary_file( fp )
        nsO = O.Neuropsych('CBST', cbstO.data)
        nsO.store()
    sourceO = O.SourceInfo('neuropsych', list(zip(cbst_files,datemods)), 'CBST')
    sourceO.store()

def eeg_behavior():
    # ~30 seconds per ~1000 subs
    path = '/active_projects/mike/ph4bl_tb'
    mats = glob(path+'/*.mat')
    datemods = []
    for mat in mats:
        datemods.append(datetime.fromtimestamp(os.path.getmtime(mat)))
        erpbeh_obj = FH.erpbeh_mat(mat)
        record_obj = O.EEGBehavior(erpbeh_obj.data)
        record_obj.store()
    sourceO = O.SourceInfo( 'EEGbehavior', list(zip(mats, datemods) ))
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