''' tools for adding questionnaire data to mongo '''

import os
from collections import defaultdict
from datetime import datetime, timedelta
from glob import glob

import numpy as np
import pandas as pd
from sas7bdat import SAS7BDAT
from tqdm import tqdm

from .compilation import buildframe_fromdocs
from .organization import Mdb, Questionnaire, SSAGA
from .utils.dates import parse_date, parse_date_apply_pd
from .utils.df import df_fromcsv

# note we defaultly use this dateformat because pandas sniffs to this format
def_info = {'date_lbl': ['ADM_Y', 'ADM_M', 'ADM_D'],
            'na_val': '',
            'dateform': '%Y-%m-%d',
            'file_ext': '.sas7bdat.csv',
            # 'file_ext': '.sas7bdat',
            'max_fups': 6,
            'id_lbl': 'ind_id',
            'capitalize': False,
            }

# these zork urls may change from distribution to distribution
# change when needed
coga_master_ph123 = '/processed_data/zork/zork-phase123/subject/master/master.sas7bdat.csv'
core_pheno = '/pheno_all/core_pheno_20161129.zip'
fam_url = '/family_data/allfam_sas_3-20-12.zip'
ach_url = '/Phase_IV/Achenbach%20January%202016%20Distribution.zip'
cal_url = '/Phase_IV/CAL%20Forms%20Summer-Fall%202015%20Harvest_sas.zip'

# still missing: cal
# for non-ssaga questionnaires, if there are multiple file_pfixes,
# the files are assumed to be basically non-overlapping in terms of individuals
# (one for adults, and one for adolescents)
# for daily, dependence, and sensation should capitalize on import for phase4
map_ph4 = {
    'achenbach': {'file_pfixes': ['asr4', 'ysr4'],
                  'zip_name': 'Achenbach',
                  'date_lbl': 'datefilled',
                  'drop_keys': ['af_', 'bp_'],
                  'zork_url': ach_url},
    'aeq': {'file_pfixes': ['aeqascore4', 'aeqscore4'],
            'zip_name': 'aeq4',
            'zork_url': '/Phase_IV/aeq4.zip'},
    'bis': {'file_pfixes': ['bis_a_score4', 'bis_score4'],
            'zip_name': 'biq4',
            'zork_url': '/Phase_IV/biq4.zip'},
    'cal': {'file_pfixes': 'scored',
            'zip_name': 'CAL',
            'zork_url': cal_url},
    'craving': {'file_pfixes': ['crv4'],
                'zip_name': 'crv',
                'zork_url': '/Phase_IV/crv4.zip'},
    'daily': {'file_pfixes': ['daily4'],
              'zip_name': 'daily',
              'zork_url': '/Phase_IV/daily4.zip',
              'capitalize': True},
    'dependence': {'file_pfixes': ['dpndnce4'],
                   'zip_name': 'dpndnce',
                   'zork_url': '/Phase_IV/dpndnce4.zip',
                   'capitalize': True},
    'neo': {'file_pfixes': ['neo4'],
            'zip_name': 'neo',
            'zork_url': '/Phase_IV/neo4.zip'},
    'sensation': {'file_pfixes': ['ssvscore4'],
                  'zip_name': 'ssv',
                  'zork_url': '/Phase_IV/ssvscore4.zip',
                  'capitalize': True},
    'sre': {'file_pfixes': ['sre_score4'],
            'zip_name': 'sre4',
            'zork_url': '/Phase_IV/sre4.zip'},
}

# for ssaga questionnaires, the multiple file_fpixes are perfectly overlapping,
# so we end up joining them
# capitalize all DX on import
map_ph4_ssaga = {
    'cssaga': {'file_pfixes': ['cssaga4', 'dx_cssaga4'],
               'date_lbl': 'IntvDate',
               'id_lbl': 'IND_ID',
               'zip_name': 'cssaga_dx',
               'zork_url': '/Phase_IV/cssaga_dx.zip'},
    'pssaga': {'file_pfixes': ['pssaga4', 'dx_pssaga4'],
               'date_lbl': 'IntvDate',
               'id_lbl': 'ind_id',
               'zip_name': 'cssagap_dx',
               'zork_url': '/Phase_IV/cssagap_dx.zip'},
    'ssaga': {'file_pfixes': ['ssaga4', 'dx_ssaga4'],
              'date_lbl': 'IntvDate',
              'id_lbl': 'IND_ID',
              'zip_name': 'ssaga_dx',
              'zork_url': '/Phase_IV/ssaga_dx.zip'}
}

# for subject-specific info, used by quest_retrieval.py
map_subject = {'core': {'file_pfixes': 'core',
                        'zip_name': 'core',
                        'zork_url': core_pheno},
               'fams': {'file_pfixes': 'allfamilies',
                        'zip_name': 'allfam',
                        'zork_url': fam_url},
               'fham': {'file_pfixes': 'bigfham4',
                        'zip_name': 'bigfham4',
                        'zork_url': '/Phase_IV/bigfham4.zip'},
               'rels': {'file_pfixes': 'all_rels',
                        'zip_name': 'allrels',
                        'zork_url': '/family_data/allrels_sas.zip'},
               'vcuext': {'file_pfixes': ['vcu'],
                          'zip_name': 'vcu',
                          'zork_url': '/vcu_ext_pheno/vcu_ext_all_121112_sas.zip'},
               'master': {'file_pfixes': 'master4',
                          'zip_name': 'master4',
                          'zork_url': '/Phase_IV/master4_sas.zip'}
               }

# note these have variegated date labels!
# for aeq, the score is not available for phases <4
# for sensation, the score is not available for phase 2
map_ph123 = {'aeq': {'file_pfixes': ['aeq', 'aeqa', 'aeq3', 'aeqa3'],
                     'followup': {'aeq': 'p2', 'aeq3': 'p3', 'aeqa': 'p2', 'aeqa3': 'p3'},
                     'date_lbl': {'aeq': 'AEQ_DT', 'aeqa': 'AEQA_DT', 'aeq3': 'AEQ3_DT', 'aeqa3': 'AEQA3_DT'},
                     'id_lbl': 'IND_ID'},
             'craving': {'file_pfixes': ['craving', 'craving3'],
                         'followup': {'craving': 'p2', 'craving3': 'p3', },
                         'date_lbl': {'craving': 'QSCL_DT', 'craving3': 'QSCL3_DT'},
                         'id_lbl': 'IND_ID'},
             'daily': {'file_pfixes': ['daily', 'daily3'],
                       'followup': {'daily': 'p2', 'daily3': 'p3', },
                       'date_lbl': {'daily': 'DAILY_DT', 'daily3': 'DLY3_DT'},
                       'id_lbl': 'IND_ID'},
             'dependence': {'file_pfixes': ['dpndnce', 'dpndnce3'],
                            'followup': {'dpndnce': 'p2', 'dpndnce3': 'p3', },
                            'date_lbl': {'dpndnce': 'QSCL_DT', 'dpndnce3': 'QSCL3_DT'},
                            'id_lbl': 'IND_ID'},
             'neo': {'file_pfixes': ['neo', 'neo3'],
                     'followup': {'neo': 'p2', 'neo3': 'p3', },
                     'date_lbl': {'neo': 'NEO_DT', 'neo3': 'NEO3_DT'},
                     'id_lbl': 'IND_ID'},
             'sensation': {'file_pfixes': ['sssc', 'ssvscore', 'sssc3'],
                           'followup': {'sssc': 'p2', 'sssc3': 'p3', 'ssvscore': 'p3'},
                           'date_lbl': {'sssc': 'SSSC_DT', 'sssc3': 'SSSC3_DT', 'ssvscore': 'ZUCK_DT'},
                           'id_lbl': 'IND_ID'},
             'sre': {'file_pfixes': ['sre', 'sre3'],
                     'followup': {'sre': 'p2', 'sre3': 'p3', },
                     'date_lbl': {'sre': 'SRE_DT', 'sre3': 'SRE3_DT'},
                     'id_lbl': 'IND_ID'},
             }

map_ph123_ssaga = {'cssaga': {'file_pfixes': ['cssaga', 'csaga2', 'csaga3', 'dx_csaga', 'dx_csag2', 'dx_csag3'],
                              'followup': {'cssaga': 'p1', 'csaga2': 'p2', 'csaga3': 'p3',
                                           'dx_csaga': 'p1', 'dx_csag2': 'p2', 'dx_csag3': 'p3'},
                              'date_lbl': {'cssaga': 'CSAGA_COMB_DT', 'csaga2': 'CSAG2_DT', 'csaga3': 'CSAG2_DT',
                                           'dx_csaga': None, 'dx_csag2': None, 'dx_csag3': None},
                              'joindate_from': {'dx_csaga': 'cssaga', 'dx_csag2': 'csaga2', 'dx_csag3': 'csaga3'},
                              'id_lbl': 'IND_ID',
                              'dateform': '%m/%d/%Y', },
                   'pssaga': {'file_pfixes': ['pssaga', 'psaga2', 'psaga3', 'dx_psaga', 'dx_psag2', 'dx_psag3'],
                              'followup': {'pssaga': 'p1', 'psaga2': 'p2', 'psaga3': 'p3',
                                           'dx_psaga': 'p1', 'dx_psag2': 'p2', 'dx_psag3': 'p3'},
                              'date_lbl': {'pssaga': 'CSAGP_DT', 'psaga2': 'CSGP2_DT', 'psaga3': 'CSGP2_DT',
                                           'dx_psaga': None, 'dx_psag2': None, 'dx_psag3': None},
                              'joindate_from': {'dx_psaga': 'pssaga', 'dx_psag2': 'psaga2', 'dx_psag3': 'psaga3'},
                              'id_lbl': 'IND_ID',
                              'dateform': '%m/%d/%Y', },
                   'ssaga': {'file_pfixes': ['ssaga', 'ssaga2', 'ssaga3', 'dx_ssaga', 'dx_saga2rv', 'dx_saga3rv'],
                             'followup': {'ssaga': 'p1', 'ssaga2': 'p2', 'ssaga3': 'p3',
                                          'dx_ssaga': 'p1', 'dx_saga2rv': 'p2', 'dx_saga3rv': 'p3'},
                             'date_lbl': {'ssaga': None, 'ssaga2': None, 'ssaga3': None,
                                          'dx_ssaga': None, 'dx_saga2rv': None, 'dx_saga3rv': None},
                             'joindate_lbl': {'ssaga': 'SSAGA_DT', 'ssaga2': 'SAGA2_DT', 'ssaga3': 'SAGA3_DT',
                                              'dx_ssaga': 'SSAGA_DT', 'dx_saga2rv': 'SAGA2_DT',
                                              'dx_saga3rv': 'SAGA3_DT'},
                             'joindate_from': {'ssaga': None, 'ssaga2': None, 'ssaga3': None,
                                               'dx_ssaga': None, 'dx_saga2rv': None, 'dx_saga3rv': None},
                             'id_lbl': 'IND_ID',
                             'dateform': '%m/%d/%Y', }
                   }

map_ph123_ssaga['ssaga']['joindate_from'] = {k: coga_master_ph123 for k in map_ph123_ssaga['ssaga']['date_lbl'].keys()}


def sasdir_tocsv(target_dir):
    ''' convert a directory filled with *.sas7bdat files to *.csv '''

    sas_files = glob(target_dir + '*.sas7bdat')

    for sf in sas_files:
        sf_contents = SAS7BDAT(sf)
        sf_df = sf_contents.to_data_frame()
        sf_df.to_csv(sf + '.csv', index=False)


def quest_pathfollowup(path, file_pfixes, file_ext, max_fups):
    ''' given path to parent directoy, a list of file prefixes, a file extension, and max number of followups,
        build a dict which maps valid filepaths to the corresponding followup number '''

    fn_dict = {}
    fn_infolder = glob(path + '*' + file_ext)

    for fp in file_pfixes:
        for followup in range(max_fups + 1):
            if followup == 0:
                fstr = fp + file_ext
            else:
                fstr = fp + '_f' + str(followup) + file_ext
            fpathstr = os.path.join(path, fstr)
            if fpathstr in fn_infolder:
                fn_dict.update({fpathstr: followup})

    return fn_dict


def quest_pathfollowup_ssaga(path, file_pfixes, file_ext, max_fups):
    ''' given path to parent directoy, a list of file prefixes, a file extension, and max number of followups,
        build a dict which maps a followup number to a list of corresponding valid filepaths '''

    fn_dict = defaultdict(list)
    fn_infolder = glob(path + '*' + file_ext)

    for fp in file_pfixes:
        for followup in range(max_fups + 1):
            if followup == 0:
                fstr = fp + file_ext
            else:
                fstr = fp + '_f' + str(followup) + file_ext
            fpathstr = os.path.join(path, fstr)
            # print(fpathstr)
            if fpathstr in fn_infolder:
                fn_dict[followup].append(fpathstr)

    return fn_dict


def build_inputdict(name, knowledge_dict):
    ''' for a given questionnaire, build dict from defaults and specifics '''

    idict = def_info.copy()
    idict.update(knowledge_dict[name])
    return idict


def import_questfolder_ph4(qname, kmap, path):
    ''' import all questionnaire data in one folder '''

    # build inputs
    i = build_inputdict(qname, kmap)
    i['path'] = path
    print(i)
    subfolder = qname

    # get dict of filepaths and followup numbers
    file_dict = quest_pathfollowup(i['path'] + subfolder + '/',
                                   i['file_pfixes'], i['file_ext'], i['max_fups'])
    print(file_dict)
    if not file_dict:
        print('There were no files in the path specified.')

    # for each file
    for file, followup_num in file_dict.items():

        # read csv in as dataframe
        tmp_path = os.path.join(i['path'], file)
        print(tmp_path)
        df = df_fromcsv(tmp_path, i['id_lbl'], i['na_val'])

        if df.empty:
            continue

        if i['capitalize']:
            capitalizer = {col: col.upper() for col in df.columns if col not in [i['date_lbl'], i['id_lbl']]}
            df.rename(columns=capitalizer, inplace=True)

        # df = df_fromsas( os.path.join(i['path'], f), i['id_lbl'])

        # if date_lbl is a list, replace columns with one strjoined column
        try:
            if type(i['date_lbl']) == list:
                newdatecol_name = '-'.join(i['date_lbl'])
                new_datecol = pd.Series([''] * df.shape[0], index=df.index)
                hyphen_col = pd.Series(['-'] * df.shape[0], index=df.index)
                new_datecol += df[i['date_lbl'][0]].apply(int).apply(str)
                for e in i['date_lbl'][1:]:
                    new_datecol += hyphen_col
                    new_datecol += df[e].apply(int).apply(str)
                df[newdatecol_name] = new_datecol
            else:
                newdatecol_name = i['date_lbl']
            df['date'] = df[newdatecol_name]
        except KeyError:
            print('expected date columns were not present')
            df['date'] = np.nan

        # attempt to convert columns to dates
        for c in df:
            try:
                df[c] = df[c].apply(parse_date, args=(i['dateform'],))
            except:
                pass

        # convert to records and store in mongo coll, noting followup_num
        for drec in df.to_dict(orient='records'):
            ro = Questionnaire(qname, followup_num, info=drec)
            ro.storeNaTsafe()


def import_questfolder_ssaga_ph4(qname, kmap, path):
    ''' import all questionnaire data in one folder,
        joining multiple files of the same type '''

    # build inputs
    i = build_inputdict(qname, kmap)
    i['path'] = path
    print(i)

    # get dict of filepaths and followup numbers
    file_dict = quest_pathfollowup_ssaga(i['path'] + qname + '/',
                                         i['file_pfixes'], i['file_ext'], i['max_fups'])
    print(file_dict)
    if not file_dict:
        print('There were no files in the path specified.')

    for followup_num, files in file_dict.items():

        join_df = pd.DataFrame()

        for f in files:
            fname = os.path.split(f)[1]
            print(f)
            # read csv in as dataframe
            df = df_fromcsv(os.path.join(i['path'], f),
                            i['id_lbl'], i['na_val'])
            # df = df_fromsas( os.path.join(i['path'], f), i['id_lbl'])

            if fname[:3] == 'dx_':  # capitalize DX column names (for backward compatibility with phase <4)
                capitalizer = {col: col.upper() for col in df.columns if col not in [i['date_lbl'], i['id_lbl']]}
                df.rename(columns=capitalizer, inplace=True)

            # attempt to convert columns to dates
            for c in df:
                try:
                    df[c] = df[c].apply(parse_date, args=(i['dateform'],))
                    # print('date format changed for', c)
                except:
                    pass
                    # print('date format failed for', c)
                    # if c == 'IntvDate':
                    # raise

            # df.set_index('ID', inplace=True)
            df.drop(i['id_lbl'], axis=1, inplace=True)

            if fname[:3] == 'dx_':
                dx_cols = df.columns
            else:
                nondx_cols = df.columns

            if join_df.empty:
                join_df = df
            else:
                join_df = join_df.join(df, rsuffix='extra')

        join_df['date'] = join_df[i['date_lbl']]

        join_df.drop('ID', axis=1, inplace=True)
        join_df.reset_index(inplace=True)
        print(join_df.shape)
        # convert to records and store in mongo coll, noting followup_num,
        # and separately for the full questionnaire and diagnoses

        for drec in join_df[list(nondx_cols) + ['ID', 'date']].to_dict(orient='records'):
            ro = SSAGA(qname, followup_num, info=drec)
            ro.storeNaTsafe()

        for drec in join_df[list(dx_cols) + ['ID', 'date', i['date_lbl']]].to_dict(orient='records'):
            ro = SSAGA('dx_' + qname, followup_num, info=drec)
            ro.storeNaTsafe()


def import_questfolder_ph123(qname, kmap, path):
    ''' import all questionnaire data in one folder,
        joining multiple files of the same type '''

    # build inputs
    i = build_inputdict(qname, kmap)
    i['path'] = path
    print(i)

    # get dict of filepaths and followup numbers

    # build input dict here
    file_list = [i['path'] + qname + '/' + fpx + i['file_ext'] for fpx in i['file_pfixes']]
    print(file_list)
    if not file_list:
        print('There were no files in the path specified.')

    for f in file_list:

        fname = os.path.split(f)[1]
        fpx = fname.split('.')[0]
        print(f)
        # read csv in as dataframe
        df = df_fromcsv(os.path.join(i['path'], f),
                        i['id_lbl'], i['na_val'])

        date_col = i['date_lbl'][fpx]
        if not date_col:
            joiner = i['joindate_from'][fpx]
            if '/' in joiner:
                date_col = i['joindate_lbl'][fpx]
                join_path = joiner
                join_df = df_fromcsv(join_path, 'IND_ID')
            else:
                date_col = kmap[qname]['date_lbl'][joiner]
                join_path = i['path'] + qname + '/' + joiner + i['file_ext']
                join_df = df_fromcsv(join_path, 'IND_ID')
            df = df.join(join_df[date_col])

        # create normalized date column, converting to datetime
        df['date'] = df[date_col].apply(parse_date_apply_pd, args=(i['dateform'],))

        # df.set_index('ID', inplace=True)
        df.drop(i['id_lbl'], axis=1, inplace=True)

        followup = i['followup'][fpx]
        for drec in df.to_dict(orient='records'):
            ro = Questionnaire(qname, followup, info=drec)
            ro.storeNaTsafe()


harm_info_csv = '/processed_data/zork/harmonization/harmonization-combined-format.csv'


def create_ssagaharm_renamer(ssaga_type, phase):
    if ssaga_type == 'pssaga':
        ssaga_type = 'cssaga'

    if phase == 'p3':
        phase = 'p2'

    from_col = phase + '_' + ssaga_type
    to_col = 'p4_' + ssaga_type

    harm_df = pd.read_csv(harm_info_csv)
    harm_df_na = harm_df[[from_col, to_col]].dropna(how='any')

    renamer = {a: b for a, b in zip(harm_df_na[from_col].values, harm_df_na[to_col].values) if a != b}

    return renamer


def import_questfolder_ssaga_ph123(qname, kmap, path):
    ''' import all questionnaire data in one folder,
        joining multiple files of the same type '''

    # build inputs
    i = build_inputdict(qname, kmap)
    i['path'] = path
    print(i)

    # get dict of filepaths and followup numbers

    # build input dict here
    file_list = [i['path'] + qname + '/' + fpx + i['file_ext'] for fpx in i['file_pfixes']]
    print(file_list)
    if not file_list:
        print('There were no files in the path specified.')

    for f in file_list:

        fname = os.path.split(f)[1]
        fpx = fname.split('.')[0]
        followup = i['followup'][fpx]

        print(f)
        # read csv in as dataframe
        df = df_fromcsv(os.path.join(i['path'], f),
                        i['id_lbl'], i['na_val'])
        date_col = i['date_lbl'][fpx]

        if fname[:3] == 'dx_':  # capitalize DX column names (for backward compatibility with phase <4)
            capitalizer = {col: col.upper() for col in df.columns if col not in [date_col, i['id_lbl']]}
            df.rename(columns=capitalizer, inplace=True)

        if not date_col:
            joiner = i['joindate_from'][fpx]
            if '/' in joiner:
                date_col = i['joindate_lbl'][fpx]
                join_path = joiner
                join_df = df_fromcsv(join_path, 'IND_ID')
            else:
                date_col = kmap[qname]['date_lbl'][joiner]
                join_path = i['path'] + qname + '/' + joiner + i['file_ext']
                join_df = df_fromcsv(join_path, 'IND_ID')
            df = df.join(join_df[date_col])

        # create normalized date column, converting to datetime
        df['date'] = df[date_col].apply(parse_date_apply_pd, args=(i['dateform'],))

        # df.set_index('ID', inplace=True)
        df.drop(i['id_lbl'], axis=1, inplace=True)

        if fname[:3] == 'dx_':
            for drec in df.to_dict(orient='records'):
                ro = SSAGA('dx_' + qname, followup, info=drec)
                ro.storeNaTsafe()
        else:

            # convert to harmonize SSAGA item names
            harm_renamer = create_ssagaharm_renamer(qname, followup)
            df = df.rename(columns=harm_renamer)

            for drec in df.to_dict(orient='records'):
                ro = SSAGA(qname, followup, info=drec)
                ro.storeNaTsafe()


def match_fups2sessions(qname, knowledge_dict, path, q_collection):
    ''' for each record in a questionnaire's subcollection,
        find the nearest session '''

    # build inputs
    i = build_inputdict(qname, knowledge_dict)
    i['path'] = path
    # determine matching session letters (across followups)
    if type(i['date_lbl']) == list:
        i['date_lbl'] = '-'.join(i['date_lbl'])
    session_datecols = [letter + '-date' for letter in 'abcdefgh']
    s = Mdb['subjects']
    q = Mdb[q_collection]
    if 'ssaga' in qname:
        qc = q.find({'questname': {'$in': [qname, 'dx_' + qname]}})
    else:
        qc = q.find({'questname': qname})
    print('matching fups to sessions')
    for qrec in tqdm(qc):
        try:
            testdate = qrec[i['date_lbl']]
        except TypeError:  # if the i['date_lbl'] is itself a dict (unhashable)
            testdate = qrec['date']
        except KeyError:
            print('no date for ' + qrec['ID'])
            continue
        # try to access subject record, notify if not possible
        try:
            s_rec = s.find({'ID': qrec['ID']})[0]
        except IndexError:
            print('could not find ' + qrec['ID'] + ' in subjects collection')
            q.update_one({'_id': qrec['_id']}, {'$set': {'session': None}})
            continue
        session_date_diffs = []
        for sdc in session_datecols:
            if type(s_rec[sdc]) == datetime:
                session_date_diffs.append(abs(s_rec[sdc] - testdate))
            else:
                session_date_diffs.append(timedelta(100000))
        if all(np.equal(timedelta(100000), session_date_diffs)):
            best_match = None
        else:
            min_ind = np.argmin(session_date_diffs)
            best_match = session_datecols[min_ind][0]
        q.update_one({'_id': qrec['_id']}, {'$set': {'session': best_match}})


def match_fups2sessions_fast(qname, knowledge_dict, path, q_collection):
    ''' for each record in a questionnaire's subcollection,
        find the nearest session '''

    print('matching fups to sessions')

    # build inputs
    i = build_inputdict(qname, knowledge_dict)
    i['path'] = path
    # determine matching session letters (across followups)
    if type(i['date_lbl']) == list:
        i['date_lbl'] = '-'.join(i['date_lbl'])

    quest_collection = Mdb[q_collection]
    quest_proj = {'_id': 1, 'ID': 1, 'date': 1, 'followup': 1, 'questname': 1, }
    if 'ssaga' in qname:
        quest_docs = quest_collection.find({'questname': {'$in': [qname, 'dx_' + qname]}}, quest_proj)
    else:
        quest_docs = quest_collection.find({'questname': qname}, quest_proj)
    quest_df = buildframe_fromdocs(quest_docs, inds=['ID'])

    if quest_df.empty:
        print('nothing in that collection, skipping')

    quest_df.rename(columns={'date': 'quest_date', '_id': 'quest__id'}, inplace=True)
    IDs = quest_df.index.tolist()
    followups = quest_df['followup'].unique()
    if 'p1' in followups:
        followups.remove('p1')  # don't match phase 1 at this point
    quests = quest_df['questname'].unique()

    session_proj = {'_id': 1, 'ID': 1, 'session': 1, 'date': 1}
    session_docs = Mdb['sessions'].find({'ID': {'$in': IDs}}, session_proj)
    session_df = buildframe_fromdocs(session_docs, inds=['ID'])
    session_df.rename(columns={'date': 'session_date', '_id': 'session__id'}, inplace=True)

    resolve_df = session_df.join(quest_df)
    resolve_df['date_diff'] = (resolve_df['session_date'] - resolve_df['quest_date']).abs()
    resolve_df.set_index('session', append=True, inplace=True)
    # resolve_df['nearest_session'] = np.nan

    IDfupQN_session_map = dict()
    for ID in IDs:
        ID_df = resolve_df.ix[resolve_df.index.get_level_values('ID') == ID, :]
        if ID_df.empty:
            continue
        for fup in followups:
            fup_df = ID_df[ID_df['followup'] == fup]
            if fup_df.empty:
                continue
            for q in quests:
                q_df = fup_df[fup_df['questname'] == q]
                if q_df.empty:
                    continue
                try:
                    best_session = q_df['date_diff'].argmin()[1]
                except TypeError:  # trying to subscript a nan
                    best_session = None
                IDfupQN_session_map[(ID, fup, q)] = best_session

    # update quest collection based on the above
    IDfupQN_idx = pd.MultiIndex.from_tuples(list(IDfupQN_session_map.keys()), names=('ID', 'followup', 'questname'))
    IDfupQN_series = pd.Series(list(IDfupQN_session_map.values()), index=IDfupQN_idx, name='nearest_session')

    quest_df_forupdate = quest_df.set_index(['followup', 'questname'], append=True)
    quest_df_forupdate = quest_df_forupdate.join(IDfupQN_series)

    for ind, qrow in quest_df_forupdate.iterrows():
        quest_collection.update_one({'_id': qrow['quest__id']}, {'$set': {'session': qrow['nearest_session']}})

        # update sessions collection (or an analogue)


def match_fups2sessions_fast_multi(q_collection, followups=None, quests=None):
    ''' for a questionnaire collection, match its followups to sessions '''

    print('matching fups to sessions')

    quest_collection = Mdb[q_collection]
    if not followups:
        followups = quest_collection.distinct('followup')
    if not quests:
        quests = quest_collection.distinct('questname')

    for fup in followups:
        for questname in quests:
            match_fups_sessions_generic(quest_collection, fup, questname)


def match_fups_sessions_generic(q_collection, fup, questname):
    print('matching fups for', fup, questname)

    quest_collection = Mdb[q_collection]
    quest_query = {'followup': fup, 'questname': questname}
    quest_proj = {'_id': 1, 'ID': 1, 'date': 1, 'followup': 1, 'questname': 1, }
    quest_docs = quest_collection.find(quest_query, quest_proj)
    quest_df = buildframe_fromdocs(quest_docs, inds=['ID'])
    if quest_df.empty:
        print('no questionnaire docs of this kind found')
        return
    if 'date' not in quest_df.columns:
        print('no date info available for this questionnaire-fup combination')
        return
    quest_df.rename(columns={'date': 'quest_date', '_id': 'quest__id'}, inplace=True)
    IDs = quest_df.index.tolist()

    session_proj = {'_id': 1, 'ID': 1, 'session': 1, 'date': 1}
    session_docs = Mdb['sessions'].find({'ID': {'$in': IDs}}, session_proj)
    session_df = buildframe_fromdocs(session_docs, inds=['ID'])
    if session_df.empty:
        print('no session docs of this kind found')
        return
    session_df.rename(columns={'date': 'session_date', '_id': 'session__id'}, inplace=True)

    resolve_df = session_df.join(quest_df)
    resolve_df['date_diff'] = (resolve_df['session_date'] - resolve_df['quest_date']).abs()
    resolve_df.set_index('session', append=True, inplace=True)
    # resolve_df['nearest_session'] = np.nan

    ID_session_map = dict()
    for ID in IDs:
        ID_df = resolve_df.ix[resolve_df.index.get_level_values('ID') == ID, :]
        if ID_df.empty:
            continue
        try:
            best_session = ID_df['date_diff'].argmin()[1]
        except TypeError:  # trying to subscript a nan
            best_session = None
        ID_session_map[ID] = best_session

    ID_session_series = pd.Series(ID_session_map, name='nearest_session')
    ID_session_series.index.name = 'ID'

    quest_df_forupdate = quest_df.join(ID_session_series)

    for ind, qrow in tqdm(quest_df_forupdate.iterrows()):
        quest_collection.update_one({'_id': qrow['quest__id']},
                                    {'$set': {'session': qrow['nearest_session']}})
