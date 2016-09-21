# formalize process by which a questionnaire gets added to a collection
# and all things that happen to it after that

import os
from datetime import datetime, timedelta
from glob import glob
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sas7bdat import SAS7BDAT

from .organization import Mdb, Questionnaire, SSAGA, SourceInfo

quest_sparser_sub = {'achenbach': ['af_', 'bp_']}

# note we defaultly use this dateformat because pandas sniffs to this format

def_info = {'date_lbl': ['ADM_Y', 'ADM_M', 'ADM_D'],
            'na_val': '',
            'dateform': '%Y-%m-%d',
            'file_ext': '.sas7bdat.csv',
            # 'file_ext': '.sas7bdat',
            'max_fups': 5,
            'id_lbl': 'ind_id',
            }

# still missing: cal
# for non-ssaga questionnaires, if there are multiple file_pfixes,
# the files are assumed to be basically non-overlapping in terms of individuals
# (one for adults, and one for adolescents)
map_ph4 = {'achenbach':   {'file_pfixes': ['asr4', 'ysr4'],
                             'date_lbl': 'datefilled',
                             'drop_keys': ['af_', 'bp_']},
             'aeq':         {'file_pfixes': ['aeqa4', 'aeq4']},
             'bis':         {'file_pfixes': ['bis_a_score4', 'bis_score4']},
             'craving':     {'file_pfixes': ['crv4']},
             'daily':       {'file_pfixes': ['daily4']},
             'dependence':  {'file_pfixes': ['dpndnce4']},
             'neo':         {'file_pfixes': ['neo4']},
             'sensation':   {'file_pfixes': ['sssc4', 'ssv4']},
             'sre':         {'file_pfixes': ['sre_score4']},
             }

# for ssaga questionnaires, the multiple file_fpixes are perfectly overlapping,
# so we end up joining them
map_ph4_ssaga = {
             'cssaga':      {'file_pfixes': ['cssaga4', 'dx_cssaga4'],
                             'date_lbl': 'IntvDate',
                             'id_lbl': 'IND_ID',
                             'join_fupfiles': True},
             'pssaga':      {'file_pfixes': ['pssaga4', 'dx_pssaga4'],
                             'date_lbl': 'IntvDate',
                             'id_lbl': 'ind_id',
                             'join_fupfiles': True},
             'ssaga':       {'file_pfixes': ['ssaga4', 'dx_ssaga4'],
                             'date_lbl': 'IntvDate',
                             'id_lbl': 'IND_ID',
                             'join_fupfiles': True},
             }

map_ph123 = {'aeq':         {'file_pfixes': ['aeq', 'aeqa', 'aeq3', 'aeqa3'],
                             'id_lbl': 'IND_ID'},
             'craving':     {'file_pfixes': ['craving', 'craving3']},
             'daily':       {'file_pfixes': ['daily', 'daily3']},
             'dependence':  {'file_pfixes': ['dpndnce', 'dpndnce3']},
             'neo':         {'file_pfixes': ['neo', 'neo3']},
             'sensation':   {'file_pfixes': ['sssc', 'ssvscore', 'sssc3']},
             'sre':         {'file_pfixes': ['sre', 'sre3']},
             }

def sasdir_tocsv(target_dir):
    ''' convert a directory filled with *.sas7bdat files to *.csv '''

    sas_files = glob.glob(target_dir+'*.sas7bdat')

    for sf in sas_files:
        sf_contents = SAS7BDAT(sf)
        sf_df = sf_contents.to_data_frame()
        sf_df.to_csv(sf+'.csv', index=False)

def quest_pathfollowup(path, file_pfixes, file_ext, max_fups):
    ''' build dict of followups to filepaths '''

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

def quest_pathfollowup_join(path, file_pfixes, file_ext, max_fups):
    ''' build dict of followups to filepaths '''

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

def parse_date(dstr, dateform):
    ''' parse date column '''

    dstr = str(dstr)

    if dstr != 'nan':
        return datetime.strptime(dstr, dateform)
    else:
        return None

def df_fromcsv(fullpath, id_lbl='ind_id', na_val=''):
    ''' convert csv into dataframe, converting ID column to standard '''

    # read csv in as dataframe
    df = pd.read_csv(fullpath, na_values=na_val)

    # convert id to str and save as new column
    df[id_lbl] = df[id_lbl].apply(int).apply(str)
    df['ID'] = df[id_lbl]

    return df


def df_fromsas(fullpath, id_lbl='ind_id'):
    ''' convert .sas7bdat to dataframe.
        unused because fails on incorrectly formatted files. '''

    # read csv in as dataframe
    df = pd.read_sas(fullpath, format='sas7bdat')

    # convert id to str and save as new column
    df[id_lbl] = df[id_lbl].apply(int).apply(str)
    df['ID'] = df[id_lbl]
    
    return df


def build_inputdict(name, knowledge_dict):
    ''' for a given questionnaire, build dict from defaults and specifics '''

    idict = def_info.copy()
    idict.update(knowledge_dict[name])
    return idict


def import_questfolder(qname, kmap, path):
    ''' import all questionnaire data in one folder '''

    # build inputs
    i = build_inputdict(qname, kmap)
    i['path'] = path
    print(i)

    # get dict of filepaths and followup numbers
    file_dict = quest_pathfollowup(i['path'] + qname + '/',
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
        # df = df_fromsas( os.path.join(i['path'], f), i['id_lbl'])

        # if date_lbl is a list, replace columns with one strjoined column
        try:
            if type(i['date_lbl']) == list:
                new_col = pd.Series([''] * df.shape[0], index=df.index)
                hyphen_col = pd.Series(['-'] * df.shape[0], index=df.index)
                new_col += df[i['date_lbl'][0]].apply(int).apply(str)
                for e in i['date_lbl'][1:]:
                    new_col += hyphen_col
                    new_col += df[e].apply(int).apply(str)
                df['-'.join(i['date_lbl'])] = new_col
        except:
            print('expected date cols were not present')

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
        datemod = datetime.fromtimestamp(os.path.getmtime(i['path']))
        sourceO = SourceInfo('questionnaires',
                                (i['path'], datemod), qname)
        sourceO.store()


def import_questfolder_ssaga(qname, kmap, path):
    ''' import all questionnaire data in one folder,
        joining multiple files of the same type '''

    # build inputs
    i = build_inputdict(qname, kmap)
    i['path'] = path
    print(i)

    # get dict of filepaths and followup numbers
    file_dict = quest_pathfollowup_join(i['path'] + qname + '/',
                                i['file_pfixes'],i['file_ext'], i['max_fups'])
    print(file_dict)
    if not file_dict:
        print('There were no files in the path specified.')

    for followup_num, files in file_dict.items():
    
        join_df = pd.DataFrame()
        
        for f in files:
            print(f)
            # read csv in as dataframe
            df = df_fromcsv(os.path.join(i['path'], f),
                                i['id_lbl'], i['na_val'])
            # df = df_fromsas( os.path.join(i['path'], f), i['id_lbl'])

            # if date_lbl is a list, replace columns with one strjoined column
            if type(i['date_lbl']) == list:
                new_col = pd.Series([''] * df.shape[0], index=df.index)
                hyphen_col = pd.Series(['-'] * df.shape[0], index=df.index)
                new_col += df[i['date_lbl'][0]].apply(int).apply(str)
                for e in i['date_lbl'][1:]:
                    new_col += hyphen_col
                    new_col += df[e].apply(int).apply(str)
                df['-'.join(i['date_lbl'])] = new_col

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
            
            df.set_index('ID', inplace=True)
            df.drop(i['id_lbl'], axis=1, inplace=True)
                
            if join_df.empty:
                join_df = df
            else:
                join_df = join_df.join(df, rsuffix='extra')
        
        join_df.reset_index(inplace=True)
        print(join_df.shape)
        # convert to records and store in mongo coll, noting followup_num
        for drec in join_df.to_dict(orient='records'):
            ro = SSAGA(qname, followup_num, info=drec)
            ro.storeNaTsafe()
        datemod = datetime.fromtimestamp(os.path.getmtime(i['path']))
        sourceO = SourceInfo('ssaga', (i['path'], datemod), qname)
        sourceO.store()


def match_fups2sessions(qname, knowledge_dict, path, q_collection):
    ''' for each record in a questionnaire's subcollection,
        find the nearest session '''

    # build inputs
    i = build_inputdict(qname, knowledge_dict)
    i['path'] = path
    # determine matching session letters (across followups)
    if type(i['date_lbl']) == list:
        i['date_lbl'] = '-'.join(i['date_lbl'])
    session_datecols = [letter + '-date' for letter in 'abcdefg']
    s = Mdb['subjects']
    q = Mdb[q_collection]
    qc = q.find({'questname': qname})
    print('matching fups to sessions')
    for qrec in tqdm(qc):
        testdate = qrec[i['date_lbl']]
        # try to access subject record, notify if not possible
        try:
            s_rec = s.find({'ID': qrec['ID']})[0]
        except:
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
