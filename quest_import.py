# formalize process by which a questionnaire gets added to a collection
# and all things that happen to it after that

import os
from datetime import datetime, timedelta
from glob import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import organization as O
from tqdm import tqdm

sparser_sub = {'achenbach': ['af_', 'bp_']}

# note we defaultly use this dateformat because pandas sniffs to this format

def_info = {  # 'path': '/processed_data/zork/zork-65/session/',
            'path': '/active_projects/mike/zork-ph4-65-bl/session/',
            'version': 4,
            'date_lbl': ['ADM_Y', 'ADM_M', 'ADM_D'],
            'na_val': '',
            'dateform': '%Y-%m-%d',
            'file_ext': '.sas7bdat.csv',
            # 'file_ext': '.sas7bdat',
            'max_fups': 5,
            'id_lbl': 'ind_id',
            'join_fupfiles': False
            }

# still missing: cal, cssaga, pssaga, ssaga
# note that if there are multiple file_pfixes, the files are assumed to be
# basically non-overlapping in terms of individuals
knowledge = {'achenbach':   {'file_pfixes': ['asr4', 'ysr4'],
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

knowledge_ssaga = {
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
    ''' convert .sas7bdat to dataframe '''
    # read csv in as dataframe
    df = pd.read_sas(fullpath, format='sas7bdat')
    # convert id to str and save as new column
    df[id_lbl] = df[id_lbl].apply(int).apply(str)
    df['ID'] = df[id_lbl]
    print(df)
    return df


def build_inputdict(name, knowledge_dict):
    ''' for a given questionnaire, build dict from defaults and specifics '''
    idict = def_info.copy()
    idict.update(knowledge_dict[name])
    return idict


def import_questfolder(qname):
    ''' import all questionnaire data in one folder '''

    # build inputs
    i = build_inputdict(qname, knowledge)
    print(i)

    # get dict of filepaths and followup numbers
    file_dict = quest_pathfollowup(i['path'] + qname + '/',
                                i['file_pfixes'], i['file_ext'], i['max_fups'])
    print(file_dict)
    if not file_dict:
        print('There were no files in the path specified.')

    # for each file
    for followup_num, files in file_dict.items():

        for f in files:
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
                except:
                    pass

            # convert to records and store in mongo coll, noting followup_num
            for drec in df.to_dict(orient='records'):
                ro = O.Questionnaire(qname, followup_num, info=drec)
                ro.storeNaTsafe()
            datemod = datetime.fromtimestamp(os.path.getmtime(i['path']))
            sourceO = O.SourceInfo('questionnaires',
                                    (i['path'], datemod), qname)
            sourceO.store()


def import_questfolder_ssaga(qname):
    ''' import all questionnaire data in one folder,
        joining multiple files of the same type '''

    # build inputs
    i = build_inputdict(qname, knowledge_ssaga)
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
            ro = O.SSAGA(qname, followup_num, info=drec)
            ro.storeNaTsafe()
        datemod = datetime.fromtimestamp(os.path.getmtime(i['path']))
        sourceO = O.SourceInfo('ssaga', (i['path'], datemod), qname)
        sourceO.store()


def match_fups2sessions(qname, knowledge_dict, q_collection):
    ''' for each record in a questionnaire's subcollection,
        find the nearest session '''

    # build inputs
    i = build_inputdict(qname, knowledge_dict)
    # determine matching session letters (across followups)
    if type(i['date_lbl']) == list:
        i['date_lbl'] = '-'.join(i['date_lbl'])
    session_datecols = [letter + '-date' for letter in 'abcdefg']
    s = O.Mdb['subjects']
    q = O.Mdb[q_collection]
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
