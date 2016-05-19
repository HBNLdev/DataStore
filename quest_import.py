# formalize process by which a questionnaire gets added to a collection
# and all things that happen to it after that

import os
from datetime import datetime, timedelta
from glob import glob
import numpy as np
import pandas as pd
import organization as O

sparser_sub = {'achenbach': ['af_', 'bp_']}

def_info = {# 'path': '/processed_data/zork/zork-65/session/',
            'path': '/active_projects/mike/zork-ph4-65-bl/session/',
            'version': 4,
            'date_lbl': ['ADM_Y','ADM_M','ADM_D'],
            'na_val': '',
            'dateform': '%Y-%m-%d',
            'file_ext': '.sas7bdat.csv',
            # 'file_ext': '.sas7bdat',
            'max_fups': 5,
            'id_lbl': 'ind_id',
            }

knowledge = {'achenbach': {'file_pfixes': ['asr4', 'ysr4'],
                           'date_lbl': 'datefilled',
                           'drop_keys': ['af_','bp_']},
             'bis':       {'file_pfixes': ['bis_a_score4', 'bis_score4']},
             'neo':       {'file_pfixes': ['neo4']},
             'sensation': {'file_pfixes': ['sssc4','ssv4']},
             'daily':     {'file_pfixes': ['daily4']},
             'aeq':       {'file_pfixes': ['aeqa4','aeq4']},
             'craving':   {'file_pfixes': ['crv4']},
             'dependence':{'file_pfixes': ['dpndnce4']},
             'sre':       {'file_pfixes': ['sre_score4']},
             }

def quest_pathfollowup(path, file_pfixes, file_ext, max_fups):
    fn_dict = {}
    fn_infolder = glob(path+'*'+file_ext)
    for fp in file_pfixes:
        for followup in range(max_fups+1):
            if followup == 0:
                fstr = fp + file_ext
            else:
                fstr = fp + '_f' + str(followup) + file_ext
            fpathstr = os.path.join(path, fstr)
            if fpathstr in fn_infolder:                
                fn_dict.update( {fpathstr: followup} )
    return fn_dict

def parse_date(dstr, dateform):
    dstr = str(dstr)
    return datetime.strptime(dstr,dateform) if dstr != 'nan' else None

def df_fromcsv( fullpath, id_lbl='ind_id', na_val='' ):
    # read csv in as dataframe
    df = pd.read_csv( fullpath, na_values=na_val)
    # convert id to str and save as new column
    df[id_lbl] = df[id_lbl].apply(int).apply(str)
    df['ID'] = df[id_lbl]
    return df

def df_fromsas( fullpath, id_lbl='ind_id' ):
    # read csv in as dataframe
    df = pd.read_sas( fullpath, format='sas7bdat' )
    # convert id to str and save as new column
    df[id_lbl] = df[id_lbl].apply(int).apply(str)
    df['ID'] = df[id_lbl]
    print(df)
    return df

def build_inputdict( name ):
    idict = def_info.copy()
    idict.update( knowledge[name] )
    return idict

def import_questfolder(qname):
    # build inputs
    i = build_inputdict( qname )
    # get dict of filepaths and followup numbers
    file_dict = quest_pathfollowup(i['path']+qname+'/', i['file_pfixes'],
                    i['file_ext'], i['max_fups'])
    print(i)
    if not file_dict:
        print('There were no files in the path specified.')
    # for each file
    for f, followup_num in file_dict.items():
        # read csv in as dataframe
        df = df_fromcsv( os.path.join(i['path'], f), i['id_lbl'], i['na_val'])
        # df = df_fromsas( os.path.join(i['path'], f), i['id_lbl'])
        # if date_lbl is a list, replace columns with one strjoined column
        if type(i['date_lbl']) == list:
            new_col = pd.Series(['']*df.shape[0], index=df.index)
            hyphen_col = pd.Series(['-']*df.shape[0], index=df.index)
            new_col += df[i['date_lbl'][0]].apply(int).apply(str)
            for e in i['date_lbl'][1:]:
                new_col += hyphen_col
                new_col += df[e].apply(int).apply(str)
            df['-'.join(i['date_lbl'])] = new_col
        # attempt to convert columns to dates
        for c in df:
            try:
                df[c] = df[c].apply( parse_date, args=(i['dateform'],) )
            except:
                pass
        # convert to records and store in mongo coll, noting followup_num
        for drec in df.to_dict(orient='records'):
            ro = O.Questionnaire(qname, followup_num, drec)
            ro.storeNaTsafe()
        datemod = datetime.fromtimestamp(os.path.getmtime(i['path']))
        sourceO = O.SourceInfo('questionnaires', (i['path'], datemod), qname)
        sourceO.store()

def match_fups2sessions(qname):
    # build inputs
    i = build_inputdict( qname )
    # determine matching session letters (across followups)
    if type(i['date_lbl']) == list:
        i['date_lbl'] = '-'.join(i['date_lbl'])
    session_datecols = [letter+'-date' for letter in 'abcdefg']
    s = O.Mdb['subjects']
    q = O.Mdb['questionnaires']
    qc = q.find( {'questname': qname} )
    for qrec in qc:
        testdate = qrec[i['date_lbl']]
        # try to access subject record, notify if not possible
        try:
            s_rec = s.find( {'ID': qrec['ID']} )[0]
        except:
            print('could not find '+qrec['ID']+' in subjects collection')
            q.update_one({'_id': qrec['_id']}, {'$set':{'session': None} })
            continue
        session_date_diffs = []
        for sdc in session_datecols:
            if type(s_rec[sdc]) == datetime:
                session_date_diffs.append(abs(s_rec[sdc] - testdate))
            else:
                session_date_diffs.append(timedelta(100000))
        if all( np.equal( timedelta(100000), session_date_diffs ) ):
            best_match = None
        else:
            min_ind = np.argmin(session_date_diffs)
            best_match = session_datecols[min_ind][0]
        q.update_one({'_id': qrec['_id']}, {'$set':{'session': best_match} })