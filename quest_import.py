# formalize process by which a questionnaire gets added to a collection
# and all things that happen to it after that

import os
from datetime import datetime, timedelta
from glob import glob
import numpy as np
import pandas as pd
import organization as O

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
    return datetime.strptime(dstr,dateform) if dstr != 'nan' else np.nan

def import_questfolder(qname, path, file_pfixes, id_lbl, date_lbl, na_val = '',
    dateform = '%Y-%m-%d', file_ext='.sas7bdat.csv', max_fups = 5):
    # get dict of filepaths and the followup number
    file_dict = quest_pathfollowup(path, file_pfixes, file_ext, max_fups)

    # for each file
    for f, followup_num in file_dict.items():
        # read csv in as dataframe, converting id field and missing vals
        df = pd.read_csv( os.path.join(path,f), na_values=na_val )
        # attempt to convert columns to dates
        for c in df:
            try:
                df[c] = df[c].apply( parse_date, args=(dateform,) )
            except:
                pass
        # convert to records and store in mongo coll, noting followup_num
        for rec in df.to_dict(orient='records'):
            ro = O.Questionnaire(qname, followup_num, rec)
            ro.storeNaTsafe()

    # determine matching session letters (across followups)
    session_datecols = [letter+'-date' for letter in 'abcdefg']
    s = O.Mdb['subjects']
    q = O.Mdb['questionnaires']
    query = {'questname':qname}
    qc = q.find(query)
    for rec in qc:
        testdate = rec[date_lbl]
        ID = str(int(float(rec[id_lbl])))
        s_rec = s.find( {'ID': ID} )[0]
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
        q.update_one({'_id': rec['_id']},{'$set':{'session':best_match}})