''' tools for adding questionnaire data to mongo '''

import os
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd

import db.database as D
from .knowledge.questionnaires import def_info, harmonization_path
from .utils.compilation import df_fromcsv
from .utils.dates import convert_date_fallback


def quest_pathfollowup(path, file_pfixes, file_ext, max_fups, phase=4):
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
                followup_lbl = 'p' + str(phase) + 'f' + str(followup)
                fn_dict.update({fpathstr: followup_lbl})

    return fn_dict


def quest_pathfollowup_ssaga(path, file_pfixes, file_ext, max_fups, phase=4):
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
                followup_lbl = 'p' + str(phase) + 'f' + str(followup)
                fn_dict[followup_lbl].append(fpathstr)

    return fn_dict


def build_inputdict(name, knowledge_dict):
    ''' for a given questionnaire, build dict from defaults and specifics '''

    idict = def_info.copy()
    idict.update(knowledge_dict[name])
    return idict


def import_questfolder_ph4(qname, kmap, path, quest_collection_name):
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
    for file, followup_lbl in file_dict.items():

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
            df['date'] = df[newdatecol_name].apply(convert_date_fallback, args=(i['dateform'],))
        except KeyError:
            print('expected date columns were not present')
            df['date'] = np.nan

        # attempt to convert columns to dates
        # for c in df:
        #     try:
        #         df[c] = df[c].apply(convert_date_fallback, args=(i['dateform'],))
        #     except:
        #         pass

        # convert to records and store in mongo coll, noting followup_lbl
        for drec in df.to_dict(orient='records'):
            ro = D.Questionnaire(quest_collection_name, qname, followup_lbl, data=drec)
            ro.storeNaTsafe()


def import_questfolder_ssaga_ph4(qname, kmap, path, ssaga_collection_name):
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

    for followup_lbl, files in file_dict.items():
        print(followup_lbl)
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

            # # attempt to convert columns to dates
            # for c in df:
            #     try:
            #         df[c] = df[c].apply(convert_date_fallback, args=(i['dateform'],))
            #         # print('date format changed for', c)
            #     except:
            #         pass
            #         # print('date format failed for', c)
            #         # if c == 'IntvDate':
            #         # raise

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

        join_df['date'] = join_df[i['date_lbl']].apply(convert_date_fallback, args=(i['dateform'],))

        join_df.drop('ID', axis=1, inplace=True)
        join_df.reset_index(inplace=True)
        print(join_df.shape)
        # convert to records and store in mongo coll, noting followup_lbl,
        # and separately for the full questionnaire and diagnoses
        all_vars = set()
        for drec in join_df[list(nondx_cols) + ['ID', 'date']].to_dict(orient='records'):
            ro = D.Questionnaire(ssaga_collection_name, qname, followup_lbl, data=drec)
            all_vars.update(drec.keys())
            ro.storeNaTsafe()

        for drec in join_df[list(dx_cols) + ['ID', 'date', i['date_lbl']]].to_dict(orient='records'):
            ro = D.Questionnaire(ssaga_collection_name, 'dx_' + qname, followup_lbl, data=drec)
            ro.storeNaTsafe()

    return all_vars

def import_questfolder_ph123(qname, kmap, path, quest_collection_name):
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
        df['date'] = df[date_col].apply(convert_date_fallback, args=(i['dateform'],))

        # df.set_index('ID', inplace=True)
        df.drop(i['id_lbl'], axis=1, inplace=True)

        followup_lbl = i['followup'][fpx]
        for drec in df.to_dict(orient='records'):
            ro = D.Questionnaire(quest_collection_name, qname, followup_lbl, data=drec)
            ro.storeNaTsafe()


def create_ssagaharm_renamer(ssaga_type, phase):
    if ssaga_type == 'pssaga':
        ssaga_type = 'cssaga'

    if phase == 'p3':
        phase = 'p2'

    from_col = phase + '_' + ssaga_type
    to_col = 'p4_' + ssaga_type

    harm_df = pd.read_csv(harmonization_path)
    harm_df_na = harm_df[[from_col, to_col]].dropna(how='any')

    renamer = {a: b for a, b in zip(harm_df_na[from_col].values, harm_df_na[to_col].values) if a != b}

    return renamer


def import_questfolder_ssaga_ph123(qname, kmap, path, ssaga_collection_name):
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
    all_fields = set()
    for f in file_list:

        fname = os.path.split(f)[1]
        fpx = fname.split('.')[0]
        followup_lbl = i['followup'][fpx]

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
        df['date'] = df[date_col].apply(convert_date_fallback, args=(i['dateform'],))

        # df.set_index('ID', inplace=True)
        df.drop(i['id_lbl'], axis=1, inplace=True)
        print(df.shape)
        if fname[:3] == 'dx_':
            for drec in df.to_dict(orient='records'):
                ro = D.Questionnaire(ssaga_collection_name, 'dx_' + qname, followup_lbl, data=drec)
                ro.storeNaTsafe()
        else:

            # convert to harmonize SSAGA item names
            harm_renamer = create_ssagaharm_renamer(qname, followup_lbl)
            df = df.rename(columns=harm_renamer)

            for drec in df.to_dict(orient='records'):
                ro = D.Questionnaire(ssaga_collection_name, qname, followup_lbl, data=drec)
                ro.storeNaTsafe()
                all_fields.update(drec.keys())
    return all_fields