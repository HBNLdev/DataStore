from db import database as D
import db.compilation as C
import pandas as pd
import sys


#knowledge
neuropsych_variables = ['motivCBST', 'motivTOLT','mim_3b', 'mom_3b', 'em_3b', 'ao_3b', 'apt_3b', 'atoti_3b', 
                        'ttrti_3b', 'atrti_3b','mim_4b', 'mom_4b', 'em_4b', 'ao_4b', 'apt_4b', 'atoti_4b', 'ttrti_4b', 
                        'atrti_4b', 'mim_5b', 'mom_5b', 'em_5b', 'ao_5b', 'apt_5b', 'atoti_5b', 'ttrti_5b', 'atrti_5b',
                        'mim_tt', 'mom_tt', 'em_tt', 'ao_tt', 'apt_tt', 'atoti_tt', 'ttrti_tt', 'atrti_tt',
                        'otr_3b', 'otr_4b', 'otr_5b', 'otr_tt', 'tc_f', 'span_f', 'tcat_f', 'tat_f',
                        'tc_b', 'span_b', 'tcat_b', 'tat_b']

missing = ['id', 'dob', 'gender', 'hand', 'testdate', 'sessioncode']

tolt_conds = ['4b', '3b', 'tt', '5b']
tolt_measures = ['otr', 'atoti', 'atrti', 'ao', 'em', 'apt', 'ttrti', 'mim', 'mom']

vst_measures = ['tcat', 'tc', 'span', 'tat']
vst_conds = ['b', 'f']


#default projections
default_neuro_proj = {'ID': 1, 'np_session':1, 'gender': 1, 
                      'age': 1, 'hand' :1, 'testdate':1, '_id':0}

default_neuro_admin = {'ID':1, 'np_session':1, 'filepath': 1, 
                       '_id' : 1, 'insert_time': 1}


def remap_neuro_variables(neuro_var_names):
    
    '''creates a mapping between old neuropsych variable names and more intuitive way of
       representing variable names -- exp_condition_measure'''

    neuro_var_dict = {}
    for var in neuropsych_variables:
        split = var.split('_')
        if len(split) == 2:
            if len(split[1]) == 2:
                tolt = 'tolt' + '_' + split[1] + '_' + split[0]
                neuro_var_dict[var] = tolt
            if len(split[1]) == 1:
                cbst = 'vst' + '_' + split[1] + '_' + split[0]
                neuro_var_dict[var] = cbst
        if len(split) == 1:
            motiv = split[0][0:5]
            name = split[0][5:]
            if 'TOLT' in name:
                new_tolt = 'tolt_' + motiv
                neuro_var_dict[var] = new_tolt
            if 'CBST' in name:
                new_vst = 'vst_' + motiv
                neuro_var_dict[var] = new_vst

    return neuro_var_dict


def rename_neuropsych_cols(df, neuro_variables_dict):
    '''replaces neuropsych variable names in data frame'''
    return df.rename(columns={k:v for k,v in neuro_variables_dict.items() if k in df.columns})


def check_neuro_dict(neuro_proj_dict): 
    
    '''checks entire dictionary against existing neuropsych conds and measures.
       checks spelling and capitalization too'''
    
    exps_lst = []
    conds_lst = []
    measures_lst= []
    for k,v in neuro_proj_dict.items():
        split_key = k.split('_')
        exps_lst.append(split_key[0])
        conds_lst.append(split_key[1])
        measures_lst.append(split_key[2])
        
    #check if exps exist
    all_exps = ['TOLT', 'VST']
    exps_set = set(exps_lst) & set(all_exps)
    
    if exps_set != set(exps_lst):
        bad_exps = tuple(exps_set ^ set(exps_lst))
        print('This experiment wasnt found/make capitalized: ', bad_exps)
        sys.exit(1)
        
    #check if tuple values exist 
    all_measures = vst_measures + tolt_measures
    measures_set = set(measures_lst) & set(all_measures)

    if measures_set != set(measures_lst):
        bad_measures = tuple(measures_set ^ set(measures_lst))
        print('This measure was not found/change to lowercase: ', bad_measures)
        sys.exit(1)

    #check if list values exist          
    all_conds = vst_conds + tolt_conds
    conds_set = set(conds_lst) & set(all_conds)
    
    if conds_set != set(conds_lst):
        bad_cond = tuple(conds_set ^ set(conds_lst))
        print('This condition was not found/change to lowercase: ', bad_cond)
        sys.exit(1)
        
    return neuro_proj_dict


def neuro_dict_proj(neuro_dict):
    
    '''created projections from user created dictionary.
   contains function that checks entire dictionary against existing neuropsych conds and measures'''

    proj_lst = []    
    for k,v in neuro_dict.items():
        for i in v:
            value_len = len(i)
            for dict_val_lst in i[-1]: #get the list
                for tup in range(value_len -1):
                    concat = k + '_' + i[tup] + '_' + dict_val_lst
                    proj_lst.append(concat)

    proj = ({i: 1 for i in proj_lst})
    formatted_proj = check_neuro_dict(proj)
    
    return formatted_proj


def get_neuro_df(uids_lst, neuro_exp_only=None,
                neuro_dict=None,
                admin=False,
                flatten_df=False):
    
    '''Can query by: 1) uids_lst + neuro_exp_only to get ALL conds/measure for experiment
                     2) uids_lst + neuro_dict to get specific conds/measures for experiments'''
    
    query = {'uID': {'$in': uids_lst}}
    proj = default_neuro_proj.copy()
    remap_cols_dict = remap_neuro_variables(neuropsych_variables)
    
    if neuro_exp_only:
        if neuro_exp_only.upper() == 'VST':
            add_proj = {k:1 for k,v in remap_cols_dict.items() if 'VST' in v}
        if neuro_exp_only.upper() == 'TOLT':
            add_proj = {k:1 for k,v in remap_cols_dict.items() if 'TOLT' in v}
            
    if neuro_dict:
        need_to_format_proj = neuro_dict_proj(neuro_dict=neuro_dict)
        add_proj = {k:1 for k,v in remap_cols_dict.items() if v in need_to_format_proj.keys()}
    
    if admin:
        admin_proj = default_neuro_admin.copy()
        proj.update(admin_proj)
        docs = D.Mdb['neuropsych'].find(query, admin_proj)
        df = C.buildframe_fromdocs(docs, inds=['ID', 'np_session'])
        return df
        
    proj.update(add_proj)
    docs = D.Mdb['neuropsych'].find(query, proj)
    df = C.buildframe_fromdocs(docs, inds=['ID', 'np_session'])
    df1 = rename_neuropsych_cols(df, remap_cols_dict)
    return df1