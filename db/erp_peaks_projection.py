from db import database as D
import db.compilation as C
import pandas as pd
from collections import defaultdict
import sys

erp_condpeaks = {'aod': ['nt_N1', 'nt_P2', 't_N1', 't_P3'],
                 'vp3': ['nt_N1', 'nt_P3', 't_N1', 't_P3', 'nv_N1', 'nv_P3'],
                 'ant': ['a_N4', 'a_P3', 'j_N4', 'j_P3', 'w_N4', 'w_P3']}

electrodes_62 = ['FP1', 'FP2', 'F7' , 'F8' , 'AF1', 'AF2', 'FZ' ,
 'F4' , 'F3' , 'FC6', 'FC5', 'FC2', 'FC1', 'T8' , 'T7' , 'CZ' ,
  'C3' , 'C4' , 'CP5', 'CP6', 'CP1', 'CP2', 'P3' , 'P4' , 'PZ' ,
   'P8' , 'P7' , 'PO2', 'PO1', 'O2' , 'O1' , 'AF7', 'AF8',
    'F5' , 'F6' , 'FT7', 'FT8', 'FPZ', 'FC4', 'FC3', 'C6' , 'C5' ,
     'F2' , 'F1' , 'TP8', 'TP7', 'AFZ', 'CP3', 'CP4', 'P5' ,
      'P6' , 'C1' , 'C2' , 'PO7', 'PO8', 'FCZ', 'POZ', 'OZ' ,
       'P2' , 'P1' , 'CPZ'] #removed X

default_measures = ['amp', 'lat']
default_ERPproj = {"ID":1, "session":1, "_id":0}

def mt_arg_check(experiments=None, cond_peaks=None, channels=None,
                 exp_cond_peaks=None,
                 exp_condpeaks_chans=None):
    
    ''' checks compatibility of create_matdf args and exits program if invalid
       combination is found '''
    
    if (experiments or cond_peaks) and (exp_cond_peaks or exp_condpeaks_chans):
        print('Error: Invalid argument combination.  Either use experiments and/or '
              'cond_peaks and/or channels.  Exp_cond_peaks and exp_condpeaks_chans are '
              'essentially dictionaries that contain the same data in 1 object.')
        sys.exit(1)
        
    if channels and exp_condpeaks_chans:
        print('Error: Invalid argument combination.  Channels should already '
               'be in exp_condpeaks_chans dictionary.')
        sys.exit(1)
        
    if exp_cond_peaks and exp_condpeaks_chans:
        print('Error: Invalid argument combination.  Use either exp_cond_peaks '  
               'or exp_condpeaks_chans, not both.')
        sys.exit(1)

def parse_mt_args(experiment, cond_peaks=None, channels=None, measure=None):
    
    ''' parses exp, cond_peaks, channels, measure 
        arguments for create_matdf function and returns the projection '''
   
    if not cond_peaks:
        cond_peaks = erp_condpeaks[experiment]
   
    if not channels:
        channels = electrodes_62.copy()
   
    if not measure:
        measure = default_measures.copy()
    
    proj = C.format_ERPprojection(experiment, cond_peaks, channels, measure)
   
    return proj

def parse_mtdict_arg(exp_condpeakschans, measures = None):
    
    ''' parses exp_condpeaks_chans arg in create_matdf
        Example args: erp_exp_condpeakschans['aod'] = [('nt', 'N1', ['FZ', 'PZ']),
                                                       ('nt', 'P2', ['CZ', 'F4', 'F3'])]  
                      measures = ['amp'] '''
    
    if not measures:
        
        for exp_name, cond_peaks_chans in exp_condpeakschans.items():
            proj = C.format_ERPprojection_tups(exp_name, cond_peaks_chans, default_measures)

    if measures:
        
        for exp_name, cond_peaks_chans in exp_condpeakschans.items():
            proj = C.format_ERPprojection_tups(exp_name, cond_peaks_chans, measures)
        
    return proj

def create_mtdf(uIDs,
                 experiments=None, cond_peaks=None, channels=None,
                 exp_cond_peaks=None,
                 exp_condpeaks_chans=None,
                 measures=None,
                 flatten_df=False):
    ''' 
        1) experiment, cond_peak, channels as 3 separate lists ,
        2) if user knows experiment & conds_peaks of interest -- use exp_condpeaks argument
           > add chans/measures arg to filter further
        3) if user wants different channels for different cond_peaks combinations -- use exp_condpeaks_chans arg
    '''
    #check for invalid argument combinations
    mt_arg_check(experiments=experiments, cond_peaks=cond_peaks, channels=channels,
                 exp_cond_peaks=exp_cond_peaks,
                 exp_condpeaks_chans=exp_condpeaks_chans)
    
    #check experiments arg
    if type(experiments) == str:
        experiments = [experiments]
    if type(experiments) == tuple:
        experiments = list(experiments)
    if type(experiments) == dict:
        print("Error: Experiments should be a list, not a dictionary")
        sys.exit(1)
        
    #check cond_peaks arg    
    if type(cond_peaks) == str:
        cond_peaks = [cond_peaks]
    if type(cond_peaks) == tuple:
        cond_peaks = list(cond_peaks)
    if type(cond_peaks) == dict:
        print("Error: Cond_peaks should be a list, not a dictionary")
        sys.exit(1)
    
    #check channels arg
    if type(channels) == str:
        channels = [channels]
    if type(channels) == tuple:
        channels = list(channels)
    if type(channels) == dict:
        print('Error: Channels should be a list, not a dictionary')
        sys.exit(1)
        
    query = {'uID': {'$in': uIDs}}
    proj = default_ERPproj.copy()
    
    if experiments:

        for e in experiments:
            add_proj = parse_mt_args(e, cond_peaks, channels, measures)
            add_query = {e: {'$exists': True}}
            proj.update(add_proj)
            query.update(add_query)

            
    if exp_cond_peaks:
        
        if type(exp_cond_peaks) is not dict:
            print('Error: exp_cond_peaks should be a dictionary')
            sys.exit(1)
        
        for e, cp_lst in exp_cond_peaks.items():
            
            add_proj = parse_mt_args(e, cp_lst, channels, measures)
            add_query = {e: {'$exists': True}}
            proj.update(add_proj)
            query.update(add_query)
        
    if exp_condpeaks_chans:
        
        if type(exp_condpeaks_chans) is not dict:
            print('Error: exp_condpeaks_chans should be a dictionary')
            sys.exit(1)
        
        add_proj = parse_mtdict_arg(exp_condpeaks_chans, measures=measures)
        proj.update(add_proj)
        
        for e in exp_condpeaks_chans.keys():
            add_query = {e: {'$exists': True}}
            query.update(add_query)
        
    docs = D.Mdb['ERPpeaks'].find(query, proj)
    df = C.buildframe_fromdocs(docs, inds=['ID', 'session'])
    
    
    if flatten_df:
        return df
    else:
        df.columns = pd.MultiIndex.from_tuples([tuple(col_name.split('_')) for col_name in df.columns],
                                              names=['experiment', 'condition', 'peak', 'channel', 'measures'])
        return df



def create_dict_of_tuples(experiment, cond_peaks, channel_lst):

    '''returns dict: {'aod': [('nt', 'P3', ['af1', 'af2', 'af3']),
                              ('t', 'P3', ['af1', 'af2', 'af3'])]} '''
    
    d = {}
    for e in experiment:
        l = []
        for cp in cond_peaks: 
            cond = cp.split('_')[0]
            peak = cp.split('_')[1]
            cond_peak_tup = cond, peak, channel_lst
            l.append(cond_peak_tup)
        d[e] = l
    return d

def update_tuple_dict(dict_of_tuples, experiment, condition, peak, channel_lst):
  
    '''for specific condition-peak tuple value, updates list of channels'''
    
    for k,v in dict_of_tuples.items():
        if experiment in k:
            for n, tpl in enumerate(v):
                if condition in tpl and peak in tpl:
                    v[n] = tuple(list(tpl)[:-1] + [channel_lst])
    return dict_of_tuples
