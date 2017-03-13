''' knowledge about questionnaire data and attributes of their CSV files
    as they are found on the zork website. change over time as needed. '''

# defining maximum followups

max_fups = 6

# definitions of zork URLs

base_url = 'https://zork5.wustl.edu/coganew/data/available_data'
core_url = '/pheno_all/core_pheno_20161129.zip'
fam_url = '/family_data/allfam_sas_3-20-12.zip'
ach_url = '/Phase_IV/Achenbach%20January%202016%20Distribution.zip'
cal_url = '/Phase_IV/CAL%20Forms%20Summer-Fall%202015%20Harvest_sas.zip'

# definitions of locations in filesystem

harmonization_path = '/processed_data/zork/harmonization/harmonization-combined-format.csv'

zork_p123_path = '/processed_data/zork/zork-phase123/'
p123_master_path = zork_p123_path + 'subject/master/master.sas7bdat.csv'

zork_p4_path = '/processed_data/zork/zork-phase4-72/'
p4_master_path = zork_p4_path + 'subject/master/master4_30nov2016.sas7bdat.csv'

internalizing_dir = '/processed_data/zork/zork-phase4-69/subject/internalizing/'
internalizing_file = 'INT_Scale_All-Total-Scores_n11281.csv'

externalizing_dir = '/processed_data/zork/zork-phase4-69/subject/vcuext/'
externalizing_file = 'vcu_ext_all_121112.sas7bdat.csv'

allrels_file = 'allrels_30nov2016.sas7bdat.csv'

fham_file = 'bigfham4.sas7bdat.csv'

# master file columns

p1_cols = ['SSAGA_DT', 'CSAGA_DT', 'CSAGC_DT', 'FHAM_DT', 'TPQ_DT',
           'ZUCK_DT', 'ERP_DT', ]
p2_cols = ['SAGA2_DT', 'CSGA2_DT', 'CSGC2_DT', 'ERP2_DT', 'FHAM2_DT',
           'AEQ_DT', 'AEQA_DT', 'QSCL_DT', 'DAILY_DT', 'NEO_DT',
           'SRE_DT', 'SSSC_DT']
p3_cols = ['SAGA3_DT', 'CSGA3_DT', 'CSGC3_DT', 'ERP3_DT', 'FHAM3_DT',
           'AEQ3_DT', 'AEQA3_DT', 'QSCL3_DT', 'DLY3_DT', 'NEO3_DT',
           'SRE3_DT', 'SSSC3_DT']
p4_col_prefixes = ['aeqg', 'aeqy', 'crv', 'cssaga', 'dp', 'hass',
                   'neo', 'ssaga', 'sssc', 'ssv']

# note we defaultly use this dateformat because pandas sniffs to this format
def_info = {'date_lbl': ['ADM_Y', 'ADM_M', 'ADM_D'],
            'na_val': '',
            'dateform': '%Y-%m-%d',
            'file_ext': '.sas7bdat.csv',
            # 'file_ext': '.sas7bdat',
            'max_fups': max_fups,
            'id_lbl': 'ind_id',
            'capitalize': False,
            }

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
                        'zork_url': core_url},
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

map_ph123_ssaga['ssaga']['joindate_from'] = {k: p123_master_path for k in map_ph123_ssaga['ssaga']['date_lbl'].keys()}
