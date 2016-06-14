import numpy as np
from collections import defaultdict

default_field_eponyms = {'ID', 'famID', 'mID', 'fID', 'sex', 'twin'}
default_dx_sxc = {'dx4': 'alc_dep_dx', 'dx5': 'cor_ald5dx',
                  'sxc4': 'cor_alc_dep_max_sx_cnt',
                  'sxc5': 'cor_ald5sx_max_cnt'}
def_fields = {f:f for f in default_field_eponyms}
def_fields.update(default_dx_sxc)
sex2pf = {'m':'fID', 'f':'mID'}
score_map = {'self': 1, 'twin': 1,
             'mother':0.5, 'father':0.5, 'fullsib':0.5, 'child':0.5,
             'mgm': 0.25, 'mgf': 0.25, 'fgm': 0.25, 'fgf': 0.25,
             'msib': 0.25, 'fsib': 0.25,
             'gp': 0.25, 'parentsib': 0.25, 'gc': 0.25, 'sibchild': 0.25,
             'halfsib': 0.25}

def conv_159(v):
    if v==1:
        return 0
    elif v==5:
        return 1
    else:
        return np.nan

def harmonize_fields(row, field1, field2):
    if row[field1]==row[field2] or (np.isnan(row[field1]) and np.isnan(row[field1])):
        return row[field1]
    else:
        return max(row[field1], row[field2])
    
def prepare_for_fhd(in_df):
    df = in_df.copy()
    df['cor_alc_dep_dx'] = df['cor_alc_dep_dx'].apply(conv_159)
    df['aldep1'] = df.apply(
        harmonize_fields, axis=1, args=['alc_dep_dx', 'cor_alc_dep_dx'])
    df.reset_index('session', inplace=True)
    df['ID'] = df.index.get_level_values('ID')
    drop_lst = ['alc_dep_dx', 'fID', 'famID', 'mID', 'sex', 'cor_alc_dep_dx',
                'cor_ald5dx', 'cor_sex', 'aldep1', 'ID']
    drop_cols = [col for col in df.columns if col not in drop_lst]
    return df.drop(drop_cols, axis=1)

def calc_fhd(sDF, in_fDF, field_dict=None):
    ''' main function to apply to a DF, adds columns '''
    fd = def_fields.copy()
    if field_dict:
        fd.update(field_dict)
    fDF = in_fDF.rename(columns={v:k for k,v in fd.items()})
    #sDF['fhd_dx4'], sDF['fhd_dx5'], sDF['fhd_sxc4'], sDF['fhd_sxc5'], \
    #    sDF['n_rels'] = zip(*sDF.apply(calc_fhd_row, axis=1, args=[fDF]))
    sDF['fhd_dx4_ratio'], sDF['fhd_dx4_sum'], sDF['n_rels'] = zip(*sDF.apply(calc_fhd_row, axis=1, args=[fDF]))
    #sDF['fhd_dx4_nrels'] = sDF.apply(calc_fhd_row, axis=1, args=[fDF])

def calc_fhd_row(row, df, degrees=[1, 2], descend=False, cat_norm=True):
    ''' row-wise apply function '''
    famDF = df[ df['famID'] == row['famID'] ]
    I = Individual(row, famDF)
    if 0 in degrees:
        I.zeroth_degree() # 1
    if 1 in degrees:
        I.first_degree(descend) # 0.5
    if 2 in degrees:
        I.second_degree(descend) # 0.25
    print('.',end='')
    return I.ratio_score('dx4', 1, cat_norm), I.sum_score('dx4', 1, cat_norm), \
                I.count('dx4')

class Individual:

    def __init__(s, row, df):
        s.info = row
        s.df = df
        try:
            s.parent_field = sex2pf[row['sex']]
        except:
            s.parent_field = None

        s.rel_set = set()
        s.rel_dict = defaultdict(list)

    def add_rel(s, ID, relation):
        if ID not in s.rel_set and isinstance(ID, str):
            s.rel_set.update({ID})
            s.rel_dict[relation].append(ID)

    def ratio_score(s, ckfield, ckfield_max, cat_norm=True):
        score_num = score_denom = 0
        for rel_type, rel_IDs in s.rel_dict.items():
            weight = score_map[rel_type]
            tmp_score = tmp_count = 0
            for rel_ID in rel_IDs:
                try:
                    rel_val = s.df.loc[rel_ID, ckfield]
                    if np.isnan(rel_val): # subject found, missing value
                        continue
                    else: # subject found, has value
                        tmp_score += weight * rel_val/ckfield_max
                        tmp_count += 1
                except: # subject not found at all
                    pass
            if cat_norm: # normalize in each relative category
                if tmp_count > 0:
                    score_num += tmp_score / tmp_count
                    score_denom += weight
            else:
                score_num += tmp_score
                score_denom += weight * tmp_count
        if score_denom == 0: # no rels known or nothing known about rels
            return np.nan
        else:
            return score_num / score_denom

    def sum_score(s, ckfield, ckfield_max, cat_norm=True):
        score_num = 0
        for rel_type, rel_IDs in s.rel_dict.items():
            weight = score_map[rel_type]
            tmp_score = tmp_count = 0
            for rel_ID in rel_IDs:
                try:
                    rel_val = s.df.loc[rel_ID, ckfield]
                    if np.isnan(rel_val): # subject found, missing value
                        continue
                    else: # subject found, has value
                        tmp_score += weight * rel_val/ckfield_max
                        tmp_count += 1
                except: # subject not found at all
                    pass
            if cat_norm: # normalize in each relative category
                if tmp_count > 0:
                    score_num += tmp_score / tmp_count
            else:
                score_num += tmp_score
        return score_num

    def count(s, ckfield):
        count = 0
        for rel_type, rel_IDs in s.rel_dict.items():
            for rel_ID in rel_IDs:
                try:
                    rel_val = s.df.loc[rel_ID, ckfield]
                    if np.isnan(rel_val): # subject found, missing value
                        continue
                    else: # subject found, has value
                        count+=1
                except: # subject not found at all
                    pass
        return count

    def zeroth_degree(s): # worth 1
        s.add_rel(s.info['ID'], 'self') # self
        if s.info['twin'] != 2:
            return
        twinDF = s.df[s.df['twin']==2] # twin
        for twin_ID, twin in twinDF.iterrows():
            if s.info['fID'] == twin['fID'] and s.info['mID'] == twin['mID']:
                s.add_rel(twin_ID, 'twin')

    def first_degree(s, descend=False): # worth 0.5
        # parents
        s.add_rel(s.info['mID'], 'mother')
        s.add_rel(s.info['fID'], 'father')

        # full siblings
        if s.rel_dict['mother'] and s.rel_dict['father']:
            fullsibDF = s.df[(s.df['mID']==s.rel_dict['mother'][0]) &
                             (s.df['fID']==s.rel_dict['father'][0])]
            for fullsib_ID, fullsib in fullsibDF.iterrows():
                s.add_rel(fullsib_ID, 'fullsib')

        if descend:
            # children
            if s.parent_field:
                childDF = s.df[s.df[s.parent_field]==s.info['ID']]
                for child_ID, child in childDF.iterrows():
                    s.add_rel(child_ID, 'child')

    def second_degree(s, descend=False):

        for name in ['mother', 'father']:
            if s.rel_dict[name]:
                try:
                    parentrow = s.df.loc[s.rel_dict[name][0],:]
                except:
                    continue
                # grandparents
                s.add_rel(parentrow['mID'], name[0]+'gm')
                s.add_rel(parentrow['fID'], name[0]+'gf')
                # full siblings of parents (aunts/uncles)
                parentfsibDF = s.df[(s.df['mID']==parentrow['mID']) &
                                    (s.df['fID']==parentrow['fID'])]
                for parentfsib_ID, parentfsib in parentfsibDF.iterrows():
                    s.add_rel(parentfsib_ID, name[0]+'sib')

        if descend:
            # grandchildren
            if s.rel_dict['child']:
                gcDF = s.df[(s.df['mID'].isin(s.rel_dict['child'])) |
                            (s.df['fID'].isin(s.rel_dict['child']))]
                for gc_ID, gc in gcDF.iterrows():
                    s.add_rel(gc_ID, 'gc')

            # children of siblings (nieces/nephews)
            if s.rel_dict['fullsib']:
                nDF = s.df[(s.df['mID'].isin(s.rel_dict['fullsib'])) |
                           (s.df['fID'].isin(s.rel_dict['fullsib']))]
                for n_ID, n in nDF.iterrows():
                    s.add_rel(n_ID, 'sibchild')

        # half siblings
        halfsibDF = s.df[(s.df['mID']==s.info['mID']) |
                         (s.df['fID']==s.info['fID'])]
        for halfsib_ID, halfsib in halfsibDF.iterrows():
            s.add_rel(halfsib_ID, 'halfsib')