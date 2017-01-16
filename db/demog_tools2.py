''' tools for demographic info, mainly calculating family history density '''

from collections import defaultdict
from itertools import combinations

import networkx as nx

import numpy as np

default_field_eponyms = {'ID', 'famID', 'mID', 'fID', 'sex', 'twin'}
# default_dx_sxc = {'cor_alc_dep_dx_fam': 'fam_dx4',
#                   'cor_alc_dep_dx': 'dx4',
#                   'cor_ald5dx': 'dx5',
#                   'cor_alc_dep_max_sx_cnt': 'sxc4',
#                   'cor_ald5sx_max_cnt': 'sxc5'}

def_fields = {f:f for f in default_field_eponyms}
# def_fields.update(default_dx_sxc)

sex_to_parentfield = {'m':'fID', 'f':'mID', 'M':'fID', 'F':'mID'}
score_map = {'self': 1, 'twin': 1,
             'mother':0.5, 'father':0.5, 'fullsib':0.5, 'child':0.5,
             'mgm': 0.25, 'mgf': 0.25, 'fgm': 0.25, 'fgf': 0.25,
             'msib': 0.25, 'fsib': 0.25,
             'gp': 0.25, 'parentsib': 0.25, 'gc': 0.25, 'sibchild': 0.25,
             'halfsib': 0.25}

def conv_159(v):
    ''' convert COGA coding to regular coding '''
    if v==1:
        return 0
    elif v==5:
        return 1
    else:
        return np.nan

def harmonize_fields_max(row, field1, field2):
    ''' harmonize two columns by return the max of the two '''
    if row[field1]==row[field2] or \
        (np.isnan(row[field1]) and np.isnan(row[field2])):
        return row[field1]
    else:
        return max(row[field1], row[field2])

def harmonize_fields_left(row, field1, field2):
    ''' harmonize two columns such that field1 overrules field2, but field2 is accepted if field1 is missing '''
    if np.isnan(row['field1']):
        return row['field2']
    else:
        return row['field1']
    
def prepare_for_fhd(in_df, extra_cols=[], do_conv_159=True):
    ''' prepare a dataframe for FHD calculation by removing irrelevant columns
        and assuring the necessary columns are present '''

    # reduce to only necessary columns
    check_cols = list(default_field_eponyms)
    if extra_cols:
        check_cols.extend(extra_cols)
    drop_cols = [col for col in in_df.columns if col not in check_cols]
    df = in_df.drop(drop_cols, axis=1)

    # make sure indexed only by ID
    df.reset_index(inplace=True)
    df_tmp = df.set_index('ID', drop=False) # keep a copy of ID column for convenience
    g = df_tmp.groupby(level='ID') # make sure no duplicates by ID
    df = g.last() # note this means that affectedness data should have been joined on ID, or else it may get dropped

    if all(col in df.columns for col in check_cols):
        print('df has all requisite columns')
    else:
        print('df missing requisite columns')
        raise

    for col in extra_cols:
        if do_conv_159 and (0 not in df[col].unique()): # expecting column to be in the COGA 159 format here
            df[col] = df[col].apply(conv_159)
    
    return df


def prepare_dfs(in_sDF, in_fDF, aff_col='cor_alc_dep_dx', do_conv_159=True, rename_cols=None):

    if rename_cols:
        fd = def_fields.copy()
        fd.update(rename_cols)
        rename_dict = {k:v for k,v in fd.items()}
        fDF = in_fDF.rename(columns=rename_dict)

    sDF = prepare_for_fhd(in_sDF)
    fDF = prepare_for_fhd(in_fDF, [aff_col], do_conv_159)

    return sDF, fDF


def calc_fhd(in_sDF, in_fDF, aff_col='cor_alc_dep_dx', do_conv_159=True,
                        degrees=[1, 2], descend=False, cat_norm=True, rename_cols=None):
    ''' main function to apply to a DF '''
    sDF, fDF = prepare_dfs(in_sDF, in_fDF, aff_col, do_conv_159, rename_cols)
    
    # drop missing data of affectedness column from fDF
    # don't do this because they tell us about family structure even if they lack affectedness info
    # fDF = fDF.dropna(subset=[aff_col])

    sDF['fhd_dx4_ratio'], sDF['fhd_dx4_sum'], sDF['n_rels'] = \
            zip(*sDF.apply(calc_fhd_row, axis=1, args=[fDF, aff_col, degrees, descend, cat_norm]))

    return sDF
    
def calc_fhd_row(row, df, aff, degrees=[1, 2], descend=False, cat_norm=True):
    ''' row-wise apply function '''
    famDF = df[ df['famID'] == row['famID'] ]
    I = Individual(row, famDF)
    if 0 in degrees:
        I.zeroth_degree() # 1
    if 1 in degrees:
        I.first_degree(descend) # 0.5
    if 2 in degrees:
        I.second_degree(descend) # 0.25
    print('.', end='')
    I.count(aff)
    return I.ratio_score(aff, 1, cat_norm), I.sum_score(aff, 1, cat_norm), I.n_rels


def careful_add(d, k, v):
    if k not in d.keys():
        d[k] = v


famscore_map = {'Mpred':0.5, 'Fpred':0.5, 'sibs':0.5,
             'MpredMpred': 0.25, 'MpredFpred': 0.25,
             'FpredFpred': 0.25, 'FpredMpred': 0.25,
             'Mpredsibs': 0.25, 'Fpredsibs': 0.25,
             'hsibs': 0.25}

class Family:

    def __init__(s, famDF):
        s.df = famDF
        s.dx_dict = famDF['cor_alc_dep_dx'].dropna().to_dict()
        s.sires_dams()
        s.build_graph3()
        s.define_predecessors3()
        s.convert_IDpreddict3()
        s.calc_ratio2()

    def sires_dams(s):
        sire_dams = defaultdict(list)
        for i, sire, dam in s.df[['ID', 'fID', 'mID']].values:
            try:
                int(sire)
                int(dam)
                sire_dams[(sire, dam)].append(i)
            except ValueError:
                pass

        s.sire_dams = sire_dams

    def build_graph(s):

        G = nx.DiGraph()

        for ID, affectedness in s.df[['ID', 'cor_alc_dep_dx']].values:
            G.add_node(ID, a=affectedness)

        for sd, sib_lst in s.sire_dams.items():
            sire, dam = sd
            G.add_edges_from([(sire, sib) for sib in sib_lst])
            G.add_edges_from([(dam, sib) for sib in sib_lst])
            for sib1, sib2 in combinations(sib_lst, 2):
                G.add_edge(sib1, sib2)
                G.add_edge(sib2, sib1)

        s.G = G

    def build_graph2(s):

        G = nx.DiGraph()

        for ID, affectedness in s.df[['ID', 'cor_alc_dep_dx']].values:
            G.add_node(ID, a=affectedness)

        for sd, sib_lst in s.sire_dams.items():
            sire, dam = sd
            G.add_edges_from([(sire, sib) for sib in sib_lst])
            G.add_edges_from([(dam, sib) for sib in sib_lst])

        s.G = G

    def build_graph3(s):

        G = nx.DiGraph()

        for ID, sex, affectedness in s.df[['ID', 'sex', 'cor_alc_dep_dx']].values:
            G.add_node(ID, sex=sex, a=affectedness)

        for sd, sib_lst in s.sire_dams.items():
            sire, dam = sd
            G.add_edges_from([(sire, sib) for sib in sib_lst])
            G.add_edges_from([(dam, sib) for sib in sib_lst])

        s.G = G

    def define_predecessors(s):

        ID_preds_dict = defaultdict(dict)

        for node in s.G.nodes():
            for pred in s.G.predecessors(node):
                careful_add(ID_preds_dict[node], pred, 0.5)

        for node in s.G.nodes():
            for pred in s.G.predecessors(node):
                for pred_pred in s.G.predecessors(pred):
                    careful_add(ID_preds_dict[node], pred_pred, 0.25)
            try:
                del ID_preds_dict[node][node] # make sure the self does not count as predecessor
            except KeyError:
                pass

        s.ID_preds_dict = ID_preds_dict


    def define_predecessors2(s):

        ID_preds_dict = defaultdict(dict)

        for node in s.G.nodes():
            # predecessors (parents)
            for pred in s.G.predecessors(node):
                careful_add(ID_preds_dict[node], pred, 0.5)

        for node in s.G.nodes():
            for pred in s.G.predecessors(node):
                # predecessors of predecessors (grandparents)
                for pred_pred in s.G.predecessors(pred):
                    careful_add(ID_preds_dict[node], pred_pred, 0.25)
                    # successors of predecessors of predecessors (aunts and uncles)
                    for pred_pred_succ in s.G.successors(pred_pred):
                        careful_add(ID_preds_dict[node], pred_pred_succ, 0.25)
                # successors of predecessors (siblings)
                for pred_succ in s.G.successors(pred):
                    careful_add(ID_preds_dict[node], pred_succ, 0.5)
            try:
                del ID_preds_dict[node][node] # make sure the self does not count as predecessor
            except KeyError:
                pass

        s.ID_preds_dict = ID_preds_dict

    def define_predecessors3(s):

        ID_preds_dict = defaultdict(lambda: defaultdict(set))

        for node in s.G.nodes():

            # predecessors (parents)
            for pred in s.G.predecessors(node):
                try:
                    pred_sex = s.G.node[pred]['sex']
                except KeyError:
                    pred_sex = '?'
                pred_lbl = pred_sex + 'pred'
                ID_preds_dict[node][pred_lbl].add(pred)

                # predecessors of predecessors (grandparents)
                for pred_pred in s.G.predecessors(pred):
                    try:
                        predpred_sex = s.G.node[pred_pred]['sex']
                    except KeyError:
                        predpred_sex = '?'
                    predpred_lbl = pred_lbl + predpred_sex + 'pred'
                    ID_preds_dict[node][predpred_lbl].add(pred_pred)

                    # successors of predecessors of predecessors (aunts and uncles)
                    for pred_pred_succ in s.G.successors(pred_pred):
                        predpredsucc_lbl = predpred_lbl + 'succ'
                        ID_preds_dict[node][predpredsucc_lbl].add(pred_pred_succ)

                # successors of predecessors (siblings)
                for pred_succ in s.G.successors(pred):
                    predsucc_lbl = pred_lbl + 'succ'
                    ID_preds_dict[node][predsucc_lbl].add(pred_succ)

        s.ID_preds_dict = ID_preds_dict


    def convert_IDpreddict3(s):

        ID_preds_dict_conv = s.ID_preds_dict.copy()

        for ID, cat_dict in s.ID_preds_dict.items():
            
            ID_preds_dict_conv[ID]['sibs'] = cat_dict['Fpredsucc'] & cat_dict['Mpredsucc']
            try:
                ID_preds_dict_conv[ID]['sibs'].remove(ID)
            except KeyError:
                pass
            ID_preds_dict_conv[ID]['hsibs'] = cat_dict['Fpredsucc'] ^ cat_dict['Mpredsucc']
            ID_preds_dict_conv[ID]['Fpredsibs'] = cat_dict['FpredFpredsucc'] & cat_dict['FpredMpredsucc']
            try:
                ID_preds_dict_conv[ID]['Fpredsibs'].remove(next(iter(ID_preds_dict_conv[ID]['Fpred'])))
            except:
                pass
            ID_preds_dict_conv[ID]['Mpredsibs'] = cat_dict['MpredFpredsucc'] & cat_dict['MpredMpredsucc']
            try:
                ID_preds_dict_conv[ID]['Mpredsibs'].remove(next(iter(ID_preds_dict_conv[ID]['Mpred'])))
            except:
                pass
            
            del ID_preds_dict_conv[ID]['Fpredsucc']
            del ID_preds_dict_conv[ID]['Mpredsucc']
            del ID_preds_dict_conv[ID]['FpredFpredsucc']
            del ID_preds_dict_conv[ID]['FpredMpredsucc']
            del ID_preds_dict_conv[ID]['MpredFpredsucc']
            del ID_preds_dict_conv[ID]['MpredMpredsucc']

        s.ID_preds_dict_conv = ID_preds_dict_conv


    def calc_ratio2(s):

        fhdratio_dict = dict()
        fhdsum_dict = dict()
        count_dict = dict()

        for ID, cat_dict in s.ID_preds_dict_conv.items():
            fhd_num = 0
            fhd_denom = 0
            rel_count = 0

            for cat, pred_set in cat_dict.items():
                try:
                    weight = famscore_map[cat]
                except KeyError:
                    continue
                tmp_fhd_num = 0
                tmp_count = 0
                for pred in pred_set:
                    try:
                        tmp_fhd_num += s.dx_dict[pred]
                        tmp_count += 1
                    except KeyError:
                        pass
                try:
                    fhd_num += (tmp_fhd_num / tmp_count) * weight
                    fhd_denom += weight
                    rel_count += tmp_count
                except ZeroDivisionError:
                    pass

            count_dict[ID] = rel_count
            try:
                fhdratio_dict[ID] = fhd_num / fhd_denom
                fhdsum_dict[ID] = fhd_num
            except ZeroDivisionError:
                pass

        s.fhdratio_dict = fhdratio_dict
        s.fhdsum_dict = fhdsum_dict
        s.count_dict = count_dict


    def calc_ratio(s):

        fhdratio_dict = dict()
        fhdsum_dict = dict()
        count_dict = dict()

        for ID, pred_dict in s.ID_preds_dict.items():
            fhd_num = 0
            fhd_denom = 0
            rel_count = 0
            for pred, weight in pred_dict.items():
                try:
                    fhd_num += weight * s.dx_dict[pred]
                    fhd_denom += weight
                    rel_count += 1
                except KeyError:
                    pass
            count_dict[ID] = rel_count
            try:
                fhdratio_dict[ID] = fhd_num / fhd_denom
                fhdsum_dict[ID] = fhd_num
            except ZeroDivisionError:
                pass

        s.fhdratio_dict = fhdratio_dict
        s.fhdsum_dict = fhdsum_dict
        s.count_dict = count_dict


class Individual:
    ''' represents one person, can calculate density of some affectedness
        based on family members '''

    def __init__(s, row, fam_df):
        s.info = row
        s.ID = row['ID']
        s.df = fam_df
        try:
            s.parent_field = sex_to_parentfield[row['sex']]
        except KeyError:
            s.parent_field = None

        s.rel_set = set()
        s.rel_dict = defaultdict(list)

    def add_rel(s, ID, relation):
        ''' add a relative to the dict of relatives '''
        if ID != s.ID and ID not in s.rel_set and isinstance(ID, str):
            s.rel_set.update({ID})
            s.rel_dict[relation].append(ID)

    def ratio_score(s, ckfield, ckfield_max, cat_norm=True):
        ''' score FHD as a ratio '''
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

    def ratio_score_rob(s, ckfield, ckfield_max, cat_norm=True, prop=0.3):
        ''' score FHD as a ratio, removing a random proportion of scores '''
        n_rels_known = s.n_rels
        remove_n = round(n_rels_known * prop)
        remove_lst = [False]*remove_n + [True]*(n_rels_known - remove_n)
        np.random.shuffle(remove_lst)
        remove_lst_iter = iter(remove_lst)

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
                        if next(remove_lst_iter):
                            tmp_score += weight * rel_val/ckfield_max
                            tmp_count += 1
                        else:
                            continue
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
        ''' score FHD as a sum '''
        score_num = known_count = 0
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
                        known_count += 1
                except: # subject not found at all
                    pass
            if cat_norm: # normalize in each relative category
                if tmp_count > 0:
                    score_num += tmp_score / tmp_count
            else:
                score_num += tmp_score
        if known_count == 0: # if nothing known
            return np.nan
        else:
            return score_num

    def sum_score_rob(s, ckfield, ckfield_max, cat_norm=True, prop=0.3):
        ''' score FHD as a sum, removing a random proportion of scores '''
        n_rels_known = s.n_rels
        remove_n = round(n_rels_known * prop)
        remove_lst = [False]*remove_n + [True]*(n_rels_known - remove_n)
        np.random.shuffle(remove_lst)
        remove_lst_iter = iter(remove_lst)

        score_num = known_count = 0
        for rel_type, rel_IDs in s.rel_dict.items():
            weight = score_map[rel_type]
            tmp_score = tmp_count = 0
            for rel_ID in rel_IDs:
                try:
                    rel_val = s.df.loc[rel_ID, ckfield]
                    if np.isnan(rel_val): # subject found, missing value
                        continue
                    else: # subject found, has value
                        if next(remove_lst_iter):
                            tmp_score += weight * rel_val/ckfield_max
                            tmp_count += 1
                            known_count += 1
                        else:
                            continue
                except: # subject not found at all
                    pass
            if cat_norm: # normalize in each relative category
                if tmp_count > 0:
                    score_num += tmp_score / tmp_count
            else:
                score_num += tmp_score
        if known_count == 0: # if nothing known
            return np.nan
        else:
            return score_num

    def count(s, ckfield):
        ''' count relatives '''
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
        s.n_rels = count

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