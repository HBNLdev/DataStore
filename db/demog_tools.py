''' tools for demographic info, mainly calculating family history density '''

from collections import defaultdict
from itertools import combinations

import graphviz as gv
import networkx as nx
import numpy as np
import pandas as pd

from .utils.compilation import conv_159

default_field_eponyms = {'ID', 'famID', 'mID', 'fID', 'sex', 'twin'}
# default_dx_sxc = {'cor_alc_dep_dx_fam': 'fam_dx4',
#                   'cor_alc_dep_dx': 'dx4',
#                   'cor_ald5dx': 'dx5',
#                   'cor_alc_dep_max_sx_cnt': 'sxc4',
#                   'cor_ald5sx_max_cnt': 'sxc5'}

def_fields = {f: f for f in default_field_eponyms}
# def_fields.update(default_dx_sxc)

sex_to_parentfield = {'m': 'fID', 'f': 'mID', 'M': 'fID', 'F': 'mID'}
indscore_map = {'self': 1, 'twin': 1,
                'mother': 0.5, 'father': 0.5, 'fullsib': 0.5, 'child': 0.5,
                'mgm': 0.25, 'mgf': 0.25, 'fgm': 0.25, 'fgf': 0.25,
                'msib': 0.25, 'fsib': 0.25,
                'gp': 0.25, 'parentsib': 0.25, 'gc': 0.25, 'sibchild': 0.25,
                'halfsib': 0.25}

# here, M means male, F means female, pred means predecessor, and sib means sibling
famscore_map = {'Mpred': 0.5, 'Fpred': 0.5, 'sibs': 0.5,
                'MpredMpred': 0.25, 'MpredFpred': 0.25,
                'FpredFpred': 0.25, 'FpredMpred': 0.25,
                'Mpredsibs': 0.25, 'Fpredsibs': 0.25,
                'hsibs': 0.25}


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
    df_tmp = df.set_index('ID', drop=False)  # keep a copy of ID column for convenience
    g = df_tmp.groupby(level='ID')  # make sure no duplicates by ID
    df = g.last()  # note this means that affectedness data should have been joined on ID, or else it may get dropped

    if all(col in df.columns for col in check_cols):
        print('df has all requisite columns')
    else:
        missing_cols = [col for col in check_cols if col not in df.columns]
        print(missing_cols, 'are missing')
        return

    for col in extra_cols:
        if do_conv_159 and (0 not in df[col].unique()):  # expecting column to be in the COGA 159 format here
            df[col] = df[col].apply(conv_159)

    return df


def prepare_dfs(in_sDF, in_fDF, aff_col='cor_alc_dep_dx', do_conv_159=True, rename_cols=None):
    ''' prepare the main dataframes used by calc_fhd functions so that they are fit for usage! '''

    if rename_cols:
        fd = def_fields.copy()
        fd.update(rename_cols)
        rename_dict = {k: v for k, v in fd.items()}
        in_fDF = in_fDF.rename(columns=rename_dict)

    sDF = prepare_for_fhd(in_sDF)
    fDF = prepare_for_fhd(in_fDF, extra_cols=[aff_col], do_conv_159=do_conv_159)

    return sDF, fDF


def calc_fhd(in_sDF, in_fDF, aff_col='cor_alc_dep_dx', do_conv_159=True,
             degrees=[1, 2], descend=False, cat_norm=True, rename_cols=None):
    '''

    inputs:

    in_sDF:     a dataframe indexed by ID or ID + session, including the individuals for whom you want to calculate FHD
    in_fDF:     a dataframe indexed by ID, including the whole family of the individuals in in_sDF. must include
                    'ID', 'famID', 'mID', 'fID', 'sex', and aff_col
    aff_col:    the affectedness column for which to calculate density
    conv_159:   if True, convert aff_col from a (1, 5, 9) to a (0, 1, nan) coding
    degrees:    the degrees of relatives to include in density calculations. 1 = primary, 2 = secondary
    descend:    if True, include descendants in density calculations (i.e. children)
    cat_norm:   if True, normalize the score from potentially large categories to not exceed its maximum
                    (this mainly refers to sibling categories)
    rename_cols: a dict mapping old column names to new column names if the DFs use a different naming scheme

    outputs:

    sDF:        a copy of in_sDF with 3 columns added:
                    fhd_dx4_ratio   ratio of family history density
                    fhd_dx4_sum     sum of family history density
                    n_rels          number of relatives for whom affectedness was known

    '''

    sDF, fDF = prepare_dfs(in_sDF, in_fDF, aff_col, do_conv_159, rename_cols)

    # drop missing data of affectedness column from fDF
    # don't do this because they tell us about family structure even if they lack affectedness info
    # fDF = fDF.dropna(subset=[aff_col])

    sDF['fhd_dx4_ratio'], sDF['fhd_dx4_sum'], sDF['n_rels'] = \
        zip(*sDF.apply(calc_fhd_row, axis=1, args=[fDF, aff_col, degrees, descend, cat_norm]))

    return sDF


def calc_fhd_fast(in_sDF, in_fDF, aff_col='cor_alc_dep_dx', do_conv_159=True, rename_cols=None):
    '''

    fast version of calc_fhd that always uses only primary and secondary forebears,
    and normalizes the score within each sibling category

    inputs:

    in_sDF:     a dataframe indexed by ID or ID + session, including the individuals for whom you want to calculate FHD
    in_fDF:     a dataframe indexed by ID, including the whole family of the individuals in in_sDF. must include
                    'ID', 'famID', 'mID', 'fID', 'sex', and aff_col
    aff_col:    the affectedness column for which to calculate density
    conv_159:   if True, convert aff_col from a (1, 5, 9) to a (0, 1, nan) coding
    rename_cols: a dict mapping input column names to new column names if the DFs use a different naming scheme

    outputs:

    sDF:        a copy of in_sDF with 3 columns added:
                    fhd_dx4_ratio   ratio of family history density
                    fhd_dx4_sum     sum of family history density
                    n_rels          number of relatives for whom affectedness was known

    '''

    sDF, fDF = prepare_dfs(in_sDF, in_fDF, aff_col, do_conv_159, rename_cols)

    all_counts, all_sums, all_ratios = fam_fhd(fDF)

    count_series = pd.Series(all_counts, name='n_rels')
    sum_series = pd.Series(all_sums, name='fhd_dx4_sum')
    ratio_series = pd.Series(all_ratios, name='fhd_dx4_ratio')

    sDF['fhd_dx4_ratio'] = ratio_series
    sDF['fhd_dx4_sum'] = sum_series
    sDF['n_rels'] = count_series

    sDF.ix[sDF['n_rels'].isnull(), 'n_rels'] = 0

    return sDF


def calc_fhd_fast2(in_sDF, in_fDF, aff_col='cor_alc_dep_dx', do_conv_159=True, rename_cols=None):
    '''

    second version of calc_fhd_fast that uses a different algorithm

    '''

    sDF, fDF = prepare_dfs(in_sDF, in_fDF, aff_col, do_conv_159, rename_cols)

    all_counts, all_sums, all_ratios = fam_fhd2(fDF)

    count_series = pd.Series(all_counts, name='n_rels')
    sum_series = pd.Series(all_sums, name='fhd_dx4_sum')
    ratio_series = pd.Series(all_ratios, name='fhd_dx4_ratio')

    sDF['fhd_dx4_ratio'] = ratio_series
    sDF['fhd_dx4_sum'] = sum_series
    sDF['n_rels'] = count_series

    sDF.ix[sDF['n_rels'].isnull(), 'n_rels'] = 0

    return sDF


def fam_fhd(fDF):
    ''' given a family DF, return dictionaries mapping the IDs of its members to the number of relatives for whom
        affectedness is known, the FHD sum scores, and the FHD ratio scores '''

    all_counts = dict()
    all_sums = dict()
    all_ratios = dict()

    fams = fDF['famID'].unique()

    for fam in fams:
        famDF = fDF[fDF['famID'] == fam]
        famO = Family(famDF)
        famO.define_rels()
        famO.calc_famfhd()
        all_counts.update(famO.count_dict)
        all_sums.update(famO.fhdsum_dict)
        all_ratios.update(famO.fhdratio_dict)

    return all_counts, all_sums, all_ratios


def fam_fhd2(fDF):
    ''' given a family DF, return dictionaries mapping the IDs of its members to the number of relatives for whom
        affectedness is known, the FHD sum scores, and the FHD ratio scores '''

    all_counts = dict()
    all_sums = dict()
    all_ratios = dict()

    fams = fDF['famID'].unique()

    for fam in fams:
        famDF = fDF[fDF['famID'] == fam]
        famO = Family(famDF)
        famO.G2 = famO.build_graph2()
        famO.calc_famfhd2()
        all_counts.update(famO.count_dict)
        all_sums.update(famO.fhdsum_dict)
        all_ratios.update(famO.fhdratio_dict)

    return all_counts, all_sums, all_ratios


def calc_fhd_row(row, df, aff, degrees=[1, 2], descend=False, cat_norm=True):
    ''' row-wise apply function for calc_fhd '''
    famDF = df[df['famID'] == row['famID']]
    I = Individual(row, famDF)
    if 0 in degrees:
        I.zeroth_degree()  # 1
    if 1 in degrees:
        I.first_degree(descend)  # 0.5
    if 2 in degrees:
        I.second_degree(descend)  # 0.25
    print('.', end='')
    I.count(aff)
    return I.ratio_score(aff, 1, cat_norm), I.sum_score(aff, 1, cat_norm), I.n_rels


# attributes for the graphviz pedigree figures
sex_shape = {'M': 'square', 'F': 'circle', '?': 'polygon'}
dx_fillcolor = {0: '#FFFFFF', 1: '#000000'}
dx_wordcolor = {0: '#000000', 1: '#FFFFFF'}


class Family:
    ''' given a famDF containing columns of ID, sex ('M', 'F'), fatherID, and motherID
        represents a family as a directed graph '''

    def __init__(s, famDF):
        s.df = famDF
        s.dx_dict = famDF['cor_alc_dep_dx'].dropna().to_dict()
        s.G = s.build_graph()
        # s.G2 = s.build_graph2()

    def build_graph(s):
        ''' builds the graph. individuals are nodes, and edges are directed from parents to children. '''

        G = nx.DiGraph()

        for ID, IDsex, fID, mID in s.df[['ID', 'sex', 'fID', 'mID']].values:
            G.add_node(ID, sex=IDsex)
            # G.add_node(fID, sex='M')
            # G.add_node(mID, sex='F')
            G.add_edge(fID, ID)
            G.add_edge(mID, ID)

        return G

    def calc_sires_dams(s):
        sire_dams = defaultdict(list)
        for i, sire, dam in s.df[['ID', 'fID', 'mID']].values:
            try:
                int(sire)
                int(dam)
                sire_dams[(sire, dam)].append(i)
            except ValueError:
                pass

        s.sire_dams = sire_dams

    def build_graph2(s):

        s.calc_sires_dams()

        G = nx.Graph()

        for ID, affectedness in s.df[['ID', 'cor_alc_dep_dx']].values:
            G.add_node(ID, a=affectedness)

        for sd, sib_lst in s.sire_dams.items():
            sire, dam = sd
            G.add_edges_from([(sire, sib) for sib in sib_lst])
            G.add_edges_from([(dam, sib) for sib in sib_lst])
            for sib1, sib2 in combinations(sib_lst, 2):
                G.add_edge(sib1, sib2)

        return G

    def build_graph_gv(s):
        ''' builds the graph in graphviz (for plotting) '''

        D = gv.Digraph(format='svg')

        for ID, fID, mID, sex in s.df[['ID', 'fID', 'mID', 'sex']].values:
            try:
                fill_color = dx_fillcolor[s.dx_dict[ID]]
                word_color = dx_wordcolor[s.dx_dict[ID]]
                style = 'filled'
            except KeyError:
                fill_color = '#808080'
                word_color = '#000000'
                style = 'filled,dashed'
            D.node(ID, label=ID, shape=sex_shape[sex], fontcolor=word_color,
                   fillcolor=fill_color, style=style)
            # D.node(ID, label='', shape=sex_shape[sex], fontcolor=word_color,
            # fillcolor=fill_color, style=style)
            try:
                # D.node(fID, label=fID, shape='square')
                # D.node(mID, label=mID, shape='circle')
                # D.node(fID, label='', shape='square')
                # D.node(mID, label='', shape='square')
                D.edge(fID, ID)
                D.edge(mID, ID)
            except AttributeError:
                pass

        return D

    def indiv_subset(s, indiv_ID):

        if 'ID_rels_dict_conv' not in dir(s):
            s.define_rels()

        indiv_reldict = s.ID_rels_dict_conv[indiv_ID]

        indiv_rels = {indiv_ID}
        for rel_cat, cat_set in indiv_reldict.items():
            for rel in cat_set:
                try:  # fast test for nan
                    rel + '1'
                except TypeError:
                    continue
                indiv_rels.add(rel)

        return indiv_rels

    def build_graph_gv_indiv(s, indiv_ID):
        ''' builds the graph in graphviz (for plotting) '''

        indiv_rels = s.indiv_subset(indiv_ID)
        indiv_df = s.df.loc[indiv_rels, :]

        D = gv.Digraph(format='svg')

        for ID, fID, mID, sex in indiv_df[['ID', 'fID', 'mID', 'sex']].values:
            try:
                fill_color = dx_fillcolor[s.dx_dict[ID]]
                word_color = dx_wordcolor[s.dx_dict[ID]]
                style = 'filled'
            except KeyError:
                fill_color = '#808080'
                word_color = '#000000'
                style = 'filled,dashed'
            D.node(ID, label=ID, shape=sex_shape[sex], fontcolor=word_color,
                   fillcolor=fill_color, style=style)
            # D.node(ID, label='', shape=sex_shape[sex], fontcolor=word_color,
            # fillcolor=fill_color, style=style)
            try:
                # D.node(fID, label=fID, shape='square')
                # D.node(mID, label=mID, shape='circle')
                # D.node(fID, label='', shape='square')
                # D.node(mID, label='', shape='square')
                D.edge(fID, ID)
                D.edge(mID, ID)
            except AttributeError:
                pass

        return D

    def define_rels(s):
        ''' builds a dict of dicts of sets in which:
                - outer keys are IDs
                - inner keys are relative categories
                - inner values are sets of IDs that belong in that category '''

        ID_rels_dict = defaultdict(lambda: defaultdict(set))

        for node in s.G.nodes():

            # predecessors (parents)
            for pred in s.G.predecessors(node):
                try:
                    pred_sex = s.G.node[pred]['sex']
                except KeyError:
                    pred_sex = '?'
                try:
                    pred_lbl = pred_sex + 'pred'
                except TypeError:
                    pred_lbl = '?pred'
                ID_rels_dict[node][pred_lbl].add(pred)

                # predecessors of predecessors (grandparents)
                for pred_pred in s.G.predecessors(pred):
                    try:
                        predpred_sex = s.G.node[pred_pred]['sex']
                    except KeyError:
                        predpred_sex = '?'
                    predpred_lbl = pred_lbl + predpred_sex + 'pred'
                    ID_rels_dict[node][predpred_lbl].add(pred_pred)

                    # successors of predecessors of predecessors (aunts and uncles)
                    for pred_pred_succ in s.G.successors(pred_pred):
                        predpredsucc_lbl = predpred_lbl + 'succ'
                        ID_rels_dict[node][predpredsucc_lbl].add(pred_pred_succ)

                # successors of predecessors (siblings)
                for pred_succ in s.G.successors(pred):
                    predsucc_lbl = pred_lbl + 'succ'
                    ID_rels_dict[node][predsucc_lbl].add(pred_succ)

        s.ID_rels_dict = ID_rels_dict

        s.convert_IDrelsdict()

    def convert_IDrelsdict(s):

        ''' convert the rels dict by collapsing categories containing successors of parents and grandparents
            into sibling, half-sibling, and parental sibling categories '''

        ID_rels_dict_conv = s.ID_rels_dict.copy()

        for ID, cat_dict in s.ID_rels_dict.items():

            ID_rels_dict_conv[ID]['sibs'] = cat_dict['Fpredsucc'] & cat_dict['Mpredsucc']
            try:
                ID_rels_dict_conv[ID]['sibs'].remove(ID)
            except KeyError:
                pass
            ID_rels_dict_conv[ID]['hsibs'] = cat_dict['Fpredsucc'] ^ cat_dict['Mpredsucc']
            ID_rels_dict_conv[ID]['Fpredsibs'] = cat_dict['FpredFpredsucc'] & cat_dict['FpredMpredsucc']
            try:
                ID_rels_dict_conv[ID]['Fpredsibs'].remove(next(iter(ID_rels_dict_conv[ID]['Fpred'])))
            except:
                pass
            ID_rels_dict_conv[ID]['Mpredsibs'] = cat_dict['MpredFpredsucc'] & cat_dict['MpredMpredsucc']
            try:
                ID_rels_dict_conv[ID]['Mpredsibs'].remove(next(iter(ID_rels_dict_conv[ID]['Mpred'])))
            except:
                pass

            del ID_rels_dict_conv[ID]['Fpredsucc']
            del ID_rels_dict_conv[ID]['Mpredsucc']
            del ID_rels_dict_conv[ID]['FpredFpredsucc']
            del ID_rels_dict_conv[ID]['FpredMpredsucc']
            del ID_rels_dict_conv[ID]['MpredFpredsucc']
            del ID_rels_dict_conv[ID]['MpredMpredsucc']

        s.ID_rels_dict_conv = ID_rels_dict_conv

    def calc_famfhd(s):

        ''' using converted rels dict, calculate FHD creating 3 dicts which contain the results '''

        fhdratio_dict = dict()
        fhdsum_dict = dict()
        count_dict = dict()

        for ID, cat_dict in s.ID_rels_dict_conv.items():

            fhd_num = 0
            fhd_denom = 0
            rel_count = 0

            for cat, pred_set in cat_dict.items():
                if not pred_set:
                    continue
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

    def calc_famfhd2(s):

        ''' using converted rels dict, calculate FHD creating 3 dicts which contain the results '''

        fhdratio_dict = dict()
        fhdsum_dict = dict()
        count_dict = dict()

        for ID in s.G2.nodes():

            fhd_num = 0
            fhd_denom = 0
            rel_count = 0

            for rel, dx in s.dx_dict.items():

                try:
                    distance = nx.shortest_path_length(s.G2, source=ID, target=rel)
                except nx.NetworkXNoPath:
                    continue
                weight = 1 / (2 ** distance)
                val = weight * dx
                # print(rel, distance, dx, weight, val)

                if distance > 0:
                    fhd_num += val
                    fhd_denom += weight
                    rel_count += 1

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
            weight = indscore_map[rel_type]
            tmp_score = tmp_count = 0
            for rel_ID in rel_IDs:
                try:
                    rel_val = s.df.loc[rel_ID, ckfield]
                    if np.isnan(rel_val):  # subject found, missing value
                        continue
                    else:  # subject found, has value
                        tmp_score += weight * rel_val / ckfield_max
                        tmp_count += 1
                except:  # subject not found a-t all
                    pass
            if cat_norm:  # normalize in each relative category
                if tmp_count > 0:
                    score_num += tmp_score / tmp_count
                    score_denom += weight
            else:
                score_num += tmp_score
                score_denom += weight * tmp_count
        if score_denom == 0:  # no rels known or nothing known about rels
            return np.nan
        else:
            return score_num / score_denom

    def ratio_score_rob(s, ckfield, ckfield_max, cat_norm=True, prop=0.3):
        ''' score FHD as a ratio, removing a random proportion of scores '''
        n_rels_known = s.n_rels
        remove_n = round(n_rels_known * prop)
        remove_lst = [False] * remove_n + [True] * (n_rels_known - remove_n)
        np.random.shuffle(remove_lst)
        remove_lst_iter = iter(remove_lst)

        score_num = score_denom = 0
        for rel_type, rel_IDs in s.rel_dict.items():
            weight = indscore_map[rel_type]
            tmp_score = tmp_count = 0
            for rel_ID in rel_IDs:
                try:
                    rel_val = s.df.loc[rel_ID, ckfield]
                    if np.isnan(rel_val):  # subject found, missing value
                        continue
                    else:  # subject found, has value
                        if next(remove_lst_iter):
                            tmp_score += weight * rel_val / ckfield_max
                            tmp_count += 1
                        else:
                            continue
                except:  # subject not found at all
                    pass
            if cat_norm:  # normalize in each relative category
                if tmp_count > 0:
                    score_num += tmp_score / tmp_count
                    score_denom += weight
            else:
                score_num += tmp_score
                score_denom += weight * tmp_count
        if score_denom == 0:  # no rels known or nothing known about rels
            return np.nan
        else:
            return score_num / score_denom

    def sum_score(s, ckfield, ckfield_max, cat_norm=True):
        ''' score FHD as a sum '''
        score_num = known_count = 0
        for rel_type, rel_IDs in s.rel_dict.items():
            weight = indscore_map[rel_type]
            tmp_score = tmp_count = 0
            for rel_ID in rel_IDs:
                try:
                    rel_val = s.df.loc[rel_ID, ckfield]
                    if np.isnan(rel_val):  # subject found, missing value
                        continue
                    else:  # subject found, has value
                        tmp_score += weight * rel_val / ckfield_max
                        tmp_count += 1
                        known_count += 1
                except:  # subject not found at all
                    pass
            if cat_norm:  # normalize in each relative category
                if tmp_count > 0:
                    score_num += tmp_score / tmp_count
            else:
                score_num += tmp_score
        if known_count == 0:  # if nothing known
            return np.nan
        else:
            return score_num

    def sum_score_rob(s, ckfield, ckfield_max, cat_norm=True, prop=0.3):
        ''' score FHD as a sum, removing a random proportion of scores '''
        n_rels_known = s.n_rels
        remove_n = round(n_rels_known * prop)
        remove_lst = [False] * remove_n + [True] * (n_rels_known - remove_n)
        np.random.shuffle(remove_lst)
        remove_lst_iter = iter(remove_lst)

        score_num = known_count = 0
        for rel_type, rel_IDs in s.rel_dict.items():
            weight = indscore_map[rel_type]
            tmp_score = tmp_count = 0
            for rel_ID in rel_IDs:
                try:
                    rel_val = s.df.loc[rel_ID, ckfield]
                    if np.isnan(rel_val):  # subject found, missing value
                        continue
                    else:  # subject found, has value
                        if next(remove_lst_iter):
                            tmp_score += weight * rel_val / ckfield_max
                            tmp_count += 1
                            known_count += 1
                        else:
                            continue
                except:  # subject not found at all
                    pass
            if cat_norm:  # normalize in each relative category
                if tmp_count > 0:
                    score_num += tmp_score / tmp_count
            else:
                score_num += tmp_score
        if known_count == 0:  # if nothing known
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
                    if np.isnan(rel_val):  # subject found, missing value
                        continue
                    else:  # subject found, has value
                        count += 1
                except:  # subject not found at all
                    pass
        s.n_rels = count

    def zeroth_degree(s):  # worth 1
        s.add_rel(s.info['ID'], 'self')  # self
        if s.info['twin'] != 2:
            return
        twinDF = s.df[s.df['twin'] == 2]  # twin
        for twin_ID, twin in twinDF.iterrows():
            if s.info['fID'] == twin['fID'] and s.info['mID'] == twin['mID']:
                s.add_rel(twin_ID, 'twin')

    def first_degree(s, descend=False):  # worth 0.5
        # parents
        s.add_rel(s.info['mID'], 'mother')
        s.add_rel(s.info['fID'], 'father')

        # full siblings
        if s.rel_dict['mother'] and s.rel_dict['father']:
            fullsibDF = s.df[(s.df['mID'] == s.rel_dict['mother'][0]) &
                             (s.df['fID'] == s.rel_dict['father'][0])]
            for fullsib_ID, fullsib in fullsibDF.iterrows():
                s.add_rel(fullsib_ID, 'fullsib')

        if descend:
            # children
            if s.parent_field:
                childDF = s.df[s.df[s.parent_field] == s.info['ID']]
                for child_ID, child in childDF.iterrows():
                    s.add_rel(child_ID, 'child')

    def second_degree(s, descend=False):

        for name in ['mother', 'father']:
            if s.rel_dict[name]:
                try:
                    parentrow = s.df.loc[s.rel_dict[name][0], :]
                except:
                    continue
                # grandparents
                s.add_rel(parentrow['mID'], name[0] + 'gm')
                s.add_rel(parentrow['fID'], name[0] + 'gf')
                # full siblings of parents (aunts/uncles)
                parentfsibDF = s.df[(s.df['mID'] == parentrow['mID']) &
                                    (s.df['fID'] == parentrow['fID'])]
                for parentfsib_ID, parentfsib in parentfsibDF.iterrows():
                    s.add_rel(parentfsib_ID, name[0] + 'sib')

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
        halfsibDF = s.df[(s.df['mID'] == s.info['mID']) |
                         (s.df['fID'] == s.info['fID'])]
        for halfsib_ID, halfsib in halfsibDF.iterrows():
            s.add_rel(halfsib_ID, 'halfsib')
