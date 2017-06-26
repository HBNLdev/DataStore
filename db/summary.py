import os
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt
from pandas.tools.plotting import table

from db.utils.text import multi_filter

# Info for labelling, etc
# Race and ethnicity definitions
count_field = 'uID'

RaceEthnicityDefs = {'hispanic': {'h': 'hispanic', 'n': 'non-hispanic', 'u': 'unknown'},
                     'core': {'1': 'Indigeneous American', '2': 'Asian', '3': 'Pacific Islander',
                              '4': 'Black', '6': 'White', '8': 'unknown', '9': 'mixed/other'}}

DiagSymp = {'ssa_ALD4D1': 'Tolerance',
            'ssa_ALD4D2': 'Withdrawal ',
            'ssa_ALD4D3': 'More than intended ',
            'ssa_ALD4D4': 'desire to cut down ',
            'ssa_ALD4D5': 'Excessive time used ',
            'ssa_ALD4D6': 'Use dominates life ',
            'ssa_ALD4D7': 'physical/psych. probs'}

QtyVars = {'ssa_AL4a': 'Weeks drank \n last 6 mo',
           'ssa_AL8d': 'Consecutive \n "heavy week"',
           'ssa_AL6': 'Max drinks (24-hour) \n lifetime',
           'ssa_AL6a': 'Max drinks (24-hour) \n 6 months',
           'ssa_AL8a': 'Max daily drinks \n in "heavy week"',
           'ssa_AL3_ws': 'Drinks wk \n before interview',
           'ssa_AL4_ws': 'Drinks typical wk \n last 6 mo',
           'ssa_AL4_p6ms': 'Total drinks \n last 6 mo',
           'ssa_AL3_da': 'Av drinks/day \n in prev wk',
           'ssa_AL4_da': 'Av drinks/day \n in last 6 mo',
           'cor_max_dpw': 'drinks in typical wk \n (all ints)',
           'cor_max_dpw_pwk': 'drinks in prev wk \n (all ints)',
           'cor_max_drinks': 'Max drinks \n (24-hour): life',
           'ssa_AL16b': 'Number of binges',
           'ssa_AL17b': 'Number of \n blackouts',
           'ssa_AL43A': 'Number of \n 3 mo abstinence',
           'ssa_AL1': 'Ever drank \n (at int)',
           'ssa_AL8': 'Ever \n "heavy drinking wk"',
           'cor_ever_drink': 'Ever drank \n (all ints)',
           'cor_ever_got_drunk': 'Ever been drunk',
           'cor_regular_drinking': 'Ever drank 1/mo \n for 6 mo'}
DrinkingVars = DiagSymp.copy()
DrinkingVars.update(QtyVars)


# Utilities
def expand_RE_aliases(aliases, race_type):
    def trans(code):
        tr = RaceEthnicityDefs['hispanic'][code[0]]
        tr += ' ' + RaceEthnicityDefs[race_type][code[1]]
        return tr

    return [trans(a) for a in aliases]


def nums_for_labels(labels, property, dataframe):
    countD = dataframe.groupby(property).count()[count_field].to_dict()
    countD = { str(k):v for k,v in countD.items() }
    return [l + '\n(' + str(countD[l]) + ')' for l in labels]


def update_tick_labels(plot, axis, trFun, trIns=None):
    label_lk = {'x': plot.get_xticklabels,
                'y': plot.get_yticklabels}
    labels = [l.get_text() for l in label_lk[axis]()]
    if trIns:
        updated = trFun(labels, **trIns)
    else:
        updated = trFun(labels)

    ud_lk = {'x': plot.set_xticklabels,
             'y': plot.set_yticklabels}
    ud_lk[axis](updated)
    return plot


def session_counts(df):
    drop = 'ID' in df.columns
    df = df.reset_index( drop = drop)
    df = df.set_index(['ID', 'session'])
    ses_counts = df.groupby(level=0).count()
    return ses_counts


# data cleaning
def replace_outs(df, Nstds):
    df[np.abs(df - df.mean()) > Nstds * df.std()] = np.nan
    return df


def clean_df(df, Nstds, numeric_cols):
    df['group_all'] = 1
    df.loc[:, numeric_cols] = df.groupby('group_all') \
        .transform(lambda g: replace_outs(g, Nstds))


def find_num_cols(df):
    num_types = [np.float64, float, int]
    col_ck_types = {}
    for c in df.columns:
        vals = df[c].tolist()
        try:
            typ = type(vals[-1] - vals[0])
        except:
            typ = None
        col_ck_types[c] = typ

    num_cols = [c for c, t in col_ck_types.items() if t in num_types]
    return num_cols


# groupwise utils
def categorize_cols(cols, delim='_'):
    delim_cols = [c for c in cols if delim in c]
    delim_pre = set([c.split(delim)[0] for c in delim_cols])
    col_groups = {}
    for pre in delim_pre:
        st_len = len(pre) + 1
        pre_cols = [c for c in delim_cols if c[:st_len] == pre + delim]
        col_groups[pre] = pre_cols
    col_groups['misc'] = [c for c in cols if c not in delim_cols]
    return col_groups


def clean_float_bins(st):
    return float(st.replace('(', '').replace(')', '').replace('[', '').replace(']', ''))


def first_el_num_labels(sttups):
    return ['{:.1f}'.format(float(stt.split('(')[1].split(',')[0])) for stt in sttups]


def plot_cols_across_bins(gr_summ, cols, stat, bin_var):
    ''' for a given set of cols, and a statistic in the grouped summary frame,
        plot the trajectory for each available subgroup across the first
        grouping level, which is expected to be quantile bins
    '''
    ix_names = gr_summ.index.names
    sub_groups = sorted(set(gr_summ.reset_index().set_index(ix_names[1:]). \
                            index.get_values()))
    age_cents_rep = {rg: np.mean([clean_float_bins(v) for v in rg.split(',')]) \
                     for rg in gr_summ.index.levels[0]}

    for col in cols:
        # try:
        fig = plt.figure(figsize=[8, 4])
        ax = fig.gca()
        for sg in sub_groups:
            sgDF = gr_summ.loc[pd.IndexSlice[:, sg[0], sg[1]], :]
            pDF = sgDF[(col, stat)]
            pDF.index = pDF.index.set_levels(pDF.index.levels[0].map(age_cents_rep.get),
                                             pDF.index.names[0])
            pDF.plot(label=sg)
        ax.legend()
        ax.set_title(col)
        update_tick_labels(ax, 'x', first_el_num_labels)
        ax.set_xlabel(bin_var + ' bin centers')
        # except:
        #    print( 'fail for',col )


def first_char(st):
    if pd.notnull(st):
        return st[0]
    else:
        return None


def groups_for_column(df, col, max_groups=6, num_groups=2, group_labelsF=False):
    u_vals = df[col].unique()
    if len(u_vals) <= max_groups:
        groups = u_vals
        gcol = col
    else:
        gcol = 'groups_' + col
        val_type = type(u_vals[0])
        # print(val_type, np.isreal(u_vals[0]))
        if val_type == str:  # use first letter
            groups = set([first_char(u_val) for u_val in u_vals])
            df[gcol] = df[col].apply(first_char)

        elif np.isreal(u_vals[0]):
            df[gcol] = pd.qcut(df[col], num_groups, labels=group_labelsF)
            groups = df[gcol].unique()
    print(Counter(df[gcol]))
    return gcol, groups


class GroupwiseAnalysis:
    def __init__(s, data, group_cols_Ns):
        s.dataO = data
        s.group_columns_Ns = group_cols_Ns

        s.column_categories = categorize_cols(s.dataO.columns)
        s.numeric_cols = find_num_cols(s.dataO)
        s.clean_outliers()
        s.create_groups()
        s.aggregate()

    def clean_outliers(s, Nstd=4):

        s.dataC = s.dataO.copy()
        clean_df(s.dataC, Nstd, s.numeric_cols)

    def create_groups(s):
        s.group_columns_use = []
        for c_N in s.group_columns_Ns:
            if type(c_N) == str:
                s.group_columns_use.append(c_N)
            else:
                gr_lab_in = False
                if c_N[0][-4:] == '_age':
                    gr_lab_in = None
                if type(c_N[1]) == int:
                    groups_for_column(s.dataC, c_N[0], num_groups=c_N[1], group_labelsF=gr_lab_in)
                    s.group_columns_use.append('groups_' + c_N[0])
                else:
                    groups_for_column(s.dataC, c_N[0])
                    s.group_columns_use.append('groups_' + c_N[0], group_labelsF=gr_lab_in)

        s.groupedC = s.dataC.groupby(s.group_columns_use)

    def aggregate(s):
        s.group_summary = s.groupedC.agg(['count', np.sum, np.mean, np.std])

    def trajectories_for_category(s, cat, stat='mean', outs=['DT', 'fup', 'followup', 'date']):
        plot_cols_across_bins(s.group_summary,
                              multi_filter(s.column_categories[cat], outs=outs),
                              stat, bin_var=s.group_columns_use[0].replace('groups_', ''))


# Elements
def histogram(df, property, bins='auto', type='plot', continuous=False, ax=None):
    vc = df[property].value_counts()
    if len(vc) < 11 and not continuous:  # use continuous distribution
        n_bins = len(vc)
    else:
        continuous = True
        n_bins = max([10, int(np.floor(len(df) / 100))])

    if type == 'plot':
        df[property].plot(kind='hist', bins=n_bins, ax=ax)

    elif type == 'table':
        if continuous:
            counts, bins = np.histogram(df[property])
            cntS.rename('counts', inplace=True)
            return cntS

        else:
            return vc.rename('counts')


def swarm_groups(df, group_prop, info_prop, ax=None):
    plot = sns.swarmplot(y=group_prop, x=info_prop, data=df, ax=ax)
    return plot


def violin_distributions(df, group_prop, dist_prop, split_prop=None, ax=None,
                         order=None):
    plot = sns.violinplot(x=group_prop, y=dist_prop, hue=split_prop, data=df,
                          ax=ax, split=True, order=order)
    return plot


def numbered_violins(df, group_prop, dist_prop, split_prop=None, ax=None, order=None):
    plot = violin_distributions(df, group_prop, dist_prop,
                                split_prop=split_prop, ax=ax, order=order)

    plot = update_tick_labels(plot, 'x', nums_for_labels, {'property': group_prop,
                                                           'dataframe': df})
    return plot


def prop_hists_by_group(df, group_prop, hist_prop, Nbins=20):
    fig = plt.figure(figsize=(10, 3))
    vals = df[group_prop].unique()
    try:
        vals = [v for v in vals if not np.isnan(v)]
    except:
        pass
    # print(hist_prop+' by '+group_prop)

    # All members
    sp = plt.subplot(1, len(vals) + 1, 1)
    histogram(df, hist_prop, ax=sp)
    sp.set_title('All subjects (' + str(len(df)) + ')')
    sp.set_xlabel(hist_prop)

    for vi, val in enumerate(vals):
        gr_df = df[df[group_prop] == val]
        sp = plt.subplot(1, len(vals) + 1, vi + 2)
        histogram(gr_df, hist_prop, ax=sp)
        sp.set_title(str(val) + '   (' + str(len(gr_df[hist_prop].dropna())) + ')')
        sp.set_ylabel('counts')
        sp.set_xlabel(hist_prop)


def table_breakdown(df_in, prop, bd_prop, cnt_prop, int_prop=False, prop_count_label=None):
    if prop_count_label == None:
        prop_count_label = cnt_prop
    df = df_in.copy()
    if int_prop:
        df[prop] = df[prop].fillna(-1).astype(int)
    cnt_df = pd.DataFrame(df.groupby([prop, bd_prop]).count()[cnt_prop])
    level0 = prop_count_label + ' counts'
    cnt_df.columns = [level0]
    cnt_dfU = cnt_df.unstack().fillna(0).astype(int)
    cnt_dfU[level0, 'total'] = cnt_dfU[level0].sum(axis=1)

    return cnt_dfU


def add_REdefs(df, coreRaceCol='core-race'):
    df.reset_index(inplace=True)
    df['definition'] = \
        df[coreRaceCol].apply(lambda x: RaceEthnicityDefs['hispanic'][x[0]] + \
                                        ' ' + RaceEthnicityDefs['core'][x[1]])
    df.set_index(coreRaceCol, inplace=True)


# Standard groups
def sessions_info(df, folder, name, fup_order=None):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    print(len(df['ID'].unique()), 'subjects, ',
          len(df), 'sessions')

    # Sessions histogram    
    ses_counts = session_counts(df)
    fig = plt.figure(figsize=[8, 4])
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    sp = plt.subplot(gs[0])
    sp.set_title('EEG session counts')
    ses_count_hist = histogram(ses_counts, count_field, ax=sp)
    sp = plt.subplot(gs[1], frame_on=False)
    sp.xaxis.set_visible(False)
    sp.yaxis.set_visible(False)
    ses_count_table = histogram(ses_counts, count_field, type='table')
    ses_count_table.sort_index(inplace=True)
    table(sp, ses_count_table, loc='center')
    fig.savefig(os.path.join(folder, name + '_sesHist.png'))

    # age distributions by followup with sex
    if fup_order:
        order = fup_order
    else:
        order_template = ['p1', 'p2', 'p3', '-1.0', '0', '1', '2', '3', '4', '5']
        order = [g for g in order_template if g in df['followup'].unique()]
    fig = plt.figure(figsize=[8, 5]);
    ax = fig.gca()
    plot = violin_distributions(df, 'followup', 'session_age', 'sex', ax=ax,
                                order=order)
    plot = update_tick_labels(plot, 'x', nums_for_labels, {'property': 'followup',
                                                           'dataframe': df})
    plot.set_ylim([0, 38])
    fig.savefig(os.path.join(folder, name + '_followups_ages_sex.png'))

    # race + ethnicity table_breakdown
    re_ses = table_breakdown(df, 'self-reported',
                             'sex', 'uID', prop_count_label='session')
    re_sub = table_breakdown(df.groupby('ID').head(1), 'self-reported',
                             'sex', 'uID', prop_count_label='subject')
    re_comb = re_sub.join(re_ses)
    add_REdefs(re_comb, 'self-reported')
    # fig = plt.figure(figsize=[8,12])
    # ax = fig.gca()
    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)
    # table(ax,re_comb,loc='center')
    with open(os.path.join(folder, name + '_RaceEthnicity_breakdown.html'), 'w') as wf:
        re_comb.to_html(wf)


        # fig = plt.figure(figsize=[8,12])
        # ax = fig.gca()
        # sg = swarm_groups(df,'self-reported','session_age',ax=ax)
        # update_tick_labels(sg,'y',expand_RE_aliases,{'race_type':'self-reported'})
        # fig.savefig( os.path.join(folder,name+'_sessions_ethnicity.png'),
        #                 bbox_inches='tight' )
