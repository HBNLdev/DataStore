import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
from pandas.tools.plotting import table
import seaborn as sns

from collections import Counter

# Info for labelling, etc
#Race and ethnicity definitions
count_field = 'uID'

RaceEthnicityDefs = {'hispanic':{'h':'hispanic','n':'non-hispanic','u':'unknown'},
          'core':{'2':'Asian','4':'Black','6':'White','8':'unknown','9':'other'},
         'self-reported':{'1':'Indigeneous American','2':'Asian',
                        '3':'Pacific Islander',
                         '4':'Black','6':'White','8':'Unknown','9':'mixed'}}

DiagSymp = {'ssa_ALD4D1':'Tolerance',
            'ssa_ALD4D2':'Withdrawal ',
            'ssa_ALD4D3':'More than intended ',
            'ssa_ALD4D4':'desire to cut down ',
            'ssa_ALD4D5':'Excessive time used ',
            'ssa_ALD4D6':'Use dominates life ',
            'ssa_ALD4D7':'physical/psych. probs'}

QtyVars = {'ssa_AL4a': 'Weeks drank \n last 6 mo',
            'ssa_AL8d': 'Consecutive \n "heavy week"',
            'ssa_AL6':  'Max drinks (24-hour) \n lifetime',
            'ssa_AL6a': 'Max drinks (24-hour) \n 6 months',
            'ssa_AL8a': 'Max daily drinks \n in "heavy week"',
            'ssa_AL3_ws':   'Drinks wk \n before interview',
            'ssa_AL4_ws':   'Drinks typical wk \n last 6 mo',
            'ssa_AL4_p6ms': 'Total drinks \n last 6 mo',
            'ssa_AL3_da':   'Av drinks/day \n in prev wk',
            'ssa_AL4_da':   'Av drinks/day \n in last 6 mo',
            'cor_max_dpw':  'drinks in typical wk \n (all ints)',
            'cor_max_dpw_pwk':  'drinks in prev wk \n (all ints)',
            'cor_max_drinks':   'Max drinks \n (24-hour): life',
            'ssa_AL16b':    'Number of binges',
            'ssa_AL17b':    'Number of \n blackouts',
            'ssa_AL43A':    'Number of \n 3 mo abstinence',
            'ssa_AL1':  'Ever drank \n (at int)',
            'ssa_AL8':  'Ever \n "heavy drinking wk"',
            'cor_ever_drink':   'Ever drank \n (all ints)',
            'cor_ever_got_drunk':   'Ever been drunk',
            'cor_regular_drinking': 'Ever drank 1/mo \n for 6 mo'}
DrinkingVars = DiagSymp.copy()
DrinkingVars.update(QtyVars)

# Utilities
def expand_RE_aliases(aliases,race_type):

    def trans(code):
        tr = RaceEthnicityDefs['hispanic'][code[0]]
        tr += ' '+RaceEthnicityDefs[race_type][code[1]]
        return tr

    return [ trans(a) for a in aliases ]

def nums_for_labels(labels,property,dataframe):

    countD = dataframe.groupby(property).count()[count_field].to_dict()
    return [ l+'\n('+str(countD[l])+')' for l in labels]

def update_tick_labels(plot,axis,trFun,trIns):
    label_lk = {'x':plot.get_xticklabels,
                'y':plot.get_yticklabels }
    labels = [ l.get_text() for l in label_lk[axis]() ]
    updated = trFun(labels,**trIns)

    ud_lk = {'x':plot.set_xticklabels,
            'y':plot.set_yticklabels }
    ud_lk[axis]( updated )
    return plot


def session_counts(df):
    df = df.reset_index()
    df = df.set_index(['ID', 'session'])
    ses_counts = df.groupby(level=0).count()
    return ses_counts

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
            cntS.rename('counts',inplace=True)
            return cntS

        else:
            return vc.rename('counts')

def swarm_groups(df,group_prop,info_prop,ax=None):

    plot = sns.swarmplot(y=group_prop,x=info_prop,data=df,ax=ax)
    return plot

def violin_distributions(df,group_prop,dist_prop,split_prop=None,ax=None,
                    order=None):

    plot = sns.violinplot(x=group_prop,y=dist_prop,hue=split_prop,data=df,
              ax=ax,split=True, order=order)
    return plot

def numbered_violins(df,group_prop,dist_prop,split_prop=None,ax=None,order=None):

    plot = violin_distributions(df,group_prop,dist_prop,
                        split_prop=split_prop,ax=ax,order=order)

    plot = update_tick_labels(plot,'x',nums_for_labels,{'property':group_prop,
                        'dataframe':df} )
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
    cnt_df.columns = [prop_count_label + ' counts']

    return cnt_df.unstack().fillna(0).astype(int)


# Standard groups
def sessions_info(df,folder,name,fup_order=None):
    if not os.path.exists(folder):
        os.makedirs(folder,exist_ok=True)

    print( len(df['ID'].unique()), 'subjects, ',
            len(df), 'sessions' )

    # Sessions histogram    
    ses_counts = session_counts(df)
    fig = plt.figure(figsize=[8,4])
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    sp = plt.subplot(gs[0])
    sp.set_title('EEG session counts')
    ses_count_hist = histogram(ses_counts,count_field,ax=sp)
    sp = plt.subplot(gs[1],frame_on=False)
    sp.xaxis.set_visible(False)
    sp.yaxis.set_visible(False)
    ses_count_table = histogram(ses_counts,count_field,type='table')
    table(sp,ses_count_table,loc='center')
    fig.savefig( os.path.join(folder,name+'_sesHist.png') )

    # age distributions by followup with sex
    if fup_order:
        order = fup_order
    else:
        order_template = ['p1','p2','p3','-1.0','0','1','2','3','4','5']
        order = [ g for g in order_template if g in df['followup'].unique() ]
    fig = plt.figure(figsize = [8,5]); ax = fig.gca()
    plot = violin_distributions(df,'followup','session_age','sex',ax=ax,
                order=order )
    plot = update_tick_labels(plot,'x',nums_for_labels,{'property':'followup',
                        'dataframe':df} )
    plot.set_ylim([0,60])
    fig.savefig( os.path.join(folder,name+'_followups_ages_sex.png') )


    # race + ethnicity table_breakdown
    # fig = plt.figure(figsize=[8,12])
    # ax = fig.gca()
    # sg = swarm_groups(df,'self-reported','session_age',ax=ax)
    # update_tick_labels(sg,'y',expand_RE_aliases,{'race_type':'self-reported'})
    # fig.savefig( os.path.join(folder,name+'_sessions_ethnicity.png'), 
    #                 bbox_inches='tight' )
