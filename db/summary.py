import pandas as pd
import numpy as np
import os
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
    df = df.set_index(['ID','session'])
    ses_counts = df.groupby(level=0).count()
    return ses_counts


# Elements

def histogram(df,property,bins='auto',type='plot',continuous=False,ax=None):
    vc = df[property].value_counts()
    if len(vc) < 11 and not continuous: #use continuous distribution
        n_bins = len(vc)
    else:
        continuous = True
        n_bins = max( [ 10, int(np.floor(len(df)/100)) ])
        

    if type == 'plot':
        df[property].plot(kind='hist',bins=n_bins,ax=ax)

    elif type == 'table':
        if continuous:
            counts, bins = np.histogram(df[property])
            cntS = Series(counts,index=bins[:-1])
            cntS.rename('counts',inplace=True)
            return cntS
        else:
            return vc.rename('counts')

def swarm_groups(df,group_prop,info_prop,ax=None):

    plot = sns.swarmplot(y=group_prop,x=info_prop,data=df,ax=ax)
    return plot

def violin_distributions(df,group_prop,dist_prop,split_prop=None,ax=None):

    plot = sns.violinplot(x=group_prop,y=dist_prop,hue=split_prop,data=df,
              ax=ax,split=True)
    return plot

def prop_hists_by_group(df,group_prop,hist_prop,Nbins=20):
    fig = plt.figure(figsize=(10,3))
    vals = df[group_prop].unique()
    try:
        vals = [ v for v in vals if not np.isnan(v) ]
    except:
        pass
    #print(hist_prop+' by '+group_prop)
    
    # All members
    sp = plt.subplot(1,len(vals)+1,1)
    histogram(df,hist_prop,ax=sp)
    sp.set_title('All subjects ('+str(len(df))+')')
    sp.set_xlabel(hist_prop)

    for vi,val in enumerate(vals):
        gr_df = df[ df[group_prop]==val ]
        sp = plt.subplot(1,len(vals)+1,vi+2)
        histogram(gr_df,hist_prop,ax=sp)
        sp.set_title(str(val) + '   ('+str(len(gr_df[hist_prop].dropna()))+')')
        sp.set_ylabel('counts')
        sp.set_xlabel(hist_prop)

def table_breakdown(df_in,prop,bd_prop,cnt_prop,int_prop=False,prop_count_label=None):
    if prop_count_label == None:
        prop_count_label = cnt_prop
    df = df_in.copy()
    if int_prop:
        df[prop] = df[prop].fillna(-1).astype(int)
    cnt_df = pd.DataFrame( df.groupby([prop, bd_prop]).count()[cnt_prop])
    cnt_df.columns = [prop_count_label+' counts']

    return cnt_df.unstack().fillna(0).astype(int)


# Standard groups
def sessions_info(df,folder,name):
    if not os.path.exists(folder):
        os.makedirs(folder,exist_ok=True)

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

    # age distributions by session with sex
    fig = plt.figure(figsize = [8,5]); ax = fig.gca()
    plot = violin_distributions(df,'session','session_age','sex',ax=ax)
    plot = update_tick_labels(plot,'x',nums_for_labels,{'property':'session',
                        'dataframe':df} ) 
    fig.savefig( os.path.join(folder,name+'_sessions_ages_sex.png') )


    # race + ethnicity table_breakdown
    fig = plt.figure(figsize=[8,12])
    ax = fig.gca()
    sg = swarm_groups(df,'self-reported','session_age',ax=ax)
    update_tick_labels(sg,'y',expand_RE_aliases,{'race_type':'self-reported'})
    fig.savefig( os.path.join(folder,name+'_sessions_ethnicity.png'), 
                    bbox_inches='tight' )

    #