import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def session_counts(df):
    df = df.reset_index()
    df = df.set_index(['ID', 'session'])
    ses_counts = df.groupby(level=0).count()
    return ses_counts


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
            return pd.Series(counts, index=bins[:-1])
        else:
            return vc


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
