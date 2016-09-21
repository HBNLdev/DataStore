''' plotting functions that operate on an eeg.Results object
	that are designed to plot statistical information '''

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as ss
import mne
from mne.viz import plot_connectivity_circle

from .plot import measure_pps
from ._plot_utils import (subplot_heuristic, figsize_heuristic,
                    is_nonstr_sequence, nested_strjoin,
                    MidpointNormalize,
                    blank_topo, plot_arcs, ordinalize_one,
                    ordered_chans, layout, n_colors)
from ._array_utils import basic_slice, compound_take, handle_by, handle_pairs

''' initialize matplotlib backend settings '''
# print(mpl.matplotlib_fname())
mpl.rcParams['svg.fonttype'] = 'none' # none, path, or svgfont
# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Verdana',
#     'Bitstream Vera Sans', 'Lucida Grande', 'Geneva', 'Lucid',
#     'Arial', 'Avant Garde', 'sans-serif']
titlefont_sz = 16
titlefont_wt = 'bold'
stitlefont_sz = 12
stitlefont_wt = 'medium'

def contig_lims(lst):
    ''' given a list of integers, return a list of tuples containing
        the minimum and maximum values within each sequence of consecutive
        integers in the list '''
    contig_loc = 0
    minmax_lst = []
    for i, num in enumerate(lst[:-1]):
        if num+1 != lst[i+1]:
            contig_set = lst[contig_loc:i+1]
            minmax_lst.append( (np.min(contig_set), np.max(contig_set)) )
            contig_loc = i+1
            continue
        elif i==len(lst)-2:
            contig_set = lst[contig_loc:i+2]
            minmax_lst.append( (np.min(contig_set), np.max(contig_set)) )
            contig_loc = i+1
            continue
    return minmax_lst

def erp_clusters(s, stat_attr):

    stats_dict = getattr(s, stat_attr)

    T_obs = stats_dict['T_obs']
    clusters = stats_dict['clusters']
    cluster_p_values = stats_dict['cluster_p_values']

    mpl.rcParams.update({'font.size': 9})
    n_subplots = T_obs.shape[1]
    colors = list(n_colors(len(clusters)))
    sp_dims = layout.shape
    f, axarr = plt.subplots(sp_dims[0], sp_dims[1], sharex=True, sharey=True,
                           figsize=(13, 9))
    for chani in range(n_subplots):
        name = s.montage.ch_names[chani]
        pos = np.where(layout==name)
        if pos[0].size > 0:
            ls = []
            for cond in range(2):
                line = np.mean(s.erp[:, cond, chani, :], axis=0)
                l, = axarr[pos][0].plot(np.arange(len(line)), line)
                ls.append(l)
            for clusti, clust in enumerate(clusters):
                if (cluster_p_values[clusti] <= .05) and (chani in clust[1]):
                    times = clust[0][np.where(clust[1]==chani)[0]]
                    try:
                        lims = contig_lims(times)[0]
                        axarr[pos][0].axvspan(lims[0], lims[1],
                                              color=colors[clusti], alpha=0.5)
                    except:
                        print('no valid lims for channel {} and cluster {}'\
                            .format(chani, clusti))
            # axarr[pos][0].grid(True)
            axarr[pos][0].set_title(name)
            if pos[0] % sp_dims[0] == sp_dims[0] - 1:
                axarr[pos][0].set_xticks(s.time_ticks_pt_erp)
                axarr[pos][0].set_xticklabels(s.time_ticks_ms)
                axarr[pos][0].set_xlabel('Time (s)')
            if pos[1] % sp_dims[1] == 0:
                axarr[pos][0].set_ylabel('Potential (' + s.pot_units + ')')
            axarr[pos][0].axhline(0, color='k', linestyle='--')
            axarr[pos][0].axvline(s.zero, color='k', linestyle='--')
            axarr[pos][0].set_xlim(s.time_plotlims)
        else:
            axarr[pos][0].axis('off')
    f.legend(tuple(ls), tuple(s.erp_dim_lsts[1]), 'upper left')