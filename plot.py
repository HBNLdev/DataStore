''' plotting functions that operate on an eeg.Results object '''

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as ss
import mne
from mne.viz import plot_connectivity_circle

from array_utils import basic_slice, compound_take, handle_by, handle_pairs
from plot_utils import (subplot_heuristic, figsize_heuristic,
                    is_nonstr_sequence, nested_strjoin,
                    MidpointNormalize,
                    blank_topo, plot_arcs, ordinalize_one,
                    ordered_chans, n_colors)

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

''' dictionary mapping measures to their object info '''
measure_pps = {'erp':   {'data': 'erp', 'd_dims': 'erp_dims',
                         'd_dimlvls': 'erp_dim_lsts', 'units': 'pot_units',
                         'lims': 'pot_lims', 'cmap': plt.cm.RdBu_r,
                         'load': 'load_erp'},
               'power': {'data': 'power', 'd_dims': 'tf_dims',
                         'd_dimlvls': 'tf_dim_lsts', 'units': 'db_units',
                         'lims': 'db_lims', 'cmap': plt.cm.RdBu_r,
                         'load': 'load_power'},
               'itc':   {'data': 'itc', 'd_dims': 'tf_dims',
                         'd_dimlvls': 'tf_dim_lsts', 'units': 'itc_units',
                         'lims': 'itc_lims', 'cmap': plt.cm.PuOr,
                         'load': 'load_itc'},
               'itc_Z': {'data': 'itc_Z', 'd_dims': 'tf_dims',
                         'd_dimlvls': 'tf_dim_lsts', 'units': 'z_units',
                         'lims': 'z_lims', 'cmap': plt.cm.PuOr,
                         'load': 'load_itc_Z'},
               'coh':   {'data': 'coh', 'd_dims': 'coh_dims',
                         'd_dimlvls': 'coh_dim_lsts', 'units': 'coh_units',
                         'lims': 'coh_lims', 'cmap': plt.cm.PuOr,
                         'load': 'load_coh'},
               'coh_Z': {'data': 'coh_Z', 'd_dims': 'coh_dims',
                         'd_dimlvls': 'coh_dim_lsts', 'units': 'z_units',
                         'lims': 'z_lims', 'cmap': plt.cm.PuOr,
                         'load': 'load_coh_Z'},
               'phi':   {'data': 'phi', 'd_dims': 'tf_dims',
                         'd_dimlvls': 'tf_dim_lsts', 'units': 'phi_units',
                         'lims': 'phi_lims', 'cmap': plt.cm.PuOr,
                         'load': 'load_phi'},
               }

def get_data(s, measure):
    ''' given a measure, retrieve data and dimensional info '''

    measure_d = s.measure_pps[measure]
    if measure_d['data'] not in dir(s):
        getattr(s, measure_d['load'])()

    data        = getattr(s, measure_d['data'])
    d_dims      = getattr(s, measure_d['d_dims'])
    d_dimlvls   = getattr(s, measure_d['d_dimlvls'])
    
    return data, d_dims, d_dimlvls

def get_plotparams(s, measure, lims=None, cmap_override=None):
    ''' given a measure, retrieve default plotting info '''

    measure_d = s.measure_pps[measure]
    units       = getattr(s, measure_d['units'])

    if lims:
        if isinstance(lims, str):
            if lims not in ['absmax', 'minmax']:
                print('lims incorrectly specified')
                raise
        elif isinstance(lims, list):
            if len(lims) != 2:
                print('lims incorrectly specified')
                raise
        else:
            lims = [-lims, lims] # if numeric, set lims as -num to +num
    else:
        lims = getattr(s, measure_d['lims'])

    if cmap_override:
        cmap = cmap_override
    else:
        cmap = measure_d['cmap']

    return units, lims, cmap

def save_fig(s, savedir, ptype, measure, label, form='svg'):
    ''' name and save the current figure to a target directory
        by default, we use SVG because it typically works the best '''

    figname = gen_figname(s, ptype, measure, label)+'.'+form
    outpath = os.path.expanduser(os.path.join(savedir, figname))
    plt.savefig(outpath, format=form, dpi=1000)

def gen_figname(s, ptype, measure, label):
    ''' generate a figure name from the plot type, measure, and label '''
    
    parts = [s.params['Batch ID'], ptype, measure, label]

    cleaned_parts = []
    for p in parts:
        if is_nonstr_sequence(p):
            cleaned_parts.append(nested_strjoin(p))
        else:
            cleaned_parts.append(p)

    return '_'.join(cleaned_parts)

# plot functions that accept an eeg results object

def erp(s, figure_by={'channel': ['FZ', 'CZ', 'PZ']},
           subplot_by={'POP': 'all'},
           glyph_by={'condition': 'all'},
           savedir=None):
    ''' plot ERPs as lines '''
    ptype = 'line'
    measure = 'erp'

    if 'erp' not in dir(s):
        s.load_erp()
    data = s.erp
    d_dims = s.erp_dims
    d_dimlvls = s.erp_dim_lsts

    f_dim, f_vals, f_lbls = handle_by(s, figure_by, d_dims, d_dimlvls)
    sp_dim, sp_vals, sp_lbls = handle_by(s, subplot_by, d_dims, d_dimlvls)
    g_dim, g_vals, g_lbls = handle_by(s, glyph_by, d_dims, d_dimlvls)

    sp_dims = subplot_heuristic(len(sp_vals))
    figsize = figsize_heuristic(sp_dims)
    for fi, fval in enumerate(f_vals):
        f, axarr = plt.subplots(sp_dims[0], sp_dims[1],
                                sharex=True, sharey=True, figsize=figsize)
        f.suptitle(f_lbls[fi], fontsize=titlefont_sz, fontweight=titlefont_wt)
        try:
            axarr = axarr.ravel()
        except:
            axarr = [axarr]
        for spi, spval in enumerate(sp_vals):
            for gi, gval in enumerate(g_vals):
                vals = [fval, spval, gval]
                dims = [f_dim[fi], sp_dim[spi], g_dim[gi]]
                dimval_tups = [(d,v) for d,v in zip(dims, vals)]
                try:
                    line = basic_slice(data, dimval_tups)
                    print('slice')
                except:
                    line = compound_take(data, dimval_tups)
                    print('compound take')
                while len(line.shape) > 1:
                    line = line.mean(axis=0)
                    err = ss.sem(line, axis=0)
                    # err = np.std(line, axis=0, ddof=1)
                    print(line.shape)
                l, = axarr[spi].plot(np.arange(len(line)), line,
                                label=g_lbls[gi])
                axarr[spi].fill_between(np.arange(len(line)),
                                line - err, line + err,
                                alpha=0.5, linewidth=0,
                                facecolor=l.get_color())
            axarr[spi].grid(True)
            axarr[spi].set_title(sp_lbls[spi], fontweight=stitlefont_wt)
            axarr[spi].legend(loc='upper left')
            axarr[spi].set_xticks(s.time_ticks_pt_erp)
            axarr[spi].set_xticklabels(s.time_ticks_ms)
            axarr[spi].set_xlabel('Time (s)', fontweight=stitlefont_wt)
            if spi % sp_dims[1] == 0:
                axarr[spi].set_ylabel('Potential (' + s.pot_units + ')',
                                                fontweight=stitlefont_wt)
            axarr[spi].axhline(0, color='k', linestyle='--')
            axarr[spi].axvline(s.zero, color='k', linestyle='--')
            axarr[spi].set_xlim(s.time_plotlims)
        if savedir:
            save_fig(s, savedir, ptype, measure, f_lbls[fi])

def tf(s, measure='power',
          figure_by={'POP': 'all', 'channel': ['FZ']},
          subplot_by={'condition': 'all'},
          lims='absmax', cmap_override=None,
          savedir=None):
    ''' plot time-frequency data as a rectangular contour image '''
    ptype = 'tf'

    if measure in ['power', 'itc', 'itc_Z', 'phi', 'coh', 'coh_Z']:
        data, d_dims, d_dimlvls = get_data(s, measure)
        units, lims, cmap = get_plotparams(s, measure, lims, cmap_override)
    else:
        print('data not recognized')
        return

    f_dim, f_vals, f_lbls = handle_by(s, figure_by, d_dims, d_dimlvls)
    sp_dim, sp_vals, sp_lbls = handle_by(s, subplot_by, d_dims, d_dimlvls)

    sp_dims = subplot_heuristic(len(sp_vals))
    figsize = figsize_heuristic(sp_dims)
    for fi, fval in enumerate(f_vals):
        f, axarr = plt.subplots(sp_dims[0], sp_dims[1],
                                sharex=True, sharey=True, figsize=figsize)
        f.suptitle(f_lbls[fi], fontsize=titlefont_sz, fontweight=titlefont_wt)
        try:
            axarr = axarr.ravel()
        except:
            axarr = [axarr]
        rect_lst = []
        c_lst = []
        for spi, spval in enumerate(sp_vals):
            vals = [fval, spval]
            dims = [f_dim[fi], sp_dim[spi]]
            dimval_tups = [(d,v) for d,v in zip(dims, vals)]
            try:
                print(dimval_tups)
                rect = basic_slice(data, dimval_tups)
                print('slice')
                print(rect.shape)
            except:
                rect = compound_take(data, dimval_tups)
                print('done compound take')
            while len(rect.shape) > 2:
                rect = rect.mean(axis=0)
                print(rect.shape)
            ''' contour '''
            c = axarr[spi].contourf(rect, 8, cmap=cmap)
            # c = axarr[spi].imshow(rect, aspect='auto', origin='lower',
            #                             cmap=cmap,
            #                             interpolation='sinc')
            # c = axarr[spi].contour(rect, cmap=plt.cm.RdBu_r,
            #                         vmin=-4, vmax=4)
            # plt.clabel(c, inline=1, fontsize=9)
            # decent interpolations include none and sinc
            c_lst.append(c)
            rect_lst.append(rect)
            
            # cbar = plt.colorbar(c, ax=axarr[spi])
            # cbar.ax.set_ylabel(units, rotation=270)
            ''' ticks and grid '''
            axarr[spi].set_xticks(s.time_ticks_pt_tf)
            axarr[spi].set_xticklabels(s.time_ticks_ms)
            axarr[spi].set_yticks(s.freq_ticks_pt)
            axarr[spi].set_yticklabels(s.freq_ticks_hz)
            axarr[spi].grid(True)
            axarr[spi].axvline(s.zero_tf, color='k', linestyle='--')
            axarr[spi].set_xlim(s.time_tf_plotlims)
            ''' labels and title '''
            axarr[spi].set_xlabel('Time (s)', fontweight=stitlefont_wt)
            if spi % sp_dims[1] == 0:
                axarr[spi].set_ylabel('Frequency (Hz)', fontweight=stitlefont_wt)
            axarr[spi].set_title(sp_lbls[spi], fontweight=stitlefont_wt)

        norm = None
        rects = np.stack(rect_lst, axis=-1)
        if lims == 'absmax':
            vmax = np.max(np.fabs(rects))
            vmin = -vmax
        elif lims == 'minmax':
            vmax, vmin = np.max(rects), np.min(rects)
        else:
            vmax, vmin = lims[1], lims[0]
        
        if vmin != -vmax:
            norm = MidpointNormalize(vmin, vmax, 0)
        print('vmin', vmin, 'vmax', vmax)

        for c, spi in zip(c_lst, range(len(sp_vals))):
            c.set_clim(vmin, vmax)
            if norm:
                c.set_norm(norm)
            cbar = plt.colorbar(c, ax=axarr[spi])
            if spi % sp_dims[1] == sp_dims[1] - 1:
                cbar.ax.set_ylabel(units, rotation=270, va='bottom',
                                        fontweight=stitlefont_wt)

        ''' colorbar '''
        # plt.subplots_adjust(right=0.85)
        # cbar_ax = f.add_axes([0.88, 0.12, 0.03, 0.75])
        # cbar = plt.colorbar(c, cax=cbar_ax)
        # cbar.ax.set_ylabel(units, rotation=270)
        # plt.colorbar(c, ax=axarr[spi])
        if savedir:
            save_fig(s, savedir, ptype, measure, f_lbls[fi])


def topo(s, measure='erp', times=list(range(0, 501, 125)),
            figure_by={'POP': ['C']},
            row_by={'condition': 'all'},
            lims='absmax', cmap_override=None,
            savedir=None):
    ''' plot data as topographic maps at specific timepoints '''
    ptype = 'topo'
    final_dim = 'channel'

    if measure in ['erp', 'power', 'itc', 'itc_Z', 'phi']:
        data, d_dims, d_dimlvls = get_data(s, measure)
        units, lims, cmap = get_plotparams(s, measure, lims, cmap_override)
    else:
        print('data not recognized')
        return

    info = mne.create_info(s.montage.ch_names, s.params['Sampling rate'],
                           'eeg', s.montage)

    f_dim, f_vals, f_lbls = handle_by(s, figure_by, d_dims, d_dimlvls)
    r_dim, r_vals, r_lbls = handle_by(s, row_by, d_dims, d_dimlvls)
    time_by = {'timepoint': times}
    t_dim, t_vals, t_lbls = handle_by(s, time_by, d_dims, d_dimlvls)

    sp_dims = (len(r_vals), len(times))
    figsize = figsize_heuristic(sp_dims)
    for fi, fval in enumerate(f_vals):
        f, axarr = plt.subplots(sp_dims[0], sp_dims[1],
                                sharex=True, sharey=True, figsize=figsize)
        f.suptitle(f_lbls[fi], fontsize=titlefont_sz, fontweight=titlefont_wt)
        try:
            axarr = axarr.ravel()
        except:
            axarr = [axarr]
        ax_dum = -1
        topo_lst = []
        im_lst = []
        for ri, rval in enumerate(r_vals):
            for ti, tval in enumerate(t_vals):
                vals = [fval, rval, tval]
                dims = [f_dim[fi], r_dim[ri], t_dim[ti]]
                # try:
                dimval_tups = [(d,v) for d,v in zip(dims, vals)]
                try:
                    topo = basic_slice(data, dimval_tups)
                    print('slice')
                except:
                    topo = compound_take(data, dimval_tups)
                    print('compound take')
                mean_dims = np.where([d!=final_dim for d in d_dims])
                topo = topo.mean(axis=tuple(mean_dims[0]))
                print(topo.shape)
                ax_dum += 1
                im, cn = mne.viz.plot_topomap(topo, info,
                                    cmap=cmap, axes=axarr[ax_dum],
                                    contours=8, show=False)
                topo_lst.append(topo)
                im_lst.append(im)
                ''' labels '''
                axarr[ax_dum].text(-.5, .5, t_lbls[ti])  # time label
                if ti == 0:
                    axarr[ax_dum].text(-1.5, 0, r_lbls[ri])  # row label
        
        norm = None
        topos = np.stack(topo_lst, axis=-1)
        if lims == 'absmax':
            vmax = np.max(np.fabs(topos))
            vmin = -vmax
        elif lims == 'minmax':
            vmax, vmin = np.max(topos), np.min(topos)
        else:
            vmax, vmin = lims[1], lims[0]
        
        if vmin != -vmax:
            norm = MidpointNormalize(vmin, vmax, 0)
        print('vmin', vmin, 'vmax', vmax)

        for im in im_lst:
            im.set_clim(vmin, vmax)
            if norm:
                im.set_norm(norm)

        ''' colorbar '''
        plt.subplots_adjust(right=0.85)
        cbar_ax = f.add_axes([0.9, 0.15, 0.03, 0.75])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.ax.set_ylabel(units, rotation=270, fontweight=stitlefont_wt)
        if savedir:
            save_fig(s, savedir, ptype, measure, f_lbls[fi])


def arctopo(s, measure='coh',
               pairs='all',
               times=list(range(0, 501, 125)),
               figure_by={'POP': ['C']},
               row_by={'condition': 'all'},
               lims='absmax', cmap_override=None,
               savedir=None):
    ''' plot coherence as topographically-arranged colored arcs '''
    ptype = 'arctopo'
    final_dim = 'pair'

    if measure in ['coh', 'coh_Z']:
        data, d_dims, d_dimlvls = get_data(s, measure)
        units, lims, cmap = get_plotparams(s, measure, lims, cmap_override)
    else:
        print('data not recognized')
        return

    info = mne.create_info(s.montage.ch_names, s.params['Sampling rate'],
                           'eeg', s.montage)

    f_dim, f_vals, f_lbls = handle_by(s, figure_by, d_dims, d_dimlvls)
    # print('~~~~')
    # print(f_dim, '\n', f_vals, '\n', f_lbls)
    # print('~~~~')
    r_dim, r_vals, r_lbls = handle_by(s, row_by, d_dims, d_dimlvls)
    time_by = {'timepoint': times}
    t_dim, t_vals, t_lbls = handle_by(s, time_by, d_dims, d_dimlvls)
    
    pair_inds = handle_pairs(s, pairs)
    pair_chaninds = [s.cohpair_inds[pi] for pi in pair_inds]
    pair_dim = d_dims.index('pair')
    data = basic_slice(data, [(pair_dim, pair_inds)])


    sp_dims = (len(r_vals), len(times))
    figsize = figsize_heuristic(sp_dims)
    for fi, fval in enumerate(f_vals):
        f, axarr = plt.subplots(sp_dims[0], sp_dims[1],
                                sharex=True, sharey=True, figsize=figsize)
        f.suptitle(f_lbls[fi], fontsize=titlefont_sz, fontweight=titlefont_wt)
        try:
            axarr = axarr.ravel()
        except:
            axarr = [axarr]
        ax_dum = -1
        arcs_lst = []
        arch_lst = []
        for ri, rval in enumerate(r_vals):
            for ti, tval in enumerate(t_vals):
                vals = [fval, rval, tval]
                dims = [f_dim[fi], r_dim[ri], t_dim[ti]]
                # try:
                dimval_tups = [(d,v) for d,v in zip(dims, vals)]
                try:
                    arcs = basic_slice(data, dimval_tups)
                    print('slice')
                except:
                    arcs = compound_take(data, dimval_tups)
                    print('compound take')
                mean_dims = np.where([d!=final_dim for d in d_dims])
                arcs = arcs.mean(axis=tuple(mean_dims[0]))
                print(arcs.shape)
                ax_dum += 1
                ax, im, cn, pos_x, pos_y = blank_topo(axarr[ax_dum], info)
                arches = plot_arcs(arcs, ax, pair_chaninds,
                                    pos_x, pos_y, cmap=cmap)
                arcs_lst.append(arcs)
                arch_lst.extend(arches)
                ''' labels '''
                axarr[ax_dum].text(-.5, .5, t_lbls[ti])  # time label
                if ti == 0:
                    axarr[ax_dum].text(-1.5, 0, r_lbls[ri])  # row label
        
        arcs_stack = np.stack(arcs_lst, axis=-1)
        if lims == 'absmax':
            vmax = np.max(np.fabs(arcs_stack))
            vmin = -vmax
        elif lims == 'minmax': # won't work with 2color asymmetric cmap
            vmax, vmin = np.max(arcs_stack), np.min(arcs_stack)
        else:
            vmax, vmin = lims[1], lims[0]
        
        if vmin < 0 and vmax > 0:
            mid = 0
        else:
            mid = None
        
        print('vmin', vmin, 'vmax', vmax)

        cmap_array = cmap(range(cmap.N))
        for arch in arch_lst:
            val, arc_h = arch
            color_ind, abs_prop = ordinalize_one(val, size=cmap.N,
                                                lims=[vmin, vmax], mid=mid)
            arc_h.set_color(cmap_array[color_ind, :3])
            # arc_h.set_alpha((abs_prop+1)/2)
            arc_h.set_alpha(abs_prop)

        ''' colorbar '''
        plt.subplots_adjust(right=0.85)
        cbar_ax = f.add_axes([0.9, 0.15, 0.03, 0.75])
        mapper = plt.cm.ScalarMappable(cmap=cmap)
        mapper.set_array([vmin, vmax])
        cbar = plt.colorbar(mapper, cax=cbar_ax)
        cbar.ax.set_ylabel(units, rotation=270, fontweight=stitlefont_wt)
        if savedir:
            save_fig(s, savedir, ptype, measure, f_lbls[fi])

def connectivity_circle(s, measure='coh',
                           pairs='all',
                           times=list(range(0, 501, 125)),
                           figure_by={'POP': ['C']},
                           row_by={'condition': 'all'},
                           lims='absmax', cmap_override=None,
                           savedir=None):
    ''' plot coherence as generic circle of nodes connected by colored arcs '''
    ptype = 'arctopo'
    final_dim = 'pair'

    if measure in ['coh', 'coh_Z']:
        data, d_dims, d_dimlvls = get_data(s, measure)
        units, lims, cmap = get_plotparams(s, measure, lims, cmap_override)
    else:
        print('data not recognized')
        return

    f_dim, f_vals, f_lbls = handle_by(s, figure_by, d_dims, d_dimlvls)
    r_dim, r_vals, r_lbls = handle_by(s, row_by, d_dims, d_dimlvls)
    time_by = {'timepoint': times}
    t_dim, t_vals, t_lbls = handle_by(s, time_by, d_dims, d_dimlvls)
    
    pair_inds = handle_pairs(s, pairs)
    pair_chaninds = [s.cohpair_inds[pi] for pi in pair_inds]

    uniq_chaninds = np.unique(np.array(pair_chaninds).ravel())
    uniq_chanlabels = [s.montage.ch_names[ci] for ci in uniq_chaninds]
    uniq_chanlabels.sort(key=lambda x: ordered_chans.index(x))

    cohpair_array = []
    for chan_pair in np.array(s.cohpair_inds):
        cohpair_array.append(
            [uniq_chanlabels.index(s.montage.ch_names[int(chan_pair[0])]),
             uniq_chanlabels.index(s.montage.ch_names[int(chan_pair[1])])])
    cohpair_array = np.array(cohpair_array)
    indices = (cohpair_array[:, 0], cohpair_array[:, 1])

    pair_dim = d_dims.index('pair')
    data = basic_slice(data, [(pair_dim, pair_inds)])

    sp_dims = (len(r_vals), len(times))
    figsize = figsize_heuristic(sp_dims)
    for fi, fval in enumerate(f_vals):
        f, axarr = plt.subplots(sp_dims[0], sp_dims[1],
                                sharex=True, sharey=True, figsize=figsize)
        f.suptitle(f_lbls[fi], fontsize=titlefont_sz, fontweight=titlefont_wt)
        try:
            axarr = axarr.ravel()
        except:
            axarr = [axarr]
        ax_dum = -1
        for ri, rval in enumerate(r_vals):
            for ti, tval in enumerate(t_vals):
                vals = [fval, rval, tval]
                dims = [f_dim[fi], r_dim[ri], t_dim[ti]]
                # try:
                dimval_tups = [(d,v) for d,v in zip(dims, vals)]
                try:
                    arcs = basic_slice(data, dimval_tups)
                    print('slice')
                except:
                    arcs = compound_take(data, dimval_tups)
                    print('compound take')
                mean_dims = np.where([d!=final_dim for d in d_dims])
                arcs = arcs.mean(axis=tuple(mean_dims[0]))
                print(arcs.shape)
                ax_dum += 1
                
                # column = (ax_dum + 1) % sp_dims[1]
                # row = floor(ax_dum / sp_dims[1])
                # plot here
                plot_connectivity_circle(arcs, uniq_chanlabels, indices,
                        # node_angles=node_angles,
                        facecolor='white',
                        textcolor='black',
                        colormap=plt.cm.hot_r,
                        title='title',
                        colorbar_size=0.4,
                        colorbar_pos=(-0.5,0.5),
                        fig=f,
                        subplot=(sp_dims[0], sp_dims[1], ax_dum + 1))

                axarr[ax_dum].text(-.5, .5, t_lbls[ti])  # time label
                if ti == 0:
                    axarr[ax_dum].text(-1.5, 0, r_lbls[ri])  # row label
        
        if savedir:
            save_fig(s, savedir, ptype, measure, f_lbls[fi])


# plot functions that worked on saved means
# can distribute as:
    # figures
    # subplots
        # rows
        # columns
    # glyph x-position
    # glyph color (lineplot)

def bar(s, measure, figure_by=None, subplot_by=None, set_by=None,
                    member_by=None, figsize_override=None):
    ptype = 'bar'
    alpha = 0.4
    error_config = {'ecolor': '0.3'}

    if measure not in dir(s):
        print('measure not found')
        return
    else:
        data, d_dims, d_dimlvls = get_data(s, measure)
        units, lims, cmap = get_plotparams(s, measure)

    f_dim, f_vals, f_lbls = handle_by(s, figure_by, d_dims, d_dimlvls)
    sp_dim, sp_vals, sp_lbls = handle_by(s, subplot_by, d_dims, d_dimlvls)
    s_dim, s_vals, s_lbls = handle_by(s, set_by, d_dims, d_dimlvls)
    m_dim, m_vals, m_lbls = handle_by(s, member_by, d_dims, d_dimlvls)

    n_sets = len(s_vals)
    swid = len(m_vals)
    width = 0.7 / swid
    colors = list(n_colors(len(m_vals)))

    sp_dims = subplot_heuristic(len(sp_vals))
    if figsize_override:
        figsize = figsize_override
    else:
        figsize = figsize_heuristic(sp_dims)
        
    for fi, fval in enumerate(f_vals):
        f, axarr = plt.subplots(sp_dims[0], sp_dims[1],
                                sharex=True, sharey=True, figsize=figsize)
        f.suptitle(f_lbls[fi], fontsize=titlefont_sz, fontweight=titlefont_wt)
        try:
            axarr = axarr.ravel()
        except:
            axarr = [axarr]
        for spi, spval in enumerate(sp_vals):
            for si, sval in enumerate(s_vals):
                r_lst = []
                for mi, mval in enumerate(m_vals):
                    vals = [fval, spval, sval, mval]
                    dims = [f_dim[fi], sp_dim[spi], s_dim[si], m_dim[mi]]
                    dimval_tups = [(d,v) for d,v in zip(dims, vals)]
                    try:
                        point = basic_slice(data, dimval_tups)
                        print('slice')
                    except:
                        point = compound_take(data, dimval_tups)
                        print('compound take')
                    pm = point.squeeze().mean(axis=0)
                    pe = ss.sem(point.squeeze(), axis=0)
                    xpos = si + (mi*width)
                    r = axarr[spi].bar(xpos, pm, width, color=colors[mi], yerr=pe,
                                       alpha=alpha, error_kw=error_config)
                    r_lst.append(r)
            axarr[spi].set_title(sp_lbls[spi], fontweight=stitlefont_wt)
            # axarr[spi].legend(loc='upper left')
            axarr[spi].set_xticks(np.arange(n_sets) + width)
            axarr[spi].set_xticklabels(s_lbls)
            axarr[spi].set_xlabel(d_dims[s_dim[si]], fontweight=stitlefont_wt)
            if spi % sp_dims[1] == 0:
                axarr[spi].set_ylabel(units, fontweight=stitlefont_wt)
            if spi % sp_dims[1] == len(sp_vals) - 1:
                axarr[spi].legend(r_lst, m_lbls)
            

