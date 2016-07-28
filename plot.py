''' plotting functions that operate on an eeg.Results object '''

import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import mne

from array_utils import basic_slice, compound_take, handle_by
from plot_utils import (subplot_heuristic, figsize_heuristic,
                    is_nonstr_sequence, nested_strjoin,
                    MidpointNormalize, blank_topo, plot_arcs, ordinalize_one)

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
               'coh':   {'data': 'coh', 'd_dims': 'coh_dims',
                         'd_dimlvls': 'coh_dim_lsts', 'units': 'coh_units',
                         'lims': 'coh_lims', 'cmap': plt.cm.PuOr,
                         'load': 'load_coh'},
               'phi':   {'data': 'phi', 'd_dims': 'tf_dims',
                         'd_dimlvls': 'tf_dim_lsts', 'units': 'phi_units',
                         'lims': 'phi_lims', 'cmap': plt.cm.PuOr,
                         'load': 'load_phi'},
               }

def get_plotparams(s, measure, lims=None, cmap_override=None):
    ''' given a measure, retrieve data and get plotting info '''

    measure_d = measure_pps[measure]
    if measure_d['data'] not in dir(s):
        getattr(s, measure_d['load'])()

    data        = getattr(s, measure_d['data'])
    d_dims      = getattr(s, measure_d['d_dims'])
    d_dimlvls   = getattr(s, measure_d['d_dimlvls'])
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

    return data, d_dims, d_dimlvls, units, lims, cmap

def save_fig(s, savedir, ptype, measure, label, form='eps'):
    ''' name and save the current figure to a target directory '''

    figname = s.gen_figname(ptype, measure, label)+'.'+form
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
           subplot_by={'POP': None},
           glyph_by={'condition': None},
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
        f.suptitle(f_lbls[fi])
        try:
            axarr = axarr.ravel()
        except:
            axarr = [axarr]
        for spi, spval in enumerate(sp_vals):
            for gi, gval in enumerate(g_vals):
                vals = [fval, spval, gval]
                dims = [f_dim[fi], sp_dim[spi], g_dim[gi]]
                try:
                    dimval_tups = [(d,v) for d,v in zip(dims, vals)]
                    line = basic_slice(data, dimval_tups)
                    print('slice')
                except:
                    line = compound_take(data, vals, dims)
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
            axarr[spi].set_title(sp_lbls[spi])
            axarr[spi].legend(loc='upper left')
            axarr[spi].set_xticks(s.time_ticks_pt_erp)
            axarr[spi].set_xticklabels(s.time_ticks_ms)
            axarr[spi].set_xlabel('Time (s)')
            axarr[spi].set_ylabel('Potential (' + s.pot_units + ')')
            axarr[spi].axhline(0, color='k', linestyle='--')
            axarr[spi].axvline(s.zero, color='k', linestyle='--')
            axarr[spi].set_xlim(s.time_plotlims)
        if savedir:
            s.save_fig(savedir, ptype, measure, f_lbls[fi])

def tf(s, measure='power',
          figure_by={'POP': None, 'channel': ['FZ']},
          subplot_by={'condition': None},
          lims='absmax', cmap_override=None,
          savedir=None):
    ''' plot time-frequency data as a rectangular contour image '''
    ptype = 'tf'

    if measure in ['power', 'itc', 'phi', 'coh']:
        data, d_dims, d_dimlvls, units, lims, cmap = \
            get_plotparams(s, measure, lims, cmap_override)
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
        f.suptitle(f_lbls[fi])
        try:
            axarr = axarr.ravel()
        except:
            axarr = [axarr]
        rect_lst = []
        c_lst = []
        for spi, spval in enumerate(sp_vals):
            vals = [fval, spval]
            dims = [f_dim[fi], sp_dim[spi]]
            try:
                dimval_tups = [(d,v) for d,v in zip(dims, vals)]
                print(dimval_tups)
                rect = basic_slice(data, dimval_tups)
                print('slice')
                print(rect.shape)
            except:
                rect = compound_take(data, vals, dims)
                print('done compound take')
            while len(rect.shape) > 2:
                rect = rect.mean(axis=0)
                print(rect.shape)
            ''' contour '''
            c = axarr[spi].contourf(rect, 8, cmap=cmap)
                                    # vmin=lims[0], vmax=lims[1])
            c_lst.append(c)
            rect_lst.append(rect)
            # c = axarr[spi].contour(rect, cmap=plt.cm.RdBu_r,
            #                         vmin=-4, vmax=4)
            # plt.clabel(c, inline=1, fontsize=9)
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
            axarr[spi].set_xlabel('Time (s)')
            axarr[spi].set_ylabel('Frequency (Hz)')
            axarr[spi].set_title(sp_lbls[spi])

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
            cbar.ax.set_ylabel(units, rotation=270)

        ''' colorbar '''
        # plt.subplots_adjust(right=0.85)
        # cbar_ax = f.add_axes([0.88, 0.12, 0.03, 0.75])
        # cbar = plt.colorbar(c, cax=cbar_ax)
        # cbar.ax.set_ylabel(units, rotation=270)
        # plt.colorbar(c, ax=axarr[spi])
        if savedir:
            s.save_fig(savedir, ptype, measure, f_lbls[fi])


def topo(s, measure='erp', times=list(range(0, 501, 125)),
            figure_by={'POP': ['C']},
            row_by={'condition': None},
            lims='absmax', cmap_override=None,
            savedir=None):
    ''' plot data as topographic maps at specific timepoints '''
    ptype = 'topo'
    final_dim = 'channel'

    if measure in ['erp', 'power', 'itc', 'phi']:
        data, d_dims, d_dimlvls, units, lims, cmap = \
            get_plotparams(s, measure, lims, cmap_override)
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
        f.suptitle(f_lbls[fi])
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
                    topo = compound_take(data, vals, dims)
                    print('compound take')
                mean_dims = np.where([d!=final_dim for d in d_dims])
                topo = topo.mean(axis=tuple(mean_dims[0]))
                print(topo.shape)
                ax_dum += 1
                im, cn = mne.viz.plot_topomap(topo, info,
                                    cmap=cmap, axes=axarr[ax_dum],
                                    show=False)
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
        cbar.ax.set_ylabel(units, rotation=270)
        if savedir:
            s.save_fig(savedir, ptype, measure, f_lbls[fi])


def arctopo(s, times=list(range(0, 501, 125)),
               figure_by={'POP': ['C']},
               row_by={'condition': None},
               lims='absmax', cmap_override=None,
               savedir=None):
    ''' plot data as topographic maps at specific timepoints '''
    measure='coh'
    ptype = 'arctopo'
    final_dim = 'pair'

    data, d_dims, d_dimlvls, units, lims, cmap = \
        get_plotparams(s, measure, lims, cmap_override)

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
        f.suptitle(f_lbls[fi])
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
                    arcs = compound_take(data, vals, dims)
                    print('compound take')
                mean_dims = np.where([d!=final_dim for d in d_dims])
                arcs = arcs.mean(axis=tuple(mean_dims[0]))
                print(arcs.shape)
                ax_dum += 1
                ax, im, cn, pos_x, pos_y = blank_topo(axarr[ax_dum], info)
                arches = plot_arcs(arcs, ax, s.cohpair_inds,
                                        pos_x, pos_y, cmap=cmap)
                arcs_lst.append(arcs)
                arch_lst.extend(arches)
                ''' labels '''
                axarr[ax_dum].text(-.5, .5, t_lbls[ti])  # time label
                if ti == 0:
                    axarr[ax_dum].text(-1.5, 0, r_lbls[ri])  # row label
        
        arcs_stack = np.stack(arcs_lst, axis=-1)
        mid = None
        if lims == 'absmax':
            vmax = np.max(np.fabs(arcs_stack))
            vmin = -vmax
        elif lims == 'minmax': # won't work with 2color asymmetric cmap
            vmax, vmin = np.max(arcs_stack), np.min(arcs_stack)
        else:
            vmax, vmin = lims[1], lims[0]
        
        # if vmin < 0 and vmax > 0:
        #     mid = 0
        
        print('vmin', vmin, 'vmax', vmax)

        cmap_array = cmap(range(cmap.N))
        for arch in arch_lst:
            val, arc_h = arch
            color_ind = ordinalize_one(val, size=cmap.N, lims=[vmin, vmax],
                                        mid=mid)
            arc_h.set_color(cmap_array[color_ind, :3])

        ''' colorbar '''
        plt.subplots_adjust(right=0.85)
        cbar_ax = f.add_axes([0.9, 0.15, 0.03, 0.75])
        mapper = plt.cm.ScalarMappable(cmap=cmap)
        mapper.set_array([vmin, vmax])
        cbar = plt.colorbar(mapper, cax=cbar_ax)
        cbar.ax.set_ylabel(units, rotation=270)
        if savedir:
            s.save_fig(savedir, ptype, measure, f_lbls[fi])