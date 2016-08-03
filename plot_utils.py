''' utils that aid plotting but don't fit in plot.py '''

import collections

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc

def subplot_heuristic(n):
    ''' for n subplots, determine best grid layout dimensions '''
    def isprime(n):
        for x in range(2, int(np.sqrt(n)) + 1):
            if n % x == 0:
                return False
        return True
    if n > 6 and isprime(n):
        n += 1
    num_lst, den_lst = [n], [1]
    for x in range(2, int(np.sqrt(n)) + 1):
        if n % x == 0:
            den_lst.append(x)
            num_lst.append(n // x)
    ratios = np.array([a / b for a, b in zip(num_lst, den_lst)])
    best_ind = np.argmin(ratios - 1.618)  # most golden
    if den_lst[best_ind] < num_lst[best_ind]: # always have more rows than cols
        return den_lst[best_ind], num_lst[best_ind]
    else:
        return num_lst[best_ind], den_lst[best_ind]

def figsize_heuristic(sp_dims):
    ''' given subplot dims (rows, columns), return (width, height) of figure '''
    max_width, max_height = 15, 10
    r = sp_dims[1] / sp_dims[0]
    phi = 1.618
    if r >= phi:
        out_width = max_width
        sp_width = out_width / sp_dims[1]
        sp_height = sp_width / phi
        out_height = sp_height * sp_dims[0]
    else:
        out_height = max_height
        sp_height = max_height / sp_dims[0]
        sp_width = sp_height * phi
        out_width = sp_width * sp_dims[1]
    return out_width, out_height

# text-processing for processively generating figure names

def is_nonstr_sequence(obj):
    ''' return True for all sequences except strings '''
    if isinstance(obj, str):
        return False
    return isinstance(obj, collections.Sequence)

def nested_strjoin(obj, delim='_'):
    ''' recursively convert obj contents to strings and join '''
    js = ''
    for thing in obj:
        if is_nonstr_sequence(thing):
            js += nested_strjoin(thing)
            js = js[:-1]
        else:
            js += str(thing) + delim
    return js

# plot utilities

def blank_topo(in_ax, info):
    ''' create a blank headplot '''
    topo = np.empty(61)
    cmap = LinearSegmentedColormap.from_list('mycmap', ['white', 'white'] )
    ax, im, cn, pos_x, pos_y = plot_topomap(topo, info, cmap=cmap,
                                            contours=0, axes=in_ax,
                                            show=False)
    return ax, im, cn, pos_x, pos_y

def create_arc(pt1, pt2, color=(0, 0, 0), linewidth=1, alpha=1):
    ''' given two x, y coordinate-tuples, return arc between them '''
    x0, y0 = pt1
    x1, y1 = pt2
    midpoint = ((x0 + x1)/2, (y0 + y1)/2)
    vdist = np.linalg.norm(np.array((x0, y0)) - np.array((x1, y1)))
    hdist = 0.1 + (vdist / 10) # this works pretty good?
    angle = np.degrees(np.arctan2((x1 - x0), -(y1 - y0)))
    return Arc(midpoint, hdist, vdist, angle, 90, 270, color=color,
              linewidth=linewidth, alpha=alpha)

def ordinalize_one(num, size, lims, mid=None):
    ''' given datum, ordinalize it to a given index size '''
    
    vmin, vmax = lims

    data_prop = (num - vmin) / (vmax -  vmin)
    data_prop_ind = int(round(data_prop * (size-1)))

    if mid is not None:
        if num < 0:
            abs_prop = np.fabs( (num - mid) / vmin )
        else:
            abs_prop = np.fabs( (num - mid) / vmax )

    if data_prop_ind < 0:
        data_prop_ind = 0
    elif data_prop_ind > size - 1:
        data_prop_ind = size - 1
    return data_prop_ind, abs_prop

def ordinalize_many(data, size=256, lims=[0, 0.25]):
    ''' given data, ordinalize it to a given index size '''
    if lims:
        vmin, vmax = lims
    else:
        vmin, vmax = data.min(), data.max()
    data_prop = (data - vmin) / (vmax -  vmin)
    data_prop_inds = (data_prop * (size-1)).round().astype(int)
    data_prop_inds[data_prop_inds < 0] = 0
    data_prop_inds[data_prop_inds > size - 1] = size - 1
    return data_prop_inds, vmin, vmax

def plot_arcs(arcs, ax, pair_inds, pos_x, pos_y, cmap, lims=[0, 0.25]):
    ''' given 1d array of connection strengths, plot as colored arcs '''
    arc_inds, vmin, vmax = ordinalize_many(arcs, cmap.N, lims)
    lims = np.array( [vmin, vmax] )
    cmap_array = cmap(range(cmap.N))
    arch_lst = []
    for pind, pair in enumerate(pair_inds):
        ch1, ch2 = pair
        pt1 = pos_x[ch1], pos_y[ch1]
        pt2 = pos_x[ch2], pos_y[ch2]
        arc = create_arc(pt1, pt2, cmap_array[arc_inds[pind], :3])
        arch = ax.add_patch(arc)
        arch_lst.append((arcs[pind], arch))
    return arch_lst


# colormap utilities

class MidpointNormalize(colors.Normalize):
    ''' create asymmetric norm '''

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

class ImageFollower(object):
    ''' update image in response to changes in clim or cmap on another image '''

    def __init__(self, follower):
        self.follower = follower

    def __call__(self, leader):
        self.follower.set_cmap(leader.get_cmap())
        self.follower.set_clim(leader.get_clim())


# borrowed code

def plot_topomap(data, pos, vmin=None, vmax=None, cmap=None, sensors=True,
                 res=64, axes=None, names=None, show_names=False, mask=None,
                 mask_params=None, outlines='head', image_mask=None,
                 contours=6, image_interp='bilinear', show=True,
                 head_pos=None, onselect=None, axis=None):
    ''' see the docstring for mne.viz.plot_topomap,
        which i've simply modified to return more objects '''

    from matplotlib.widgets import RectangleSelector
    from mne.io.pick import (channel_type, pick_info, _pick_data_channels)
    from mne.utils import warn
    from mne.viz.utils import (_setup_vmin_vmax, plt_show)
    from mne.defaults import _handle_default
    from mne.channels.layout import _find_topomap_coords
    from mne.io.meas_info import Info
    from mne.viz.topomap import _check_outlines, _prepare_topomap, _griddata, _make_image_mask, _plot_sensors, _draw_outlines

    data = np.asarray(data)

    if isinstance(pos, Info):  # infer pos from Info object
        picks = _pick_data_channels(pos)  # pick only data channels
        pos = pick_info(pos, picks)

        # check if there is only 1 channel type, and n_chans matches the data
        ch_type = set(channel_type(pos, idx)
                      for idx, _ in enumerate(pos["chs"]))
        info_help = ("Pick Info with e.g. mne.pick_info and "
                     "mne.channels.channel_indices_by_type.")
        if len(ch_type) > 1:
            raise ValueError("Multiple channel types in Info structure. " +
                             info_help)
        elif len(pos["chs"]) != data.shape[0]:
            raise ValueError("Number of channels in the Info object and "
                             "the data array does not match. " + info_help)
        else:
            ch_type = ch_type.pop()

        if any(type_ in ch_type for type_ in ('planar', 'grad')):
            # deal with grad pairs
            from ..channels.layout import (_merge_grad_data, find_layout,
                                           _pair_grad_sensors)
            picks, pos = _pair_grad_sensors(pos, find_layout(pos))
            data = _merge_grad_data(data[picks]).reshape(-1)
        else:
            picks = list(range(data.shape[0]))
            pos = _find_topomap_coords(pos, picks=picks)

    if data.ndim > 1:
        raise ValueError("Data needs to be array of shape (n_sensors,); got "
                         "shape %s." % str(data.shape))

    # Give a helpful error message for common mistakes regarding the position
    # matrix.
    pos_help = ("Electrode positions should be specified as a 2D array with "
                "shape (n_channels, 2). Each row in this matrix contains the "
                "(x, y) position of an electrode.")
    if pos.ndim != 2:
        error = ("{ndim}D array supplied as electrode positions, where a 2D "
                 "array was expected").format(ndim=pos.ndim)
        raise ValueError(error + " " + pos_help)
    elif pos.shape[1] == 3:
        error = ("The supplied electrode positions matrix contains 3 columns. "
                 "Are you trying to specify XYZ coordinates? Perhaps the "
                 "mne.channels.create_eeg_layout function is useful for you.")
        raise ValueError(error + " " + pos_help)
    # No error is raised in case of pos.shape[1] == 4. In this case, it is
    # assumed the position matrix contains both (x, y) and (width, height)
    # values, such as Layout.pos.
    elif pos.shape[1] == 1 or pos.shape[1] > 4:
        raise ValueError(pos_help)

    if len(data) != len(pos):
        raise ValueError("Data and pos need to be of same length. Got data of "
                         "length %s, pos of length %s" % (len(data), len(pos)))

    norm = min(data) >= 0
    vmin, vmax = _setup_vmin_vmax(data, vmin, vmax, norm)
    if cmap is None:
        cmap = 'Reds' if norm else 'RdBu_r'

    pos, outlines = _check_outlines(pos, outlines, head_pos)

    if axis is not None:
        axes = axis
        warn('axis parameter is deprecated and will be removed in 0.13. '
             'Use axes instead.', DeprecationWarning)
    ax = axes if axes else plt.gca()
    pos_x, pos_y = _prepare_topomap(pos, ax)
    if outlines is None:
        xmin, xmax = pos_x.min(), pos_x.max()
        ymin, ymax = pos_y.min(), pos_y.max()
    else:
        xlim = np.inf, -np.inf,
        ylim = np.inf, -np.inf,
        mask_ = np.c_[outlines['mask_pos']]
        xmin, xmax = (np.min(np.r_[xlim[0], mask_[:, 0]]),
                      np.max(np.r_[xlim[1], mask_[:, 0]]))
        ymin, ymax = (np.min(np.r_[ylim[0], mask_[:, 1]]),
                      np.max(np.r_[ylim[1], mask_[:, 1]]))

    # interpolate data
    xi = np.linspace(xmin, xmax, res)
    yi = np.linspace(ymin, ymax, res)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = _griddata(pos_x, pos_y, data, Xi, Yi)

    if outlines is None:
        _is_default_outlines = False
    elif isinstance(outlines, dict):
        _is_default_outlines = any(k.startswith('head') for k in outlines)

    if _is_default_outlines and image_mask is None:
        # prepare masking
        image_mask, pos = _make_image_mask(outlines, pos, res)

    mask_params = _handle_default('mask_params', mask_params)

    # plot outline
    linewidth = mask_params['markeredgewidth']
    patch = None
    if 'patch' in outlines:
        patch = outlines['patch']
        patch_ = patch() if callable(patch) else patch
        patch_.set_clip_on(False)
        ax.add_patch(patch_)
        ax.set_transform(ax.transAxes)
        ax.set_clip_path(patch_)

    # plot map and countour
    im = ax.imshow(Zi, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
                   aspect='equal', extent=(xmin, xmax, ymin, ymax),
                   interpolation=image_interp)

    # This tackles an incomprehensible matplotlib bug if no contours are
    # drawn. To avoid rescalings, we will always draw contours.
    # But if no contours are desired we only draw one and make it invisible .
    no_contours = False
    if contours in (False, None):
        contours, no_contours = 1, True
    cont = ax.contour(Xi, Yi, Zi, contours, colors='k',
                      linewidths=linewidth)
    if no_contours is True:
        for col in cont.collections:
            col.set_visible(False)

    if _is_default_outlines:
        from matplotlib import patches
        patch_ = patches.Ellipse((0, 0),
                                 2 * outlines['clip_radius'][0],
                                 2 * outlines['clip_radius'][1],
                                 clip_on=True,
                                 transform=ax.transData)
    if _is_default_outlines or patch is not None:
        im.set_clip_path(patch_)
        if cont is not None:
            for col in cont.collections:
                col.set_clip_path(patch_)

    if sensors is not False and mask is None:
        _plot_sensors(pos_x, pos_y, sensors=sensors, ax=ax)
    elif sensors and mask is not None:
        idx = np.where(mask)[0]
        ax.plot(pos_x[idx], pos_y[idx], **mask_params)
        idx = np.where(~mask)[0]
        _plot_sensors(pos_x[idx], pos_y[idx], sensors=sensors, ax=ax)
    elif not sensors and mask is not None:
        idx = np.where(mask)[0]
        ax.plot(pos_x[idx], pos_y[idx], **mask_params)

    if isinstance(outlines, dict):
        _draw_outlines(ax, outlines)

    if show_names:
        if names is None:
            raise ValueError("To show names, a list of names must be provided"
                             " (see `names` keyword).")
        if show_names is True:
            def _show_names(x):
                return x
        else:
            _show_names = show_names
        show_idx = np.arange(len(names)) if mask is None else np.where(mask)[0]
        for ii, (p, ch_id) in enumerate(zip(pos, names)):
            if ii not in show_idx:
                continue
            ch_id = _show_names(ch_id)
            ax.text(p[0], p[1], ch_id, horizontalalignment='center',
                    verticalalignment='center', size='x-small')

    plt.subplots_adjust(top=.95)

    if onselect is not None:
        ax.RS = RectangleSelector(ax, onselect=onselect)
    plt_show(show)
    return ax, im, cont, pos_x, pos_y