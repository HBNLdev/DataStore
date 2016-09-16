''' utils for manipulating and slicing arrays '''

import numpy as np
import itertools
from collections import OrderedDict

# array functions
def permute_data(a, a_dimnames, out_dimnames):
    ''' given array a and a_dimnames, a tuple naming its dimensions,
        permute it so that its dimensions become out_dimnames '''
    if set(a_dimnames) != set(out_dimnames):
        print('the dimensions specifications don''t match')
        raise
    transpose_lst = []
    for dim in a_dimnames:
        transpose_lst.append(out_dimnames.index(dim))
    out_a = a.transpose(transpose_lst)
    return out_a, out_dimnames

def reverse_dimorder(array):
    ''' return version of array with dimensions in reversed order '''
    return np.transpose(array, list(range(len(array.shape)-1, -1, -1)))

def baseline_sub(array, pt_lims, along_dim=-1):
    ''' baseline array in a subtractive way '''
    return array - array.take(range(pt_lims[0], pt_lims[1]+1), axis=along_dim)\
                        .mean(axis=along_dim, keepdims=True)

def baseline_div(array, pt_lims, along_dim=-1):
    ''' baseline array in a divisive way '''
    return 10 * np.log10(array / array.take(range(pt_lims[0], pt_lims[1] + 1),
                                            axis=along_dim)
                         .mean(axis=along_dim, keepdims=True))

def convert_ms(time_array, ms):
    ''' given time array, find index nearest to given time value '''
    return np.argmin(np.fabs(time_array - ms))

def compound_take(a, dimval_tups):
    ''' given array, apply multiple indexing operations '''
    def apply_take(a, d, v):
        if isinstance(v, int) or isinstance(v, np.int64):
            return a.take([v], d) # stand-in for expand_dims
        else:
            return a.take(v, d)
    print(a.shape)
    for d, v in dimval_tups:
        if isinstance(v, tuple): # reserved for itertools.product-takes
            for dp, vp in zip(d, v):
                if isinstance(vp, dict): # reserved for operation-takes
                    (op, lvls), = vp.items()
                    if op == 'minus':
                        a = apply_take(a, dp, lvls[0]) - \
                            apply_take(a, dp, lvls[1])
                        # print('level subtraction')
                    elif op == 'mean':
                        print('mean')
                        a = apply_take(a, dp, lvls).mean(dp)
                        print(a.shape)
                        # a = np.expand_dims(a, axis=dp)
                        print(a.shape)
                else:
                    a = apply_take(a, dp, vp)
        elif isinstance(v, dict): # reserved for operation-takes
            (op, lvls), = v.items()
            if op == 'minus':
                try: # assume singleton indices
                    a = apply_take(a, d, lvls[0]) - apply_take(a, d, lvls[1])
                except ValueError:
                    print('level subtraction with mean')
                    a = apply_take(a, d, lvls[0]).mean(d) - \
                        apply_take(a, d, lvls[1]).mean(d)
                    a = np.expand_dims(a, axis=d)
                    print(a.shape)
            elif op == 'mean':
                print('mean')
                a = apply_take(a, d, lvls).mean(d)
                print(a.shape)
                # a = np.expand_dims(a, axis=d)
                print(a.shape)
        else:
            a = apply_take(a, d, v)
        # print(a.shape)
    return a

def basic_slice(a, in_dimval_tups):
    ''' given array a and list of (dim, val) tuples, basic-slice a '''
    slicer = [slice(None)]*len(a.shape) # initialize slicer

    # if the elements of the tuples are tuples, unpack and put in series
    dimval_tups = []
    for dvt in in_dimval_tups:
        if isinstance(dvt[0], tuple):
            for d, v in zip(*dvt):
                dimval_tups.append((d, v))
        else:
            dimval_tups.append(dvt)

    # build the slice list
    dimval_tups.sort(reverse=True) # sort descending by dims
    print('~~~slice time~~~')
    for d, v in dimval_tups:
        try:
            v[1] # for non-singleton vals
            if type(v) == range:
                print('got a range')
                slicer[d] = slice(v.start, v.stop, v.step)
            else:
                slicer[d] = v
        except: # for singleton vals
            print('singleton dim', d)
            if type(v) == np.ndarray:
                slicer[d] = v[0]
            else:
                slicer[d] = v
            slicer.insert(d+1, np.newaxis)

    print(slicer)
    return a[tuple(slicer)]


def handle_pairs(s, pairs_arg):
    ''' handle a pairs argument for plot.arctopo '''
    if isinstance(pairs_arg, list):
        return list(map(s.cohpair_lbls.index, pairs_arg))
    elif pairs_arg == 'all':
        return list(range(len(s.cohpair_inds)))
    elif pairs_arg in s.cohpair_sets:
        return list(map(s.cohpair_lbls.index, s.cohpair_sets[pairs_arg]))
    else:
        print('pairs incorrectly specified')
        raise

def handle_by(s, by_stage, d_dims, d_dimlvls, ordered=False):
    ''' handle a 'by' argument, which tells a plotting functions what parts
        of the data will be distributed across a plotting object.
        returns lists of the dimension, indices, and labels requested.
        if given a list, create above lists as products of requests '''

    if len(by_stage) > 1:
        # create list versions of the dim, vals, and labels
        tmp_dims, tmp_vals, tmp_labels, stage_lens = [], [], [], []
        for bs in by_stage.items():
            dims, vals, labels = interpret_by(s, bs, d_dims, d_dimlvls)
            tmp_dims.append(dims)
            tmp_vals.append(vals)
            tmp_labels.append(labels)
            stage_lens.append(len(dims))
        all_dims = list(itertools.product(*tmp_dims))
        all_vals = list(itertools.product(*tmp_vals))
        all_labels = list(itertools.product(*tmp_labels))
        if ordered:
            return all_dims, all_vals, all_labels, stage_lens
        else:
            return all_dims, all_vals, all_labels
    else:
        return interpret_by(s, tuple(by_stage.items())[0], d_dims, d_dimlvls)

def interpret_by(s, by_stage, data_dims, data_dimlvls):
    ''' by_stage: 2-tuple of variable name and requested levels
        data_dims: n-tuple describing the n dimensions of the data
        data_dimlvls: n-tuple of lists describing levels of each dim '''

    print('by stage is', by_stage[0])
    if by_stage[0] in data_dims:  # if variable is in data dims
        dim = data_dims.index(by_stage[0])
        print('data in dim', dim)
        if by_stage[1] == 'all': # if levels were not specified
            labels = data_dimlvls[dim]
            vals = list(range(len(labels)))
            print('iterate across available vals including', vals)
        else: # if levels were specified
            labels = by_stage[1]
            if data_dimlvls[dim].dtype == np.float64: # if array data
                vals = []
                for lbl in labels:
                    if isinstance(lbl, list):
                        if len(lbl) == 2:
                            tmp_inds = range(np.argmin(np.fabs(\
                                data_dimlvls[dim] - lbl[0])),
                                            np.argmin(np.fabs(\
                                data_dimlvls[dim] - lbl[1]))+1)
                        else:
                            tmp_inds = [np.argmin(np.fabs(\
                                data_dimlvls[dim] - lp)) for lp in lbl]
                    else:
                        tmp_inds = np.argmin(np.fabs(\
                            data_dimlvls[dim] - lbl))
                    vals.append(tmp_inds)
            else: # if non-array data (labeled, like conditions)
                vals = []
                for lbl in labels:
                    if isinstance(lbl, dict):
                        (op, lvls), = lbl.items()
                        vals.append({op: [np.where(data_dimlvls[dim]==l)[0]
                                                    for l in lvls]})
                    elif isinstance(lbl, list):
                        vals.append([np.where(data_dimlvls[dim]==lb)[0][0]
                                                    for lb in lbl])
                    else:
                        vals.append(np.where(data_dimlvls[dim]==lbl)[0])
            print('vals to iterate on are', vals)
    elif by_stage[0] in s.demog_df.columns:  # if variable in demog dims
        dim = data_dims.index('subject')
        print('demogs in', dim)
        if by_stage[1] == 'all':
            labels = s.demog_df[by_stage[0]].unique()
            vals = [np.where(s.demog_df[by_stage[0]].values == lbl)[0]
                    for lbl in labels]
            print('iterate across available vals including', vals)
        else:
            labels = by_stage[1]
            vals = []
            for lbl in labels:
                if isinstance(lbl, dict):
                    (op, lvls), = lbl.items()
                    vals.append({op: [np.where(s.demog_df[by_stage[0]]==l)[0]
                                                for l in lvls]})
                elif isinstance(lbl, list):
                    vals.append([np.where(s.demog_df[by_stage[0]]==lb)[0][0]
                                                for lb in lbl])
                else:
                    vals.append(np.where(s.demog_df[by_stage[0]]==lbl)[0])
            print('vals to iterate on are', vals)
    else:
        print('variable not found in data or demogs')
        raise
    dims = [dim] * len(vals)
    return dims, vals, labels