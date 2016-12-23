''' statistics functions that operate on an eeg.results object, creating
	attributes that contain nd-arrays of test statistics and/or the results
    of clustering those test-statistics given a threshold '''

import numpy as np
from sklearn import linear_model

import mne
from mne.stats import (permutation_cluster_1samp_test,
                       permutation_cluster_test,
                       spatio_temporal_cluster_1samp_test,
                       spatio_temporal_cluster_test)

from ._plot_utils import measure_pps
from ._array_utils import get_data, permute_data



def regress_linear(xvals, yvals):
    ''' given equal length vectors, do a linear regression. returns the coefficient (slope) and variance explained '''

    if len(xvals.shape) < 2:
        xvals = xvals.reshape(-1, 1)
    if len(yvals.shape) < 2:
        yvals = yvals.reshape(-1, 1)
    regr = linear_model.LinearRegression()
    regr.fit(xvals, yvals)

    pred_y = regr.predict(xvals)

    return pred_y, regr.coef_[0][0], regr.score(xvals, yvals)


# should be able to do:
# - 1-sample deviation from 0 (OK for ERP, power)
# - 1-sample deviation from phase uniformity (OK for ITC, ISPS)
# - t-test comparing conditions on any measure
# - t-test comparing groups on any measure
# - 3+ conditions/groups --> use one-way F
# - mixed linear model?

def onesample(s, measure):
    ''' given a measure, perform a one-sample deviation from 0 test '''

    if measure in measure_pps.keys():
        data, d_dims, d_dimlvls = get_data(s, measure)
    else:
        print('data not recognized')
        raise


def multisample(s, measure, test_dim='condition', spec_dict=None,
                threshold=30, tail=0):
    ''' given a measure, and a short dimension (condition or group),
        perform a t- or F-test across that dimension) for all time/frequency
        points and channels.
        all other data specifications must be provided by spec_dict.
        since this is not using spatial information about channels, channels
        should not be used as a clustering dimension. '''

    if measure in measure_pps.keys():
        data, d_dims, d_dimlvls = get_data(s, measure)
    else:
        print('data not recognized')
        raise

    # first do any takes you might need
    if spec_dict:
        # do something
        pass

    # then permute the data to be in the right order
    want_dims = ('subject', 'timepoint', 'channel', 'condition')
    data, d_dims = permute_data(data, d_dims, want_dims)

    if test_dim == 'condition':  # should be last dim
        X = [data[..., cond] for cond in range(data.shape[-1])]

    T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test(X,
                                 threshold=threshold,
                                 tail=tail,
                                 connectivity=s.ch_connectivity,
                                 n_jobs=1)

    stat_dict = {'T_obs': T_obs, 'clusters': clusters,
                 'cluster_p_values': cluster_p_values, 'H0': H0}
    new_attrname = measure + '_stats'
    setattr(s, new_attrname, stat_dict)


def onesample_spatial(s, measure):
    ''' given a measure, perform a one-sample deviation from 0 test '''

    if measure in measure_pps.keys():
        data, d_dims, d_dimlvls = get_data(s, measure)
    else:
        print('data not recognized')
        raise


def multisample_spatial(s, measure, test_dim='condition', spec_dict=None,
                        threshold=30, tail=0):
    ''' given a measure, and a short dimension (condition or group),
        perform a t- or F-test across that dimension) for all time/frequency
        points and channels.
        all other data specifications must be provided by spec_dict. '''

    if measure in measure_pps.keys():
        data, d_dims, d_dimlvls = get_data(s, measure)
    else:
        print('data not recognized')
        raise

    # first do any takes you might need
    if spec_dict:
        # do something
        pass

    # then permute the data to be in the right order
    want_dims = ('subject', 'timepoint', 'channel', 'condition')
    data, d_dims = permute_data(data, d_dims, want_dims)

    if test_dim == 'condition':  # should be last dim
        X = [data[..., cond] for cond in range(data.shape[-1])]

    T_obs, clusters, cluster_p_values, H0 = \
        spatio_temporal_cluster_test(X,
                                     threshold=threshold,
                                     tail=tail,
                                     connectivity=s.ch_connectivity,
                                     n_jobs=1)

    stat_dict = {'T_obs': T_obs, 'clusters': clusters,
                 'cluster_p_values': cluster_p_values, 'H0': H0}
    new_attrname = measure + '_stats_ms_spatial'
    setattr(s, new_attrname, stat_dict)
