''' doing math '''

import numpy as np
from statsmodels.robust.scale import huber


def convert_scale(scale_array, scale_val):
    ''' given array, find index nearest to given value '''
    return np.argmin(np.fabs(scale_array - scale_val))


def robust_datemean(row):
    ''' row apply function that finds the robust mean (if possible) among all passed values, which should be dates.
        if the robust mean cannot be found, the regular mean is used. '''

    row_na = row.dropna()
    try:
        row_na_dates = row_na.astype(np.datetime64)
    except TypeError:  # already datetime
        # print('could not convert', row_na)
        row_na_dates = row_na
    date_ints = row_na_dates.values.tolist()
    try:
        rm = huber(date_ints)[0].item()  # attempt to use huber robust mean
    except ValueError:
        rm = int(np.mean(reject_outliers(np.array(date_ints))))  # use mean after rejecting outliers
        # rm = int(np.mean(date_ints))
    return rm
    # return int(np.mean(date_ints))


def reject_outliers(data, m=2.5):
    ''' return version of data only containing points inside arbitrary scaled absolute distance from the median '''
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    return data[s < m]


def largest_prime_factor(n):
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
    return n


def fix_ratio(ratio, n=50, k=10):
    ''' given a ratio measurement stemming from n observations,
        return a version in which none of the values are exactly 0 or 1. '''

    if ratio == 0:
        return 1 / (n * k)
    elif ratio == 1:
        return 1 - (1 / (n * k))
    else:
        return ratio
