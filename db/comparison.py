''' tools for comparing two dataframes '''

from collections import OrderedDict

import numpy as np
import pandas as pd


def basic_diagnostics(df):
    ''' given a dataframe, return a string with some basic diagnostics '''

    shape = df.shape
    dupes = df.index.has_duplicates
    unique = df.index.is_unique
    mono = df.index.is_monotonic

    base_str = 'shape: {} | dupes: {} | unique: {} | monotonic: {}'
    diag_str = base_str.format(shape, dupes, unique, mono)

    return diag_str


def index_sort(df1, df2):
    ''' in place '''

    if not df1.index.is_monotonic:
        print('sorting index of lefthand index')
        df1.sort_index(inplace=True)
    if not df2.index.is_monotonic:
        print('sorting index of righthand index')
        df2.sort_index(inplace=True)


def columns_sort(df1, df2):
    ''' not in place '''

    if (df1.columns != df1.columns.sort_values()).any():
        print('sorting columns of lefthand index')
        df1 = df1.reindex_axis(sorted(df1.columns), axis=1)

    if (df2.columns != df2.columns.sort_values()).any():
        print('sorting columns of lefthand index')
        df2 = df2.reindex_axis(sorted(df2.columns), axis=1)

    return df1, df2


def series_eq(s1, s2):
    ''' if the input series are exacty equal, return true. otherwise, return false,
        while printing some diagnostics '''

    len_match = len(s1) == len(s2)
    if len_match:
        print('same number of elements')
        eq_bool = (s1 == s2)
        all_match = eq_bool.all()
        if all_match:
            print('all elements match')
            return True
    else:
        print('different number of elements')

    s1_novels = s1[~s1.isin(s2)]
    s2_novels = s2[~s2.isin(s1)]
    if s1_novels.size > 0:
        print('lefthand df had', s1_novels.size, 'novel elements:', s1_novels)
    if s2_novels.size > 0:
        print('righthand df had', s2_novels.size, 'novel elements:', s2_novels)

    return False


def contents_eq(df1_in, df2_in, join_how='inner', lsuffix='_larry', rsuffix='_ricky', ind_sort=False):
    ''' given two similar dataframes with differences, make a diff dataframe
        which contains the differing rows and columns '''

    print('index diagnostics:')
    if ind_sort:
        index_sort(df1_in, df2_in)
    inds_eq = series_eq(df1_in.index, df2_in.index)
    if not inds_eq:
        print('inds differ, consider how you are joining for meaningful results')
        print('see the join_how param, which is inner by default')
    print('~~~')

    print('column diagnostics:')
    cols_eq = series_eq(df1_in.columns, df2_in.columns)
    if not cols_eq:
        print('differing columns will be appended to the diff dataframe')
    print('~~~')

    df1_tmp, df2_tmp = df1_in.fillna('NA'), df2_in.fillna('NA')
    dfj = df1_tmp.join(df2_tmp, how=join_how, lsuffix=lsuffix, rsuffix=rsuffix)

    lsl, rsl = len(lsuffix), len(rsuffix)
    nonmatch_cols = [col for col in dfj.columns if
                     col[-lsl:] != lsuffix and col[-rsl:] != rsuffix]
    df1_cols = [col for col in dfj.columns if col[-lsl:] == lsuffix]
    # df2_cols = [col for col in dfj.columns if col[-rsl:] == rsuffix]
    df2_cols = [col[:-lsl] + rsuffix for col in df1_cols]

    df1, df2 = dfj[df1_cols + nonmatch_cols], dfj[df2_cols + nonmatch_cols]

    if df1.shape != df2.shape:
        print('something went wrong')
        return

    df1.rename(columns={col: col[:-lsl] for col in df1.columns if col[-lsl:] == lsuffix}, inplace=True)
    df2.rename(columns={col: col[:-rsl] for col in df2.columns if col[-rsl:] == rsuffix}, inplace=True)

    ne_bool = df1 != df2
    total_diffs = ne_bool.sum().sum()
    print('there were a total of', total_diffs, 'differences')

    alldiff_rows, alldiff_cols = ne_bool.all(1), ne_bool.all(0)
    anydiff_rows, anydiff_cols = ne_bool.any(1), ne_bool.any(0)

    n_anydiffrows = anydiff_rows.sum()
    n_anydiffcols = anydiff_cols.sum()
    n_alldiffrows = alldiff_rows.sum()
    n_alldiffcols = alldiff_cols.sum()

    if n_alldiffcols > 0 or n_alldiffrows > 0:
        print(n_alldiffcols, 'columns and', n_alldiffrows, 'rows were totally different')
        print('the completely differing rows were', df1.index[alldiff_rows])
        print('the completely differing columns were', df1.columns[alldiff_cols])

    if n_anydiffcols > 0 or n_anydiffrows > 0:
        print(n_anydiffcols, 'columns and', n_anydiffrows, 'rows had differing vals')
        print('the differing rows were', df1.index[anydiff_rows])
        print('the differing columns were', df1.columns[anydiff_cols])

    # return a dataframe showing the differences
    out_cols = nonmatch_cols.copy()
    for col in df1.columns[anydiff_cols]:
        out_cols.append(col + lsuffix)
        out_cols.append(col + rsuffix)

    return dfj.loc[df1.index[anydiff_rows], out_cols]


def combine(df1_in, df2_in, lsuffix='_larry', rsuffix='_ricky', ind_sort=False):
    ''' given a two dataframes, match their indices and columns, then combine corresponding info such that:
            - if both values are present (but different), take the right-hand one
            - if one value is missing, take the other
    '''

    print('index diagnostics:')
    if ind_sort:
        index_sort(df1_in, df2_in)
    inds_eq = series_eq(df1_in.index, df2_in.index)
    if not inds_eq:
        print('some inds differ, differing inds will be unchanged in the outer combine results')
    print('~~~')

    print('column diagnostics:')
    cols_eq = series_eq(df1_in.columns, df2_in.columns)
    if not cols_eq:
        print('some columns differ, differing columns will be unchanged in the outer combine results')
    print('~~~')

    dfji = df1_in.join(df2_in, how='inner', lsuffix=lsuffix, rsuffix=rsuffix)
    dfjo = df1_in.join(df2_in, how='outer', lsuffix=lsuffix, rsuffix=rsuffix)

    lsl, rsl = len(lsuffix), len(rsuffix)
    nonmatch_cols = [col for col in dfji.columns if
                     col[-lsl:] != lsuffix and col[-rsl:] != rsuffix]
    df1_cols = [col for col in dfji.columns if col[-lsl:] == lsuffix]
    # df2_cols = [col for col in dfji.columns if col[-rsl:] == rsuffix]
    df2_cols = [col[:-lsl] + rsuffix for col in df1_cols]

    df1_tmp, df2_tmp = dfji[df1_cols + nonmatch_cols], dfji[df2_cols + nonmatch_cols]

    if df1_tmp.shape != df2_tmp.shape:
        print('something went wrong')

    df1_tmp.rename(columns={col: col[:-lsl] for col in df1_tmp.columns if col[-lsl:] == lsuffix}, inplace=True)
    df2_tmp.rename(columns={col: col[:-rsl] for col in df2_tmp.columns if col[-rsl:] == rsuffix}, inplace=True)

    df1_tmp_notnan, df2_tmp_notnan = df1_tmp.notnull(), df2_tmp.notnull()

    df1, df2 = df1_tmp.where(df1_tmp_notnan, other=df2_tmp), df2_tmp.where(df2_tmp_notnan, other=df1_tmp)

    ne_bool = df1 != df2

    df_diffresolved = df1.where(~ne_bool, other=df2)

    df1_uniqinds = set(df1_in.index) - set(df2_in.index)
    df2_uniqinds = set(df2_in.index) - set(df1_in.index)

    df1_uniqrows = dfjo.loc[df1_uniqinds, df1_cols + nonmatch_cols]
    df2_uniqrows = dfjo.loc[df2_uniqinds, df2_cols + nonmatch_cols]

    df1_uniqrows.rename(columns={col: col[:-lsl] for col in df1_uniqrows.columns if col[-lsl:] == lsuffix},
                        inplace=True)
    df2_uniqrows.rename(columns={col: col[:-rsl] for col in df2_uniqrows.columns if col[-rsl:] == rsuffix},
                        inplace=True)

    df_altogether = pd.concat([df_diffresolved, df1_uniqrows, df2_uniqrows]).sort_index()

    return df_altogether


def check_column(diff_df, col_name, lsuffix='_larry', rsuffix='_ricky'):
    ''' given a diff dataframe, and the name of a column,
        return the subset of the diff dataframe related to that column
        in which differences are present '''

    lcol_name = col_name + lsuffix
    rcol_name = col_name + rsuffix
    out_subset = diff_df.ix[diff_df[lcol_name] != diff_df[rcol_name], [lcol_name, rcol_name]]
    return out_subset


def take_left(diff_df, col_name, lsuffix='_larry', rsuffix='_ricky'):
    ''' given a diff dataframe, and the name of a column,
        return the subset of the diff dataframe related to that column
        in which differences are present
        but ONLY the left-hand column '''

    lcol_name = col_name + lsuffix
    rcol_name = col_name + rsuffix
    out_subset = diff_df.ix[diff_df[lcol_name] != diff_df[rcol_name], lcol_name]
    return out_subset


def check_allcoldiffs(diff_df, lsuffix='_larry', rsuffix='_ricky'):
    ''' given a diff dataframe, create a dictionary whose keys are differing columns,
        and whose values are the differing subsets related to those columns '''

    lsl = len(lsuffix)
    rsl = len(rsuffix)

    df1_cols = [col[:-lsl] for col in diff_df.columns if col[-lsl:] == lsuffix]
    df2_cols = [col[:-rsl] for col in diff_df.columns if col[-rsl:] == rsuffix]

    match_cols = list(set(df1_cols) | set(df2_cols))

    diff_dict = OrderedDict()
    for col in sorted(match_cols):
        diff_dict[col] = check_column(diff_df, col, lsuffix=lsuffix, rsuffix=rsuffix)

    return diff_dict


def twocol_MSE(df):
    return np.power(df.diff(axis=1).iloc[:, 1], 2).mean()


def print_diffdict(diff_dict, do_MSE=False):
    ''' given a diff dict, print its contents '''

    cdum = 1
    for column, df in diff_dict.items():
        print('______________________________________________________________')
        print(cdum, '|', column)
        if do_MSE:
            try:
                mse = twocol_MSE(df)
                print('MSE:', mse)
            except:
                pass
        print(df)
        print('______________________________________________________________')
        print('')
        cdum += 1
