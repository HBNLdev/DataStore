import numpy as np


def basic_diagnostics(df):
    ''' given a dataframe, return a string with some basic diagnostics '''

    shape = df.shape
    dupes = df.index.has_duplicates
    unique = df.index.is_unique
    mono = df.index.is_monotonic

    base_str = 'shape: {} | dupes: {} | unique: {} | monotonic: {}'
    diag_str = base_str.format(shape, dupes, unique, mono)

    return diag_str


def index_eq(df1_in, df2_in, do_sort=False):
    ''' given 2 dataframes, print diagnostics related to whether the two have the same indices '''

    df1 = df1_in.copy()
    df2 = df2_in.copy()

    if do_sort and not df1.index.is_monotonic and not df2.index.is_monotonic:
        print('sorting')
        df1 = df1.sort_index()
        df2 = df2.sort_index()

    index_len_match = len(df1.index) == len(df2.index)
    if index_len_match:
        print('same number of indices')
        index_bool = (df1.index == df2.index)
        all_index_match = index_bool.all()
        if all_index_match:
            print('all indices match')
        else:
            print('some indices do not match')
            diff_ilocs = np.where(~index_bool)[0]
            print('indices differ at', len(diff_ilocs), 'ilocs:', diff_ilocs)
            diff_inds1 = df1.index[~index_bool]
            diff_inds2 = df2.index[~index_bool]
            print('lefthand inds:', diff_inds1)
            print('righthand inds:', diff_inds2)
    else:
        print('different number of indices')

    df1_novelinds = df1.index[~df1.index.isin(df2.index)]
    df2_novelinds = df2.index[~df2.index.isin(df1.index)]
    if df1_novelinds.size > 0:
        print('lefthand df had', df1_novelinds.size, 'novel inds:', df1_novelinds)
    if df2_novelinds.size > 0:
        print('righthand df had', df2_novelinds.size, 'novel inds:', df2_novelinds)


def column_eq(df1, df2):
    ''' given 2 dataframes, print diagnostics related to whether the two have the same columns '''

    column_len_match = len(df1.columns) == len(df2.columns)
    if column_len_match:
        print('same number of columns')
        column_bool = (df1.columns == df2.columns)
        all_columns_match = column_bool.all()
        if all_columns_match:
            print('all columns match')
        else:
            print('some columns do not match')
            diff_ilocs = np.where(~column_bool)[0]
            print('indices differ at', len(diff_ilocs), 'ilocs:', diff_ilocs)
            diff_inds1 = df1.columns[~column_bool]
            diff_inds2 = df2.columns[~column_bool]
            print('lefthand inds:', diff_inds1)
            print('righthand inds:', diff_inds2)
    else:
        print('different number of columns')

    df1_novelcols = df1.columns[~df1.columns.isin(df2.columns)]
    df2_novelcols = df2.columns[~df2.columns.isin(df1.columns)]
    if df1_novelcols.size > 0:
        print('lefthand df had', df1_novelcols.size, 'novel inds:', df1_novelcols)
    if df2_novelcols.size > 0:
        print('righthand df had', df2_novelcols.size, 'novel inds:', df2_novelcols)


def contents_eq(df1_in, df2_in, join_how='left', lsuffix='_larry', rsuffix='_ricky'):
    ''' given two similar dataframes with differences, make a diff dataframe
        which contains the differing rows and columns '''

    if df1_in.shape != df2_in.shape:
        print('input dfs had different shapes')
        print('it is suggested to treat them with index/col conditioning first')
        print('to get more meaningful results')

    if df1_in.shape[0] < df2_in.shape[0]:
        print('lefthand df has fewer rows, a left join is suggested')
    else:
        print('righthand df has fewer rows, a right join is suggested')

    df1_tmp = df1_in.fillna('NA')
    df2_tmp = df2_in.fillna('NA')
    lsl = len(lsuffix)
    rsl = len(rsuffix)
    dfj = df1_tmp.join(df2_tmp, how=join_how, lsuffix=lsuffix, rsuffix=rsuffix)

    nonmatch_cols = [col for col in dfj.columns if
                     col[-lsl:] != lsuffix and col[-rsl:] != rsuffix]

    df1_cols = [col for col in dfj.columns if col[-lsl:] == lsuffix]
    df2_cols = [col for col in dfj.columns if col[-rsl:] == rsuffix]

    df1 = dfj[df1_cols + nonmatch_cols]
    df2 = dfj[df2_cols + nonmatch_cols]

    if df1.shape != df2.shape:
        print('something went wrong')
        return

    df1.rename(columns={col: col[:-lsl] for col in df1.columns if col[-lsl:] == lsuffix}, inplace=True)
    df2.rename(columns={col: col[:-rsl] for col in df2.columns if col[-rsl:] == rsuffix}, inplace=True)

    ne_bool = df1 != df2

    total_diffs = ne_bool.sum().sum()
    print('there were a total of', total_diffs, 'differences')

    alldiff_cols = ne_bool.all(0)
    alldiff_rows = ne_bool.all(1)

    n_alldiffcols = alldiff_cols.sum()
    n_alldiffrows = alldiff_rows.sum()

    if n_alldiffcols > 0 or n_alldiffrows > 0:
        print(n_alldiffcols, 'columns and', n_alldiffrows, 'rows were totally different')
        print('the completely differing rows were', df1.index[alldiff_cols])
        print('the completely differing columns were', df1.columns[alldiff_rows])

    anydiff_cols = ne_bool.any(0)
    anydiff_rows = ne_bool.any(1)

    n_anydiffcols = anydiff_cols.sum()
    n_anydiffrows = anydiff_rows.sum()

    if n_anydiffcols > 0 or n_anydiffrows > 0:
        print(n_anydiffcols, 'columns and', n_anydiffrows, 'rows had differing vals')
        print('the differing rows were', df1.index[anydiff_rows])
        print('the differing columns were', df1.columns[anydiff_cols])

    # return lists of differing rows / columns?
    # return df1.index[anydiff_rows].tolist(), df1.columns[anydiff_cols].tolist()

    # return a dataframe showing the differences
    out_cols = nonmatch_cols.copy()
    for col in df1.columns[anydiff_cols]:
        out_cols.append(col + lsuffix)
        out_cols.append(col + rsuffix)

    return dfj.loc[df1.index[anydiff_rows], out_cols]

def check_column(diff_df, col_name, lsuffix='_larry', rsuffix='_ricky'):
    ''' given a diff dataframe, and the name of a column,
        return the subset of the diff dataframe related to that column
        in which differences are present '''

    lcol_name = col_name + lsuffix
    rcol_name = col_name + rsuffix
    out_subset = diff_df.ix[diff_df[lcol_name] != diff_df[rcol_name], [lcol_name, rcol_name]]
    return out_subset

def check_allcoldiffs(diff_df, lsuffix='_larry', rsuffix='_ricky'):
    ''' given a diff dataframe, create a dictionary whose keys are differing columns,
        and whose values are the differing subsets related to those columns '''
    
    lsl = len(lsuffix)
    rsl = len(rsuffix)
    
    df1_cols = [col[:-lsl] for col in diff_df.columns if col[-lsl:] == lsuffix]
    df2_cols = [col[:-rsl] for col in diff_df.columns if col[-rsl:] == rsuffix]
    
    match_cols = list(set(df1_cols) & set(df2_cols))
    
    diff_dict = {}
    for col in match_cols:
        diff_dict[col] = check_column(diff_df, col, lsuffix=lsuffix, rsuffix=rsuffix)
        
    return diff_dict