''' working with dataframes '''

import pandas as pd


def df_fromcsv(fullpath, id_lbl='ind_id', na_val=''):
    ''' convert csv into dataframe, converting ID column to standard '''

    # read csv in as dataframe
    try:
        df = pd.read_csv(fullpath, na_values=na_val)
    except pd.parser.EmptyDataError:
        print('csv file was empty, continuing')
        return pd.DataFrame()

    # convert id to str and save as new column
    df[id_lbl] = df[id_lbl].apply(int).apply(str)
    df['ID'] = df[id_lbl]
    df.set_index('ID', drop=False, inplace=True)

    return df


def join_columns(row, columns):
    return '_'.join([row[field] for field in columns])


def column_split(v, ind=0, sep='_'):
    return v.split(sep)[ind]
