import os

import h5py
import pandas as pd

from .file_handling import parse_filename


def add_avgh1paths(df):
    pass


class EmptyStackError(Exception):
    def __init__(s):
        print('all files in the stack were missing')


class AVGH1Stack:
    ''' represent a list of .avg.h1's as a stacked array '''

    def __init__(s, path_lst):
        s.init_df(path_lst)

    def init_df(s, path_lst):

        row_lst = []
        missing_count = 0
        for fp in path_lst:
            if os.path.exists(fp):
                avgh1 = AVGH1(fp)
                row_lst.append(avgh1.info)
            else:
                missing_count += 1

        if row_lst:
            print(missing_count, 'files missing')
        else:
            raise EmptyStackError

        s.data_df = pd.DataFrame.from_records(row_lst)
        s.data_df.set_index(['ID', 'session', 'powertype', 'experiment', 'condition'], inplace=True)
        s.data_df.sort_index(inplace=True)


class AVGH1:
    ''' represents a single .avg.h1 file '''

    def __init__(s, filepath):
        s.filepath = filepath
        s.filename = os.path.split(s.filepath)[1]
        s.file_info = parse_filename(s.filename)

    def load(s):
        s.loaded = h5py.File(s.filepath, 'r')
        s.electrodes = [st.decode() for st in list(s.loaded['file']['run']['run'])[0][-2]]
        s.electrodes_61 = s.electrodes[0:31] + s.electrodes[32:62]
        s.samp_freq = 256
