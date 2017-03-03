''' working with text files '''

import os
from glob import glob


def txt2list(path):
    ''' given path to text file, return a list of its lines '''
    with open(path, 'r') as f:
        lst = [line.strip() for line in f]
    return lst


def list2txt(lst, filename, replace=False):
    ''' given a list, write its contents to a text file.
        if replace is True, will overwrite. '''

    if not os.path.exists(filename) or replace:
        of = open(filename, 'w')
        for item in lst:
            of.write(str(item) + '\n')
        of.close()
    return


def get_toc(target_dir, toc_str):
    ''' given dir containing toc files and string to be found in one,
        find the path of the most recently modified one matching the string '''
    pd_tocfiles = [f for f in glob(target_dir + '*.toc') if toc_str in f]
    pd_tocfiles.sort(key=os.path.getmtime)
    latest = pd_tocfiles[-1]
    return latest


def txt_tolines(path):
    ''' given path to text file, return its lines as list '''
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    return lines


def find_lines(lines, start, end):
    ''' find lines that match start and end exressions '''
    tmp_lines = [l for l in lines if l[:len(start)] == start and l[-len(end):] == end]
    return tmp_lines
