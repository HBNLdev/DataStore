''' finding files, walking directories, etc. '''

import os
import shutil
from glob import glob
from datetime import datetime

from .filename_parsing import parse_filename


def identify_files(starting_directory, filter_pattern='*', file_parameters={}, filter_list=[], time_range=()):
    ''' given a starting directory, and a glob-style filter pattern,
        recursively find all files that match the filter pattern '''

    file_list = []
    date_list = []

    for dName, sdName, fList in os.walk(starting_directory):

        for filename in fList:
            path = dName
            if 'reject' not in path:
                fullpath = os.path.join(path, filename)
                if os.path.exists(fullpath):
                    if shutil.fnmatch.fnmatch(filename, filter_pattern):
                        if file_parameters:
                            file_info = parse_filename(filename)

                            param_ck = [file_parameters[k] == file_info[k]
                                        for k in file_parameters]
                        else:
                            param_ck = [True]
                        if time_range:
                            time_ck = False
                            stats = os.stat(fullpath)
                            if time_range[0] < stats.st_ctime < time_range[1]:
                                time_ck = True
                        else:
                            time_ck = True
                        if filter_list:
                            filter_ck = any(
                                [s in filename for s in filter_list])
                        else:
                            filter_ck = True
                        if all(param_ck) and time_ck and filter_ck:
                            file_list.append(fullpath)
                            date_mod = datetime.fromtimestamp(
                                os.path.getmtime(fullpath))
                            date_list.append(date_mod)

    return file_list, date_list


def verify_files(files):
    ''' given a list of paths, return the ones that exist '''

    existent_files = []
    for f in files:
        if os.path.isfile(f):
            existent_files.append(f)
        else:
            print(f + ' does not exist')
    return existent_files


def get_dates(files):
    ''' given a list of paths, return a matching list of modified times '''

    return [os.path.getmtime(f) for f in files]


def get_dir(fp, depth=3):
    ''' from a full file_path, retrieve the directory containing it to an arbitrary depth
        e.g. get_dir('/usr/local/bin/file.py', 2) returns '/usr/local/' '''

    return '/'.join(fp.split('/')[:depth + 1])


def get_toc(target_dir, toc_str):
    ''' given dir containing toc files and string to be found in one,
        find the path of the most recently modified one matching the string '''
    pd_tocfiles = [f for f in glob(target_dir + '*.toc') if toc_str in f]
    pd_tocfiles.sort(key=os.path.getmtime)
    latest = pd_tocfiles[-1]
    return latest
