''' sas tools '''

import os
from collections import OrderedDict

import pandas as pd
from sas7bdat import SAS7BDAT

from .knowledge.questionnaires import map_ph4, map_ph4_ssaga

map_subject = {'core': {'file_pfixes': []}}

parent_dir = '/processed_data/zork/zork-phase4-69/session/'
n_header_lines = 30


def extract_descriptions(path):
    ''' given path to .sas7bdat file, returns dictionary mapping column labels
        to their verbose descriptions in the SAS header.
        dictionary will only contain an entry if there was new information present
        (if there was a description, and it was different from the label) '''
    f = SAS7BDAT(path)
    kmap = OrderedDict()
    for line in str(f.header).splitlines()[n_header_lines + 1:]:
        line_parts = line.split(maxsplit=4)
        label = line_parts[1]
        try:
            description = line_parts[4].rstrip()
            if description == label or description[0] == '$':
                continue
            else:
                kmap[label] = description
        except IndexError:
            pass
    return kmap


def exemplary_files(kdict):
    ''' given a questionnaire knowledge map,
        return a new dictionary mapping questionnaire names to the filepath
        of an exemplary .sas7bdat file for each file prefix '''
    exemplars = {}
    for test, tdict in kdict.items():
        for fpx in tdict['file_pfixes']:
            fd = parent_dir + test
            fn = fpx + '.sas7bdat'
            fp = os.path.join(fd, fn)
            if os.path.exists(fp):
                exemplars[test] = fp
            else:
                print(fp, 'did not exist')
    return exemplars


def build_labelmaps():
    ''' return a dict in which keys are questionnaires names and values are
        dictionaries mapping column labels to descriptions '''
    comb_dict = map_ph4.copy()
    comb_dict.update(map_ph4_ssaga)
    exemplars = exemplary_files(comb_dict)
    big_kmap = {}
    for test, fp in exemplars.items():
        kmap = extract_descriptions(fp)
        big_kmap[test] = kmap
    return big_kmap


def df_fromsas(fullpath, id_lbl='ind_id'):
    ''' convert .sas7bdat to dataframe.
        unused because fails on incorrectly formatted files. '''

    # read csv in as dataframe
    df = pd.read_sas(fullpath, format='sas7bdat')

    # convert id to str and save as new column
    df[id_lbl] = df[id_lbl].apply(int).apply(str)
    df['ID'] = df[id_lbl]

    return df
