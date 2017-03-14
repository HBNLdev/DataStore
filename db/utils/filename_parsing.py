''' parsing filenames in the HBNL '''

import os

import numpy as np

site_hash = {'a': 'uconn',
             'b': 'indiana',
             'c': 'iowa',
             'd': 'suny',
             'e': 'washu',
             'f': 'ucsd',
             'h': 'unknown',
             '1': 'uconn',
             '2': 'indiana',
             '3': 'iowa',
             '4': 'suny',
             '5': 'washu',
             '6': 'ucsd',
             '7': 'howard',
             '8': 'unknown'
             }
site_hash_rev = {
    v: k for k, v in site_hash.items() if k in [str(n) for n in range(1, 9)]}

rd_subdir_tosite = {'alc': 'a-subject',
                    'control': 'c-subjects',
                    'high_risk': 'h-subjects'}

system_shorthands = {'mc': 'masscomp',
                     'ns': 'neuroscan'}

current_experiments = ['eeo', 'eec', 'vp3', 'cpt', 'ern',
                       'ant', 'aod', 'ans', 'stp', 'gng']

experiment_shorthands = {'aa': 'ap3',
                         'ab': 'abk',
                         'an': 'ant',
                         'ao': 'aod',
                         'as': 'asa',
                         'av': 'avm',
                         'ax': 'axv',
                         'bl': 'blk',
                         'bt': 'btk',
                         'cl': 'clr',
                         'cn': 'cnv',
                         'co': 'cob',
                         'cp': 'cpt',
                         'cs': 'csb',
                         'ea': 'eas',
                         'ec': 'eec',
                         'eo': 'eeo',
                         'es': 'esa',
                         'et': 'etg',
                         'fa': 'fac',
                         'fb': 'fbc',
                         'fc': 'fcc',
                         'fd': 'fdc',
                         'fe': 'fec',
                         'fg': 'fgc',
                         'fh': 'fhc',
                         'fn': 'fne',
                         'fo': 'foa',
                         'fr': 'fre',
                         'ft': 'ftl',
                         'il': 'iln',
                         'im': 'imn',
                         'is': 'ish',
                         'it': 'iti',
                         'ke': 'key',
                         'mf': 'mmf',
                         'ml': 'mlg',
                         'mm': 'mmn',
                         'mn': 'mng',
                         'mo': 'mob',
                         'ms': 'mms',
                         'mt': 'mtg',
                         'ob': 'obj',
                         'om': 'oma',
                         'os': 'osa',
                         'ow': 'owa',
                         'ox': 'oxg',
                         'rf': 'rfa',
                         'rn': 'rno',
                         'ro': 'rot',
                         'rp': 'rep',
                         'sh': 'shp',
                         'sp': 'spc',
                         'st': 'str',
                         'ti': 'tim',
                         'tm': 'trm',
                         'tv': 'trv',
                         'va': 'va3',
                         'vp': 'vp3'}


def site_fromIDstr(IDstr):
    ''' given ID string, return the site '''

    if IDstr[0].isnumeric():
        return site_hash[IDstr[0]]
    else:
        return 'suny'


def parse_filename(filename, include_full_ID=False):
    ''' given an HBNL-style filename -- e.g. 'vp3_6_e1_10162024_avg.mt' or 'vp2e0157027.mt',
        extract important information from it '''

    if os.path.sep in filename:
        filename = os.path.split(filename)[1]

    # neuroscan type
    if '_' in filename:
        pieces = filename.split('_')
        experiment = pieces[0]
        version = pieces[1]
        session_piece = pieces[2]
        session_letter = session_piece[0]
        run_number = session_piece[1]

        subject_piece = pieces[3]
        system = 'neuroscan'  # preliminary
        fam_number = subject_piece[1:5]
        subject = subject_piece[5:8]
        if fam_number in ['0000', '0001'] and subject_piece[0] in 'achp':
            site = subject_piece[0] + '-subjects'
            family = np.nan
            if fam_number == '0000':  # no family
                family = 0
            # need to check master file here to see if h and p subjects are
            # masscomp or neuroscan
            if fam_number == '0001':
                # need to look here for three recordings on family 0001
                family = 0

        else:
            family = fam_number
            site = site_hash[subject_piece[0].lower()]
            if not subject_piece[0].isdigit():
                system = 'masscomp'
                subject_piece = site_hash_rev[site_hash[
                    subject_piece[0].lower()]] + subject_piece[1:]

    # masscomp
    else:
        system = 'masscomp'
        experiment_short = filename[:2]
        experiment = experiment_shorthands[experiment_short]
        site_letter = filename[3].lower()
        if filename[4:8] in ['0000', '5000']:  # no family
            site = site_letter + '-subjects'
            family = 0
        else:
            family = filename[4:8]
            site = site_hash[site_letter.lower()]

        run_number = '1'  # determine first or second run
        if filename[4] == '5':
            run_number = '2'

        if site_letter.lower() == site_letter:
            session_letter = 'a'
        else:
            session_letter = 'b'

        subject = filename[8:11]
        subject_piece = site_hash_rev[site_hash[site_letter]] + filename[4:11]
        version = filename[2]

    output = {'system': system,
              'experiment': experiment,
              'session': session_letter,
              'run': run_number,
              'site': site,
              'family': family,
              'subject': subject,
              'id': subject_piece,
              'version': version}

    if include_full_ID:
        try:
            output['ID'] = site_hash_rev[site] + family + subject
        except KeyError:
            output['ID'] = filename

    return output


def parse_filename_tester():
    ''' test function for parse_filename '''

    cases = [('vp3_6_e1_10162024_avg.mt', 'neuroscan', 'vp3', 'uconn', 162, 24, 'e', 1),
             ('vp2e0157027.mt', 'masscomp', 'vp3', 'washu', 157, 27, 'a', 1),
             ('aod_5_a1_c0000857_avg.h1', 'neuroscan',
              'aod', 'c-subjects', 0, 857, 'a', 1),
             ('vp2c5000027.mt', 'masscomp',
              'vp3', 'c-subjects', 0, 27, 'a', 2),
             ('aod_5_a2_c0000857_avg.h1', 'neuroscan',
              'aod', 'c-subjects', 0, 857, 'a', 2)
             ]
    for case in cases:
        info = parse_filename(case[0])
        if ((info['system'] != case[1]) or (info['experiment'] != case[2]) or
                (info['site'] != case[3]) or (int(info['family']) != case[4]) or (int(info['subject']) != case[5]) or
                (info['session'] != case[6]) or (int(info['run']) != case[7])):
            print(info, ' mismatch for case: ', case)


def parse_mt_name(file_or_path):
    ''' parse the filename of a *.mt file -- a text file containing results of ERP peak-picking '''

    if os.path.sep in file_or_path:
        name = os.path.split(file_or_path)[1]
    else:
        name = file_or_path
    base_ext = name.split('.')

    parts = parse_filename(base_ext[0])
    parts['case'] = base_ext[2]

    return parts


def parse_STinv_path(path):
    ''' parse the path of a *.st.mat file -- a MATLAB-produced binary data file containing the inverted result
        of trial-averaged power values created using the Stockwell Transform '''

    path, filename = os.path.split(path)

    info = {}

    path_parts = path.split(os.path.sep)
    info['prc_ver'] = path_parts[2][-2]
    info['param_string'] = path_parts[-2]
    info['n_chans'] = path_parts[-5][-2:]

    base_ext = filename.split('.')
    fn_parts = parse_filename(base_ext[0])
    fn_parts['ID'] = fn_parts['id']
    fn_parts['case'] = base_ext[2]
    info.update(fn_parts)

    return info


def parse_rd_path(filepath):
    ''' parse the path of a *.rd file --  a legacy raw EEG data format from the masscomp system '''

    path_parts = filepath.split(os.path.sep)
    full_filename = path_parts[-1]
    filename, ext = os.path.splitext(full_filename)

    system = 'masscomp'
    subdir = path_parts[3]
    if subdir == 'coga':
        site = path_parts[-3]
    elif subdir in rd_subdir_tosite.keys():
        site = rd_subdir_tosite[subdir]
    else:
        site = 'unknown'

    experiment_short = filename[:2]
    try:
        experiment = experiment_shorthands[experiment_short]
    except KeyError:
        experiment = 'unknown'

    if filename[4:8] in ['0000', '5000']:  # no family
        family = None
    else:
        family = filename[4:8]

    run_number = '1'  # determine first or second run
    if filename[4] == '5':
        run_number = '2'

    subject_piece = path_parts[-2]
    site_letter = filename[3].lower()
    if site_letter.lower() == site_letter:
        session_letter = 'a'
    else:
        session_letter = 'b'

    subject = filename[8:11]
    version = filename[2]

    output = {'system': system,
              'experiment': experiment,
              'session': session_letter,
              'run': run_number,
              'site': site,
              'family': family,
              'subject': subject,
              'ID': subject_piece,
              'version': version,
              'path': filepath}

    return output


def parse_cnt_path(filepath):
    ''' parse the path of a *.cnt file -- containing raw continuous EEG data recorded with the Neuroscan system '''

    full_filename = os.path.split(filepath)[1]
    filename, ext = os.path.splitext(full_filename)

    pieces = filename.split('_')
    experiment = pieces[0]
    version = pieces[1]
    session_piece = pieces[2]
    session_letter = session_piece[0]
    run_number = session_piece[1]

    subject_piece = pieces[3]
    system = 'neuroscan'  # preliminary
    fam_number = subject_piece[1:5]
    subject = subject_piece[5:8]

    if fam_number in ['0000', '0001'] and subject_piece[0] in 'acghp':
        site = subject_piece[0] + '-subjects'
        family = np.nan
        if fam_number == '0000':  # no family
            family = 0
        if fam_number == '0001':
            family = 0
    else:
        family = fam_number
        site = site_hash[subject_piece[0].lower()]
        if not subject_piece[0].isdigit():
            system = 'masscomp'
            subject_piece = site_hash_rev[site_hash[
                subject_piece[0].lower()]] + subject_piece[1:]

    try:
        bitrate_or_note = pieces[4]
    except IndexError:
        bitrate_or_note = None

    try:
        note = pieces[5]
    except IndexError:
        note = None

    try:
        int(bitrate_or_note)
        bitrate = bitrate_or_note
    except ValueError:
        note = bitrate_or_note
        bitrate = None

    output = {'path': filepath,
              'system': system,
              'experiment': experiment,
              'session': session_letter,
              'run': run_number,
              'site': site,
              'family': family,
              'subject': subject,
              'ID': subject_piece,
              'version': version,
              'bitrate': bitrate,
              'note': note}

    return output


def parse_cnth1_path(filepath):
    ''' parse the path of a *.cnt.h1 file -- containing resampled, raw, continuous EEG data as well as lots of useful
        metadata regarding the recording '''

    full_dir, full_filename = os.path.split(filepath)

    dir_pieces = full_dir.split(os.path.sep)
    rec_type = dir_pieces[6]
    n_chans = rec_type[-2:]

    filename, ext = os.path.splitext(full_filename)

    pieces = filename.split('_')
    experiment = pieces[0]
    version = pieces[1]
    session_piece = pieces[2]
    session_letter = session_piece[0]
    run_number = session_piece[1]

    subject_piece = pieces[3]
    system = 'neuroscan'  # preliminary
    fam_number = subject_piece[1:5]
    subject = subject_piece[5:8]

    if fam_number in ['0000', '0001'] and subject_piece[0] in 'acghp':
        site = subject_piece[0] + '-subjects'
        family = np.nan
        if fam_number == '0000':  # no family
            family = 0
        if fam_number == '0001':
            family = 0
    else:
        family = fam_number
        site = site_hash[subject_piece[0].lower()]
        if not subject_piece[0].isdigit():
            system = 'masscomp'
            subject_piece = site_hash_rev[site_hash[
                subject_piece[0].lower()]] + subject_piece[1:]

    try:
        bitrate = pieces[4]
    except IndexError:
        bitrate = None

    output = {'filepath': filepath,
              'system': system,
              'experiment': experiment,
              'session': session_letter,
              'run': run_number,
              'site': site,
              'family': family,
              'subject': subject,
              'ID': subject_piece,
              'uID': subject_piece + '_' + session_letter,
              'version': version,
              'bitrate': bitrate,
              'n_chans': n_chans,
              'rec_type': rec_type}

    return output


def parse_avgh1_path(filepath):
    ''' parse the path of a *.avg.h1 file -- containing ERP data, which are trial-averaged amplitudes after
        filtering, trial rejection, and baselining procedures '''

    full_dir, full_filename = os.path.split(filepath)

    dir_pieces = full_dir.split(os.path.sep)
    rec_type = dir_pieces[6]
    param_str = dir_pieces[4]
    n_chans = rec_type[-2:]

    filename, ext = os.path.splitext(full_filename)

    pieces = filename.split('_')
    experiment = pieces[0]
    version = pieces[1]
    session_piece = pieces[2]
    session_letter = session_piece[0]
    run_number = session_piece[1]

    subject_piece = pieces[3]
    system = 'neuroscan'  # preliminary
    fam_number = subject_piece[1:5]
    subject = subject_piece[5:8]

    if fam_number in ['0000', '0001'] and subject_piece[0] in 'acghp':
        site = subject_piece[0] + '-subjects'
        family = np.nan
        if fam_number == '0000':  # no family
            family = 0
        if fam_number == '0001':
            family = 0
    else:
        family = fam_number
        site = site_hash[subject_piece[0].lower()]
        if not subject_piece[0].isdigit():
            system = 'masscomp'
            subject_piece = site_hash_rev[site_hash[
                subject_piece[0].lower()]] + subject_piece[1:]

    output = {'filepath': filepath,
              'system': system,
              'experiment': experiment,
              'session': session_letter,
              'run': run_number,
              'site': site,
              'family': family,
              'subject': subject,
              'ID': subject_piece,
              'uID': subject_piece + '_' + session_letter,
              'version': version,
              'param_str': param_str,
              'n_chans': n_chans,
              'rec_type': rec_type}

    return output
