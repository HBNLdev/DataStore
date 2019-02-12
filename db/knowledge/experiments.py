''' knowledge about experiments, their conditions, and regions of interest '''

from collections import defaultdict

# maps for experiments to the list of conditions
exp_cases = {'ant': ['a', 'j', 'w', 'p'],  # need renaming (for masscomp files, tttt --> ajwp)
             'aod': ['tt', 'nt'],  # need renaming (t --> tt)
             'vp3': ['tt', 'nt', 'nv'],
             'ans': ['r1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8'],
             'cpt': ['cg', 'c', 'cn', 'un', 'dn', 'dd'],
             'ern': ['n50', 'n10', 'p10', 'p50'],
             'err': ['p', 'n'],
             'gng': ['g', 'ng'],
             'stp': ['c', 'in'],
             'cas':['Con','Inc','Neg','Alc','Ntr','Ins','Prc'] }

exp_cases_desc= {'ant': {'a':'antonym',
                          'j':'jumble',
                          'w':'word',
                          'p':'prime'},  # need renaming (for masscomp files, tttt --> ajwp)
             'aod': {'tt':'target',
                    'nt':'non-target'},  # need renaming (t --> tt)
             'vp3': {'tt':'target',
                      'nt':'non-target',
                      'nv':'novel'},
             'ans': {'r1':'tone1',
                      'f2':'tone2',
                      'f3':'tone3',
                      'f4':'tone4',
                      'f5':'tone5',
                      'f6':'tone6',
                      'f7':'tone7',
                      'f8':'tone8'},
             'cpt': {'cg':'cue go',
                      'c':'cue',
                      'cn':'cue no-go',
                      'un':'uncued no-go',
                      'dn':'distractor no-go',
                      'dd':'distractor distractor'},
             'ern': {'n50':'negative 50',
                      'n10':'negative 10',
                      'p10':'positive 10',
                      'p50':'positive 50'},
             'err': {'p':'positive',
                      'n':'negative'},
             'gng': {'g':'go',
                      'ng':'no-go'},
             'stp': {'c':'congruent',
                    'in':'incongruent'},
              'cas':{'Alc': 'Alcohol',
                     'Con': 'Congruent',
                     'Inc': 'Incongruent',
                     'Neg': 'Negative',
                     'Ntr': 'Neutral',
                     'Ins':'Instruc',
                     'Prc':'Practice'}
                    }

# maps experiments to possible #s of chans
exp_nchans = {'ant': ['21', '32', '64'],
              'aod': ['21', '32', '64'],
              'vp3': ['21', '32', '64'],
              'ans': ['32', '64'],
              'ern': ['32', '64'],
              'err': ['32', '64'],
              'gng': ['64'],
              'cpt': ['64'],
              'stp': ['64'], }

# maps from new condition order for ant to the old (fix)
ant_cind_map = {'1': '3', '2': '1', '3': '4', '4': '2'}

# chans

interesting_chans = ['C3', 'C4', 'CZ', 'F3', 'F4', 'FZ', 'OZ', 'P3', 'P4', 'PO7', 'PO8', 'PZ']
center_nine = ['C3', 'C4', 'CZ', 'F3', 'F4', 'FZ', 'P3', 'P4', 'PZ']

# ERP

erp_exp_condpeakchans = dict()
erp_exp_condpeakchans['aod'] = [('nt', 'N1', ['FZ', 'PZ']),
                                ('nt', 'P2', ['CZ', 'F4', 'F3']),
                                ('t', 'N1', ['FZ', 'PZ']),
                                ('t', 'P3', ['PZ']), ]
erp_exp_condpeakchans['vp3'] = [('nt', 'N1', ['FZ', 'PZ']),
                                ('nt', 'P3', ['PZ', 'F4', 'F3']),
                                ('nv', 'N1', ['P4', 'PZ', 'P3']),
                                ('nv', 'P3', ['PZ']),
                                ('t', 'N1', ['P4', 'PZ', 'P3']),
                                ('t', 'P3', ['CZ', 'PZ']), ]
erp_exp_condpeakchans['ant'] = [('a', 'N4', ['FZ', 'PZ']), ]

# ERO

ero_exp_info = defaultdict(dict)

ero_exp_info['ant']['cases'] = ['a', 'j', 'w', 'p']
ero_exp_info['ant']['tf_wins'] = [(200, 500, 1, 3), (200, 400, 4, 7), (50, 200, 8, 13)]
ero_exp_info['ant']['elec'] = ['CZ', 'FZ', 'PO7', 'PO8']
ero_exp_info['ant']['tfwin_elecs'] = [(200, 500, 1, 3, ['CZ', 'F3', 'F4']),
                                      (100, 200, 4, 7, ['PZ', 'OZ']),
                                      (200, 400, 4, 7, ['FZ']),
                                      (50, 200, 8, 13, ['PO7', 'PO8'])]

ero_exp_info['aod']['cases'] = ['tt', 'nt']
ero_exp_info['aod']['tf_wins'] = [(200, 400, 1, 3), (200, 300, 4, 7), (100, 200, 8, 13), (50, 100, 15, 24),
                                  (350, 450, 15, 24)]
ero_exp_info['aod']['elec'] = ['FZ', 'PZ']
ero_exp_info['aod']['tfwin_elecs'] = [(200, 400, 1, 3, ['PZ']),
                                      (200, 300, 4, 7, ['FZ']),
                                      (100, 200, 8, 13, ['FZ']),
                                      (50, 100, 15, 24, ['FZ']),
                                      (350, 450, 15, 24, ['PZ'])]

ero_exp_info['vp3']['cases'] = ['tt', 'nt', 'nv']
ero_exp_info['vp3']['tf_wins'] = [(200, 600, 1, 3), (200, 400, 4, 7), (50, 200, 8, 13), (400, 500, 15, 24)]
ero_exp_info['vp3']['elec'] = ['FZ', 'PZ', 'PO7', 'PO8']
ero_exp_info['vp3']['tfwin_elecs'] = [(200, 600, 1, 3, ['PZ']),
                                      (200, 400, 4, 7, ['FZ']),
                                      (400, 700, 4, 7, ['P3', 'PZ', 'P4']),
                                      (400, 700, 8, 13, ['P3', 'PZ', 'P4']),
                                      (400, 500, 15, 24, ['P3', 'PZ', 'P4']),
                                      (50, 200, 8, 13, ['PO7', 'PO8']), ]

ero_exp_info['cpt']['cases'] = ['cg', 'cn', 'un', ]
ero_exp_info['cpt']['tf_wins'] = [(200, 400, 1, 3), (100, 200, 4, 7), (200, 400, 4, 7)]
ero_exp_info['cpt']['elec'] = ['FZ', 'PZ', 'PO7', 'PO8']
ero_exp_info['cpt']['tfwin_elecs'] = [(200, 400, 1, 3, ['PZ']),
                                      (300, 400, 1, 3, ['FZ']),
                                      (200, 300, 4, 7, ['FZ']),
                                      (400, 500, 8, 13, ['C3', 'C4']),
                                      (100, 200, 4, 7, ['PO7', 'PO8'])]

ero_exp_info['gng']['cases'] = ['g', 'ng', ]
ero_exp_info['gng']['tf_wins'] = [(200, 500, 1, 3), (200, 400, 4, 7), (50, 150, 8, 13), (300, 500, 15, 24)]
ero_exp_info['gng']['elec'] = ['FZ', 'PZ', 'C3', 'C4', 'PO7', 'PO8']
ero_exp_info['gng']['tfwin_elecs'] = [(200, 500, 1, 3, ['PZ']),
                                      (200, 400, 4, 7, ['FZ']),
                                      (300, 500, 15, 24, ['C3', 'C4']),
                                      (50, 150, 8, 13, ['PO7', 'PO8'])]

ero_exp_info['err']['cases'] = ['p', 'n', ]
ero_exp_info['err']['tf_wins'] = [(200, 500, 1, 3), (200, 400, 4, 7), (100, 200, 4, 7), (100, 200, 8, 13)]
ero_exp_info['err']['elec'] = ['FZ', 'CZ', 'PZ', 'PO7', 'PO8']
ero_exp_info['err']['tfwin_elecs'] = [(200, 500, 1, 3, ['CZ', 'PZ']),
                                      (200, 400, 4, 7, ['FZ']),
                                      (100, 200, 8, 13, ['FZ', 'PO7', 'PO8']),
                                      (100, 200, 4, 7, ['PO7', 'PO8']), ]

ero_exp_info['stp']['cases'] = ['c', 'in', ]
ero_exp_info['stp']['tf_wins'] = [(100, 300, 1, 3), (200, 400, 4, 7), ]
ero_exp_info['stp']['elec'] = ['FZ', 'OZ']
ero_exp_info['stp']['tfwin_elecs'] = [(100, 300, 1, 3, ['PZ', 'OZ']),
                                      (200, 300, 4, 7, ['FZ']), ]

ero_exp_info['ans']['cases'] = ['r1', 'f2', 'f3', 'f4', ]
ero_exp_info['ans']['tf_wins'] = [(200, 500, 1, 3), (200, 500, 4, 7), (100, 200, 8, 13)]
ero_exp_info['ans']['elec'] = ['FZ', 'CZ', ]
ero_exp_info['ans']['tfwin_elecs'] = [(200, 500, 1, 3, ['CZ', 'PZ']),
                                      (200, 500, 4, 7, ['FZ']),
                                      (100, 200, 8, 13, ['FZ'])]
