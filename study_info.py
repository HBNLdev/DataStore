studies= ['Wave12', 'COGA4500', 'ccGWAS', 'EAfamGWAS', 'COGA11k', 'ExomeSeq', 'AAfamGWAS', 'PhaseIV', 'fMRI-NKI-bd1', 'fMRI-NKI-bd2', 'fMRI-NYU-hr', 'a-subjects', 'c-subjects', 'h-subjects', 'p-subjects', 'smokeScreen']
experiments_parts= {
  'ans': ['r1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8'],
  'ant': ['a', 'j', 'w', 'p'],
  'aod': ['tt', 'nt'],
  'cpt': ['cg', 'c' ,'cn', 'un', 'dn', 'dd'],
  'ern': ['n50', 'n10', 'p10', 'p50'],
  'err': ['p', 'n'],
  'gng': ['g', 'ng'],
  'stp': ['c', 'in'],
  'vp3': ['tt', 'nt', 'nv'] }
time_windows = [(300,700), (300,500), (200,600), (200,500), (200,400), (200,300), (100,300), (100,200), (75,175), (50,150), (0,150)]
processing_types= [('v4','center9'), ('v6', 'center9'), ('v6', 'all')]

site_hash = {'a':'uconn',
             'b':'indiana',
             'c':'iowa',
             'd':'suny',
             'e':'washu',
             'f':'ucsd',
             '1':'uconn',
             '2':'indiana',
             '3':'iowa',
             '4':'suny',
             '5':'washu',
             '6':'ucsd',
             '7':'howard'
            }
sites = set(list(site_hash.values()))

Niklas_experiments = ['aod','vp3']
