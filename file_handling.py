'''Handling HBNL files
'''

import os, shutil, subprocess
import utils
from collections import OrderedDict

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
site_hash_rev = { v:k for k,v in site_hash.items() if k in [str(n) for n in range(1,8)] }
			
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
							
def parse_filename(filename,include_full_ID=False):

	if os.path.sep in filename:
		filename = os.path.split()[1]

	#neuroscan type
	if '_' in filename:
		pieces = filename.split('_')
		experiment = pieces[0]
		version = pieces[1]
		session_piece = pieces[2]
		session_letter = session_piece[0]
		run_number = session_piece[1]
		
		subject_piece = pieces[3]
		system = 'neuroscan' # preliminary
		fam_number = subject_piece[1:5]
		subject = subject_piece[5:8]
		if fam_number in ['0000','0001']:
			site = subject_piece[0]+'-subjects'
			if  fam_number == '0000': # no family
				family = 0 
				# need to check master file here to see if h and p subjects are masscomp or neuroscan
			if fam_number == '0001':
				family = 0; # need to look here for three recordings on family 0001
				
		else:
			family = fam_number
			site = site_hash[ subject_piece[0].lower() ]
			if not subject_piece[0].isdigit():
				system = 'masscomp'
			 
	# masscomp
	else:
		system = 'masscomp'
		experiment_short = filename[:2]
		experiment = experiment_shorthands[ experiment_short ]
		site_letter = filename[3]
		if filename[4:8] in ['0000','5000']: # no family
			site = site_letter+'-subjects'
			family = 0
		else:
			family = filename[4:8]
			site = site_hash[ site_letter.lower() ]
		
		run_number='1' # determine first or second run
		if filename[4] =='5':
			run_number='2'
		
		if site_letter.lower() == site_letter:
			session_letter = 'a'
		else: session_letter = 'b'
			
		subject = filename[8:11]

	output = {'system':system,
			'experiment':experiment,
			'session': session_letter,
			'run':run_number,
			'site':site,
			'family': family,
			'subject': subject,
			'id': subject_piece,
			'version': version}

	if include_full_ID:
		try:
			output['ID'] = site_hash_rev[site]+family+subject
		except: output['ID'] = filename

	return output

def parse_filename_tester():

	cases = [ ('vp3_6_e1_10162024_avg.mt','neuroscan','vp3','uconn',162,24,'e',1), 
			 ('vp2e0157027.mt','masscomp','vp3','washu',157,27,'a',1),
			 ('aod_5_a1_c0000857_avg.h1','neuroscan','aod','c-subjects',0,857,'a',1),
			('vp2c5000027.mt','masscomp','vp3','c-subjects',0,27,'a',2),
			 ('aod_5_a2_c0000857_avg.h1','neuroscan','aod','c-subjects',0,857,'a',2)
			]
	for case in cases:
		info = parse_filename(case[0])
		if ( (info['system'] != case[1]) or (info['experiment'] != case[2]) or
			(info['site'] != case[3]) or (int(info['family']) != case[4]) or (int(info['subject']) != case[5]) or
			(info['session'] != case[6]) or (int(info['run']) != case[7] ) ):
			print ( info, ' mismatch for case: ', case )

def identify_files(starting_directory, filter_pattern='*', file_parameters={}, filter_list=[], time_range=()):
	 
	file_list = []
	 
	for dName, sdName, fList in os.walk(starting_directory):
		 
		for filename in fList:
			path = dName
			if 'reject' not in path:
				fullpath = os.path.join(path,filename)
				if os.path.exists(fullpath):
					if shutil.fnmatch.fnmatch( filename, filter_pattern ):
						if file_parameters:
							file_info = parse_filename( filename )
						
							param_ck = [file_parameters[k]==file_info[k] for k in file_parameters]
						else: param_ck = [True]
						if time_range:
							time_ck = False
							stats = os.stat( fullpath )
							if time_range[0] < stats.st_ctime < time_range[1]:
								time_ck = True
						else: time_ck = True
						if filter_list:
							filter_ck = any([s in filename for s in filter_list])
						else: filter_ck = True	
						if all(param_ck) and time_ck and filter_ck:
							file_list.append( fullpath )

	return file_list



##############################
##
##		EEG
##
##############################

def parse_maybe_numeric(st):
	proc = st.replace('-','')
	dec = False
	if '.' in st:
		dec = True
		proc = st.replace('.','')
	if proc.isnumeric():
		if dec:
			return float(st)
		else:
			return int(st)
	return st

class cnth1_file:

	def __init__(s,filepath):
		s.filepath = filepath
		s.filename = os.path.split(filepath)[1]
		s.file_info = parse_filename(s.filename)

	def read_trial_info(s,nlines=-1):
		h5header = subprocess.check_output(['/opt/bin/print_h5_header',s.filepath])
		head_lines = h5header.decode().split('\n')
		hD = {}
		for L in head_lines[:nlines]:
			if L[:8] == 'category':
				cat = L[9:].split('"')[1]
				hD[cat]= {}
				curD = hD[cat]
			elif L[:len(cat)] == cat:
				subcat = L.split(cat)[1].strip()
				hD[cat][subcat] = {}
				curD = hD[cat][subcat]
			else:
				parts = L.split(';')
				var = parts[0].split('"')[1]
				val = parse_maybe_numeric( parts[1].split(',')[0].strip() )

				curD[var] = val
			
		s.trial_info = hD



class mt_file:
	''' manually picked files from eeg experiments
		initialization only parses the filename, call parse_file to load data
	'''
	columns = ['subject_id', 'experiment', 'version', 'gender', 'age', 'case_num',
	 'electrode', 'peak', 'amplitude', 'latency', 'reaction_time']
	
	cases_peaks_by_experiment = {'aod': {(1,'tt'):['N1','P3'],
										(2,'nt'):['N1','P2'] 
										},
								'vp3':{(1,'tt'):['N1','P3'],
									  (2,'nt'):['N1','P3'],
									  (3,'nv'):['N1','P3']
									  },
								'ant':{(1,'j'):['P3','N4'],
										#(2,'p'):['P3','N4'],
										(3,'a'):['P3','N4'],
										(4,'w'):['P3','N4']
										}
							  }
	data_structure = '{(case#,peak):{electrodes:(amplitude,latency),reaction_time:time} }' # string for reference
	
	def __init__(s,filepath):
		s.fullpath = filepath
		s.filename = os.path.split(filepath)[1]
		s.header = {'cases_peaks':{}}
		
		s.parse_fileinfo()
		s.parse_header()
	
	def parse_fileinfo(s):
		s.file_info = parse_filename( s.filename )

	def __repr__(s):
		return '<mt-file object '+str(s.file_info)+' >' 

	def parse_header(s):
		of = open(s.fullpath,'r')
		reading_header = True; 
		s.header_lines = 0
		while reading_header:
			file_line = of.readline()
			if file_line[0] != '#':
				reading_header = False
				continue
			s.header_lines +=1
			
			line_parts = [ pt.strip() for pt in file_line[1:-1].split(';') ]
			if 'nchans' in line_parts[0]:
				s.header['nchans'] = int(line_parts[0].split(' ')[1])
			elif 'case' in line_parts[0]:
				cs_pks = [ lp.split(' ') for lp in line_parts]
				if cs_pks[1][0] != 'npeaks':
					s.header['problems'] = True
				else: 
					s.header['cases_peaks'][int(cs_pks[0][1])] = int(cs_pks[1][1])
					
		of.close()
	
	def parse_file(s):
		of = open(s.fullpath,'r')
		data_lines = of.readlines()[s.header_lines:]
		of.close()
		s.data = OrderedDict()
		for L in data_lines:
			Ld = { c:v for c,v in zip( s.columns,L.split() )  }
			key = (Ld['case_num'], Ld['peak'])
			if key not in s.data:
				s.data[key] = OrderedDict()
			s.data[key][ Ld['electrode'].upper() ] = (Ld['amplitude'],Ld['latency'])
			if 'reaction_time' not in s.data[key]:
				s.data[key]['reaction_time'] = Ld['reaction_time']
		return
		
	def build_header(s):
		if 'data' not in dir(s):
			s.parse_file()
		cases_peaks = list( s.data.keys() )
		cases_peaks.sort()
		header_data = OrderedDict()
		for cp in cases_peaks:
			if cp[0] not in header_data:
				header_data[ cp[0] ] = 0
			header_data[ cp[0] ] += 1
		
		s.header_text = '#nchans '+str(len( s.data[ cases_peaks[0] ] )-1 )+'\n' # one less for reaction_time
		for cs, ch_count in header_data.items(): 
			s.header_text += '#case '+ str(cs) + '; npeaks '+ str(ch_count) +';\n'
	
		print(s.header_text)

	def build_file(s):
		pass
	
	def check_header_for_experiment(s):
		expected = s.cases_peaks_by_experiment[s.file_info['experiment']]
		if len(expected) != len(s.header['cases_peaks']):
			return 'Wrong number of cases'
		case_problems = []
		for pknum_name,pk_list in expected.items():
			if s.header['cases_peaks'][pknum_name[0]] != len(pk_list):
				case_problems.append('Wrong number of peaks for case '+str(pknum_name))
		if case_problems:
			return str(case_problems)
		
		return True
		
	def check_peak_identities(s):
		if 'data' not in dir(s):
			s.parse_file()
		for case, peaks in s.cases_peaks_by_experiment[s.file_info['experiment']].items():
			if ( str(case[0]), peaks[0] ) not in s.data:
				return (False, 'case '+str(case)+ ' missing '+ peaks[0]+' peak')
			if ( str(case[0]), peaks[1] ) not in s.data:
				return (False, 'case '+str(case)+ ' missing '+ peaks[1]+' peak')
		return True
	
	def check_peak_orderNmax_latency(s,latency_thresh=1000):
		if 'data' not in dir(s):
			s.parse_file()
		for case, peaks in s.cases_peaks_by_experiment[s.file_info['experiment']].items():
			try:
				latency1 = float(s.data[(str(case[0]),peaks[0])]['FZ'][1])
				latency2 = float(s.data[(str(case[0]),peaks[1])]['FZ'][1])
			except:
				print( s.fullpath+ ': '+ str(s.data[(str(case[0]),peaks[0])].keys()) )
			if latency1 > latency_thresh:
				return (False, str(case)+' '+peaks[0]+' '+'exceeds latency threshold ('+str(latency_thresh)+'ms)')
			if latency2 > latency_thresh:
				return (False, str(case)+' '+peaks[1]+' '+'exceeds latency threshold ('+str(latency_thresh)+'ms)')
			if  latency1 > latency2:
				return (False, 'Wrong order for case '+str(case) )
		return True


##############################
##
##		Neuropsych
##
##############################

class neuropsych_summary:
	
	def __init__(s,filepath):
		
		s.filepath = filepath
		s.path_parts = filepath.split(os.path.sep)
		s.filename = os.path.splitext( s.path_parts[-1] )[0]
		s.fileparts = s.filename.split('_')

		s.site = s.path_parts[-3]
		s.subject_id = s.fileparts[0]
		s.session = s.fileparts[3][0]
		s.motivation = int(s.fileparts[3][1])
		
		s.data = {'subject':s.subject_id,
				'site':s.site,
				'session':s.session,
				'motivation':s.motivation,
				}
		
	def read_file(s):
		
		of = open(s.filepath)
		lines = [ l.strip() for l in of.readlines() ]
		of.close()
		
		#find section line numbers
		section_beginnings =  [ lines.index(k) for k in s.section_header_funs_names ] + [ -1 ]
		ind = -1
		for sec, fun_nm in s.section_header_funs_names.items():
			ind +=1
			sec_cols = lines[ section_beginnings[ind]+1 ].split('\t')
			sec_lines = [L.split('\t') for L in lines[ section_beginnings[ind]+2:section_beginnings[ind+1]-1 ] ]
			s.data[fun_nm[1]] = eval('s.'+fun_nm[0])( sec_cols, sec_lines )
		

def parse_value_with_info(val, column, integer_columns, float_columns, boolean_columns={}):
	if column in integer_columns:
		val = int(val)
	elif column in float_columns:
		val = float(val)
	elif column in boolean_columns:
		val = bool( boolean_columns[column].index(val) )
	return val

class tolt_summary_file(neuropsych_summary):
	integer_columns = ['PegCount','MinimumMoves','MovesMade','ExcessMoves']
	float_columns = ['AvgPickupTime','AvgTotalTime','AvgTrialTime','%AboveOptimal','TotalTrialsTime','AvgTrialsTime']
	#boolean_columns = {}
	
	section_header_funs_names = {'Trial Summary':('parse_trial_summary','trials'),
								'Test Summary':('parse_test_summary','tests')}
	
	def parse_trial_summary(s, trial_cols, trial_lines):
		trials = {}
		for trial_line in trial_lines:
			trialD = {}
			for col,val in zip( trial_cols, trial_line ):
				val = parse_value_with_info(val,col,s.integer_columns,s.float_columns)
				if col == 'TrialNumber':
					trial_num = val
				else:
					trialD[col] = val
			trials[trial_num] = trialD
		return trials
	
	def parse_test_summary(s, test_cols, test_lines):
		#summary data is transposed
		for lnum,tl in enumerate(test_lines):
			if tl[0][0] == '%':
				test_lines[lnum] = [tl[0]] + [ st[:-1] if '%' in st else st for st in tl[1:] ]
			#print(type(tl),tl)
			#print([ st[:-1] if '%' in st else st for st in tl[1:] ])
			#tlinesP.append( tl[0] + [ st[:-1] if '%' in st else st for st in tl[1:] ] )
		test_data = { line[0]:[ parse_value_with_info(val,line[0],s.integer_columns,s.float_columns)\
							for val in line[1:] ] for line in test_lines }
		caseD = {}#case:{} for case in test_cols[1:] }
		for cnum,case in enumerate(test_cols[1:]):
			caseD[case] = { stat:data[cnum] for stat,data in test_data.items() }
		return caseD

	def __init__(s,filepath):
		neuropsych_summary.__init__(s,filepath)
		s.rdead_file()

class cbst_summary_file(neuropsych_summary):
	
	integer_columns = ['Trials','TrialsCorrect']
	float_columns = ['TrialTime','AverageTime']
	boolean_columns = {'Direction':['Backward','Forward'],'Correct':['-','+']} # False, True 
	
	section_header_funs_names = {'Trial Summary':('parse_trial_summary','trials'),
								 'Test Summary':('parse_test_summary','tests')}
	
	def parse_trial_summary(s, trial_cols, trial_lines):
		trials = {}
		for trial_line in trial_lines:
			trialD = {}
			for col,val in zip( trial_cols, trial_line ):
				val = parse_value_with_info(val,col,s.integer_columns,s.float_columns,s.boolean_columns)
				if col == 'TrialNum':
					trial_num = val
				else:
					trialD[col] = val
			trials[trial_num] = trialD
		return trials
	
	def parse_test_summary(s, test_cols, test_lines):
		tests = {'Forward':{},'Backward':{}}
		for test_line in test_lines:
			testD = {}
			for col,val in zip( test_cols, test_line ):
				if col == 'Direction':
					dirD = tests[val]
				else:
					val = parse_value_with_info(val,col,s.integer_columns,s.float_columns,s.boolean_columns)
					if col == 'Length':
						test_len = val
					else:
						testD[col] = val
			dirD[test_len] = testD
		return tests
	
	def __init__(s,filepath):
		neuropsych_summary.__init__(s,filepath)
		s.read_file()


def move_picked_files_to_processed( from_base, from_folders, working_directory, filter_list=[], do_now=False ):
	''' utility for moving processed files - places files in appropriate folders based on filenames
		
		inputs:
			from_base - folder containing all from_folders
			from_folders - list of subfolders
			working_directory - folder to store delete list (/active_projects can only be modified by exp)
			filter_list - a list by which to limit the files
 
			do_now - must be set to true to execute - by default, just a list of proposed copies is returned
	'''
	
	to_base = '/processed_data/mt-files/'
	to_copy = []
	counts = {'non coga':0, 'total':0, 'to move':0, 'masscomp':0, 'neuroscan':0}
	if do_now:
		delete_file = open( os.path.join(working_directory, utils.next_file_with_base(working_directory,'picked_files_copied_to_processed','lst') ), 'w')

	for folder in from_folders:
		for reject in [False, True]:
			from_folder = os.path.join( from_base, folder )
			if reject:
				from_folder += os.path.sep+'reject'
			if not os.path.exists(from_folder):
				print( from_folder+ ' Does Not Exist' )
				continue
			
			print( 'checking: '+from_folder )
			files = [ f for f in os.listdir(from_folder) if not os.path.isdir( os.path.join(from_folder,f) )]
			if filter_list:
				print(len(files))
				files = [ f for f in files if any([s in f for s in filter_list]) ] 
				print(len(files))
			for file in files:
				counts['total'] +=1
				if not ('.lst' in file or '.txt' in file or '_list' in file):
					try: 
						file_info = parse_filename(file)
						if 'subjects' in file_info['site']:
							counts['non coga'] += 1

						if file_info['system'] == 'masscomp':
							counts['masscomp'] +=1
							type_short = 'mc'
							session_path = None

						else:
							counts['neuroscan'] +=1
							type_short = 'ns'
							session_path = file_info['session']+'-session'
			
						to_path = to_base+file_info['experiment']+os.path.sep+file_info['site']+os.path.sep+type_short +os.path.sep 
						
						if session_path:
							to_path += session_path+os.path.sep
					
						if reject:
							to_path += 'reject'+os.path.sep
					
						to_copy.append( (from_folder+os.path.sep+file, to_path ) ) 
						counts['to move'] +=1


					except: print('uninterpretable file: '+file)

		print( str(counts['total']) + ' total ('+str(counts['masscomp'])+' masscomp, '+str(counts['neuroscan'])+' neuroscan) '+ str(counts['to move'])+' to move' ) 

	print( 'total non coga: '+ str(counts['non coga']) )
	
	if do_now:
		for cf_dest in to_copy:
			delete_file.write(cf_dest[0]+'\n')
			if not os.path.exists(cf_dest[1]):
				os.makedirs( cf_dest[1] )
			shutil.copy2(cf_dest[0],cf_dest[1])
		delete_file.close()
	
	return to_copy
