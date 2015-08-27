'''EEGmaster
'''

master_path = '/export/home/mort/projects/EEG-master-file/EEG-master-file-08.csv' #'/processed_data/master-file/EEG-master-file-08.csv'
master= None

import datetime
import pandas as pd

def calc_date_w_Qs(dstr):
	''' assumes date of form mm/dd/yyyy
	'''
	dstr = str(dstr)
	if dstr == 'nan':
		return None
	if '?' in dstr:
		if dstr[:2] == '??':
			if dstr[3:5] == '??':
				if dstr[5:7] == '??':
					return None
				else: dstr = '6/1'+dstr[5:]
			else: dstr= '6'+dstr[2:]
		else:
			if dstr[3:5] == '??':
				dstr = dstr[:2]+'/15'+dstr[5:]
	try:
		return datetime.datetime.strptime(dstr,'%m/%d/%Y')
	except:
		print('problem with date: '+dstr)
		return None 

def load_master( preloaded= None, force_reload=False):
	global master
	if type(preloaded) == pd.core.frame.DataFrame:
		master = preloaded
		return
			
	if not type(master)==pd.core.frame.DataFrame  or force_reload:	
		master = pd.read_csv(master_path, 
							converters={'ID':str}, na_values=['.'],low_memory=False)
	
		master.set_index('ID',drop=False,inplace=True)#verify_integrity=True)
		for dcol in ['DOB'] + [col for col in master.columns if '-date' in col]:
			master[dcol] = master[dcol].map(calc_date_w_Qs).apply( pd.to_datetime )
	
	return

def ids_with_exclusions():
	return master[ master['no-exp'].notnull() ]['ID'].tolist()
	
def excluded_experiments(id):
	ex_str = master.ix[ id ]['no-exp']
	excluded = []
	if type( ex_str ) == str:
		excluded = [ tuple(excl.split('-')) for excl in ex_str.split('_') ]
	return excluded
	
def subjects_for_study(study, return_series=False):
	'''for available studies, see study_info module
	'''
	
	study_symbol_dispatch = {'Wave12':('Wave12','x'),
							'COGA4500':('4500','x'), # could also be 'x-'
							'ccGWAS':('ccGWAS','ID number'),
							'EAfamGWAS':('EAfamGWAS','x'),
							'COGA11k':('COGA11k-fam','ID number'),
							'ExomeSeq':('ExomeSeq','x'),
							'AAfamGWAS':('AAfamGWAS','x'),
							'PhaseIV':('Phase4-session',('a','b','c','d')),
							'fMRI-NKI-bd1':('fMRI',('1a','1b')),
							'fMRI-NKI-bd2':('fMRI',('2a','2b')),
							'fMRI-NYU-hr':('fMRI',('3a','3b')),
							'a-subjects':'use ID',
							'c-subjects':'use ID',
							'h-subjects':'use ID',
							'p-subjects':'use ID',
							'smokeScreen':('SmS','x')
							}
	
	study_identifier_info = study_symbol_dispatch[study]
	if study_identifier_info == 'use ID':
		id_series = master['ID'].str.contains(study[0]) # works for -subjects studies
	elif study_identifier_info[1]  == 'ID number':
		id_series = master[ study_identifier_info[0] ] > 0
	elif type( study_identifier_info[1] ) == tuple:
		id_series = master['ID'] == 'dummy'
		for label in study_identifier_info[1]:
			id_series = id_series | ( master[ study_identifier_info[0] ] == label )
	else: 
		id_series = master[ study_identifier_info[0] ] == study_identifier_info[1]

	if return_series:
		return id_series
	else:
		return master.ix[id_series]['ID'].tolist()

def frame_for_study( study ):
	
	id_series = subjects_for_study( study )
		
	study_frame = master.ix[id_series]
	
	return study_frame
	


session_letters = 'abcdefghijk'
def sessions_for_subject_experiment( subject_id, experiment ):
	sessions = []
	subject = master.ix[subject_id]
	excluded = excluded_experiments( subject_id )
	next_session = True; ses_ind = 0
	while not pd.isnull(next_session) and ses_ind < len(session_letters):
		ses_letter = session_letters[ses_ind]
		next_session = subject[ses_letter + '-run']
		if not pd.isnull(next_session) and (ses_letter, experiment) not in excluded:
			sessions.append( ( next_session, subject[ses_letter+'-raw'], subject[ses_letter+'-date'] ) )
		ses_ind +=1
		
	return sessions
		
def protect_dates(filepath):

	inf = open(filepath)
	outf = open( 'tmp.csv','w')
	for line in inf:
		lparts = line.split(',')
		newline = []
		for part in lparts:
			if '/' in part:
				newline.append("'"+part)
			else: 
				newline.append(part)
		newline = ','.join(newline)
		outf.write(newline)

	inf.close()
	outf.close()
