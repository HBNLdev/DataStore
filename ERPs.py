'''ERP processing
'''

import os, shutil
import numpy as np

import file_handling as FH
import study_info as SI
import master_info as MI

MI.load_master()

threshold_values = { ('reaction_time','max'):1000, #ms
					('latency','max'):1000 #ms
					}


 # Identify files to use
identify_files = FH.identify_files 

mt_file = FH.mt_file

def process_ERPs_for_study( study ):
	
	subjectsDF = MI.frame_for_study( study )
	
	for sub in subjectsDF['ID']:


class analysis_file:
	
	def __init__(s,fullpath):
		s.fullpath = fullpath
		s.filename = os.path.split(fullpath)[1]
		
		s.length_problems = []
		s.question_problems = []
	
	def parse_problem_line( line ):
		pass
	
	def parse_file(s, problems_only = False ):
		''' use problems_only=True to just identify problem files
		'''
		of = open(s.fullpath,'r')
		data_lines = of.readlines()
		of.close()
		s.data = {}
		for L in data_lines:
			splitline = L.split(s.delim)
			if len(splitline) != len(s.columns):
				s.length_problems.append(L)
			elif '?' in L: 
				s.question_problems.append(L)
			else:
				if not problems_only:
					Ld = { c:v for c,v in zip( s.columns, splitline )  }
					key = (Ld['ID'], Ld['session'])

					s.data[key] = [ Ld[col] for col in columns[2:] ]
			
		return

	def create_line(s,subject_id,session, master_frame):
		line = []
		# lookup info
		Minfo = master_frame.ix[subject_id]
		for column in s.columns:
			if column in ['ID', 'sex','age','POP','wave12-race','4500-race','ccGWAS-race','COGA11k-race','alc_dep_dx','alc_dep_ons']:
				datum = Minfo[column]
			elif 'age' in column:
				age = Minfo['age']
				if column == 'lnage': datum = np.log(age)
				if column == 'age2': datum = age**2
				if column == 'sqrtage': datum = age**(0.5)
				
			line.append(str(datum))
		
		return line

class clean_list(analysis_file):
	columns = ['ID','session','sex','age','POP','wave12-race','4500-race','ccGWAS-race','COGA11k-race','alc_dep_dx','alc_dep_ons','case','electrode','peak','amplitude','latency','reaction_time','path']
	delim = ' '

class csv_analysis(analysis_file):
	
	columns = ['ID','session','sex','age','lnage','age2','sqrtage','POP','COGA11k-race','alc_dep_dx','alc_dep_ons','case','electrode','peak','amplitude','latency','reaction_time']
	delim = ','


def check_thresholds(mtt_obj):
	pass
	
def check_ages(csv_analysis_obj):
	pass
	
	
