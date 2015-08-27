'''ERP processing
'''

import file_handling as FH
import os, shutil


threshold_values = { ('reacttion_time','min'):150,
					('amplitude','max'):150
					}


 # Identify files to use
 
def identify_files(starting_directory, filter_pattern='*', file_parameters= {}):
	 
	file_list = []
	 
	for dName, sdName, fList in os.walk(starting_directory):
		 
		for filename in fList:
			path = dName
			if 'reject' not in path:
				if shutil.fnmatch.fnmatch( filename, filter_pattern ):
					#file_info = FH.parse_filename( filename )
					
					#param_ck = [file_parameters[k]==file_info[k] for k in file_parameters]
					#if all(param_ck):
					file_list.append( os.path.join(path,filename) )

	return file_list

class csv_analysis:
	
	columns = ['ID','session','sex','age','lnage','age2','sqrtage','POP','COGA11k-race','alc_dep_dx','alc_dep_ons','case','electrode','peak','amplitude','latency','reaction_time']

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
			splitline = L.split(',')
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

	def write_line(s,subject,session):
		pass

def check_thresholds(mtt_obj):
	pass
	
def check_ages(csv_analysis_obj):
	pass
	
	
