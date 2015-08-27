import master_info as mi
import study_info as si
from sas7bdat import SAS7BDAT as sas
import pandas as pd
import numpy as np
import os


ssaga_files_dir = '/export/home/mort/projects/comb/'

ssaga_vars= ['DM1','AGE','TBd4dpsx','TBd4dpdx','FTNDscore','ald4dpsx','ald4dpdx','ald4abdx','ald4absx','mjd4dpsx','mjd4dpdx','mjd4abdx','mjd4absx','asd4dx','asd4a','asd4c','ODd4dx','ODd4sx','ADd4dx','ADd4a1sx','ADd4a2Hypsx','ADd4a2Impsx','ADd4a2sx']

############    GENERAL FUNCTIONS    ##############

def extend_df(main_df,extender_df, join_col, extend_cols):
    extend_frame = extender_df[ [join_col]+extend_cols ]
    extended_df = pd.merge( main_df, extend_frame, left_on=join_col, right_on=join_col )
    return extended_df

def rename_col(col,add,skip, sep='-'):
    if col not in skip:
        return col+sep+add
    return col

def repeats_to_columns(DF, repcol, index_col, common_cols):
    #find unique values and compose separate dataframes
    vals = DF[repcol].unique()
    vals.sort()
    vDFs = {}
    for val in vals:
        vDFs[val] = DF[ DF[repcol]==val ].copy()
        vDFs[val].columns = [ rename_col(c,val,[index_col]+common_cols) for c in vDFs[val].columns ]
        if val == vals[0]:
            comb_df = vDFs[val].copy()
        else:
            for cc in common_cols:
                #vDFs[val]= vDFs[val].copy()
                vDFs[val].drop(cc, axis=1, inplace=True)
            comb_df = pd.merge(comb_df,vDFs[val],how='left',left_on=index_col,right_on=index_col)
    
    for v in vals:
        comb_df.drop(repcol+'-'+v, axis=1, inplace=True)
    
    return comb_df
    
def limit_columns( dataframe, column_prefixes_to_keep ):
	for c in list( dataframe.columns ):
		if not any( [ base in c for base in column_prefixes_to_keep ] ):
			dataframe.drop( c, axis=1, inplace=True)


############    STUDY SPECIFIC FUNCTIONS    #############


def load_and_combine_ssaga():
	ssaga_files = [f.split('.')[0] for f in os.listdir(ssaga_files_dir)]
	ssaga_files.sort(key=lambda b: (len(b), int(b[-1] if b[-2]=='f' else 0))  )

	ssaga_dfs = {} 
	for f in ssaga_files:
		with sas(ssaga_files_dir+f+'.sas7bdat') as of:
			ssaga_dfs[f] = of.to_data_frame()
			ssaga_dfs[f]['IND_ID']= ssaga_dfs[f]['IND_ID'].astype(str)
			
	# combine all ssaga followups
	dxdfs = [f for f in ssaga_files if 'dx_' in f]
	for fname in dxdfs:
		date_frame_name = '_'.join( fname.split('_')[1:] )
		dx_dfWdate= extend_df( ssaga_dfs[fname], ssaga_dfs[date_frame_name], 'IND_ID', ['IntvDate'] )
		fparts = fname.split('_')
		if len(fparts) < 3:
			num = '0'
		else: num = fparts[2][1]
			
		dx_dfWdate.columns = [ rename_col(c, 'f'+num, ['IND_ID'],  sep='_') for c in dx_dfWdate.columns ]   
		
		if num == '0':
			ssagaCombDXDF = dx_dfWdate.copy()
		else:
			ssagaCombDXDF = pd.merge( ssagaCombDXDF, dx_dfWdate, how='left', left_on='IND_ID', right_on='IND_ID' )
			
	ssagaCombDXDF['IND_ID'] = ssagaCombDXDF['IND_ID'].map( lambda x: x[:-2] if '.' in x else x)


''' Old combining using compiled PhaseIV results
 building the combined data frame

        electrophys columns have the structure dataType_experiment-session-case-freqTimeWindow
'''

'''
study_dir = '/processed_data/csv-analysis-files/v6.0_center9/PhaseIV/'
file_ending = '.PhaseIV.2015-06-19.csv'
power_type = 'tot'
freq_time_windows = [('t','3.0-7.5_300-700'),('a','8.0-12.0_200-400'),('g','29.0-45.0_50-150')]

common_cols = ['sex']

ssagaNepDFc = ssagaCombDXDF.copy()
#ssagaNepDFc.rename(columns={'IND_ID':'ID'}, inplace=True)
ssagaNepDFc.set_index('ID',drop=True,inplace=True)

# limit columns
for c in list(ssagaNepDFc.columns):
    if not any( [ base in c for base in ssaga_vars+['ID'] ] ):
        ssagaNepDFc.drop( c, axis=1, inplace=True)    

# calculate follow up ages
'''
'''
ssagaNepDFc = ssagaNepDFc.join( mi.master['DOB'], how='left' )
ssagaNepDFc['DOB'] =ssagaNepDFc['DOB'].apply( mi.calc_date_w_Qs )
ssagaNepDFc['DOB'] = ssagaNepDFc['DOB'].apply( lambda x: datetime.date(x.year,x.month,x.day) )
ssagaNepDFc['DOB'] = ssagaNepDFc['DOB'].apply( pd.to_datetime )
date_cols = [c for c in ssagaNepDFc.columns if 'Intv' in c]
print( date_cols )
for c in date_cols:
    print(c)
    #ssagaNepDFc[ c ] = ssagaNepDFc[ c ].apply( pd.to_datetime )
    age_col = c.replace('IntvDate','AGE')
    ssagaNepDFc[c] - ssagaNepDFc['DOB']
    ssagaNepDFc[ age_col ] = ssagaNepDFc[c] - ssagaNepDFc['DOB']
    if '5' in c:
        ssagaNepDFc[ age_col ][19:] = ssagaNepDFc[ age_col ][19:].apply(pd.tslib.Timedelta).astype('timedelta64[s]')/(365*24*60*60)
    else:
        ssagaNepDFc[ age_col ] = ssagaNepDFc[ age_col ].apply(lambda x: pd.to_timedelta(x,unit='s',coerce=True) )/(365*24*60*60)
        #.apply(pd.tslib.Timedelta).astype('timedelta64[s]')/(365*24*60*60)
'''
'''
cnt = 0
exps = ['ant','vp3','aod','gng','ern','err']
for exp in exps: #si.experiments_parts:
    study_exp_dir = os.path.join(study_dir,exp)
    for case in si.experiments_parts[exp]:
        ec_str = exp+'-'+case
        exp_case_dir =  os.path.join(study_exp_dir,exp+'-'+case)
        for nm,ftw in freq_time_windows:
            cnt +=1
            if cnt < 10000:
                
                
                possible_files = [f for f in os.listdir(exp_case_dir) if ec_str in f and ftw in f and power_type in f and file_ending in f ]
                if len(possible_files) > 1:
                    print(exp_case_dir,possible_files)
                elif len(possible_files) == 0:
                    print( 'no files for ', ec_str, nf_str )
                else:
                    expDFses = pd.read_csv( os.path.join( exp_case_dir,possible_files[0] ), converters={'ID':str} )
                    
                    expDF = repeats_to_columns(expDFses, 'session', 'ID', ['COGA11k-race','POP', 'sex'])
                    
                    age_cols = []
                    #print( sorted(list(expDF.columns)) )
                    for ckcol in list(expDF.columns):
                        
                        if tryNaNcheck(expDF[ckcol].max()):
                            #print('nancol: '+c)
                            expDF.drop( ckcol, axis=1, inplace=True )
                        else:
                            if 'age' in ckcol:
                                #if cnt == 1:
                                #    age_cols.append( ckcol )
                                #else:
                                #    ageDF= expDF.rename(columns={ckcol:ckcol+'1'})
                                #    ageDF = ssagaNepDFc[[ckcol]].join(ageDF[[ckcol+'1']], how='left')
                                #    adfs[ckcol] = ageDF.copy()
                                    #ageDF[ckcol] = ageDF.apply( take_best_value )
                                    #ssagaNepDFc[ckcol] = ageDF[ckcol]
                                expDF.drop( ckcol, axis=1, inplace=True )

                    expDF.columns = [ rename_col(c,'_'.join([exp,case,nm]),age_cols+common_cols) for c in expDF.columns ]
                    
                    if cnt > 1:
                        for cc in common_cols:
                            expDF.drop( cc, axis=1, inplace=True )
                    
                    ssagaNepDFc=ssagaNepDFc.join(expDF, how='left')#, lsuffix='_ss', rsuffix='_ep')
'''
