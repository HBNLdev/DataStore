''' Morton Ehrenberg's HBNL utilities
'''

import os

def next_file_with_base(directory,base,ext):
    files = [ f for f in os.listdir(directory) if base in f and '.'+ext in f ]
    if files:
        numbers = [ int(os.path.splitext(f)[0].split('_')[-1]) for f in files]
        next_num = max(numbers)+1
    else: next_num = 1
    next_file = base+'_'+str(next_num)+'.'+ext
    return next_file
    
def list_to_file( lst, filename, replace=False ):
    if not os.path.exists(filename) or replace:
        of = open(filename,'w')
        for item in lst:
            of.write( str(item)+'\n' )
        of.close()
    return
