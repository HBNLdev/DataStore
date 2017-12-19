from collections import defaultdict
from glob import glob
import os
import shutil
import sys


def move_neuro_files(new_site_data, neuro_path_server, site_name):
    """
    given a path to incoming neuropsych data, moves files to correct directory in /raw_data/neuropsych/
    data must be checked with .exe files before moving. 
    
    new_site_data = path on server to new np data
    neuro_path_server = /raw_data/neuropsych/__sitename__
    site_name = double check for correct site directory 
    """
    
    
    if neuro_path_server.split('/')[-1] != site_name:
        print("Site name & site name in neuropsych path don't match -- exiting")
        sys.exit(1)
    
    # get dir names from neuropsych disc
    disc_dir_names = []
    for r,d,f in os.walk(new_site_data):
        for n in d:
            disc_dir_names.append(n)
                
    # search dirs from neuropsych disc on server to see if they exist 
    dirs_exist = []
    for i in disc_dir_names:
        neuro_dirs = glob(neuro_path_server + '/' + i)
        for nd in neuro_dirs:
            if os.path.exists(nd) == True:
                dirs_exist.append(nd)

    # here are the dirs from disc that exist on server
    dirs_found = [i.split('/')[-1] for i in dirs_exist]
    
    # here are the dirs that need to be created
    to_be_created = set(disc_dir_names) ^ set(dirs_found)
    
    # zip lengths MUST BE EQUAL -- for all the dirs that were found, multiply by 5
    dirs_found_concat = ['/raw_data/neuropsych/'+ site_name + '/' + i for i in dirs_found]
    dirs_exist_all = sorted(dirs_found_concat *5)
    
    
    # get all files from disc of dirs that were found on server ==> dirs exist
    files_to_move = []
    for r,d,f in os.walk(new_site_data):
        for n in d:
            if n in dirs_found:
                path = os.path.join(r,n)
                files = os.listdir(path)
                for fi in files:
                    goods = r + '/'+ n + '/' + fi
                    files_to_move.append(goods)
                    
    # check for dirs exist
    if len(dirs_exist_all)!= len(files_to_move):
        print('STOP RIGHT NOW')
        sys.exit(1)
        
    # create directories for all new neuro subs -- dirs don't exist 
    # if 
    new_dirs = []
    if len(list(to_be_created)) != 0:
        for i in list(to_be_created):
            new_dir = ['/raw_data/neuropsych/'+ site_name + '/' + i]
            for dirs in new_dir:
                if os.path.exists(dirs):
                    print('dir already exists,.. closing')
                    sys.exit(1)
                else:
                    os.makedirs(dirs)
                    print('making new dir...\n{}\n'.format(dirs))
                    new_dirs.append(dirs) 
                    
    # prep new dirs for zip & get SUB ID  -- dirs that dont exist 
    new_dirs_server = new_dirs *5   
    new_dir_check = []
    for i in new_dirs_server:
        new_d = i.split('/')[-1]
        new_dir_check.append(new_d)
        
    # check new site data for dirs with name that isn't found on server  -- dirs that dont exist 
    files_to_be_added = []
    for r,d,f in os.walk(new_site_data):
        for n in d:
            if n in list(set(new_dir_check)):
                good_dir = os.path.join(r,n)
                to_be_added = os.listdir(good_dir)
                for i in to_be_added:
                    full_paths = good_dir +'/'+ i
                    files_to_be_added.append(full_paths)

                    
    # key is neuropsych dir on server & values are full paths to site data files 
    neuro_dict = {}
    for server, site in zip(sorted(dirs_exist_all), sorted(files_to_move)):
        neuro_dict.setdefault(server, []).append(site)
        
    # this is for NEWLY created dirs 
    neuro_dict_to_be_added = {}
    for server, site in zip(sorted(new_dirs_server), sorted(files_to_be_added)):
        neuro_dict_to_be_added.setdefault(server, []).append(site)

    neuro_dict.update(neuro_dict_to_be_added)
    
    # move some stuff
    count = 0
    for x in range(len(neuro_dict)):
        for idx, (k,v) in enumerate(neuro_dict.items()):
            check_server = k.split('/')[-1]
            if idx % 5 == 0:
                count+=1
                input('\n\nPress enter to start moving files in sets of 5.\n\n')
            if idx == x:
                for i in v:
                    fname = os.path.basename(i)
                    trg = k +'/' + fname
                    if not os.path.exists(trg):
                        #shutil.move(i, trg)
                        print('\nSource - {}\nDestination - {}\n'.format(i, trg))
                    else:
                        print(i)
                        ans = input('\nThis file above already exists.\nEnter "i" to ignore or "q" to quit\n')
                        if 'i' in ans:
                            pass
                        else:
                            print('closing down...')
                            sys.exit(1)

                
    print('\n\nTotal of {} subjects moved from {} to {}\n\n'.format(count, neuro_path_server, new_site_data))
    
    
dont_forget = input("\nDon't forget to uncomment shutil.move - line 123.  Press q to quit or enter to continue.\n> ")
new_site_data_path = input('\nEnter path to new neuropsych data that has already been reviewed.\n> ')
neuro_path_server = input('\nEnter path to neuropsych directory files belong in.\n> ')
np_site = input('\nEnter site name.\n> ')

if 'q' in dont_forget:
    print('\nYou forgot to uncomment shutil.move -- exiting program.\n')
    sys.exit(1)
    
move_neuro_files(new_site_data_path, neuro_path_server, np_site)