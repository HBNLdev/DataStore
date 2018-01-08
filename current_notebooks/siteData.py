from PyQt5.QtWidgets import (QMainWindow, QApplication, QPushButton, QWidget, QAction, 
                             QTabWidget,QVBoxLayout, QHBoxLayout, QInputDialog, QLineEdit, QLabel,
                             QFileDialog, QMainWindow, QPushButton, QTextEdit, QMessageBox)
from PyQt5.QtGui import QIcon, QTextCursor
from PyQt5.QtCore import pyqtSlot, QCoreApplication, QProcess, QObject, pyqtSignal
import sys
import os
import re
from datetime import datetime
import hashlib
from collections import defaultdict, Counter
from glob import glob
import shutil
import subprocess


############################################## NEURO BUTTON FUNCTIONS/CLASSES ##############################################


def np_run_exists(path_to_new_data, site):
    '''given a path to new data and str(site) -- 
       tells if file already exists in directory'''
    
    np_basename = '/vol01/raw_data/neuropsych/'
    
    files = []
    for r,d,f in os.walk(path_to_new_data):
        for n in f:
            files.append(n)
            
         
    np_full_path = np_basename + site
    
    duplicate_files = []  
    for r,d,f in os.walk(np_full_path):
        for n in f:
            if n in files:
                path = os.path.join(r,n)
                duplicate_files.append(path)
    
    if len(duplicate_files) == 0:
        print('All files are unique')
    else:
        #pass
        #print('im here')
        #print('The following files already exist in', np_full_path, ' --->\n\n', [i for i in duplicate_files])
        print('The following files already exist in {}...\n'.format(np_full_path))
        count = 0
        for idx, i in enumerate(duplicate_files):
            if idx % 5 == 0:
                print('\n{}'.format(i))
            else:
                print(i)


class neuropsych_check:


    def create_neuro_dict(self, key, neuro_dict, inner_key, inner_inner_key, value):
        """
        formats nested dictionary of lists
        """
        
        self.key = key
        self.neuro_dict = neuro_dict
        self.inner_key = inner_key
        self.inner_inner_key = inner_inner_key
        self.value = value
            
        return neuro_dict.setdefault(key, {}).setdefault(inner_key, {}).setdefault(inner_inner_key, []).append(value)



    def parse_neuro_files(self, path):
        """
        parse all relevant info from sum.txt/txt/xml file NAMES into nested dictionary
        """
        self.path = path
        
        key = path.split('/')[-1]

        neuro_dict = {}
        for f in os.listdir(path):
            if f.endswith('_sum.txt'):
                sum_txt_split = f.split('_')
                self.create_neuro_dict(key, neuro_dict, 'sum.txt', 'exp_name', sum_txt_split[1])
                self.create_neuro_dict(key, neuro_dict, 'sum.txt', 'run_letter', sum_txt_split[3][0])
                self.create_neuro_dict(key, neuro_dict, 'sum.txt', 'num_files', sum_txt_split[-1])
                self.create_neuro_dict(key, neuro_dict, 'sum.txt', 'id', sum_txt_split[0])
            if not f.endswith('_sum.txt') and not f.endswith('xml'):
                txt_split = re.split(r'[_.]', f)
                self.create_neuro_dict(key, neuro_dict, 'txt', 'exp_name', txt_split[1])
                self.create_neuro_dict(key, neuro_dict, 'txt', 'run_letter', txt_split[3][0])
                self.create_neuro_dict(key, neuro_dict, 'txt', 'num_files', txt_split[-1])
                self.create_neuro_dict(key, neuro_dict, 'txt', 'id', txt_split[0])
            ### need to create optarg for this below
            if f.endswith('xml'):
                xml_split = re.split(r'[_.]', f)
                neuro_dict.setdefault(key, {}).setdefault('xml', {})['id']=xml_split[0]
                neuro_dict.setdefault(key, {}).setdefault('xml', {})['num_files']=xml_split[-1]
                neuro_dict.setdefault(key, {}).setdefault('xml', {})['run_letter']=xml_split[1]
                
        return neuro_dict


    def neuro_dict_check(self, neuro_dict, file_ext):
        """
        take dictionary from parse_neuro_files  & check for errors
        """
        self.neuro_dict = neuro_dict
        self.file_ext = file_ext
        
        exp_lst = ['TOLT', 'CBST']
        
        error_list = []
        for k,v in neuro_dict.items():
            for k1,v1 in v.items():
                if k1 == file_ext:
                    for k2,v2 in v1.items():
                        if k2 == 'exp_name':
                            for exp in v2:
                                if exp not in exp_lst:
                                    error_list.append('Error: Missing experiment for a {} file for {}'.format(file_ext, k))
                        if k2 == 'id':
                            if len(set(v2)) != 1:
                                error_list.append('Error: Incorrect ID in {} file for '.format(file_ext, k))
                            for sub_id in v2:
                                if len(sub_id) != 8:
                                     error_list.append('Error: Sub ID incorrect length in {} file for {}'.format(file_ext, k))
                        if k2 == 'num_files':

                            if v2.count(file_ext) != 2:
                                error_list.append('Error: Missing a {} file for {}'.format(file_ext, k))
        return error_list

                                

    def parse_inside_xml(self, path):
        """
        create dictionary of important stuff INSIDE XML FILE 
        """
        self.path = path
        
        key = path.split('/')[-1]
        
        xml = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('xml')]


        xml_dict = {}
        with open(''.join(xml)) as f:
            for line in f:
                if line.startswith('  <Sub'):
                    sub = ''.join(re.findall(r'<SubjectID>(.*?)</SubjectID>', line))
                    xml_dict.setdefault(key, {})['id']=sub
                if line.startswith('  <Sess'):
                    run = ''.join(re.findall(r'<SessionCode>(.*?)</SessionCode>', line))
                    xml_dict.setdefault(key, {})['run_letter'] = run
                if line.startswith('  <Motivation>'):
                    motiv= ''.join(re.findall(r'<Motivation>(.*?)</Motivation>', line))
                    xml_dict.setdefault(key, {})['motiv'] = motiv
                if line.startswith('  <DOB>'):
                    dob =  ''.join(re.findall(r'<DOB>(.*?)</DOB>', line))
                    xml_dict.setdefault(key, {})['dob'] = dob
                if line.startswith('  <TestDate>'):
                    test_date = ''.join(re.findall(r'<TestDate>(.*?)</TestDate>', line))
                    xml_dict.setdefault(key, {})['test_date'] = test_date
                if line.startswith('  <Gender>'):
                    gender =  ''.join(re.findall(r'<Gender>(.*?)</Gender>', line))
                    xml_dict.setdefault(key, {})['gender'] = gender
                if line.startswith('  <Hand>'):
                    hand = ''.join(re.findall(r'<Hand>(.*?)</Hand>', line))
                    xml_dict.setdefault(key, {})['hand'] = hand
        return xml_dict



    def inside_xml_error_check(self, inside_xml_dict):
        """
        check stuff INSIDE XML FILE -- except id & run letter & motivation score 
        """

        self.inside_xml_dict = inside_xml_dict
        
        error_list = []
        for k,v in inside_xml_dict.items():
            for k1,v1 in v.items():
                # is sub ID 8 characters long?
                if k1 == 'id':
                    if len(v1) != 8:
                        error_list.append('Error: Sub ID incorrect length in {} file'.format(k))
                # is dob later than 2010??
                if k1 == 'dob':
                    date_split = v1.split('/')
                    year = int(date_split[-1])
                    if year > 2010:
                        error_list.append('Error: Check DOB in xml file for {}'.format(k))
                # is test year later than current year? __________ probably needs revamping      
                if k1 == 'test_date':
                    test_split = v1.split('/')
                    test_year = int(test_split[-1])
                    if test_year < datetime.now().year:
                        error_list.append('Error: Check test date in xml file for {}'.format(k))
                # is gender & handedness capitalized??
                if k1 == 'gender':
                    if not v1[0].isupper(): #maybe str.istitle() would be better?
                        error_list.append('Error: Make gender uppercase in xml file for {}'.format(k))
                if k1 == 'hand':
                    if not v1[0].isupper():
                        error_list.append('Error: Make handedness uppercase in xml file for {}'.format(k))

        return error_list



    def xml_check(self, path, xml_dict, neuro_dict):
        """
        checks sub ID & run letter between inside & outside of xml file
        """
        self.xml_dict = xml_dict
        self.neuro_dict = neuro_dict
        self.path = path

        
        key = path.split('/')[-1]
        
        error_list = []
        
        # xml checks -- id
        xml_inside_id = xml_dict[key]['id']
        xml_outside_id = neuro_dict[key]['xml']['id']

        if xml_inside_id != xml_outside_id:
            error_list.append("Error: Subject ID inside xml doesn't match ID outside xml for {}".format(key))

        # xml checks -- run letter
        xml_inside_run = xml_dict[key]['run_letter']
        xml_outside_run = neuro_dict[key]['xml']['run_letter']

        if xml_inside_run != xml_outside_run:
             error_list.append("Error: Run Letter inside xml doesn't match run letter outside xml for {}".format(key))
        
        # check run letter between xml filename & sum.txt/txt files 
        sum_txt_run = set(neuro_dict[key]['sum.txt']['run_letter'])
        txt_run = set(neuro_dict[key]['txt']['run_letter'])

        if txt_run != sum_txt_run:
            error_list.append("Error: Run letter in txt file doesn't match sum.txt file for {}".format(key))
          
        return error_list
    
    
    def md5(self, path):
        """
        generate md5 -- read file in binary mode 
        """
        self.path = path

        with open(path, "rb") as f:
            data = f.read()
            md5_return = hashlib.md5(data).hexdigest()
            return md5_return
    

    def md5_check_walk(self, path):
        """
        return any file pairs with matching checksums 
        """
        self.path = path
        
        md5_dict = defaultdict(list)
        for r,d,f in os.walk(path):
            for n in f:
                fp = os.path.join(r,n)
                md5_dict[self.md5(fp)].append(fp)
        
        dupes_list = []
        for k,v in md5_dict.items():
            if len(v) > 1:  #multiple values for the same key
                dupes_list.append("Error: Identical files found:\n")
                for dupe in v:
                    dupes_list.append("Filename: {}\nChecksum: {}\n".format(dupe,k))
                   
        if len(dupes_list) != 0:
            for dupe in dupes_list:
                print(i)
        else:
            print('No duplicates found!')

            
    def run_all(self, path):
        """
        combines all methods into 1 giant method 
        """
        
        self.path = path
        
        file_ext = ['txt', 'sum.txt']
        
        errors = []
        errors_duplicates = []
        for ext in file_ext:
        
            # check for txt/sum.txt files
            neuro_dict = self.parse_neuro_files(path)
            err1 = self.neuro_dict_check(neuro_dict, ext)
            errors.extend(err1)

            # check for xml files
            xml_dict = self.parse_inside_xml(path)
            err2 = self.inside_xml_error_check(xml_dict)
            errors.extend(err2)
            
            err3 = self.xml_check(path, xml_dict, neuro_dict)
            errors.extend(err3)

        unique_errors = set(errors)
        for i in list(unique_errors):
            if i is None:
                pass
            else:
                print("{}\n\n".format(i))

                            
        
np = neuropsych_check()


def move_neuro_files(new_site_data, site_name):
    """
    given a path to incoming neuropsych data, moves files to correct directory in /raw_data/neuropsych/
    data must be checked with .exe files before moving. 
    
    new_site_data = path on server to new np data
    site_name = double check for correct site directory 
    """
    full_neuro_path = '/vol01/raw_data/neuropsych/' + site_name
    
    if not os.path.exists(full_neuro_path):
        print("{} doesn't exist...exiting".format(full_neuro_path))
        sys.exit(1)
    
    # get dir names from neuropsych disc
    disc_dir_names = []
    for r,d,f in os.walk(new_site_data):
        for n in d:
            disc_dir_names.append(n)
                
    # to search dirs from neuropsych disc on server to see if they exist 
    dirs_exist = []
    for i in disc_dir_names:
        neuro_dirs = glob(full_neuro_path)
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
                        shutil.move(i, trg)
                        print('\nSource - {}\nDestination - {}\n'.format(i, trg))
                    else:
                        print(i)
                        ans = input('\nThis file above already exists.\nEnter "i" to ignore or "q" to quit\n')
                        if 'i' in ans:
                            pass
                        else:
                            print('closing down...')
                            sys.exit(1)

                
    print('\n\nTotal of {} subjects moved from {} to {}\n\n'.format(count, full_neuro_path, new_site_data))



############################################## ERP FUNCTIONS/CLASSES FOR BUTTONS ##############################################

class erp_data:

    def __init__(self):

        # avg + ps exp names are the same
        self.avg_and_ps_exps = ['vp3', 'cpt', 'ern', 'ant', 'aod', 'anr', 'stp', 'gng']
        # number of files associated with avg extension 
        self.avg_exp_nums = [3,6,4,4,2,2,2,2]
        # cnt exp names
        self.cnt_exp_list = ['eeo', 'eec', 'vp3', 'cpt', 'ern', 'ant', 'aod', 'ans', 'stp', 'gng']
        # versions associated with erp 
        self.version_list = ['4', '4', '6', '4', '9', '6', '7', '5', '3', '3']
        # dat exp names 
        self.dat_exps = ['vp3', 'cpt', 'ern', 'ant', 'aod', 'ans', 'stp', 'gng']
        # only 1 exp for dat/cnt/ps
        self.exp_nums_single = [1] * len(self.cnt_exp_list)


    def check_erp_version(self, path, exp_name, version_num): 
        """
        returns whether ERP experiment version is correct 
        """
        self.path = path
        self.exp_name = exp_name
        self.version_num = version_num

        for r,d,f in os.walk(path):
            for n in f:
                split = n.split('_')
                if n.startswith((exp_name)):
                    if split[1] != version_num:
                        return 'Check version for {}'.format(n)


    def iter_check_version(self, path, cnt_exp_list, version_list):
        """
        iterator version of check_erp_version()
        """
        self.path = path


        for exp, version in zip(self.cnt_exp_list, self.version_list):
            out = self.check_erp_version(path, exp, version)
            if out is None:
                pass
            else:
                return str(out)


    def parse_site_data(self, path):
        """
        returns a nested dictionary of common/uncommon file extensions by experiment for a sub
        """
        self.path = path

        key = path.split('/')[-1]

        nested_dict = {}
        for r,d,f in os.walk(path):
            for n in f:
                if n.endswith('_32.cnt'):
                    cnts = n.split('_')
                    nested_dict.setdefault(key, {}).setdefault('cnt', []).append(cnts[0])
                if n.endswith('dat'):
                    dats = n.split('_')
                    nested_dict.setdefault(key, {}).setdefault('dat', []).append(dats[0])
                if n.endswith('_avg.ps'):
                    pss = n.split('_')
                    nested_dict.setdefault(key, {}).setdefault('ps', []).append(pss[0])
                if n.endswith('.avg'):
                    avgs = re.split(r'[_.]', n)
                    nested_dict.setdefault(key, {}).setdefault('avg', []).append(avgs[0])
                if n.endswith('_orig.cnt'):
                    origs = n.split('_')
                    nested_dict.setdefault(key, {}).setdefault('orig_cnt', []).append(origs[0])
                if n.endswith('_32_original.cnt'):
                    bad_orig = n.split('_')
                    nested_dict.setdefault(key, {}).setdefault('bad_orig', []).append(bad_orig[0])
                if n.endswith('_rr.cnt'):
                    reruns = n.split('_')
                    nested_dict.setdefault(key, {}).setdefault('rerun', []).append(reruns[0])
                if n.endswith('_cnt.h1'):
                    cnt_h1 = n.split('_')
                    nested_dict.setdefault(key, {}).setdefault('cnt_h1', []).append(cnt_h1[0])
                if n.endswith('_avg.h1'):
                    avg_h1 = n.split('_')
                    nested_dict.setdefault(key, {}).setdefault('h1', []).append(avg_h1[0])
                if n.endswith('_avg.h1.ps'):
                    h1_ps = n.split('_')
                    nested_dict.setdefault(key, {}).setdefault('h1_ps', []).append(h1_ps[0])

        return nested_dict


    def remove_wild_files(self, path):
        """
        prompts user to delete file extensions that don't belong in ns folders
        """
        self.path = path

        wild_files = []
        for r,d,f in os.walk(path):
            for n in f:
                if n.endswith(('_32.cnt', '_orig.cnt', 'avg', 'avg.ps', 'dat', 'txt', 'sub')):
                    pass
                else:
                    fp = os.path.join(r,n)
                    wild_files.append(fp)
        count = 0
        for wild in wild_files:
            count+=1
            print("\n{} || {}".format(count, wild))
            ans= input('\nDo you want to delete this file?\n> ')
            if ans in ['y', 'Y', 'Yes', 'yes']:
                #os.remove(wild)
                print('deleting')
            else:
                pass


    def get_ext_count(self, path, nested_dict, ext_type, exp_name, number_files):
        """
        checks nested dictionary for number of extensions associated with each experiment
        e.g. checks that there's 6 CPT avg files or 1 EEC cnt file...
        """

        self.path = path
        self.nested_dict = nested_dict
        self.ext_type = ext_type
        self.exp_name = exp_name
        self.number_files = number_files

        output = []
        for k,v in nested_dict.items():
            for k1, v1 in v.items():
                if ext_type == k1:
                    num_files = v1.count(exp_name)
                    if num_files != number_files:
                        return 'Incorrect number of {} {} files in {}'.format(exp_name, ext_type, path)


    # get_ext_count()
    def iter_exps(self, path, nested_dict, exp_list, exp_list_avgs, ext_type):
        """
        iterator version of iter_exps()
        """
        self.path = path
        self.exp_list = exp_list
        self.exp_list_avgs = exp_list_avgs
        self.ext_type = ext_type

        for exp, avg_nums in zip(exp_list, exp_list_avgs):
            out = self.get_ext_count(path, nested_dict, ext_type, exp, avg_nums)
            if out is None:
                pass
            else:
                return str(out)


    def check_id_and_run(self, path):
        """
        checks to see if important file extensions have same sub ID & run letter
        """
        self.path = path

        folder = path.split('/')[-1]

        sub_id_list = []
        run_letter_list = []

        for file in glob(os.path.join(path, '*.*')):
            if file.endswith(('_32.cnt', '_orig.cnt', '_avg.ps', '.avg', 'dat')):
                fname = os.path.basename(file)
                if not fname.endswith(('avg', 'dat')):
                    sub_id = fname.split('_')[3]
                    sub_id_list.append(sub_id)
                else:
                    avg_dat_ids = re.split(r'[_.A-Z]', fname)
                    sub_id_list.append(avg_dat_ids[3])

                # append run letters 
                run_letter_list.append(fname.split('_')[2][0])


        unique_ids = list(set(sub_id_list))
        unique_run_letters = list(set(run_letter_list))

        if  len(unique_ids) > 1:
            return str("Folder {} has more than one sub ID => {}".format(folder, unique_ids))

        if len(unique_run_letters) > 1:
            return str("Folder {} has more than one run letter => {}".format(folder, unique_run_letters))


    def print_erp_version(self, path, cnt_exp_list, version_list):
        """
        check erp version 
        """

        self.path = path

        print('\n\nERP VERSION CHECK:')

        iter_check = self.iter_check_version(path, self.cnt_exp_list, self.version_list)
        if  iter_check is None:
            print('All versions check out!')
        else:
            print(iter_check)


    def print_file_counts(self, nested_data_dict):
        """
        returns counts of file extensions
        """

        self.nested_data_dict = nested_data_dict

        print('\n\nFILES COUNT:')
        for k,v in nested_data_dict.items():
            for k1,v1 in v.items():
                print("There are {} {} files".format(len(v1), k1.upper()))


    def print_missing_exps(self, path, nested_data_dict):
        """
        returns file type that's missing, if any 
        """
        self.path = path
        self.nested_data_dict = nested_data_dict

        print('\n\nMissing Experiments:')


        avg_counts = self.iter_exps(path, nested_data_dict, self.avg_and_ps_exps, self.avg_exp_nums, 'avg')

        if avg_counts is None:
            print('All avg files found!')
        else:
            print(avg_counts)


        cnt_counts = self.iter_exps(path, nested_data_dict, self.cnt_exp_list, self.exp_nums_single, 'cnt')
        if cnt_counts is None:
            print('All cnt files found!')
        else:
            print(cnt_counts)


        ps_counts = self.iter_exps(path, nested_data_dict, self.avg_and_ps_exps, self.exp_nums_single, 'ps')
        if ps_counts is None:
            print('All ps files found!')
        else:
            print(ps_counts)


        dat_counts = self.iter_exps(path, nested_data_dict, self.dat_exps, self.exp_nums_single, 'dat')
        if dat_counts is None:
            print('All dat files found!')
        else:
            print(dat_counts)


    def print_wild_files(self, path):
        """
        returns files that don't belong
        """
        self.path = path

        print("\n\nFILES THAT DON'T BELONG IN NS FOLDERS:")
        self.remove_wild_files(path)


    def print_id_and_letter(self, path):
        """
        returns ID & run letter, should both be unique
        """
        self.path = path

        print('\n\nCHECK SUBJECT ID & RUN LETTER:')

        if self.check_id_and_run(path) is None:
            print('All IDs & run letters check out!')
        else:
            print(self.check_id_and_run(path))


    def run_all(self, path):
        """
        2nd to last step 
        """
        self.path = path

        nested_data_dict = self.parse_site_data(path)

        self.print_erp_version(path, self.cnt_exp_list, self.version_list)
        self.print_file_counts(nested_data_dict)
        self.print_missing_exps(path, nested_data_dict)
        self.print_id_and_letter(path)


    def execute_all(self, path):
        """
        last step
        """

        self.path = path

        test = [os.path.join(r,n) for r,d,f in os.walk(path) for n in d]

        if len(test) == 0:
            self.run_all(path)
        else:
            fps = [os.path.join(r,n) for r,d,f in os.walk(path) for n in d]
            count=0
            for i in fps:
                print("\n\n{} || {}".format(count, i))
                self.run_all(i)
                count+=1
                

    # anything below here doesn't get executed in execute_all()
    def shell_filesize_check(self, path):
        """
        return stdout & stderr from David's ERP shell scripts 
        """
        self.path = path

        erp_check = subprocess.Popen('ERP-raw-data_check.sh {}'.format(path), shell=True, 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=path)
        result_erp = erp_check.communicate()[0]
        print("\nERP CHECK: {} {}".format(path, result_erp.decode('ascii')))
        
        
        size_check = subprocess.Popen('DVD-file-size_check.sh {}'.format(path), shell=True, 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=path)
        result_size = size_check.communicate()[0]
        print('FILE SIZE CHECK: {} {}\n'.format(path, result_size.decode('ascii')))


    def iter_shell_check(self, path):
        """
        iterate over directories checking with shell scripts 
        """

        self.path = path

        test = [os.path.join(r,n) for r,d,f in os.walk(path) for n in d]

        if len(test) == 0:
            self.shell_filesize_check(path)
        else:
            fps = [os.path.join(r,n) for r,d,f in os.walk(path) for n in d]
            count=0
            for i in fps:
                print("\n{} || {}".format(count, i))
                self.shell_filesize_check(i)
                count+=1


ep = erp_data()


class site_data:
    #get_h1s()
    def __init__(self):
    
        self.ant_set = {'ant'}
        self.ans_set = {'ans'}
        self.other_exps = {'vp3', 'cpt', 'ern', 'aod', 'stp', 'gng'}
                
    def check_cnt_copy(self, path, exp_tuple):
        """
        checks to see if files you want to h1 are actually in the directory to begin with
        """

        self.path = path
        self.exp_tuple = exp_tuple

        cnt_dict = {}
        for r,d,f in os.walk(path):
            for n in f:
                if n.startswith((exp_tuple)) and n.endswith('_32.cnt'):
                    split = n.split('_')
                    cnt_dict.setdefault(split[3], []).append(split[0])


        for k,v in cnt_dict.items():
            if len(v) is not len(exp_tuple):
                missing_exp = '.'.join(str(s) for s in (set(v) ^ set(list(exp_tuple))))
                print('\n\nLOOK HERE!!!{} cnt file missing from {}\n\n'.format(missing_exp.upper(), path))
                
    #get_h1s()                
    def rename_cnts(self, path, skip=False):
        """
        rename file ending with _rr.cnt if it doesn't already exist
        """
        self.path = path
        self.skip = skip

        for r,d,f in os.walk(path):
            for n in f:
                if n.endswith('_rr.cnt'):
                    new_name = os.path.join(r, n[:-7] + '.cnt')
                    if not os.path.exists(new_name):
                        path = os.path.join(r,n)
                        os.rename(os.path.join(r,n), new_name)
                        print("Renaming {}".format(n))
                    else:
                        print("{} already exists!!".format(new_name))
        
        if skip:
            for r,d,f in os.walk(path):
                for n in f:
                    if n.endswith('_rr.cnt'):
                        print("Re-run found for {} -- create manually".format(n))
    #get_h1s()                    
    def create_cnth1(self, path):
        """
        create cnt.h1 files from shell script
        """
        self.path = path
        
        print('\n\n>>> MAKING CNT.H1 FILES <<<\n') 
        for r,d,f in os.walk(path):
            for n in f:
                if n.startswith(self.cnth1_tups) and n.endswith('_32.cnt'):
                    path = os.path.join(r,n)
                    p = subprocess.Popen("create_cnthdf1_from_cntneuroX.sh {}".format(path),shell=True, 
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.dirname(path))
                    result = p.communicate()[1]
                    print(result.decode('ascii'))

    #get_h1s()
    def create_avgh1(self, path):
        """
        create avg.h1 files from shell script
        """
        self.path = path

        print('\n\n>>> Making AVG.H1 FILES <<<\n') 
        for r,d,f in os.walk(path):
            for n in f:
                if n.startswith(self.ant_tup) and n.endswith('cnt.h1'):
                    ant_path = os.path.join(r,n)
                    p_ant = subprocess.Popen('create_avghdf1_from_cnthdf1X -lpfilter 8 -hpfilter 0.03 -thresh 75 -baseline_times -125 0 {}'.format(ant_path), 
                                             shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.dirname(ant_path))
                    result_ant = p_ant.communicate()[1]
                    print(result_ant.decode('ascii'))
                if n.startswith(self.ans_tup) and n.endswith('cnt.h1'):
                    ans_path = os.path.join(r,n)
                    p_ans = subprocess.Popen('create_avghdf1_from_cnthdf1X -lpfilter 16 -hpfilter 0.03 -thresh 100 -baseline_times -125 0 {}'.format(ans_path), 
                                            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.dirname(ans_path))
                    result_ans = p_ans.communicate()[1]
                    print(result_ans.decode('ascii'))
                if n.startswith((self.others_tup)) and n.endswith('cnt.h1'):
                    path=os.path.join(r,n)
                    p_others = subprocess.Popen('create_avghdf1_from_cnthdf1X -lpfilter 16 -hpfilter 0.03 -thresh 75 -baseline_times -125 0 {}'.format(path), 
                                                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.dirname(path))
                    result_others = p_others.communicate()[1]
                    print(result_others.decode('ascii'))
    #get_h1s()
    def create_avgps(self, path):
        """
        create avg.h1.ps files from shell script
        """
        
        self.path = path

        print('\n\n>>> Making AVG.PS FILES <<<\n') 
        for path, subdirs, files in os.walk(path):
            for name in files:
                if name.endswith("avg.h1"):
                    ps_paths = os.path.join(path, name)
                    subprocess.Popen("plot_hdf1_data.sh {}".format(name), shell=True, cwd=os.path.dirname(ps_paths))
                    print("creating ps files.. " + name)
                    
        #get_h1s()                
    def delete_bad_files(self, path, exts_to_keep=None, to_be_deleted_set=None):
        ''' returns any extension not in exts_to_keep & prompts user to delete '''
    
        self.path = path
        self.exts_to_keep = exts_to_keep
        self.to_be_deleted_set = to_be_deleted_set


        
        print("\n\n>>> REMOVING FILES <<<\n") 
        if exts_to_keep:

            for r,d,f in os.walk(path):
                for n in f:
                    if n.endswith(('_cnt.h1')):
                        os.remove(os.path.join(r,n))
                        print('Removing {}'.format(n))

        if to_be_deleted_set:
            
            for r,d,f in os.walk(path):
                for n in f:
                    if n.endswith(('_32.cnt', '_cnt.h1')):
                        os.remove(os.path.join(r,n))
                        print('Removing {}'.format(n))



    
    def get_h1s(self, path, set_of_exps, del_ext=None, ps=None, trg_dir=None):
        '''combines all these commands together'''
        
        self.path = path
        self.set_of_exps = set_of_exps
        self.ps = ps
        self.trg_dir = trg_dir
        
        #use in create_cnth1()
        self.cnth1_tups = tuple(set_of_exps)
        
        if_ant = self.ant_set & set_of_exps
        if_ans = self.ans_set & set_of_exps
        all_others = self.other_exps & set_of_exps
        
        #used in create_avgh1()
        self.ant_tup = tuple(if_ant)
        self.ans_tup = tuple(if_ans)
        self.others_tup = tuple(all_others)

        #if being used for peak picking, create new directories and move all cnt files to the correct folder...and so on
        if trg_dir:
            #create new directories
            exps_list = list(set_of_exps)
            for exp in exps_list:
                new_dirs = os.path.join(trg_dir, exp)
                if not os.path.exists(new_dirs):
                    os.makedirs(new_dirs)
                    print(">>> Creating {} <<<".format(new_dirs))
                else:
                    #print("{} already exist".format(new_dirs))
                    pass
            #copy cnt files to newly created directories       
            count = 0
            print('\n>>> COPYING CNT FILES <<<\n')
            for r,d,f in os.walk(path):
                for n in f:
                    if n.startswith(self.cnth1_tups) and n.endswith('_32.cnt'):
                        count+=1
                        print('Copying {}'.format(n))
                        shutil.copy(os.path.join(r,n), os.path.join(trg_dir, n[:3]))
            print('\nCopied {} of {} files'.format(count, len(self.cnth1_tups)))

            self.rename_cnts(path)
            self.check_cnt_copy(trg_dir, self.cnth1_tups)
            self.create_cnth1(trg_dir)
            self.create_avgh1(trg_dir)
            if ps:
                self.create_avgps(trg_dir)
            if del_ext:
                self.delete_bad_files(trg_dir, to_be_deleted_set=True)
            return True
        
        #if you want to create avg.h1s IN directory copying files...
        self.rename_cnts(path, skip=True)
        self.create_cnth1(path)
        self.create_avgh1(path)
        if ps:
            self.create_avgps(path)
            self.delete_bad_files(path, exts_to_keep=True)


sd = site_data()


class EmittingStream(QObject):

    textWritten = pyqtSignal(str)

    def write(self, text):

        self.textWritten.emit(str(text))
    
    def flush(self):
        pass

    
class App(QMainWindow):        
 
    def __init__(self):   
        super(App, self).__init__()
        
        self.title = 'Site Data Utilities'
        self.left = 0
        self.top = 0
        self.width = 600
        self.height = 400
        self.setWindowTitle(self.title)
        self.setWindowIcon(QIcon('/vol01/active_projects/anthony/brain.jpg'))
        self.setGeometry(self.left, self.top, self.width, self.height)
        
 
        # Initialize tab widget
        self.tabs = QTabWidget()
        
        # create tabs 
        self.h1Tab = QWidget()
        self.neuroTab = QWidget()
        self.redirectTab = QWidget()
        
        #################################### ERP TAB ####################################
        
        # create vertical layout for ERP tab 
        self.erpLayout = QVBoxLayout()
    
        # create directory box 
        self.dirLayout = QHBoxLayout()
        dirLabel = QLabel('Directory: ')
        self.dirInp = QLineEdit('/vol01/active_projects/anthony/ns650')
        self.dirLayout.addWidget(dirLabel)
        self.dirLayout.addWidget(self.dirInp)
        
        self.dirExcludeLayout = QHBoxLayout()
        dirExcludeLabel = QLabel('Directories to exclude: ')
        self.dirExcludeInp = QLineEdit('00000001 00000002 00000003')
        self.dirExcludeLayout.addWidget(dirExcludeLabel)
        self.dirExcludeLayout.addWidget(self.dirExcludeInp)
        
        # create experiment box 
        self.filesLayout = QHBoxLayout()
        self.filesLabel = QLabel('Space-delimited exp names: ')
        self.filesInp = QLineEdit('aod ans')
        self.filesLayout.addWidget(self.filesLabel)
        self.filesLayout.addWidget(self.filesInp)
        

        # create target directory box 
        self.trgLayout = QHBoxLayout()
        self.trgLabel = QLabel('Target Directory: ')
        self.trgInp = QLineEdit('/vol01/active_projects/anthony/test_qt')
        self.trgLayout.addWidget(self.trgLabel)
        self.trgLayout.addWidget(self.trgInp)
        
         ############################################## TEXT EDITOR ##############################################
        
        # PROCESS IS A WIDGET NOT A LAYOUT 
        #self.process  = QTextEdit()
        #self.process.moveCursor(QTextCursor.Start)
        #self.process.ensureCursorVisible()
        #self.process.setLineWrapColumnOrWidth(1000)
        #self.process.setLineWrapMode(QTextEdit.FixedPixelWidth)
        
        ############################################## END EDITOR ##############################################
        
        #################################### ERP BUTTONS 
        # add create get h1's button
        self.h1Button = QPushButton("Create h1 files ( for peak picking)")
        self.h1Button.clicked.connect(self.h1_handler)
        self.h1Button.setToolTip("""Instructions:\nINPUTS => Directory\nINPUTS => Experiments\nINPUTS => Target Directory
                                    \n\nOptional: \nDirectories to exclude => sub ID of directories you want to exclude""")
        
        # add other create h1's button
        self.h1ButtonInPWD = QPushButton("Create h1 files ( for viewing )")
        self.h1ButtonInPWD.clicked.connect(self.h1_handler_PWD)
        self.h1ButtonInPWD.setToolTip("""Instructions: \nINPUTS Same as button above, except for Target Directory""")
        
        
        # add review data button 
        self.reviewButton = QPushButton("Review Data")
        self.reviewButton.clicked.connect(self.review_handler)
        self.reviewButton.setToolTip("Instructions:\nINPUTS => Directory")

        # add david's shell scripts 
        self.davidButton = QPushButton("Run ERP shell scripts")
        self.davidButton.clicked.connect(self.erp_shell_scripts)
        self.davidButton.setToolTip("Instructions:\nINPUTS => Directory")
        self.davidButton.resize(100,32)
        
        # add a clear button for editor
        #clearButton = QPushButton('Clear all text')
        #clearButton.clicked.connect(self.clear_function)
        

        # add dirLayout, files layout, start buttons to nav layout 
        self.erpLayout.addLayout(self.dirLayout)
        self.erpLayout.addLayout(self.dirExcludeLayout)
        self.erpLayout.addLayout(self.filesLayout)
        self.erpLayout.addLayout(self.trgLayout)
        
        
        # add buttons 
        self.erpLayout.addWidget(self.h1Button)
        self.erpLayout.addWidget(self.h1ButtonInPWD)
        self.erpLayout.addWidget(self.reviewButton)
        self.erpLayout.addWidget(self.davidButton)
        
        # add nav layout to NAV TAB 
        self.h1Tab.setLayout(self.erpLayout)
        
        #################################### NEURO TAB ####################################
        
        # create vertical layout for neuropsych tab 
        self.neuroLayout = QVBoxLayout()
    
        # create directory box -- neuropsych 
        self.dirLayoutNeuro = QHBoxLayout()
        dirLabelNeuro = QLabel('Directory: ')
        self.dirInpNeuro = QLineEdit('/vol01/raw_data/staging/ucsd/oct_neuro')
        self.dirLayoutNeuro.addWidget(dirLabelNeuro)
        self.dirLayoutNeuro.addWidget(self.dirInpNeuro)
        
        
        # create site box -- neuropsych 
        self.siteLayoutNeuro = QHBoxLayout()
        siteLabelNeuro = QLabel('Enter site name: ')
        self.siteInpNeuro = QLineEdit('site here')
        self.siteLayoutNeuro.addWidget(siteLabelNeuro)
        self.siteLayoutNeuro.addWidget(self.siteInpNeuro)
        
        # add directory, month, & site box to neuroLayout    
        self.neuroLayout.addLayout(self.dirLayoutNeuro)
        self.neuroLayout.addLayout(self.siteLayoutNeuro)
        
        # add neuroLayout to neuroTab
        self.neuroTab.setLayout(self.neuroLayout)
        
        #################################### NEURO BUTTONS  
        # add check neuro raw data button 
        self.checkButtonNeuro = QPushButton("Check /raw_data/neuropsych/__sitename__")
        self.checkButtonNeuro.clicked.connect(self.check_neuro_handler)
        self.checkButtonNeuro.setToolTip('INSTRUCTIONS:\nINPUTS => Dirctory\nINPUTS => Site Name')
        
        
        # add neuro duplicates check button 
        self.neuroDupesButton = QPushButton("Check for duplicate files")
        self.neuroDupesButton.clicked.connect(self.neuro_dupes_handler)
        self.neuroDupesButton.setToolTip('INSTRUCTIONS: \nINPUTS => Directory')
        
        # add neuro review button 
        self.reviewButtonNeuro = QPushButton("Run neuropsych check")
        self.reviewButtonNeuro.clicked.connect(self.review_neuro_handler)
        self.reviewButtonNeuro.setToolTip('INSTRUCTIONS:\nINPUTS => Directory')
    
        # add move neuro files button 
        self.moveButtonNeuro = QPushButton("MOVE files to /raw_data/neuropsych/__sitename__")
        self.moveButtonNeuro.clicked.connect(self.move_neuro_handler)
        
        # add buttons to neuroLayout
        self.neuroLayout.addWidget(self.checkButtonNeuro)
        self.neuroLayout.addWidget(self.neuroDupesButton)
        self.neuroLayout.addWidget(self.reviewButtonNeuro)
        self.neuroLayout.addWidget(self.moveButtonNeuro)
        
        #################################################### END NEURO TAB ####################################################

        #################################################### new std out tab 
        
        self.processLayout = QVBoxLayout()
        
        self.process  = QTextEdit()
        
        
        # add a clear button for editor
        clearButton = QPushButton('Clear all text')
        clearButton.clicked.connect(self.clear_function)
        
        self.processLayout.addWidget(self.process)
        self.processLayout.addWidget(clearButton)
        
        self.redirectTab.setLayout(self.processLayout)
        

        
        #################################################### end new std out tab 
         
        
        # Add tabs to tab widget 
        self.tabs.addTab(self.h1Tab,"ERP")
        self.tabs.addTab(self.neuroTab,"Neuropsych")
        self.tabs.addTab(self.redirectTab, "Stdout & Stderr")

        
        self.setCentralWidget(self.tabs)
        self.show()
        
        
        sys.stdout = EmittingStream(textWritten=self.redirect_output)
        
        
    def redirect_output(self, text):

        """Append text to the QTextEdit."""

        # Maybe QTextEdit.append() works as well, but this is how I do it:

        cursor = self.process.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.process.setTextCursor(cursor)
        self.process.ensureCursorVisible()



    def __del__(self):
        sys.stdout = sys.__stdout__
    ###################################################################
        
    def clear_function(self, signal):
        self.process.clear()
        
    def test(self, signal):#### remove this function ######
        
        directoryInp = self.dirInp.text().strip()
        self.erp_and_filesize_check(directoryInp)
        

        
        
    ############################################## ERP HANDLERS ##############################################
    
    def h1_handler(self, signal):
        """ creates avg.h1 files for peak picking outside of _32.cnt directory """
        
        directoryInp = self.dirInp.text().strip()
        directorytrg = self.trgInp.text().strip()
        directoryExclude = self.dirExcludeInp.text().strip()
        
        files = self.filesInp.text().strip()
        files_lst = files.split(' ')
        files_set = set(files_lst)
        
        # create avg.h1 for only 1 directory 
        dirs = [os.path.join(r,n) for r,d,f in os.walk(directoryInp) for n in d]
        if len(dirs) == 0:
            sd.get_h1s(directoryInp, files_set, del_ext=True, trg_dir=directorytrg)
        
        # exclude directories while creating h1's
        if directoryExclude != '':
            excluded = directoryExclude.split()
            include_dirs = [os.path.join(directoryInp, i) for i in os.listdir(directoryInp) if i not in excluded]
            count = 0
            for i in include_dirs:
                count+=1
                print("\n\n{}".format(count)), sd.get_h1s(i, files_set, del_ext=True, trg_dir=directorytrg)
        # create h1's in all sub-directories 
        else:
            count = 0
            for i in dirs:
                count+=1
                print("\n\n{}".format(count)), sd.get_h1s(i, files_set, del_ext=True, trg_dir=directorytrg)
    
    def h1_handler_PWD(self, signal):
        """ creates avg.h1/h1.ps files for viewing purposes """
        
        directoryInp = self.dirInp.text().strip()
        directoryExclude = self.dirExcludeInp.text().strip()
        
        files = self.filesInp.text().strip()
        files_lst = files.split(' ')
        files_set = set(files_lst)
        
        sd.get_h1s(directoryInp, files_set, ps=True)
        
        
    def review_handler(self, signal):
        directoryInp = self.dirInp.text().strip()
        
        dirs = [os.path.join(r,n) for r,d,f in os.walk(directoryInp) for n in d]
        
        count = 0
        for i in dirs:
            count+=1
            print("\n\n{}".format(count)), ep.run_all(i)
        
            
            
    def erp_shell_scripts(self, signal):
        
        buttonReply = QMessageBox.question(self, 'Confirmation Message', "Are you sure?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            directoryInp = self.dirInp.text().strip()
            ep.iter_shell_check(directoryInp)
        else:
            print('No clicked.')
        
        
    ############################################## NEURO HANDLERS ##############################################
    
    def check_neuro_handler(self, signal):
        """Check to see if new neuropsych data already exists in /raw_data/neuropsych/__sitename__"""
        
        directoryInp = self.dirInpNeuro.text().strip()
        siteInp = self.siteInpNeuro.text().strip()
        # my script
        np_run_exists(directoryInp, siteInp)
        
        
    def neuro_dupes_handler(self, signal):
        """ Check for duplicate files across/within neuropsych folders """
        
        directoryInp = self.dirInpNeuro.text().strip()
        
        np.md5_check_walk(directoryInp)
        
        
    def review_neuro_handler(self, signal):
        """Check for file naming/xml inconsistencies in new neuropsych data from site"""
        
        directoryInp = self.dirInpNeuro.text().strip()
        # my script 
        
        dirs = [os.path.join(r,n) for r,d,f in os.walk(directoryInp) for n in d]
        for d in dirs:
            np.run_all(d)
        

    def move_neuro_handler(self, signal):
        """only after the above 2 functions are run, move new site data to /raw_data/neuropsych/__sitename__"""
        
        directoryInp = self.dirInpNeuro.text().strip()
        siteInp = self.siteInpNeuro.text().strip()
        # my script
        move_neuro_files(directoryInp, siteInp)
        
    ############################################## END NEURO HANDLERS ##############################################
        
if __name__ == '__main__':
    app = QCoreApplication.instance() ### adding this if statement prevents kernel from crashing 
    if app is None:
        app = QApplication(sys.argv)
        print(app)
    ex = App()
    sys.exit(app.exec_())