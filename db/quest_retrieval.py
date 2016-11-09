''' tools for downloading questionnaire data onto the HBNL filesystem '''

import os
import shutil
import requests
import zipfile
from time import sleep

from db.quest_import import sasdir_tocsv, map_ph4, map_ph4_ssaga, map_subject, ach_url

# combine maps for future usage
all_kmap = map_ph4.copy()
all_kmap.update(map_ph4_ssaga)
all_kmap.update(map_subject)

all_ph4_kmap = map_ph4.copy()
all_ph4_kmap.update(map_ph4_ssaga)

# achenbach specific info
ach_url_parts = ach_url.split(os.path.sep)
ach_currentname = os.path.splitext(ach_url_parts[-1])[0]
ach_currentname_spaces = ach_currentname.replace('%20', ' ')
ach_currentname_nospaces = ach_currentname.replace('%20', '')


def zork_retrieval(user_name, password, distro_num, target_base_dir='/processed_data/zork'):
    ''' given a username, password, phase 4 distribution #, downloads most recent distribution
        of questionnaire data to a target base directory. will nest files into the correct subdirs. '''

    # convert distro_num to string (if not string)
    distro_num = str(distro_num)
    distro_subdir = 'zork-phase4-' + distro_num

    path = os.path.join(target_base_dir, distro_subdir)

    succeeded = zork_download(target_base_dir, distro_subdir, user_name, password)
    if not succeeded:
        print('a download failed, aborting retrieval.')
        return False

    zork_move(target_base_dir, distro_subdir)

    recursive_unzip(path)

    zork_convert(path)

    return True


def zork_download(target_base_dir, distro_subdir, user_name, password):
    ''' downloads zork zips and places them into appropriate directories'''

    # create full urls from dictionary
    base_url = 'https://zork5.wustl.edu/coganew/data/available_data'
    url_lst = []
    for quest, quest_info in all_kmap.items():
        url_lst.append(base_url + quest_info['zork_url'])

    # create subject and session directories
    for name in 'session', 'subject':
        new_folder = os.path.join(target_base_dir, distro_subdir, name)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

    # log in and download zip files
    for url in url_lst:
        try:
            download = requests.get(url, auth=(user_name, password))
        except:
            print('download failed:', url)
            return False
        sleep(3) # pause for a few seconds to give the server a break
        # error check -- if url doesnt exist then print missing zip???
        zip_name = url.split('/')[-1]
        with open(os.path.join(target_base_dir, distro_subdir, zip_name), 'wb') as zip_pointer:
            zip_pointer.write(download.content)
            print('Downloading ' + '||' + zip_name + '||')
    
    return True

def zork_move(target_base_dir, distro_subdir):
    ''' creates directories and moves zork zips around '''

    # moves zip files to subject or session directory
    path = os.path.join(target_base_dir, distro_subdir)
    for file in os.listdir(path):
        # all phase 4 questionnaires belong in the "session" subdirectory
        for quest, quest_info in all_ph4_kmap.items():
            if file.startswith(quest_info['zip_name']):
                shutil.move(os.path.join(path, file), os.path.join(path, 'session'))
        # all subject questionnaires beloing in the "subject" subdirectory
        os.chdir(os.path.join(path, 'subject'))
        for quest, quest_info in map_subject.items():
            if file.startswith(quest_info['zip_name']):
                shutil.move(os.path.join(path, file), os.path.join(path, 'subject'))

    # create subdirectories named after questionnaires
    for file in os.listdir(os.path.join(path, 'session')):
        for quest, quest_info in all_ph4_kmap.items():
            if file.startswith(quest_info['zip_name']):
                if not os.path.exists(os.path.join(path, 'session', quest)):
                    os.makedirs(os.path.join(path, 'session', quest))
                    shutil.move(os.path.join(path, 'session', file), os.path.join(path, 'session', quest, file))
    for file in os.listdir(os.path.join(path, 'subject')):
        for quest, quest_info in map_subject.items():
            if file.startswith(quest_info['zip_name']):
                if not os.path.exists(os.path.join(path, 'subject', quest)):
                    os.makedirs(os.path.join(path, 'subject', quest))
                    shutil.move(os.path.join(path, 'subject', file), os.path.join(path, 'subject', quest, file))


def recursive_unzip(path):
    ''' given a path, recursively unzip all files within '''

    # unzip files in their directories
    for roots, dirs, files in os.walk(path):
        for name in files:
            if name.endswith('.zip'):
                zipfile.ZipFile(os.path.join(roots, name)).extractall(os.path.join(roots))
                print('Unzipping ' + '||' + name + '||')


def zork_convert(path):
    ''' convert all .sas7bdat files within path to csvs and handle some exceptions unique to zork stuff '''

    # create csvs, remove zips
    for roots, dirs, files in os.walk(path):
        for name in dirs:
            # for achenbach, zips nested in folder with spaces in name, so we have to rename it and move them up
            if name == 'achenbach':
                try:
                    old_name = os.path.join(roots, name, ach_currentname_spaces)
                    new_name = os.path.join(roots, name, ach_currentname_nospaces)
                    os.rename(old_name, new_name)
                    for file in os.listdir(new_name):
                        shutil.move(os.path.join(roots, name, ach_currentname_nospaces, file),
                                    os.path.join(roots, name, file))
                    os.rmdir(os.path.join(roots, name, ach_currentname_nospaces))
                except:
                    print('problem with renaming achenbach')
            try:
                sasdir_tocsv(os.path.join(roots, name) + '/')
                print('Creating csvs for ' + '||' + name + '||')
            except:
                print('problem with converting for', name)
        for name in files:
            if name.endswith('.zip'):
                os.remove(os.path.join(roots, name))
