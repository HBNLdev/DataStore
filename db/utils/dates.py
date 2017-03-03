''' working with dates '''

from datetime import datetime

import numpy as np


def calc_currentage(dob):
    ''' given a DOB, calculate current age '''
    return (datetime.now() - dob).days / 365.25


def convert_date(datestr, dateform='%m/%d/%Y'):
    ''' convert a date string with a given format '''
    try:
        return datetime.strptime(datestr, dateform)
    except:
        return np.nan


def calc_date_w_Qs(dstr):
    ''' given date string of form mm/dd/yyyy, potentially containing ?? in some positions,
        return datetime object '''

    dstr = str(dstr)
    if dstr == 'nan':
        return np.nan
    if '?' in dstr:
        if dstr[:2] == '??':
            if dstr[3:5] == '??':
                if dstr[5:7] == '??':
                    return None
                else:
                    dstr = '6/1' + dstr[5:]
            else:
                dstr = '6' + dstr[2:]
        else:
            if dstr[3:5] == '??':
                dstr = dstr[:2] + '/15' + dstr[5:]
    try:
        return datetime.strptime(dstr, '%m/%d/%Y')
    except:
        print('problem with date: ' + dstr)
        return np.nan


def parse_date(dstr, dateform):
    ''' parse date column '''

    dstr = str(dstr)

    if dstr != 'nan':
        return datetime.strptime(dstr, dateform)
    else:
        return None


def parse_date2_apply(dstr, dateform):
    dstr = str(dstr)

    if dstr != 'nan':
        try:
            return datetime.strptime(dstr, dateform)
        except ValueError:
            return None
    else:
        return None


def parse_date_apply_pd(dstr, dateform):
    dstr = str(dstr)

    if dstr != 'nan':
        try:
            return datetime.strptime(dstr, dateform)
        except ValueError:
            try:
                return datetime.strptime(dstr, '%Y-%m-%d')
            except ValueError:
                return None
    else:
        return None
