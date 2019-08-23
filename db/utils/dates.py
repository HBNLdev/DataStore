''' working with dates '''

from datetime import datetime

import numpy as np


def calc_age(dob,date):
    ''' given dob and date, calculate age in years '''
    return (date - dob).days / 365.25

def calc_currentage(dob):
    ''' given a DOB, calculate current age '''

    return (datetime.now() - dob).days / 365.25


def convert_date(datestr, dateform='%m/%d/%Y'):
    ''' convert a date string with a given format '''

    try:
        return datetime.strptime(datestr, dateform)
    except:
        return np.nan


def convert_date_fallback(dstr, dateform):
    ''' parse date, falling back on a default date format if it fails '''

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

def date_only(ds):
    dtime = convert_date(ds)
    try:
        return datetime(*dtime.utctimetuple()[:3])
    except: return dtime

def date_string_clean(date):
    ''' write date as string in American form without zero pads'''
    return date.strftime('%m/%d/%Y').lstrip("0").replace("/0","/")

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
                    dstr = '7/1' + dstr[5:]
            else:
                dstr = '7' + dstr[2:]
        else:
            if dstr[3:5] == '??':
                dstr = dstr[:2] + '/15' + dstr[5:]
    try:
        return datetime.strptime(dstr, '%m/%d/%Y')
    except:
        print('problem with date: ' + dstr)
        return np.nan


def datetime_fromtimestamp(stamp):
    ''' given a UTC timestamp, return a datetime '''

    ts = (stamp - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime.utcfromtimestamp(ts)
