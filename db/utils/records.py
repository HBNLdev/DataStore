''' working with record-style dictionaries '''


def flatten_dict(D, delim='_', prefix=''):
    ''' given nested dict, return un-nested dict with str-joined keys '''
    if len(prefix) > 0:
        prefix += delim
    flat = {}
    for k, v in D.items():
        if type(v) == dict:
            F = flatten_dict(v, delim, prefix + k)
            flat.update(F)
        else:
            flat[prefix + k] = v
    return flat


def unflatten_dict(D, delimiter='_', do_skip=True, skipkeys=set()):
    ''' given flat dict, nest keys given a key delimiter.
        use with caution, specify keys to skip. '''
    default_skipkeys = {'_id', 'test_type', 'ind_id', 'insert_time', 'form_id'}
    skipkeys.update(default_skipkeys)
    resultDict = dict()
    for key, value in D.items():
        if do_skip and key in skipkeys:
            d = resultDict
            d[key] = value
        else:
            parts = key.split(delimiter)
            d = resultDict
            for part in parts[:-1]:
                if part not in d:
                    d[part] = dict()
                d = d[part]
            if isinstance(d, dict):  # this OK?
                d[parts[-1]] = value
    return resultDict


def remove_NaTs(rec):
    ''' given a record-style dict, convert all NaT vals to None '''
    for k, v in rec.items():
        typ = type(v)
        if typ != dict and typ == pd.tslib.NaTType:
            rec[k] = None


def show_dict_hierarchy(d, init_space='', total=0):
    for k, v in d.items():
        space = init_space
        print(space + k)
        if isinstance(v, dict):
            space += '   '
            total = show_dict_hierarchy(v, space, total)
        elif isinstance(v, list):
            print(space + str(len(v)))
            total += len(v)
    return total
