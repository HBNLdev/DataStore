''' h5py parsing functions designed to interpret MATLAB .mat's saved using '-v7.3' '''

import h5py
import numpy as np


def show_file_hierarchy(h5_path):
    ''' given a path to an HDF-compatible file, show the hierarchy of its contents '''

    def disp_node(name, node):
        print(name)
        indent = name.count('/') * 5 * ' '
        if 'items' in dir(node):
            for k, v in node.items():
                print(indent, k, ': ', v)
        else:
            print(indent, node)
            if type(node) == h5py._hl.dataset.Dataset:
                print(indent, node.dtype)

    loaded = h5py.File(h5_path, 'r')
    loaded.visititems(disp_node)


def parse_text(dset, dset_field):
    ''' parse .mat-style h5 field that contains text '''
    dset_ref = dset[dset_field]
    return ''.join(chr(c[0]) for c in dset_ref[:])


def parse_textarray(dset, dset_field):
    ''' parse .mat-style h5 field that contains a text array '''
    dset_ref = dset[dset_field]
    array = dset_ref[:]
    return [''.join([chr(arg) for arg in args]).rstrip() for args in
            zip(*array)]


def parse_cell(dset, dset_field):
    ''' parse .mat-style h5 field that contains a cell array '''
    try:
        dset_ref = dset[dset_field]
        refs = [t[0] for t in dset_ref[:]]
        out_lst = [''.join(chr(c) for c in dset[ref][:]) for ref in refs]
        return out_lst
    except:
        return []


def parse_array(dset, dset_field):
    ''' parse .mat-style h5 field that contains a numerical array.
        not in use yet. '''
    contents = dset[dset_field][:]
    if contents.shape == (1, 1):
        return contents[0][0]
    elif contents == np.array([0, 0], dtype=np.uint64):
        return None
    else:
        return contents


ftypes_funcs = {'text': parse_text, 'cell': parse_cell, 'array': None,
                'text_array': parse_textarray}


def handle_parse(dset, dset_field, field_type):
    ''' given file pointer, field, and datatype, apply appropriate parser '''
    func = ftypes_funcs[field_type]
    return func(dset, dset_field) if func else dset[dset_field][:]
