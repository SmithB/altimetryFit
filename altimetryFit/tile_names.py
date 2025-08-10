#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 14:16:44 2025

@author: ben
"""

import glob
import re
import numpy as np
import os


def tile_centers_from_files(files, out_format='numpy'):
    """
    Get the center locations from a list of files

    Parameters
    ----------
    files : iterable
        list of files.
    out_format : str, optional
        if 'numpy', return a numpy array of coordinates
        if 'dict', return a dictionary whose keys are the tile centers and whose entries are the filenames
        The default is 'numpy'.

    Returns
    -------
    xy_tile : TYPE
        DESCRIPTION.

    """
    tile_re=re.compile('E(.*)_N(.*).h5')
    xy_tile={}
    for file in files:
        thekey =tuple(1000*np.array([*map(float, tile_re.search(os.path.basename(file)).groups())]))
        xy_tile[thekey] = file

    if out_format=='numpy':
        xy_tile=np.c_[list(xy_tile.keys())]
    if out_format=='tuples':
        xy_tile = list(xy_tile.keys())
    return xy_tile

def tile_centers_from_scripts(files, out_format='numpy'):

    xy_re=re.compile('--xy0\s+(\S+)\s+(\S+)')

    if isinstance(files, str):
        files=[files]
    xy_tile = {}
    for file in files:
        with open(file,'r') as fh:
            for line in fh:
                mm = xy_re.search(line)
                if mm is None:
                    continue
                key=tuple([*map(float, mm.groups())])
                xy_tile[key] = [file, line]
    if out_format=='numpy':
        xy_tile=np.c_[list(xy_tile.keys())]
    if out_format=='tuples':
        xy_tile = list(xy_tile.keys())
    return xy_tile
