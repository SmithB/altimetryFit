#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:04:03 2023

@author: ben
"""

import re
import requests
import shutil
import os
import json

## Is this a good place to park the code for making the PGC url dictionary?


def get_pgc_url(filename, pgc_url_file):
    """Get the bas url for a file at PGC."""
    pgc_re=re.compile('(SETSM_.*_)\d+m(_lsf_seg\d+)')
    pgc_base='2m'.join(list(pgc_re.search(filename).groups()))

    pgc_url=None
    with open(pgc_url_file,'r') as fh:
        for line in fh:
            if pgc_base in line:
                pgc_url=line.rstrip()
                break

    # complain if no entry is in the PGC url file
    if pgc_url is None:
        raise(IndexError(f'No PGC url found in {pgc_url_file} for {pgc_base}'))

    return pgc_url

def download_pgc(filename, pgc_url, extension, file_type='.tif'):
    """Download a file from PGC."""
    pgc_re=re.compile('(SETSM_.*_seg\d+)')
    pgc_base=pgc_re.search(filename).group(1)

    dst_file  = os.path.join(os.path.dirname(filename), pgc_base + extension + file_type)
    if not os.path.isfile(dst_file):
        with requests.get(pgc_url+extension+'.tif', stream=True) as r:
            with open( dst_file ,'wb') as fh:
                shutil.copyfileobj(r.raw, fh)
    return dst_file

def get_meta_from_json(filename, pgc_url):
    """Get the metadata for the filename from the pcg json file."""
    return json.loads(
        requests.get(pgc_url+'.json').content)

def get_pgc(filename, pgc_url_file, targets=['url']):
    """
    Get files or data from PGC.

    Parameters
    ----------
    filename : str
        Filename to use to search for PGC files.
    pgc_url_file : str
        Filename for a file containing URLs for each PGC DEM filename.
    targets : list of strings, optional
        Targets to read.  Can include 'url', 'masks', or 'dem'. The default is ['url'].

    Returns
    -------
    out : dict
        Dict containing filenames for saved files.

    """
    out={}
    pgc_url=get_pgc_url(filename, pgc_url_file)
    if 'url' in targets:
        out['url']=pgc_url

    if 'masks' in targets:
        dst_files={}
        # Code for downloading the full-res masks(not necessary - see the option to read the url with gdal.Warp
        for extension in ['_bitmask','_matchtag']:
            key = extension.replace('_','')
            dst_files[ key ] = download_pgc(filename, pgc_url, extension)
        out['masks']=dst_files

    if 'dem' in targets:
        extension = '_dem'
        out['dem'] = download_pgc(filename, pgc_url, extension)

    if 'mdf' in targets:
        extension = '_mdf'
        out['meta'] = download_pgc(filename, pgc_url, extension, file_type='.txt')

    if 'meta' in targets:
        out['meta']=get_meta_from_json(filename, pgc_url)

    return out
