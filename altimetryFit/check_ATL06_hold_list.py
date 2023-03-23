# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 18:26:47 2019

@author: ben
"""
import os
import glob
import re
import numpy as np
import importlib.resources

def read_ATL06_hold_files(hold_dir=None):
    if hold_dir is None:
        with importlib.resources.path('altimetryFit','package_data') as pp:
            hold_dir=os.path.join(pp, 'held_granules')
    rgt=[]; cycle=[]; subprod=[];
    for file in glob.glob(os.path.join(hold_dir, '*.csv')):
        with open(file,'r') as ff:
            for line in ff:
                try:
                     items=line.replace(',',' ').split()
                     cycle.append(int(items[0]))
                     rgt.append(int(items[1]))
                     subprod.append(int(items[2]))
                except ValueError:
                    continue
    hold_list=[item for item in zip(cycle, rgt, subprod)]
    return hold_list


def check_ATL06_hold_list(filenames, hold_list=None, hold_dir=None):
    if hold_list is None:
        hold_list=read_ATL06_hold_files(hold_dir=hold_dir)
    if isinstance(filenames, (str)):
        filenames=list(filenames)
    
    r06=re.compile('ATL.._(\d\d\d\d)(\d\d)(\d\d)(\d\d)(\d\d)(\d\d)_(\d\d\d\d)(\d\d)(\d\d)_(\d\d\d)_(\d\d).h5')
    bad=[]
    for filename in filenames:
        m=r06.search(filename)
        bad.append((int(m.group(8)), int(m.group(7)), int(m.group(9))) in hold_list)
    return bad
