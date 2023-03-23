#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 21:00:02 2019

@author: ben
"""

import matplotlib.pyplot as plt
import numpy as np
import pointCollection as pc
import scipy.ndimage as snd
import sys
import os
import re
import argparse

env_str=''
#env_str='source activate IS2'
error_str=''
#error_str=' --calc_error_for_xy'

prog='fit_OIB_aug.py'

# account for a bug in argparse that misinterprets negative agruents
argv=sys.argv
for i, arg in enumerate(argv):
    if (arg[0] == '-') and arg[1].isdigit(): argv[i] = ' ' + arg


parser=argparse.ArgumentParser(description="generate a list of commands to run fit_OIB.py")
parser.add_argument('step', type=str)
parser.add_argument('defaults_file', type=str)
parser.add_argument('--region_file', '-R', type=str)
args=parser.parse_args()


if len(sys.argv) > 2:
    defaults_file=os.path.abspath(sys.argv[2])


if args.step not in ['centers', 'edges','corners']:
    raise(ValueError('argument not known'))
    sys.exit()

if args.region_file is not None:
    line_re=re.compile('(..)\s*=\s*\[\s*(\S+),\s*(\S+)\s*]')
    temp={}
    with open(args.region_file,'r') as fh:
        for line in fh:
            m = line_re.search(line)
            temp[m.group(1)]=[np.float(m.group(2)), np.float(m.group(3))]
    XR=temp['XR']
    YR=temp['YR']

defaults_re=re.compile('(.*)\s*=\s*(.*)')
defaults={}
with open(args.defaults_file,'r') as fh:
   for line in fh:
       m=defaults_re.search(line)
       if m is not None:
           defaults[m.group(1)]=m.group(2)
#print( defaults)
#sys.exit()

Wxy=float(defaults['-W'])
Hxy=Wxy/2

mask_G=pc.grid.data().from_geotif(defaults['--mask_file'].replace('100m','1km'))
mask_G.z=snd.binary_dilation(mask_G.z, structure=np.ones([20, 1], dtype='bool'))
mask_G.z=snd.binary_dilation(mask_G.z, structure=np.ones([1, 20], dtype='bool'))

try:
    out_dir=defaults['-b']
except KeyError:
    out_dir=defaults['--base_dir']

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
step_dir=out_dir+'/'+args.step
if not os.path.isdir(step_dir):
    os.mkdir(step_dir)

if args.step=='centers':
    delta_x=[0]
    delta_y=[0]
    symbol='r*'
elif args.step=='edges':
    delta_x=[-1, 0, 0, 1.]
    delta_y=[0, -1, 1, 0.]
    symbol='bo'
elif args.step=='corners':
    delta_x=[-1, 1, -1, 1.]
    delta_y=[-1, -1, 1, 1.]
    symbol='ms'

x0=np.unique(np.round(mask_G.x/Hxy)*Hxy)
y0=np.unique(np.round(mask_G.y/Hxy)*Hxy)
x0, y0 = np.meshgrid(x0, y0)

xg=x0.ravel()
yg=y0.ravel()
good=(np.abs(mask_G.interp(xg, yg)-1)<0.1) & (np.mod(xg, Wxy)==0) & (np.mod(yg, Wxy)==0)

if XR is not None:
    good &= (xg>=XR[0]) & (xg <= XR[1]) & (yg > YR[0]) & (yg < YR[1])

#mask_G.show()
xg=xg[good]
yg=yg[good]
#[xg, yg]=pad_bins([xg, yg], 2, [Wxy/2, Wxy/2])

queued=[];
for xy0 in zip(xg, yg):
    for dx, dy in zip(delta_x, delta_y):  
        xy1=np.array(xy0)+np.array([dx, dy])*Hxy
        if tuple(xy1) in queued:
            continue
        else:
            queued.append(tuple(xy1))
        out_file=out_dir+'/%s//E%d_N%d.h5' % (args.step, xy1[0]/1000, xy1[1]/1000)  
        if not os.path.isfile(out_file):
            #plt.plot(xy0[0], xy0[1],'r*')
            #if np.sqrt((xy1[0]-320000.)**2 + (xy1[1]- -2520000.)**2) > 2.e5:
            #    continue
            #plt.plot(xy1[0], xy1[1],symbol)
            cmd='%s %d %d --%s @%s ' % (prog, xy1[0], xy1[1], args.step, defaults_file)
            print(env_str + cmd)# +';'+cmd+error_str)
            

