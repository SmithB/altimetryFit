#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 15:16:14 2025

@author: ben
"""
import numpy as np
import pointCollection as pc
import glob
from altimetryFit import SMB_corr_from_grid
import os

import argparse

def path(p):
    return os.path.abspath(os.path.expanduser(p))

def main():

    parser=argparse.ArgumentParser('determine whether a file has been run with the --calc_firn_anomaly flag set', fromfile_prefix_chars="@")
    parser.add_argument('--base_directory','-b', type=path, help='base directory')
    parser.add_argument('--xy0', type=float, nargs=2, help='tile center to check')
    parser.add_argument('--tile_file', type=path, help='file to check for the SMB correction, overrides xy0.  '
                        'Without --base_directory, the result is written to the fit directory that contains the tile')
    parser.add_argument('--firn_directory', type=path, help='directory containing firn model')
    parser.add_argument('--firn_model', type=str, help='firn model name')
    parser.add_argument('--firn_version', type=str, help='firn version')
    parser.add_argument('--firn_grid_file', type=str, help='gridded firn model file that can be interpolated directly.')
    parser.add_argument('--tide_mask_file', type=path)
    args, unk = parser.parse_known_args()

    tile_file=args.tile_file
    if tile_file is not None:
        if args.base_directory is None:
            # the result is written to the fit directory: the one that
            # contains the tile's prelim/ or matched/ directory
            tile_dir=os.path.dirname(tile_file)
            if os.path.basename(tile_dir) in ('prelim', 'matched'):
                tile_dir=os.path.dirname(tile_dir)
            args.base_directory=tile_dir
    else:
        if args.base_directory is None:
            parser.error('specify --base_directory, or --tile_file')
        if args.xy0 is not None:
            tile_file=os.path.join(args.base_directory, 'prelim', f'E{int(args.xy0[0]/1000)}_N{int(args.xy0[1]/1000)}.h5')
        if tile_file is None or not os.path.isfile(tile_file):
            tile_files=glob.glob(args.base_directory+'/prelim/E*.h5')
            if len(tile_files)==0:
                parser.error(f'no tiles found in {args.base_directory}/prelim')
            tile_file=tile_files[0]

    data=pc.data().from_h5(tile_file, group='data')
    h_firn = data.h_firn.copy()


    cfa_sigma_firn = []
    for calc_FAC_anomaly in [False, True]:
        SMB_corr_from_grid(data,
                            model_file=os.path.join(args.firn_directory,args.firn_grid_file),
                            calc_FAC_anomaly = calc_FAC_anomaly)
        delta = data.h_firn - h_firn
        cfa_sigma_firn+= [[calc_FAC_anomaly,  np.nanmean(np.abs(delta))]]
    cfa_sigma_firn = np.c_[cfa_sigma_firn]
    print(cfa_sigma_firn)
    calc_FAC_anomaly = cfa_sigma_firn[np.argmin(cfa_sigma_firn[:,1]), 0].astype(bool)

    with open(os.path.join(args.base_directory,'calc_FAC_anomaly.txt'),'w') as fh:
        fh.write(str(calc_FAC_anomaly))

    print("calc_FAC_anomaly is " +str(calc_FAC_anomaly))

if __name__=='__main__':
    main()
