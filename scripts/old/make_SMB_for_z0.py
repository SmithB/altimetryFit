#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Retired 2026-11-09, replaced by make_dem_for_time.py:
#   make_dem_for_time.py T0 OUT_FILE --z0_file Z0_FILE --SMB_file M2H_FILE \
#       --EPSG 3031 --corrections FAC,SMB
# gives the surface consistent with the fits' FAC anomaly; --FAC0_scale 0 adds
# the absolute FAC, as this script did.  Unlike this script, make_dem_for_time
# does not scale the SMB for floating ice.
"""
Created on Mon Aug 28 15:51:44 2023

@author: ben
"""

import pointCollection as pc
import numpy as np
import os
import glob
import h5py
import sys

import scipy.ndimage as snd
def smooth_corrected(z, w_smooth, set_NaN=True, mask=None, return_mask=False):
    if mask is None:
        mask=np.isfinite(z)
    mask1=snd.gaussian_filter(np.float64(mask), w_smooth, mode="constant", cval=0)
    ztemp=np.nan_to_num(z)
    ztemp[mask==0]=0.0
    zs=snd.gaussian_filter(ztemp, w_smooth, mode="constant", cval=0)
    zs[mask1>0]=zs[mask1>0]/mask1[mask1>0]
    if set_NaN:
        zs[mask1==0]=np.nan
    if return_mask:
        return zs, mask
    else:
        return zs

def fill_gaps(z, w_smooth, mask=None, set_NaN=True, return_mask=False):
    if return_mask:
        zs, mask1 = smooth_corrected(z, w_smooth, mask=mask, set_Nan=set_NaN)
        missing= mask1==0
        z[missing]=zs[missing]
        return z, mask1
    else:
        zs=smooth_corrected(z, w_smooth, mask=mask, set_NaN=set_NaN)
        missing=~np.isfinite(z)
        z[missing]=zs[missing]
        return z

def fill_SMB_gaps(z, w_smooth):
    if z.ndim==3:
        for ii in range(z.shape[2]):
            temp=z[:,:,ii]
            fill_gaps(temp, w_smooth)
            z[:,:,ii]=temp
    else:
        fill_gaps(z, w_smooth)
    return z


def make_queue(args, defaults_file):

    if args.prelim:
        step='prelim'
    elif args.matched:
        step='matched'
    else:
        raise(ValueError('Need to specify prelim or matched'))
    if args.tile_files is None:
        thedir=os.path.join(args.base_directory, step)
        files=glob.glob(thedir+'/E*.h5')
    else:
        files=args.tile_files

    if args.M2H_file is None:
        args.M2H_file=args.firn_directory+'/'+args.firn_grid_file

    for file in files:
        thestr=f'make_SMB_for_tile.py --file {file} @{defaults_file} --time {args.time}'
        for key in ['M2H_file', 'rho_water']:
            thestr += f' --{key} {getattr(args, key)}'
        for key in ['prelim', 'matched']:
            if getattr(args, key):
                thestr += f' --{key}'
        for key in ['fields', 'groups']:
            temp=' '.join(getattr(args, key))
            thestr += f' --{key} {temp}'
        print(thestr)

def main():

    def path(p):
        return os.path.abspath(os.path.expanduser(p))

    import argparse
    parser=argparse.ArgumentParser(description='Find the best offset for a DEM relative to a set of altimetry data', fromfile_prefix_chars="@")
    parser.add_argument('--file', type=str, required=False)
    parser.add_argument('--reference_time', type=float)
    parser.add_argument('--grid_spacing','-g', type=str, help='DEM grid spacing:DEM (meters)', default=100)
    parser.add_argument('--base_directory','-b', type=path, help='base directory')
    parser.add_argument('--tide_mask_file', type=path)
    parser.add_argument('--tile_file', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--M2H_file', type=str)
    parser.add_argument('--firn_directory', type=str)
    parser.add_argument('--firn_grid_file', type=str)
    parser.add_argument('--RACMO_file', type=str)
    parser.add_argument('--rho_water', type=float, default=1)
    parser.add_argument('--epsg', type=int, default=3031)
    parser.add_argument("--DEBUG", action='store_true')
    parser.add_argument("--verbose",'-v', action="store_true")
    parser.add_argument('--fields', type=str, nargs='+', default=['SMB_a','FAC'])
    args, _=parser.parse_known_args()


    for arg in sys.argv:
        if arg[0]=='@':
            defaults_file=arg[1:]

    #if args.make_queue:
    #    make_queue(args, defaults_file)
    #    sys.exit(0)

    w_smooth=1
    pad_mask=np.array([-250, 250])
    pad_f=np.array([-2.6e4, 2.6e4])
    pad_c=pad_f*3

    z0=pc.grid.data().from_geotif(args.file)
    bounds=z0.bounds()
    t_range=[args.reference_time-0.1, args.reference_time+0.1]


    smbfd=pc.grid.data().from_nc(args.M2H_file, bounds=[np.array(jj)+pad_f for jj in bounds], t_range=t_range)
    if not(np.all(np.isfinite(smbfd.SMB_a))):
        smbfd=pc.grid.data().from_nc(args.M2H_file, bounds=[np.array(jj)+pad_c for jj in bounds], t_range=t_range)
        for field in smbfd.fields:
            setattr(smbfd, field, fill_SMB_gaps(getattr(smbfd, field), w_smooth))


    floating = pc.grid.data().from_file(args.tide_mask_file, bounds=[np.array(jj)+pad_mask for jj in bounds])\
        .interp(z0.x, z0.y, gridded=True)>0.5
    floating[~np.isfinite(floating)]=0
    float_scale = (floating==0) + (args.rho_water-.917)/args.rho_water*floating


    t0=args.reference_time


    z0.z += (float_scale * np.squeeze(smbfd.interp(z0.x, z0.y, np.array(t0), field='SMB_a', gridded=True )))
    z0.z += np.squeeze(smbfd.interp( z0.x, z0.y, np.array(t0), field='FAC', gridded=True ))

    out_file=os.path.join(args.out_dir, os.path.basename(args.file))
    z0.to_geotif(out_file, srs_epsg = args.epsg)

if __name__=='__main__':
    main()
