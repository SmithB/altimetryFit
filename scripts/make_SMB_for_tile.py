#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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


def make_queue(args, defaults_files):

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

    defaults_str = " ".join(defaults_files)
    for file in files:
        thestr=f'make_SMB_for_tile.py --file {file} {defaults_str} --time {args.time}'
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
    parser.add_argument("--time", '-t', type=str)
    parser.add_argument('--reference_epoch', type=int)
    parser.add_argument('--reference_time', type=float)
    parser.add_argument('--grid_spacing','-g', type=str, help='grid spacing:DEM (meters),dh maps xy (meters),dh_maps time (years): comma-separated, no spaces', default='250.,4000.,1.')
    parser.add_argument('--base_directory','-b', type=path, help='base directory')
    parser.add_argument('--tide_mask_file', type=path)
    parser.add_argument('--make_queue','-q', action='store_true')
    parser.add_argument('--tile_files', type=str, nargs='+')
    parser.add_argument('--M2H_file', type=str)
    parser.add_argument('--firn_directory', type=str)
    parser.add_argument('--firn_grid_file', type=str)
    parser.add_argument('--RACMO_file', type=str)
    parser.add_argument('--rho_water', type=float, default=1)
    parser.add_argument("--DEBUG", action='store_true')
    parser.add_argument('--prelim', action='store_true')
    parser.add_argument('--matched', action='store_true')
    parser.add_argument("--verbose",'-v', action="store_true")
    parser.add_argument("--initial_FAC", action="store_true")
    parser.add_argument('--fields', type=str, nargs='+', default=['SMB_a','FAC'])
    parser.add_argument('--groups', type=str, nargs='+', default=['dz','z0'])
    parser.add_argument('--extrapolate_time','-e', action='store_true')
    args, _=parser.parse_known_args()

    defaults_files=[]
    for arg in sys.argv:
        if arg[0]=='@':
            defaults_files += [arg]

    if args.make_queue:
        make_queue(args, defaults_files)
        sys.exit(0)
    args.time=[*map(float, args.time.split(','))]
    args.grid_spacing=[*map(float, args.grid_spacing.split(','))]

    if args.reference_time is None:
        epoch_ind = int(args.reference_epoch)

    w_smooth=1
    pad_mask=np.array([-250, 250])
    pad_f=np.array([-2.6e4, 2.6e4])
    pad_c=pad_f*3

    dz, z0 = [None, None]
    if 'dz' in args.groups:
        dz=pc.grid.data().from_h5(args.file, group='dz')
        bounds=dz.bounds()
    else:
        z0=pc.grid.data().from_h5(args.file, group='z0')
        bounds=z0.bounds()

    smbfd=pc.grid.data().from_nc(args.M2H_file, bounds=[np.array(jj)+pad_f for jj in bounds], t_range=[args.time[0]-1, args.time[1]+1])
    if not(np.all(np.isfinite(smbfd.SMB_a))):
        smbfd=pc.grid.data().from_nc(args.M2H_file, bounds=[np.array(jj)+pad_c for jj in bounds], t_range=[args.time[0]-1, args.time[1]+1])
        for field in smbfd.fields:
            setattr(smbfd, field, fill_SMB_gaps(getattr(smbfd, field), w_smooth))

    if 'dz' in args.groups:
        floating = pc.grid.data().from_file(args.tide_mask_file, bounds=[np.array(jj)+pad_mask for jj in bounds])\
            .interp(dz.x, dz.y, gridded=True)>0.5
        floating[~np.isfinite(floating)]=0
        float_scale = (floating==0) + (args.rho_water-.917)/args.rho_water*floating

        if args.reference_time is not None:
            epoch_ind = np.argmin(np.abs(dz.t-args.reference_time))

        if args.extrapolate_time:
            if smbfd.t[-1] < dz.t[-1]:
                smbfd.t[-1] = dz.t[-1]

        for field in args.fields:
            temp=smbfd.interp(dz.x, dz.y, dz.t, field=field, gridded=True)
            if field == 'SMB_a':
                temp *= float_scale[:,:,None]
            dz.assign( {field :  temp - temp[:,:,epoch_ind][:,:,None]} )
    if 'z0' in args.groups:
        t0=np.arange(args.time[0], args.time[-1]+1, args.grid_spacing[2])[epoch_ind]

        if z0 is None:
            z0=pc.grid.data().from_h5(args.file, group='z0')
        for field in args.fields:
            z0.assign({field : smbfd.interp(z0.x, z0.y, np.array(t0), field=field, gridded=True )})

        if args.initial_FAC:
            with h5py.File(args.M2H_file,'r') as h5f:
                if 't' in h5f:
                    t01=h5f['t'][0:2]
                else:
                    t01=h5f['time'][0:2]
            FAC_0 = pc.grid.data().from_nc(args.M2h_file,
                                           bounds=[np.array(jj)+pad_f for jj in bounds],
                                           t_range=t01).interp(z0.x, z0.y, [t01], gridded=True)

    with h5py.File(args.file,'r+') as h5f:
        for field in args.fields:
            if 'dz' in args.groups:
                if field not in h5f['dz']:
                    h5f.create_dataset('dz/'+field, data=getattr(dz, field),
                       chunks=True, compression="gzip", fillvalue=dz.fill_value)
                else:
                    h5f['dz/'+field][...]=getattr(dz, field)
            if 'z0' in args.groups:
                if field not in h5f['z0']:
                    h5f.create_dataset('z0/'+field, data=getattr(z0, field),
                           chunks=True, compression="gzip", fillvalue=z0.fill_value)
                else:
                    h5f['z0/'+field][...]=getattr(z0, field)
        if args.initial_FAC:
            if 'initial_FAC' not in h5f['z0']:
                h5f.create_dataset('z0/initial_FAC', data=FAC_0,
                       chunks=True, compression="gzip", fillvalue=z0.fill_value)
            else:
                h5f['z0/initial_FAC'][...]=FAC_0

if __name__=='__main__':
    main()
