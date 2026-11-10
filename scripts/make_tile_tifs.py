#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:17:08 2024

@author: ben
"""

import numpy as np
#import matplotlib.pyplot as plt
import pointCollection as pc
import argparse
import sys
import scipy.ndimage as snd
import os
import glob

def make_tile_tifs(dz_file, epsg, verbose=True, mask_w = None, geoid_file=None, read_sigma=False, skip_z0=False):

    input_dir=os.path.dirname(os.path.dirname(dz_file))

    if isinstance(epsg,str):
        epsg=int(epsg)
    #gtif_kwargs={'driver':'COG','srs_epsg':epsg}
    gtif_kwargs={'srs_epsg':epsg}
    tif_dir=os.path.join(input_dir,'tif')
    try:
        os.makedirs(tif_dir, exist_ok=True)
    except FileExistsError:
        pass
    groups = ['epoch_count','dzdt','std_dzdt',  'epoch_count', 'cell_area']
    if read_sigma:
        groups.append('sigma_dz')
    if not skip_z0:
        groups += ['z0','dz0dx', 'count']
    if read_sigma and (not skip_z0):
        groups += ['sigma_z0']
    for group in groups:
        if not os.path.isdir(os.path.join(tif_dir, group)):
            os.makedirs(os.path.join(tif_dir, group), exist_ok=True)


    if verbose:
        print("working on "+dz_file)
    base, ext=os.path.splitext(os.path.basename(dz_file).replace('dz',''))
    if not skip_z0:
        z0_file=os.path.join(input_dir,'z0', 'z0'+base+ext)
        z0_fields=['z0','cell_area','count']
        if read_sigma:
            z0_fields += ['sigma_z0']
        if os.path.isfile(z0_file):
            z0=pc.grid.data().from_h5(z0_file, group='z0', fields = z0_fields)
            z0.z0[z0.cell_area < 10]=np.nan
            if mask_w is not None:
                mask_N = int(np.ceil(mask_w / (z0.x[1]-z0.x[0])))
                z0.assign(mask=snd.binary_dilation(\
                    snd.binary_dilation(z0.count>0.5, np.ones([mask_N,1])),\
                        np.ones([1, mask_N])).astype(float))
                z0.mask[z0.mask==0]=np.nan
                z0.z0 *= z0.mask

            if geoid_file is not None:
                geoid=pc.grid.data().from_geotif(geoid_file, bounds=z0.bounds(pad=1.e4))
                z0.z0 -= geoid.interp(z0.x, z0.y, gridded=True)
            z0.calc_gradient(field='z0')
            z0.to_geotif(os.path.join(tif_dir,'dz0dx', f'dz0dx{base}.tif'), field='z0_x', **gtif_kwargs)
            z0.to_geotif(os.path.join(tif_dir,'z0', f'z0{base}.tif'), field='z0', **gtif_kwargs)
            z0.to_geotif(os.path.join(tif_dir,'count', f'count{base}.tif'), field='count', **gtif_kwargs)
            z0.to_geotif(os.path.join(tif_dir,'cell_area', f'cell_area{base}.tif'), field='cell_area', **gtif_kwargs)
            if read_sigma:
                z0.to_geotif(os.path.join(tif_dir,'sigma_z0', f'sigma_z0{base}.tif'), field='sigma_z0', **gtif_kwargs)
    dz_file=os.path.join(input_dir,'dz','dz'+base+ext)
    if os.path.isfile(dz_file):
        dz_fields = ['dz','cell_area','count']
        if read_sigma:
            dz_fields += ['sigma_dz']
        dz=pc.grid.data().from_h5(dz_file, group='dz', fields=dz_fields)
        dz.dz[dz.cell_area<100]=np.nan

        dz.assign(epoch_count=np.nansum(dz.count > 0.25, axis=2))
        if mask_w is not None:
            mask_N = int(np.ceil(mask_w / (dz.x[1]-dz.x[0])))
            dz.assign(mask=snd.binary_dilation(dz.epoch_count>0, np.ones([mask_N, mask_N])).astype(float))
            dz.mask[dz.mask==0]=np.nan
        else:
            dz.mask=np.ones_like(dz.epoch_count)

        dz.assign(dzdt=dz.mask*(dz.dz[:,:,-1]-dz.dz[:,:,0])/(dz.t[-1]-dz.t[0]))
        dz.to_geotif(os.path.join(tif_dir,'dzdt', f'dzdt{base}.tif'), field='dzdt', **gtif_kwargs)
        dz.assign(std_dzdt=dz.mask*np.std(np.diff(dz.dz, axis=2), axis=2)/(dz.t[1]-dz.t[0]))
        dz.to_geotif(os.path.join(tif_dir, 'std_dzdt',f'std_dzdt{base}.tif'), field='std_dzdt', **gtif_kwargs)
        dz.to_geotif(os.path.join(tif_dir, 'epoch_count',f'epoch_count{base}.tif'), field='epoch_count', **gtif_kwargs)
        if 'sigma_dz' in dz_fields:
            dz.to_geotif(os.path.join(tif_dir,'sigma_dz', f'sigma_dz{base}.tif'), field='sigma_dz', **gtif_kwargs)  
    dzL_file=os.path.join(input_dir,'lagrangian_dz','lagrangian_dz'+base+ext)
    if os.path.isfile(dzL_file):
        for group in ['dzL','dzLdx','dz0_tot_dx']:
            if not os.path.isdir(os.path.join(tif_dir, group)):
                os.makedirs(os.path.join(tif_dir, group), exist_ok=True)

        dzL=pc.grid.data().from_h5(dzL_file, group='lagrangian_dz', field='dz')
        if 'mask' in z0.fields:
            dzL.dz *= z0.mask
        dzL.dz[z0.cell_area <10]=np.nan
        dzL.calc_gradient(field='dz')
        dzL.to_geotif(os.path.join(tif_dir,'dzLdx', f'dzLdx{base}.tif'), field='dz_x', **gtif_kwargs)
        dzL.to_geotif(os.path.join(tif_dir,'dzL', f'dzL{base}.tif'), field='dz', **gtif_kwargs)
        dzL.assign(z0_tot=z0.z0+dzL.dz)
        dzL.calc_gradient(field='z0_tot')
        dzL.to_geotif(os.path.join(tif_dir, 'dz0_tot_dx', f'dz0_tot_dx{base}.tif'), field='z0_tot_x', **gtif_kwargs)
    return None

def main():
    args=sys.argv
    parser=argparse.ArgumentParser()
    parser.add_argument('dz_file', type=str, help='tile file containing dz')
    parser.add_argument('epsg', type=int, help='working EPSG')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--mask_w', type=float, help='if sepcified, only points within this distance of a datapoint will have data in the output')
    parser.add_argument('--read_sigma', action='store_true', help='if specified, the errors in z0 will be included in the output')
    parser.add_argument('--geoid_file', type=str, help='geoid file to correct elevations')
    parser.add_argument('--skip_z0', action='store_true')

    args=parser.parse_args()

    for thefile in glob.glob(args.dz_file):
        make_tile_tifs(thefile, args.epsg,
                       verbose=args.verbose,
                       mask_w=args.mask_w,
                       geoid_file=args.geoid_file,
                       read_sigma=args.read_sigma,
                       skip_z0=args.skip_z0)

if __name__=='__main__':
    main()
