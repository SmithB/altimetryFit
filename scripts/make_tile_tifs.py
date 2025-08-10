#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:17:08 2024

@author: ben
"""

import numpy as np
#import matplotlib.pyplot as plt
import h5py
import pointCollection as pc
import argparse
import sys
import scipy.stats as sps
import scipy.ndimage as snd
import os
import glob

def make_tile_tifs(dz_file, epsg, verbose=True, mask_w = None, geoid_file=None, read_sigma=False):

    input_dir=os.path.dirname(os.path.dirname(dz_file))

    if isinstance(epsg,str):
        epsg=int(epsg)
    tif_dir=os.path.join(input_dir,'tif')
    if not os.path.isdir(tif_dir):
        os.mkdir(tif_dir)
    groups = ['z0','dz0dx', 'count','epoch_count','dzdt','sigma_dzdt',  'epoch_count','dz0_tot_dx']
    if read_sigma:
        groups += ['sigma_z0']
    for group in groups:
        if not os.path.isdir(os.path.join(tif_dir, group)):
            os.mkdir(os.path.join(tif_dir, group))


    if verbose:
        print("working on "+dz_file)
    base=os.path.basename(dz_file).replace('dz','')
    z0_file=os.path.join(input_dir,'z0', 'z0'+base)
    z0_fields=['z0','cell_area','count']
    if read_sigma:
        z0_fields += ['sigma_z0']
    if os.path.isfile(z0_file):
        z0=pc.grid.data().from_h5(z0_file, group='z0', fields = sigma_z0)
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
        z0.to_geotif(os.path.join(tif_dir,'dz0dx', f'dz0dx{base}.tif'), field='z0_x', srs_epsg=epsg)
        z0.to_geotif(os.path.join(tif_dir,'z0', f'z0{base}.tif'), field='z0', srs_epsg=epsg)
        z0.to_geotif(os.path.join(tif_dir,'count', f'count{base}.tif'), field='count', srs_epsg=epsg)
        if read_sigma:
            z0.to_geotif(os.path.join(tif_dir,'sigma_z0', f'sigma_z0{base}.tif'), field='sigma_z0', srs_epsg=epsg)
    dz_file=os.path.join(input_dir,'dz','dz'+base)
    if os.path.isfile(dz_file):
        dz=pc.grid.data().from_h5(dz_file, group='dz', fields=['dz','cell_area','count'])
        dz.dz[dz.cell_area<100]=np.nan

        dz.assign(epoch_count=np.nansum(dz.count > 0.25, axis=2))
        if mask_w is not None:
            mask_N = int(np.ceil(mask_w / (dz.x[1]-dz.x[0])))
            dz.assign(mask=snd.binary_dilation(dz.epoch_count>0, np.ones([mask_N, mask_N])).astype(float))
            dz.mask[dz.mask==0]=np.nan
        else:
            dz.mask=np.ones_like(dz.epoch_count)

        dz.assign(dzdt=dz.mask*(dz.dz[:,:,-1]-dz.dz[:,:,0])/(dz.t[-1]-dz.t[0]))
        dz.to_geotif(os.path.join(tif_dir,'dzdt', f'dzdt{base}.tif'), field='dzdt', srs_epsg=epsg)
        dz.assign(sigma_dzdt=dz.mask*np.std(np.diff(dz.dz, axis=2), axis=2)/(dz.t[1]-dz.t[0]))
        dz.to_geotif(os.path.join(tif_dir, 'sigma_dzdt',f'sigma_dzdt{base}.tif'), field='sigma_dzdt', srs_epsg=epsg)
        dz.to_geotif(os.path.join(tif_dir, 'epoch_count',f'epoch_count{base}.tif'), field='epoch_count', srs_epsg=epsg)
    dzL_file=os.path.join(input_dir,'lagrangian_dz','lagrangian_dz'+base)
    if os.path.isfile(dzL_file):
        for group in ['dzL','dzLdx','dz0_tot_dx']:
            if not os.path.isdir(os.path.join(tif_dir, group)):
                os.mkdir(os.path.join(tif_dir, group))

        dzL=pc.grid.data().from_h5(dzL_file, group='lagrangian_dz', field='dz')
        if 'mask' in z0.fields:
            dzL.dz *= z0.mask
        dzL.dz[z0.cell_area <10]=np.nan
        dzL.calc_gradient(field='dz')
        dzL.to_geotif(os.path.join(tif_dir,'dzLdx', f'dzLdx{base}.tif'), field='dz_x', srs_epsg=epsg)
        dzL.to_geotif(os.path.join(tif_dir,'dzL', f'dzL{base}.tif'), field='dz', srs_epsg=epsg)
        dzL.assign(z0_tot=z0.z0+dzL.dz)
        dzL.calc_gradient(field='z0_tot')
        dzL.to_geotif(os.path.join(tif_dir, 'dz0_tot_dx', f'dz0_tot_dx{base}.tif'), field='z0_tot_x', srs_epsg=epsg)
    return None

def main(*args):

    parser=argparse.ArgumentParser()
    parser.add_argument('dz_file', type=str, help='tile file containing dz')
    parser.add_argument('epsg', type=int, help='working EPSG')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--mask_w', type=float, help='if sepcified, only points within this distance of a datapoint will have data in the output')
    parser.add_argument('--read_sigma', action='store_true', help='if specified, the errors in z0 will be included in the output')
    parser.add_argument('--geoid_file', type=str, help='geoid file to correct elevations')

    args=parser.parse_args()

    for thefile in glob.glob(args.dz_file):
        make_tile_tifs(thefile, args.epsg, verbose=args.verbose, mask_w=args.mask_w, geoid_file=args.geoid_file, read_sigma=args.read_sigma)

if __name__=='__main__':
    main(*sys.argv)
