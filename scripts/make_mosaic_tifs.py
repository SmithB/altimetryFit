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
import os

def make_mosaic_tifs(input_dir, epsg, geoid_file=None):

    if isinstance(epsg,str):
        epsg=int(epsg)
    tif_dir=os.path.join(input_dir,'tif')
    if not os.path.isdir(tif_dir):
        os.mkdir(tif_dir)
    z0_file=os.path.join(input_dir, 'z0.h5')
    if os.path.isfile(z0_file):
        z0=pc.grid.data().from_h5(z0_file, group='z0', fields=['z0','cell_area','count'])
        z0.z0[z0.cell_area < 10]=np.nan
        if geoid_file is not None:
            geoid=pc.grid.data().from_geotif(geoid_file, bounds=z0.bounds())
            z0.z0 -= geoid.interp(z0.x, z0.y, gridded=True)
        z0.calc_gradient(field='z0')
        z0.to_geotif(os.path.join(tif_dir,'dz0dx.tif'), field='z0_x', srs_epsg=epsg)
        z0.to_geotif(os.path.join(tif_dir,'z0.tif'), field='z0', srs_epsg=epsg)
        z0.to_geotif(os.path.join(tif_dir,'count.tif'), field='count', srs_epsg=epsg)
    dz_file=os.path.join(input_dir,'dz.h5')
    if os.path.isfile(dz_file):
        dz=pc.grid.data().from_h5(dz_file, group='dz', fields=['dz','cell_area','count'])
        dz.dz[dz.cell_area<100]=np.nan
        dz.assign(dzdt=(dz.dz[:,:,-1]-dz.dz[:,:,0])/(dz.t[-1]-dz.t[0]))
        dz.to_geotif(os.path.join(tif_dir,'dzdt.tif'), field='dzdt', srs_epsg=epsg)
        dz.assign(sigma_dzdt=np.std(np.diff(dz.dz, axis=2), axis=2)/(dz.t[1]-dz.t[0]))
        dz.assign(epoch_count=np.nansum(dz.count > 0.25, axis=2))
        dz.to_geotif(os.path.join(tif_dir,'sigma_dzdt.tif'), field='sigma_dzdt', srs_epsg=epsg)
        dz.to_geotif(os.path.join(tif_dir,'epoch_count.tif'), field='epoch_count', srs_epsg=epsg)
    dzL_file=os.path.join(input_dir,'lag_dz.h5')
    if os.path.isfile(dzL_file):
        dzL=pc.grid.data().from_h5(dzL_file, group='lagrangian_dz', field='dz')
        if np.all(z0.shape[0:2]==dzL.shape[0:2]):
            dzL.dz[z0.cell_area <10]=np.nan
        else:
            temp=np.nan_to_num(z0.interp(dzL.x, dzL.y, field='cell_area', gridded=True))<10
            dzL.dz[temp]=np.nan
        dzL.calc_gradient(field='dz')
        dzL.to_geotif(os.path.join(tif_dir, 'dzLdx.tif'), field='dz_x', srs_epsg=epsg)
        dzL.to_geotif(os.path.join(tif_dir, 'dzL.tif'), field='dz', srs_epsg=epsg)
        if np.all( z0.shape[0:2] == dzL.shape[0:2]):
            dzL.assign( z0_tot =z0.z0 + dzL.dz )
        else:
            dzL.assign( z0_tot = z0.interp( dzL.x, dzL.y, field='z0', gridded=True) + dzL.dz)  
        dzL.calc_gradient(field='z0_tot' )
        dzL.to_geotif(os.path.join(tif_dir, 'dz0_tot_dx.tif'), field='z0_tot_x', srs_epsg=epsg)
    return None

def main(*args):
    if args is None or len(args)==0:
        args=sys.argv[1:]
    make_mosaic_tifs(*args)

if __name__=='__main__':
    main()
