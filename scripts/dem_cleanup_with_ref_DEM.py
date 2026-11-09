#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:56:33 2024

@author: ben
"""

import pointCollection as pc
import numpy as np
import matplotlib.pyplot as plt # comment for production
import argparse
import sys
import scipy.ndimage as snd
from osgeo import gdal
import os
import shutil



def parse_input_args(args):
    '''
    parse_input_args: transform input argument string into a dataspace

    Parameters
    ----------
    args : iterable
        Input arguments.  Keywords should have two hyphens at the start

    Returns
    -------
    dataspace
        input arguments formatted as a namespace

    '''

    parser = argparse.ArgumentParser(description='cull out spurious values from a filtered DEM', \
                                     fromfile_prefix_chars='@')
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('--smooth_scale','-s', type=float, default=320.)
    parser.add_argument('--ref_dem', type=str, required=True)
    parser.add_argument('--ref_dem_tol', type=float, default=20)
    parser.add_argument('--ref_dem_erode', type=float, default=50)
    parser.add_argument('--ref_dem_coarse_tol', type=float)
    parser.add_argument('--max_dem_offset', type=float, default=20)
    args=parser.parse_args()

    return args

def dem_cleanup_with_ref_DEM(*args, **kwargs):

    if isinstance(args[0], argparse.Namespace):
        args=args[0]
    elif kwargs is not None:
        for key, value in kwargs.items():
            if not key.startswith('-'):
                key='-'+key
            args += [key, str(value)]
        args=parse_input_args(args)

    if args.ref_dem_coarse_tol is None:
        args.ref_dem_coarse_tol=3*args.ref_dem_tol

    DEM=pc.grid.data().from_geotif(args.input_file)
    # the first band is cleaned up; a single-band DEM has a 2-D z
    multiband = DEM.z.ndim == 3
    z_first = DEM.z[:,:,0] if multiband else DEM.z
    dx=(DEM.x[1]-DEM.x[0])
    dec=1
    if args.smooth_scale > 16*dx:
        dec = int(np.maximum(1, np.floor(args.smooth_scale/(4*dx))))

    ref_DEM=pc.grid.data().from_geotif(args.ref_dem, bounds=DEM.bounds(pad=2*args.smooth_scale))

    DEM.assign(z_ref=ref_DEM.interp(DEM.x, DEM.y, gridded=True))
    if dec == 1:
        DEM_lowres=(DEM[:,:,0] if multiband else DEM).copy()
    else:
        DEM_lowres=DEM[::dec, ::dec, 0] if multiband else DEM[::dec, ::dec]
    delta=DEM_lowres.z-DEM_lowres.z_ref

    DEM_offset=np.nanmedian(delta)

    DEM_offset = np.maximum(-args.max_dem_offset,
                            np.minimum(args.max_dem_offset, DEM_offset))
    delta -= DEM_offset
    delta[np.abs(delta) > args.ref_dem_coarse_tol]=np.nan
    smooth_sigma = args.smooth_scale / (DEM_lowres.x[1]-DEM_lowres.x[0])
    delta_smoothed=pc.grid.smooth_corrected(delta, np.isfinite(delta), smooth_sigma)[0]
    DEM_lowres.assign(z_filled = DEM_lowres.z_ref + delta_smoothed + DEM_offset)

    if dec > 1:
        delta_filled = z_first - DEM_lowres.interp(DEM.x, DEM.y, field='z_filled', gridded=True)
    else:
        delta_filled = z_first - DEM_lowres.z_filled

    keep = (np.abs(delta_filled) < args.ref_dem_tol) & np.isfinite(delta_filled)
    N_erode = int(np.ceil(args.ref_dem_erode/dx))
    keep = snd.binary_erosion(
        snd.binary_erosion(keep, np.ones((N_erode, 1)), border_value=1),
                           np.ones((1,N_erode)), border_value=1)
    z1=z_first.copy()
    z1[keep==0]=np.nan
    if multiband:
        DEM.z[:,:,0]=z1
    else:
        DEM.z=z1

    if not os.path.isfile(args.output_file):
        shutil.copy(args.input_file, args.output_file)
    out_ds = gdal.Open(args.output_file, gdal.GA_Update)
    z_bands = DEM.z if multiband else DEM.z[:,:,None]
    for band in range(z_bands.shape[2]):
        out_ds.GetRasterBand(band+1).WriteArray(z_bands[::-1,:,band])
    out_ds=None

def main(args=None):
    if args is None:
        args=sys.argv
    dem_cleanup_with_ref_DEM(args)

if __name__=='__main__':
    main()
