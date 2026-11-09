#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:51:37 2026

@author: ben

Purpose: for z0, a reference surface contained in a tif (or a directory of 
tifs), dz, a set of surface temporal variations contained in a set of hdf5 files, 
and a SMB-correction file, calculate a DEM for a specific time.

The DEM is equal to z0(x, y) + dz(t0, x, y) + FAC(t0, x, y) - FAC0_scale * FAC0(x, y) + SMB_a(t0, x, y)
z0 comes from the z0 file (or a mosaic from a directory)
dz comes from a mosaic of dz files
FAC, SMB_a come from the SMB file
FAC0 is the FAC at the first time slice in the SMB file.

If geoid_file is specified, the geoid height is subtracted from the DEM, so
that the output is relative to the geoid rather than the ellipsoid.

There are two patterns for invoking this script: one is to specify an input z0 file,
the other is to specify a z0 directory and the bounds for the output.

Example arguments: 
t0 = 2020.0
z0_file = /home/ben/DEM_regions/AA_ref/fits/800m_100m_qyr_Eul_grounded/2019_2023/200km_tiles/tif/z0_filled/z0-1800_-1600_800_1000.tif
dz_dir = /home/ben/DEM_regions/AA_ref/fits/800m_100m_qyr_Eul_grounded/2019_2023/200km_tiles/dz
SMB_file  = /Volumes/ice1/ben/MERRA2_hybrid/v1.2.1/gsfc_fdm_v1_2_1_ais_Jul24_extrap.nc
FAC0_scale = 1.0
out_file = /home/ben/DEM_regions/AA_ref/fits/800m_100m_qyr_Eul_grounded/2019_2023/200km_tiles/tif/zsurf_filled/z0-1800_-1600_800_1000_t2020.0.tif



"""
from argparse import ArgumentParser
import pointCollection as pc
from altimetryFit.surf_for_time import surf_for_time
import glob
import os

def main():
    
    parser = ArgumentParser('Make an elevation estimate for a location')
    parser.add_argument('t0', type=float, help='reference time for output DEM, required')
    parser.add_argument('out_file', type=str, help='output filename')
    parser.add_argument('--z0_file', type=str, help='single z0 file')
    parser.add_argument('--bounds', type=str, help='4-element comma-separated, xmin,xmax,ymin,ymax, can be skipped if z0_file is specified')
    parser.add_argument('--SMB_file', type=str, help='file containing SMB and FAC corrections')
    parser.add_argument('--EPSG', type=int, required=True, help='EPSG for output (should be the same as the input)')
    parser.add_argument('--z0_dir', type=str, help='directory to search for z0 files')
    parser.add_argument('--dz_dir', type=str, help='directory containing dz tiles')
    parser.add_argument('--z0_format', type=str, default='tif')
    parser.add_argument('--dz_format', type=str, default='h5')
    parser.add_argument('--z0_group', type=str, help='group to read from the z0 file (if hdf5)')
    parser.add_argument('--z0_field', type=str, help='field to read from the z0 file (if hdf5)')
    parser.add_argument('--dz_group', type=str, default='dz', help='group to read from the dz file (if hdf5)')
    parser.add_argument('--dz_field', type=str, default='dz', help='field to read from the dz file (if hdf5)')
    parser.add_argument('--FAC0_scale', type=float, default=1., help='specify 0 for older fits that did not subtract the time-zero FAC')    
    parser.add_argument('--corrections', type=str, default='FAC,SMB,dz', help='comma-separated list of corrections to apply to z0, any of FAC, SMB, dz')
    parser.add_argument('--geoid_file', type=str, help='geoid geotif; if specified, the geoid height is subtracted from the output')
    args=parser.parse_args()

    if args.z0_file is not None:
        z0_files = [args.z0_file]
    elif args.z0_dir is not None:
        z0_files = glob.glob(os.path.join(args.z0_dir, f'*.{args.z0_format}'))
    else:
        parser.error('one of --z0_file or --z0_dir must be specified')
    if len(z0_files)==0:
        parser.error(f'no z0 files found in {args.z0_dir}')

    if args.bounds is not None:
        XR_YR = [float(val) for val in args.bounds.split(',')]
        if len(XR_YR) != 4:
            parser.error('--bounds must have four comma-separated values: xmin,xmax,ymin,ymax')
        bounds = [XR_YR[0:2], XR_YR[2:4]]
    elif args.z0_file is not None:
        z0_group = args.z0_group if args.z0_group is not None else 'z0'
        bounds = pc.grid.data().from_file(args.z0_file, group=z0_group, meta_only=True).bounds()
    else:
        parser.error('--bounds must be specified if --z0_file is not')

    if args.dz_dir is not None:
        dz_files = glob.glob(os.path.join(args.dz_dir, f'*.{args.dz_format}'))
    else:
        dz_files= None

    corrections = [item.strip() for item in args.corrections.split(',')]

    out_dir = os.path.dirname(args.out_file)
    if len(out_dir) > 0:
        os.makedirs(out_dir, exist_ok=True)

    z = surf_for_time(bounds, args.t0,
                    SMB_file = args.SMB_file,
                    dz_files = dz_files,
                    z0_files = z0_files,
                    z0_field = args.z0_field,
                    z0_group = args.z0_group,
                    dz_field = args.dz_field,
                    dz_group = args.dz_group,
                    FAC0_scale = args.FAC0_scale,
                    corrections = corrections)

    if args.geoid_file is not None:
        geoid = pc.grid.data().from_geotif(args.geoid_file, bounds=z.bounds(pad=1.e4))
        z.z -= geoid.interp(z.x, z.y, gridded=True)

    z.to_geotif(args.out_file, srs_epsg=args.EPSG)

if __name__=='__main__':
    main()
 
    