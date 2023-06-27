#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 20:07:43 2023

@author: ben
"""

import matplotlib.pyplot as plt
import numpy as np
import pyproj
import re
import os
import geopandas as gpd
import pyproj
import sparseqr
import scipy.sparse as sp
from altimetryFit import get_pgc
import LSsurf as LS
import glob
import pointCollection as pc
import geopandas
import LSsurf as LS
import json

def setup_jitter_fit(D, filename, url_list_file=None, res=500,
                     expected_rms_grad=1.e-5, expected_rms_bias=2,
                     expected_plane_slope=0.02, expected_plane_bias=5):
    # read the PGC metadata
    meta=get_pgc(os.path.basename(filename), url_list_file, targets=['meta'])['meta']
    gdf=gpd.GeoDataFrame.from_features([meta])

    # select the projection based on the first y coordinate of the bounding box
    if meta['bbox'][1] < 0:
        epsg_str='epsg:3031'
    else:
        epsg_str='epsg:3413'

    gdf=gdf.set_crs('epsg:4326').to_crs(epsg_str)

    # find the geometry with the largest area
    areas=[geom.area for geom in gdf.geometry[0].geoms]
    biggest=np.argmax(areas)
    poly = np.c_[[*gdf.geometry[0].geoms[biggest].exterior.coords]]
    ctrs=[tuple(geom.centroid.coords)[0] for geom in gdf.geometry[0].geoms]
    xy0=np.array(ctrs[biggest])

    # the along and across-track vectors are the eigenvectors of the polygon boundary segment differences
    dxy=np.diff( poly[:, 0:2], axis=0 )
    vals, vecs = np.linalg.eig(dxy.T@dxy)
    # along track vector is the eigenvector aligned with local north-south
    vec_order=np.argsort(np.abs(xy0@vecs))[::-1]
    xform = {'origin': xy0, 'basis_vectors':vecs[:,vec_order]}

    xy_atc = (np.c_[D.x, D.y] - xform['origin']) @ xform['basis_vectors']

    XlR = np.round(np.array([ np.min(xy_atc[:,0]), np.max(xy_atc[:,0]) ])/res + np.array([-1, 1]))*res
    this_grid = LS.fd_grid([XlR], [res], name='x_atc', xform=xform)

    # data fit
    Gd = LS.lin_op( grid = this_grid ).interp_mtx([D.x, D.y])
    # plane fit
    G_plane = LS.lin_op( col_0=Gd.col_N, col_N = Gd.col_N+3 )
    rr=np.arange(Gd.shape[0])
    cc=np.zeros_like(rr)+Gd.col_N
    G_plane.r = np.c_[rr, rr, rr]
    G_plane.c = np.c_[cc+0, cc+1, cc+2]
    G_plane.v = np.c_[(D.x-xform['origin'][0])/1000, (D.y-xform['origin'][1])/1000, np.ones_like(rr)]
    G_plane.N_eq=Gd.shape[0]
    G_plane.ravel()
    G_plane.__update_size_and_shape__()
    Gd.add(G_plane)

    # constraint equations
    Gc_zero = LS.lin_op(col_N=Gd.col_N, name='jitter_bias_zero').data_bias(np.arange(Gd.shape[1]), col=np.arange(Gd.shape[1]), val=np.ones(Gd.shape[1]))
    Gc_zero.expected = np.zeros(Gc_zero.shape[0])+expected_rms_bias
    Gc_slope = LS.lin_op(grid=this_grid, name='jitter_bias_grad').grad()
    Gc_slope.expected = np.zeros(Gc_slope.shape[0])+expected_rms_grad
    Gc_plane = LS.lin_op(col_N=Gd.col_N, name='plane').data_bias(np.arange(3), val=np.ones(3), col=Gd.col_N+[-3, -2, -1])
    Gc_plane.expected = np.array([expected_plane_slope, expected_plane_slope, expected_plane_bias])
    Gc = LS.lin_op().vstack([Gc_zero, Gc_slope, Gc_plane])
    return Gd, Gc, xform, this_grid, xy_atc, poly

def get_residuals(DEM_file):
    meta_file = DEM_file.replace('_dem_filt.tif','_shift_est.h5')
    print([DEM_file, meta_file])
    try:
        D=pc.data().from_h5(meta_file, group='data')
    except Exception:
        thecmd=f'register_WV_DEM_with_IS2.py --DEM_file {DEM_file} --GeoIndex_wc "/Volumes/ice1/ben/ATL06/tiles/Antarctic/005/c*/GeoIndex.h5" -m /home/ben/git_repos/ATL1415/masks/Antarctic/Greene_22_shelf_plus_10m_mask_240m.tif --DEBUG'
        os.system(thecmd)
        D=pc.data().from_h5(meta_file, group='data')
    return D


def med_spread(D, els):
    return np.nanmedian(D.z[els]), pc.RDE(D.z[els])

def est_along_track_jitter(filename, url_list_file=None, expected_rms_grad=1.e-5, expected_rms_bias=2, res=500):

    D=get_residuals(filename)
    Gd, Gc, xform, this_grid, xy_atc, poly = setup_jitter_fit(D, filename, url_list_file,
                                                              expected_rms_grad = expected_rms_grad,
                                                              expected_rms_bias = expected_rms_bias)
    G_full=LS.lin_op().vstack([Gd, Gc])
    Gd_full = Gd.toCSR()
    valid_data = np.ones(D.x.size, dtype=bool)

    sigma_extra=0
    for count in range(3):
        N_data_rows = G_full.shape[0]-np.sum(valid_data==0)
        TCinv = sp.diags(np.concatenate([1/D.sigma[valid_data], 1/Gc.expected]),  \
                                        shape=[ N_data_rows, N_data_rows ], offsets=0)
        keep=np.ones(G_full.shape[0], dtype=bool)
        keep[0:len(valid_data)] = valid_data

        m=sparseqr.solve(TCinv @ G_full.toCSR()[keep,:], TCinv @ np.concatenate([D.r0[valid_data], np.zeros(Gc.shape[0])]))

        b_est=Gd_full @ m
        sigma_extra = LS.calc_sigma_extra( D.r0 - b_est, D.sigma, valid_data )
        valid_data = np.abs(D.r0-b_est)  < 3 * np.sqrt( D.sigma**2 + sigma_extra**2 )

    return  {'R0' : pc.RDE(D.r0[valid_data]),
             'poly' : poly,
             'sigma_uncorr':np.std(D.r0[valid_data]),
             'R' : pc.RDE((D.r0-b_est)[valid_data]),
             'sigma_corr' : np.std((D.r0-b_est)[valid_data]),
             'bias' : m[:-3],
             'x_bias' : this_grid.ctrs[0],
             'tilt_model' : m[-3:],
             'data' : D,
             'valid': valid_data,
             'xy_atc' : xy_atc,
             'bias_est' : b_est,
             'N_data': np.sum(valid_data),
             'xform' : xform}

def main():
    import argparse
    parser=argparse.ArgumentParser(description='Calculate estimate the along-track bias variability in a DEM.')
    parser.add_argument('filename', type=str, help="DEM filename")
    parser.add_argument('--url_list_file','-u', type=str, help='File containing the PGC url for each DEM filename')
    parser.add_argument('--expected_rms_grad', type=float, default=1.e-5, help='Expected RMS gradient of the bias')
    parser.add_argument('--expected_rms_bias', type=float, default=2, help='Expected RMS of the bias')
    parser.add_argument('--res', type=float, default=500, help='resolution of the biases')
    parser.add_argument('--output_dir', default=None, help="directory in which to save the output")
    parser.add_argument('--output_extension', default='_AT_bias_est', help="text to add to the end of the output filename")
    args=parser.parse_args()

    out_file=args.filename.replace('.tif',args.output_extension+'.json')
    if args.output_dir is not None:
        out_file = os.path.join(args.output_dir, os.path.basename(out_file))

    S=est_along_track_jitter(args.filename, args.url_list_file,
                               expected_rms_grad=args.expected_rms_grad,
                               expected_rms_bias=args.expected_rms_bias,
                               res=args.res)

    out_precision={'R0': 1.e-3,
                   'sigma_uncorr': 1.e-4,
                   'sigma_corr': 1.e-4,
                   'bias': 1.e-3,
                   'x_bias': 1.e-3,
                   'tilt_model': 1.e-6,
                   'N_data': 1.}
    out={}
    for field, precision in out_precision.items():
        temp = np.round(S[field]/precision)*precision
        if isinstance(temp, np.ndarray):
            out[field]=list(temp)
        else:
            out[field]=temp

    out['xform']={'origin':list(S['xform']['origin']),
                  'basis_vectors':[list(item) for item in S['xform']['basis_vectors']]}

    with open(out_file,'w') as fh:
        json.dump(out, fh, indent=4)

if __name__=="__main__":
    main()
