#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 20:07:43 2023

@author: ben
"""
import sys
import re
import os

threads_re=re.compile("THREADS=(\S+)")
n_threads="1"
for arg in sys.argv:
    try:
        n_threads=str(threads_re.search(arg).group(1))
    except Exception:
        pass

os.environ["MKL_NUM_THREADS"]=n_threads
os.environ["OPENBLAS_NUM_THREADS"]=n_threads

import matplotlib.pyplot as plt
import numpy as np
import pyproj
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

def get_residuals(DEM_file):
    '''
    Calculate the residuals between point data and a DEM

    Parameters
    ----------
    DEM_file : str
        The DEM file

    Returns
    -------
    D : pointCollection.data
        point data including DEM differences.
    '''


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
    # return the median and spread of a subset of a dataset
    return np.nanmedian(D.z[els]), pc.RDE(D.z[els])

def est_along_track_jitter(filename, url_list_file=None, expected_rms_grad=1.e-5, expected_rms_bias=2, res=500):
    """
    Estimate the along-track jitter for a DEM

    Parameters
    ----------
    filename : str
        DEM filename.
    url_list_file : str optional
        Filename for a file that gives the PGC URL for each DEM. The default is None.
    expected_rms_grad : float, optional
        Expected RMS gradient of the along-track jitter. The default is 1.e-5.
    expected_rms_bias : TYPE, optional
        Expected RMS magnitude of the along-track jitter. The default is 2.
    res : TYPE, optional
        Resolution of the along-track jitter estimates. The default is 500.

    Returns
    -------
    dict
        Dictionary containing jitter parameters.

    """
    D=get_residuals(filename)
    Gd, constraint_list, xform, this_grid, xy_atc, poly = LS.setup_DEM_jitter_fit(D, filename, url_list_file,
                                                              expected_rms_grad = expected_rms_grad,
                                                              expected_rms_bias = expected_rms_bias)
    Gc = LS.lin_op().vstack(constraint_list)
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

    data_count_AT = (Gd_full.T @ valid_data)[0:len(this_grid.ctrs[0])]
    if np.any(data_count_AT > 100):
        sigma_bias = np.nanstd(m[:-3][data_count_AT > 100])
    else:
        sigma_bias = np.NaN

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
             'data_count_AT' : data_count_AT,
             'sigma_bias' : sigma_bias,
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
                   'N_data': 1.,
                   'data_count_AT':0.1,
                   'sigma_bias':0.01}
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
