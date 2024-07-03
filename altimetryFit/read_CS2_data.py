#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 14:22:56 2019

@author: ben
"""

import pointCollection as pc
import numpy as np
import json
import h5py
from pathlib import Path
import os
import glob
from scipy.interpolate import RectBivariateSpline

def radar_key():
    return {'CS2_POCA':-2,'CS2_swath':-1}

def interp_sigma(D, DEM, bias_sigma_file):
    # read the grids from the file
    with h5py.File(bias_sigma_file,'r') as h5f:
        p_ctrs=np.array(h5f['log10_power'])
        log_s_ctrs=np.array(h5f['log10_slope'])
        bias_grid=np.array(h5f['bias_sm'])
        spread_grid=np.array(h5f['spread_sm'])

    # interpolate the slope from the DEM [must have a slope_mag field]
    DEM_slope_mag=DEM.interp(D.x, D.y, field='slope_mag')
    DEM_slope_mag[DEM_slope_mag < 10**(log_s_ctrs[1]+0.001)] = 10**(log_s_ctrs[1]+0.001)
    l_p=np.log10(D.power)

    # make the interpolators
    valid_grid=(np.isfinite(bias_grid) & np.isfinite(spread_grid))
    valid_interpolator = RectBivariateSpline(p_ctrs, log_s_ctrs,
                        valid_grid.astype(float),
                        kx=1, ky=1)
    temp=bias_grid.copy()
    temp[valid_grid==0]=0
    bias_interpolator=RectBivariateSpline(p_ctrs, log_s_ctrs,  temp, kx=1, ky=1)

    temp=spread_grid.copy()
    temp[valid_grid==0]=0
    spread_interpolator=RectBivariateSpline(p_ctrs, log_s_ctrs,  temp, kx=1, ky=1)
    # interpolate the biases
    D.assign({'sigma':spread_interpolator.ev(l_p, np.log10(DEM_slope_mag)),
            'bias_est':bias_interpolator.ev(l_p, np.log10(DEM_slope_mag))})
    # subtract the bias estimate from the surface height
    D.h -= D.bias_est
    #print(f'mean finite bias={np.mean(np.isfinite(D.bias_est))}\n mean finite sigma={np.mean(np.isfinite(D.sigma))}')
    D.index(valid_interpolator.ev(l_p, np.log10(DEM_slope_mag))==1)
    D.index(np.isfinite(D.h)& np.isfinite(D.sigma))

    #print(f'mean valid bias={D.size/s0}')

def read_swath_data(xy0, W, index_files, apply_filters=True, DEM=None,
                    bias_lookup_dir=None):
    fields = ['x','y','time','h', 'power','coherence','block_count','block_h_spread']

    if isinstance(index_files, str):
        index_files=glob.glob(index_files)

    D=[]

    for index_file in index_files:
        try:
            D += pc.geoIndex().from_file(index_file).query_xy_box(xy0[0]+W['x']*np.array([-1/2, 1/2]), \
                   xy0[1]+W['y']*np.array([-1/2, 1/2]), fields=fields)
        except TypeError:
            pass

    if len(D) > 0:
        D=pc.data().from_list(D)
    else:
        return
    if D.size==0:
        return D

    D.index(D.power>0.)

    D.assign({'swath':np.ones_like(D.x, dtype=bool)})
    if bias_lookup_dir is not None:
        # note that the correlation analysis gives a swath bias of 0.25.
        # This results in a large number of edited swath measurements,
        # so a better value seems to be 0.5 m.  Further testing needed
        interp_sigma(D, DEM, os.path.join(bias_lookup_dir, 'CS_AA_swath_PS_correction.h5'))
        D.assign({'sigma_corr':np.zeros_like(D.x)+0.25})
    else:
        D.assign({'sigma':np.minimum(5, np.maximum(1, 0.95 -.4816*(np.log10(D.power)+14.3) +\
                                       1.12*np.sqrt(D.block_h_spread)))})

    if apply_filters:
        if np.any(D.block_count>1):
            D.index( (D.power > 1e-17) & (D.power < 1e-13) & \
                    (D.block_count > 3) & (D.block_h_spread < 15))
        else:
            D.index( (D.power > 1e-17) & (D.power < 1e-13) )
    return D

def read_poca_data(xy0, W, index_files, apply_filters=True, DEM=None, baseline_num=None, bias_lookup_dir=None):
    fields=['x','y','time','h', 'power','coherence']

    D=[]
    if isinstance(index_files, str):
        index_files=glob.glob(index_files)
    for index_file in index_files:
        try:
            D += pc.geoIndex().from_file(index_file).query_xy_box(xy0[0]+W['x']*np.array([-1/2, 1/2]), \
                   xy0[1]+W['y']*np.array([-1/2, 1/2]), fields=fields)
        except TypeError:
            pass

    if len(D) > 0:
        D=pc.data().from_list(D)
    else:
        return
    if D.size==0:
        return D

    D.index(D.power>0.)

    if DEM is not None:
        if bias_lookup_dir is not None:
            interp_sigma(D, DEM, os.path.join(bias_lookup_dir, 'CS_AA_POCA_PS_correction.h5'))
        else:
            DEM_slope_mag=DEM.interp(D.x, D.y, field='slope_mag')
            D.assign({'sigma':np.maximum(0.5, 50*DEM_slope_mag + np.maximum(0, -0.64*(np.log10(D.power)+14)))})
    else:
        # if no DEM is specified, use a typical value of 0.01 for the slope
        D.assign({'sigma':50*0.01+ np.maximum(0, -0.64*(np.log10(D.power)+14))})

    if apply_filters:
        try:
            D.index((D.power > 5e-17)  & (D.power < 5e-13))
        except Exception:
            D.index((D.power > 5e-17)  & (D.power < 5e-13))

    # see compare_CS_IS2_POCA.ipynb
    D.assign({'sigma_corr':0.13+np.zeros_like(D.x),
              'swath':np.zeros_like(D.x, dtype='bool')})
    return D

def read_CS2_data(xy0, W, index_files, apply_filters=True, DEM_file=None,
                  dem_tol=None, bias_lookup_dir=None, sensor_dict=None,
                  POCA_sensor=0):
    if isinstance(W, (int, float)):
        W={'x':W,'y':W}

    if DEM_file is not None:
        DEM=pc.grid.data().from_geotif(DEM_file, bounds=[xy0[0]+np.array([-W['x']/2-1.e4, W['x']/2+1.e4]), \
                                    xy0[1]+np.array([-W['y']/2-1.e4, W['y']/2+1.e4])])
        gx, gy = np.gradient(DEM.z, DEM.y, DEM.x)
        DEM.assign({'slope_mag':np.abs(gx+1j*gy)})
    else:
        DEM=None

    D=[]
    D += [read_poca_data( xy0, W, index_files['POCA'], apply_filters=apply_filters, DEM=DEM, bias_lookup_dir=bias_lookup_dir)]
    D += [read_swath_data(xy0, W, index_files['swath'], apply_filters=apply_filters, DEM=DEM, bias_lookup_dir=bias_lookup_dir)]
    # eliminate Nonetype objects within D
    D = [Di for Di in D if Di is not None]
    
    if DEM_file is not None:
        for Di in D:
            Di.assign(DEM=DEM.interp(Di.x, Di.y))
        if dem_tol is not None:
            for Di in D:
                dh_DEM = Di.h-Di.DEM
                Di.index(np.abs(dh_DEM-np.nanmedian(dh_DEM)) < dem_tol)

    for Di in D:
        if not 'abs_orbit' in Di.fields:
            Di.assign(abs_orbit=np.floor((Di.time-2010)*365.25*16))

    if apply_filters:
        try:
            with open(os.path.join(Path(__file__).parent.absolute(), \
                         'CS2_known_bad_orbits.json'), 'r') as fh:
                bad_orbits=json.load(fh)[1]#['bad orbits']
            for Di in D:
                Di.index( ~np.in1d(Di.abs_orbit.astype(int), np.array(bad_orbits, dtype='int')))
        except FileNotFoundError:
            print("\t\tno bad-orbits file found")

    if sensor_dict is None or len(sensor_dict)==0:
        sensor_dict=radar_key()
    for Di in D:
        Di.assign(sensor=sensor_dict['CS2_POCA']+Di.swath, z=Di.h)

    return D
