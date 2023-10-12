#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 17:12:48 2019

@author: ben

Revision history [Authoritative history is in the git history]
C/A 2021: Code developed by smithB, under the fit_OIB_aug.py filename
Early 2022: Renamed to fit_altimetry.py by smithB
Mid 2022: Tsutterley added lagrangian code to fit_OIB_aug.py
April 2023: SmithB merged lagrangian code into fit_altimetry.py
"""

import resource
import os
# make sure we're only using one thread
os.environ['MKL_NUM_THREADS']="1"
os.environ['NUMEXPR_NUM_THREADS']="1"
os.environ['OMP_NUM_THREADS']="1"
os.environ['OPENBLAS_NUM_THREADS']="1"


#import warnings
#warnings.simplefilter('error')

import numpy as np
#import matplotlib.pyplot as plt
from LSsurf.smooth_fit import smooth_fit
from LSsurf import fd_grid
from altimetryFit.reread_data_from_fits import reread_data_from_fits
import pointCollection as pc
from pyTMD import compute_tide_corrections
from SMBcorr import assign_firn_variable
from altimetryFit.read_optical import read_optical_data, laser_key
from CS2_fit.read_CS2_data import read_CS2_data
import pointAdvection
import h5py
import sys
import glob
import json
import re
import xarray as xr

def set_memory_limit(max_bytes):
    '''
    Limit the amount of memory the program can use.
    '''
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, hard))


def make_sensor_dict(h5_file):
    '''
    Make a dictionary matching sensor number to bias id and source.

    Input: h5f: h5py file containing the fit output
    Output: dict giving the sensor for each number
    '''
    this_sensor_dict=dict()
    with h5py.File(h5_file,'r') as h5f:
        if '/meta/sensors/' not in h5f:
            return this_sensor_dict

        sensor_re=re.compile('sensor_(\d+)')
        for sensor_key, sensor in h5f['/meta/sensors/'].attrs.items():
            sensor_num=int(sensor_re.search(sensor_key).group(1))
            this_sensor_dict[sensor_num]=sensor
    return this_sensor_dict


def get_SRS_proj4(hemisphere):
    if hemisphere==1:
        return '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
    else:
        return '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'

def custom_edits(data):
    '''
    Any custom edits go here

    Parameters
    ----------
    data : pc.data
        data structure

    Returns
    -------
    None.

    '''
    # day 731891 has bad ICESat elevations
    bad=data.day==731891
   # bad ICESat data for 2003.782
    bad=np.abs(data.time - 2003.7821) < 0.1/24/365.25
    data.index(bad==0)

def apply_tides(D, xy0, W, tide_mask_file, tide_directory, tide_model, EPSG=3031):
    # read in the tide mask (for Antarctica) and apply dac and tide to ice-shelf elements
    # the tide mask should be 1 for non-grounded points (ice shelves?), zero otherwise
    tide_mask = pc.grid.data().from_geotif(tide_mask_file, bounds=[np.array([-0.6, 0.6])*W+xy0[0], np.array([-0.6, 0.6])*W+xy0[1]])
    is_els=tide_mask.interp(D.x, D.y) > 0.5
    D.assign(floating=is_els)
    print(f"\t\t{np.mean(is_els)*100}% shelf data")
    D.assign({'tide_ocean':np.zeros_like(D.x)})
    if np.any(is_els.ravel()):
        # N.B.  removed the half-day offset in the tide correction.
        # Old version read:  (D.time-(2018+0.5/365.25))*24*3600*365.25
        # updates have made time values equivalent to years- Y2K +2000,
        # consistent between IS1 and IS2 (2/18/2022)
        D.tide_ocean = compute_tide_corrections(\
                D.x, D.y, (D.time-2000)*24*3600*365.25,
                DIRECTORY=tide_directory, MODEL=tide_model,
                EPOCH=(2000,1,1,0,0,0), TYPE='drift', TIME='utc', EPSG=EPSG)
    D.tide_ocean[is_els==0] = 0
    #D.dac[is_els==0] = 0
    D.tide_ocean[~np.isfinite(D.tide_ocean)] = 0
    #D.dac[~np.isfinite(D.dac)] = 0
    D.z -= D.tide_ocean
    return D

def mask_data_by_year(data, mask_dir):
    masks={}
    year_re=re.compile('(\d\d\d\d.+\d+).tif')
    for file in glob.glob(mask_dir+'/*.tif'):
        m=year_re.search(file)
        if m is not None:
            masks[float(m.group(1))]=pc.grid.data().from_geotif(file)
    years=np.sort(np.array(list(masks.keys())))
    years_ind=np.searchsorted(years, data.time)
    good=np.ones(data.shape, dtype=bool)
    for uy in np.unique(years_ind):
        if uy ==0:
            continue
        this_mask=masks[years[uy-1]]
        these_pts=np.flatnonzero(years_ind==uy)
        temp=this_mask.interp(data.x[these_pts], data.y[these_pts])
        good[these_pts[(temp<0.5) & np.isfinite(temp)]] = 0
    data.index(good)

def update_mask_data_andor(mask_data, and_mask_files, or_mask_files):
    if and_mask_files is not None:
        for mask_file in and_mask_files:
            mask_i = pc.grid.data().from_file(mask_file).interp(mask_data.x, mask_data.y, gridded=True)
            these=np.isfinite(mask_i)
            if mask_data.z.ndim==2:
                mask_data.z[these] = bool(mask_data.z[these]) & (mask_i[these]>0.5)
            else:
                for ind in range(mask_data.shape[2]):
                    temp=mask_data.z[:,:,ind]
                    temp[these] = temp[these].astype(bool) & (mask_i[these]>0.5)
                    mask_data.z[:,:,ind]=temp
    if or_mask_files is not None:
        for mask_file in or_mask_files:
            mask_i = pc.grid.data().from_file(mask_file).interp(mask_data.x, mask_data.y, gridded=True)
            these=np.isfinite(mask_i)
            if mask_data.z.ndim==2:
                mask_data.z[these] = mask_data[these].astype(bool) | (mask_i[these] > 0.5)
            else:
                for ind in range(mask_data.shape[2]):
                    temp=mask_data.z[:,:,ind]
                    temp[these] = temp[these].astype(bool) | (mask_i[these] > 0.5)
                    mask_data.z[:,:,ind]=temp

def setup_lagrangian(velocity_files=None, lagrangian_epoch=None, reference_epoch=None,
    SRS_proj4=None, xy0=None, Wxy=None, t_span=None, spacing=None, verbose=False,
    **kwargs):
    SPY=365.25*24*3600
    # verbose output of lagrangian parameters
    if verbose:
        print(f'Velocity File(s): {",".join(velocity_files)}')
        print(f'Advect parcels to {lagrangian_epoch:0.1f}')

    # make the ouput arguments dictionary
    out_args=kwargs
    out_args.update({
            'lagrangian_epoch':lagrangian_epoch,
            'SRS_proj4':SRS_proj4,
            'xy0':xy0,
            'Wxy':Wxy})
    if lagrangian_epoch is None:
        # calculate the lagrangian epoch based on the reference time
        lagrangian_epoch = fd_grid([t_span], [spacing['dt']], name='t').ctrs[0][reference_epoch]
        print(f"lagrangian epoch is {lagrangian_epoch}")

    bounds=[xyi + np.array([-0.5, 0.5])*Wxy for xyi in xy0]
    # create arrays of grid coordinates
    x = fd_grid([bounds[0]], [spacing['dz']], name='x').ctrs[0]
    y = fd_grid([bounds[1]], [spacing['dz']], name='y').ctrs[0]
    if 'xy01_grids' in velocity_files[0]:
        interpolator_save_file=velocity_files[0]
    else:
        interpolator_save_file=os.path.splitext(velocity_files[0])[0] +'_xy01_grids.h5'
    if os.path.isfile(interpolator_save_file):
        xy0_obj=pc.grid.data().from_h5(interpolator_save_file,group='xy0')
        xy1_obj=pc.grid.data().from_h5(interpolator_save_file,group='xy1')
        gridy, gridx, gridt = np.meshgrid(y, x,
                                          (np.array(t_span)-lagrangian_epoch)*SPY, indexing='ij')
        x1=xy1_obj.interp(gridx.ravel(), gridy.ravel(), gridt.ravel(), field='x1')
        y1=xy1_obj.interp(gridx.ravel(), gridy.ravel(), gridt.ravel(), field='y1')
        x1 = np.append(x1, bounds[0])
        y1 = np.append(y1, bounds[1])
        bounds_full=[[np.nanmin(x1), np.nanmax(x1)],[np.nanmin(y1), np.nanmax(y1)]]
        out_args.update({'xy1_obj': xy1_obj,
            'xy0_obj': xy0_obj,
            'bds':bounds_full})
        return out_args
    else:
        xy0_obj, xy1_obj = None, None

    gridx,gridy = np.meshgrid(x, y)
    # create advection object with buffer around grid
    adv = pointAdvection.advection(
        x=gridx.flatten(),
        y=gridy.flatten(),
        t=np.zeros((x.N_nodes*y.N_nodes))
    )
    # read velocity image and trim to a buffer extent around points
    # use a wide buffer to encapsulate advections in fast moving areas
    if isinstance(velocity_files, str):
        velocity_files=[velocity_files]
    if len(velocity_files) == 1:

        if 'NSIDC-0731' in os.path.basename(velocity_files[0]):
            with xr.open_dataset(velocity_files[0]) as ds:
                t_vel=np.array(
                    ds.time.data-np.datetime64('2000-01-01'),
                    dtype='timedelta64[s]').astype(float)/SPY + 2000
                if t_span is not None:
                    t_ind = np.flatnonzero(
                        (t_vel >= (t_span[0] - 1))
                        & (t_vel <= (t_span[1] + 1)))
                else:
                    t_ind = np.flatnonzero(np.ones_like(t_vel))
                adv.from_dict({'x':np.array(ds.x),
                               'y':np.array(ds.y)[::-1],
                               'U':np.array(ds.VelocitySeries[t_ind, 0, ::-1,:]),
                               'V':np.array(ds.VelocitySeries[t_ind, 1, ::-1,:]),
                               'eU':np.array(ds.VelocitySeries[t_ind, 3, ::-1,:]),
                               'eV':np.array(ds.VelocitySeries[t_ind, 4, ::-1,:]),
                               'time': t_vel[t_ind]
                               }, t_axis=0)
                lagrangian_interpolation = 'linear'
        else:
            adv.from_nc(velocity_files[0], buffer=Wxy)
            lagrangian_interpolation = 'linear'
    else:
        vlist = [pc.grid.data().from_nc(v,field_mapping=dict(U='VX', V='VY')) \
            for v in velocity_files]
        adv.from_list(vlist, buffer=Wxy)
        lagrangian_interpolation = 'linear'

    # convert velocity times to delta times from epoch
    adv.velocity.time = (adv.velocity.time - lagrangian_epoch)*SPY
    adv.fill_velocity_gaps()

    # create an xy1 interpolator to find the locations that might have contributed
    # points to the dataset
    xy1_obj = adv.xy1_interpolator(bounds=bounds,
                                   t_range=(np.array(t_span)-lagrangian_epoch)*SPY,
                                   t_step=0.25*SPY)
    bounds_full = [[np.nanmin(ii.values), np.nanmax(ii.values)] for ii in\
                           [xy1_obj.x, xy1_obj.y]]

    # create an xy0 interpolator to find final locations for each point
    xy0_obj =adv.xy0_interpolator(bounds=bounds_full,
                                   t_range=(np.array(t_span)-lagrangian_epoch)*SPY,
                                   t_step=0.25*SPY)

    # make the ouput arguments dictionary
    out_args=kwargs
    out_args.update({'advection_obj':adv,
            'method':lagrangian_interpolation,
            'lagrangian_epoch':lagrangian_epoch,
            'SRS_proj4':SRS_proj4,
            'xy0':xy0,
            'Wxy':Wxy,
            'xy1_obj': xy1_obj,
            'xy0_obj': xy0_obj,
            'bds':bounds_full})
    return out_args

def apply_lagrangian_masks(data, x, y, mask_files=None, **kwargs):
    for mask_file in mask_files:
        mask=pc.grid.data().from_geotif(mask_file, bounds=kwargs['xy0_obj'].bounds())
        if mask is None or np.any(mask.shape==0):
            continue
        year_mask = pc.grid.DEM_year(mask_file)
        t_mask=( year_mask - kwargs['lagrangian_epoch'])*24*3600*365.25
        # if we have moved the points, the location of the points at the time of the mask is
        # found by interpolating into the xy1 object
        xx = kwargs['xy1_obj'].interp(x, y, t_mask, field='x1')
        yy = kwargs['xy1_obj'].interp(x, y, t_mask, field='y1')
        mask_i = mask.interp(xx, yy)
        keep =  (~np.isfinite(mask_i)) | ( mask_i < 0.1 ) | (data.time < year_mask)
        data.index(keep)


def update_data_for_lagrangian(data, lagrangian_ref_dem=None, mask_data=None, **kwargs):
    # default keyword arguments
    kwargs.setdefault('advection_obj', pointAdvection.advection())
    kwargs.setdefault('lagrangian_interpolation', 'linear')
    kwargs.setdefault('lagrangian_epoch', 0)
    kwargs.setdefault('SRS_proj4', None)
    kwargs.setdefault('xy0', [None,None])
    kwargs.setdefault('Wxy', 0)

    # get latitude and longitude of the original data
    data.get_latlon(proj4_string=kwargs['SRS_proj4'])

    if lagrangian_ref_dem is not None:
        print(f'Subtracting reference dem from {lagrangian_ref_dem}')
        data.assign(z0=data.z.copy())
        data.z -= pc.grid.data().from_geotif(lagrangian_ref_dem,
            bounds=data.bounds()).interp(data.x, data.y)
        data.index(np.isfinite(data.z))

    # update advection object with original coordinates and times
    #adv = kwargs['advection_obj']
    #adv.x = data.x.copy()
    #adv.y = data.y.copy()
    t_lag = (data.time - kwargs['lagrangian_epoch'])*24*3600*365.25
    #adv.t = t_lag
    # calculate the number of seconds between data times and epoch
    # advect points to delta time 0
    #adv.translate_parcel(integrator='RK4', method=kwargs['lagrangian_interpolation'], t0=0)
    # xy0_interpolator(bounds=None, t_range=None, t_step=None)

    x0 = kwargs['xy0_obj'].interp(data.x, data.y, t_lag, field='x0')
    y0 = kwargs['xy0_obj'].interp(data.x, data.y, t_lag, field='y0')

    if 'move_points' in kwargs and kwargs['move_points']:
        # use the mask data to remove data points that were measured at invalid locations
        if mask_data is not None:
            if mask_data.z.ndim==2:
                temp=mask_data.interp(data.x, data.y)>0.5
            else:
                temp=mask_data.interp(data.x, data.y, data.time)>0.5
            data.index(temp)
            x0=x0[temp]
            y0=y0[temp]
        # save the original coordinates:
        data.assign(x_original=data.x.copy(), y_original=data.y.copy())
        # replace x and y with the advected coordinates
        data.x=np.copy(x0); data.y=np.copy(y0)
        # reindex to coordinates that are within the domain after advection
        domain_mask = (np.abs(data.x-kwargs['xy0'][0]) <= kwargs['Wxy']/2) & \
            (np.abs(data.y-kwargs['xy0'][1]) <= kwargs['Wxy']/2)
        data.index(domain_mask)
        if 'lagrangian_mask_files' in kwargs and kwargs['lagrangian_mask_files'] is not None:
            apply_lagrangian_masks(data, data.x, data.y, mask_files=kwargs['lagrangian_mask_files'], **kwargs)
    else:
        # save additional x_original and y_original points
        data.assign(x_lag=np.copy(x0), y_lag=np.copy(y0))
        # remove data for which the advection returned NaN
        data.index(np.isfinite(x0+y0))
        if 'lagrangian_mask_files' in kwargs and kwargs['lagrangian_mask_files'] is not None:
            apply_lagrangian_masks(data, data.x_lag, data.y_lag, mask_files=kwargs['lagrangian_mask_files'], **kwargs)
    out_args=kwargs
    return out_args

def update_output_data_for_lagrangian(S, **kwargs):

    # Advect output coordinates to output grid times.
    #print('Advect grid coordinates from Lagrangian epoch')
    if ['x_original'] in S['data'].fields:
    # reassign the unadvected locations to the data 'x' and 'y' fields
        S['data'].x=S['data'].x_original.copy()
        S['data'].y=S['data'].y_original.copy()
        for field in ['x_original','y_original']:
            S['data'].fields.remove(field)


def interp_ds(ds, scale):
    for field in ds.fields:
        delta_xy=[(ds.x[1]-ds.x[0])/scale, (ds.y[1]-ds.y[0])/scale]
        xi=np.arange(ds.x[0], ds.x[-1]+delta_xy[0], delta_xy[0])
        yi=np.arange(ds.y[0], ds.y[-1]+delta_xy[1], delta_xy[1])
        z0=getattr(ds, field)
        if len(ds.shape)==2:
            zi=pc.grid.data().from_dict({'x':ds.x, 'y':ds.y, 'z':z0}).interp(xi, yi, gridded=True)
            return pc.grid.data().from_dict({'x':xi, 'y':yi, field:zi})
        else:
            zi=np.zeros([xi.size, yi.size, ds.time.size])
            for epoch in range(ds.time.size):
                temp=pc.grid.data().from_dict({'x':ds.x, 'y':ds.y, 'z':np.squeeze(z0[:,:,epoch])})
                zi[:,:,epoch] = temp.interp(xi, yi, gridded=True)
            return pc.grid.data().from_dict({'x':xi, 'y':yi, 'time':ds.time, field:zi})

def save_fit_to_file(S,  filename, sensor_dict=None, dzdt_lags=None, reference_epoch=0):
    if os.path.isfile(filename):
        os.remove(filename)
    with h5py.File(filename,'w') as h5f:
        h5f.create_group('/data')
        for key in S['data'].fields:
            h5f.create_dataset('/data/'+key, data=getattr(S['data'], key))
        h5f.create_group('/meta')
        h5f.create_group('/meta/timing')
        for key in S['timing']:
            h5f['/meta/timing/'].attrs[key]=S['timing'][key]
        h5f.create_group('/RMS')
        for key in S['RMS']:
            h5f.create_dataset('/RMS/'+key, data=S['RMS'][key])
        h5f.create_group('E_RMS')
        for key in S['E_RMS']:
            h5f.create_dataset('E_RMS/'+key, data=S['E_RMS'][key])
        for key in S['m']['bias']:
            h5f.create_dataset('/bias/'+key, data=S['m']['bias'][key])
        if 'slope_bias' in S['m']:
            sensors=np.array(list(S['m']['slope_bias'].keys()))
            h5f.create_dataset('/slope_bias/sensors', data=sensors)
            x_slope=[S['m']['slope_bias'][key]['slope_x'] for key in sensors]
            y_slope=[S['m']['slope_bias'][key]['slope_y'] for key in sensors]
            h5f.create_dataset('/slope_bias/x_slope', data=np.array(x_slope))
            h5f.create_dataset('/slope_bias/y_slope', data=np.array(y_slope))
        if sensor_dict is not None:
            h5f.create_group('meta/sensors')
            for key in sensor_dict:
                h5f['/meta/sensors'].attrs['sensor_%d' % key]=sensor_dict[key]
        if 'sensor_bias_grids' in S['m'] and 'grid_bias' not in h5f:
            h5f.create_group('/grid_bias')
        if 'jitter_bias_grids' in S['m'] and 'jitter_bias' not in h5f:
            h5f.create_group('/jitter_bias')
    if 'sensor_bias_grids' in S['m']:
        for name, ds in S['m']['sensor_bias_grids'].items():
                ds.to_h5(filename, group='/grid_bias/'+name)
    if 'jitter_bias_grids' in S['m']:
        for name, ds in S['m']['jitter_bias_grids'].items():
                ds.to_h5(filename, group='/jitter_bias/'+name)
    for key , ds in S['m'].items():
        if isinstance(ds, pc.grid.data):
                ds.to_h5(filename, group=key)
    return

def assign_sigma_corr(D, orbital_sensors=[1, 2], airborne_sensors=[3, 4, 5]):
    delta_t_corr={'airborne':60*10/(24*3600*365.25),
                  'orbital':10/(24*3600*365.25)}
    time_corr=D.time.copy()
    these=np.flatnonzero(np.in1d(D.sensor, orbital_sensors))
    time_corr[these]=np.floor(D.time[these]/delta_t_corr['orbital'])*delta_t_corr['orbital']
    these=np.flatnonzero(np.in1d(D.sensor, airborne_sensors))
    time_corr[these]=np.floor(D.time[these]/delta_t_corr['airborne'])*delta_t_corr['airborne']
    D.assign({'time_corr':time_corr})
    _, sensor_dict=pc.unique_by_rows(np.c_[D.sensor, D.time_corr], return_dict=True)
    for key, ind in sensor_dict.items():
        #skip DEMs
        if key[0] > np.max(orbital_sensors+airborne_sensors):
            continue
        if np.any(np.isfinite(D.slope_mag[ind])):
            mean_slope=np.nanmean(D.slope_mag[ind])
        else:
            mean_slope=0
        D.sigma_corr[ind] = np.sqrt(D.sigma_corr[ind]**2 +
                                    (mean_slope*5)**2)

def save_errors_to_file( S, filename):

    for key, ds in S['E'].items():
        if isinstance(ds, pc.grid.data):
            print(key)
            if 'sensor_' in key and 'bias' in key:
                #write the sensor grid biases into their own group
                ds.to_h5(filename, group='/grid_bias/'+key.replace('sigma_',''))
            else:
                ds.to_h5(filename, group=key.replace('sigma_',''))

    with h5py.File(filename,'r+') as h5f:
        for key in S['E']['sigma_bias']:
            if 'bias/sigma' in h5f and  key in h5f['/bias/sigma']:
                print(f'{key} already exists in sigma_bias')
                h5f['/bias/sigma/'+key][...]=S['E']['sigma_bias'][key]
            else:
                h5f.create_dataset('/bias/sigma/'+key, data=S['E']['sigma_bias'][key])
    return

def fit_altimetry(xy0, Wxy=4e4, \
            reread_file=None,\
            E_RMS={}, t_span=[2003, 2020], spacing={'z0':2.5e2, 'dz':5.e2, 'dt':0.5},  \
            hemisphere=1, reference_epoch=None, reread_dirs=None, max_iterations=5, \
            dzdt_lags=[1, 4], \
            Edit_only=False, \
            E_slope_bias=1.e-5, \
            sensor_dict={}, out_name=None, \
            bias_nsigma_edit=None, bias_nsigma_iteration=3,\
            replace=False, DOPLOT=False, spring_only=False, \
            firn_fixed=False, firn_rescale=False, \
            firn_correction=None, firn_directory=None, firn_version=None,\
            GI_files=None,\
            geoid_file=None,\
            mask_file=None, \
            and_mask_files=None, \
            or_mask_files=None, \
            DEM_file=None,\
            mask_floating=False,\
            water_mask_threshold=None, \
            bm_scale=None,
            N_target=None,\
            calc_error_file=None, \
            extra_error=None,\
            repeat_res=None,\
            tide_mask_file=None,\
            error_res_scale=None,\
            avg_mask_directory=None, \
            velocity_files=None, \
            lagrangian_dict=None, \
            shelf_only=False,\
            tide_directory=None, \
            tide_model='CATS2008', \
            year_mask_dir=None, \
            avg_scales=None,\
            bias_params=['time_corr','sensor','spot'],\
            seg_diff_tol=4,\
            verbose=True,\
            reference_DEM_file = None,\
            reference_DEM_uncertainty = None,\
            DEM_grid_bias_params=None):
    """
        Wrapper for smooth_xytb_fit_aug that can find data and set the appropriate parameters
    """
    print("fit_OIB: working on %s" % out_name)

    # default:
    lagrangian_coords=None

    SRS_proj4=get_SRS_proj4(hemisphere)
    bias_model_args={}
    compute_E=False
    # set defaults for E_RMS, then update with input parameters
    E_RMS0={'d2z0_dx2':200000./3000/3000, 'd3z_dx2dt':3000./3000/3000, 'd2z_dxdt':3000/3000, 'd2z_dt2':5000}
    E_RMS0.update(E_RMS)

    W={'x':Wxy, 'y':Wxy,'t':np.diff(t_span)}
    ctr={'x':xy0[0], 'y':xy0[1], 't':np.mean(t_span)}

    bds={ dim: c_i+np.array([-0.5, 0.5])*W[dim]  for dim, c_i in ctr.items()}

    if out_name is not None:
        try:
            out_name=out_name %(xy0[0]/1000, xy0[1]/1000)
        except:
            pass

    if calc_error_file is not None:
        reread_file=calc_error_file
        compute_E=True
        max_iterations=0
        repeat_res=None

    pad=np.array([-1.e4, 1.e4])
    mask_data=pc.grid.data().from_file(mask_file,bounds=[bds['x']+pad, bds['y']+pad])
    if shelf_only:
        # mask out anything that's not shelf
        if mask_data.z.ndim==2:
            mask_data.z *= (pc.grid.data().from_file(tide_mask_file, bounds=mask_data.bounds()).interp(mask_data.x, mask_data.y, gridded=True) > 0.1)
        else:
            mask_data.z *= np.tile(\
                (pc.grid.data().from_file(tide_mask_file, bounds=mask_data.bounds())\
                                   .interp(mask_data.x, mask_data.y, gridded=True) > 0.1)[:,:,None],
                                   [1, 1, mask_data.shape[2]])
        if np.all(mask_data.z==0):
            if verbose:
                print("shelf_only specified, and no shelf in mask, returning")
            return None, None, None

    if and_mask_files is not None or or_mask_files is not None:
        update_mask_data_andor(mask_data, and_mask_files, or_mask_files)

    if lagrangian_dict is not None:
        lagrangian_dict = setup_lagrangian(
            SRS_proj4=SRS_proj4, xy0=xy0, Wxy=Wxy,
            t_span=t_span, spacing=spacing, reference_epoch=reference_epoch,
            **lagrangian_dict)
        if np.any(~np.isfinite(np.array(lagrangian_dict['bds']))):
            if verbose:
                print("fit_altimetry.py: lagrangian advection failed to produce finite bounds, returning")
            return None, None, None
        if lagrangian_dict['lagrangian_dz_spacing'] is not None:
            lagrangian_dict['move_points'] = False
            spacing['lagrangian_dz']= lagrangian_dict['lagrangian_dz_spacing']
            E_RMS0['lagrangian_dz'] = lagrangian_dict['lagrangian_dz_E_RMS']
            E_RMS0['lagrangian_dzdx'] = lagrangian_dict['lagrangian_dz_E_RMS_grad']
            lagrangian_coords=['y_lag','x_lag']
        else:
            # move points by default
            lagrangian_dict['move_points'] = True

    if reread_file is not None:
        # get xy0 from the filename
        re_match=re.compile('E(.*)_N(.*).h5').search(reread_file)
        xy0=[float(re_match.group(ii))*1000 for ii in [1, 2]]
        data=pc.data().from_h5(reread_file, group='data')
        sensor_dict=make_sensor_dict(reread_file)
    elif reread_dirs is None:
        if lagrangian_dict is not None and lagrangian_dict['move_points']:
            this_xy0=[*map(np.mean, lagrangian_dict['bds'])]
            this_W={'x':np.diff(lagrangian_dict['bds'][0]),
                           'y':np.diff(lagrangian_dict['bds'][1]),
                           't':W['t']}
        else:
            this_xy0, this_W = [xy0, W]
        D, sensor_dict, DEM_meta_dict = read_optical_data(this_xy0, this_W, GI_files=GI_files, \
                                SRS_proj4=get_SRS_proj4(hemisphere),\
                                bm_scale=bm_scale,\
                                N_target=N_target,\
                                target_area=W['x']*W['y'],
                                 mask_file=mask_file, geoid_file=geoid_file, \
                                 mask_floating=mask_floating,\
                                 water_mask_threshold=water_mask_threshold, \
                                 time_range=t_span, \
                                 DEM_file=DEM_file, \
                                 hemisphere=hemisphere,
                                 seg_diff_tol=seg_diff_tol)
        for ind, Di in enumerate(D):
            if Di is None:
                continue
            for field in ['rgt','cycle','spot']:
                if field not in Di.fields:
                    Di.assign({field:np.zeros_like(Di.x)+np.NaN})

        data=pc.data(fields=['x','y','z','time','sigma','sigma_corr','slope_mag', 'sensor','spot', 'rgt','cycle','BP']).from_list(D)
        data.assign({'day':np.floor(data.time*365.25)})
        if extra_error is not None:
            data.sigma[data.time < 2010] = np.sqrt(data.sigma[data.time<2010]**2 +extra_error**2)
        # apply the tides if a directory has been provided
        if tide_mask_file is not None:
            if hemisphere==1:
                EPSG=3413
            else:
                EPSG=3031
            if lagrangian_dict is not None and lagrangian_dict['move_points']:
                this_Wxy = np.max([*map(np.diff, data.bounds())])+5000
                this_xy0 = [*map(np.mean, data.bounds())]
            else:
                this_Wxy, this_xy0 = [Wxy, xy0]
            apply_tides(data, this_xy0, this_Wxy, tide_mask_file, tide_directory, tide_model, EPSG=EPSG)
    else:
        data, sensor_dict = reread_data_from_fits(xy0, Wxy, reread_dirs, template='E%d_N%d.h5')
    if shelf_only and hasattr(data, 'floating'):
        data.index(data.floating>0.1)
        if data.size==0:
            if verbose:
                print("No non-floating data points found, returning")
            return None, None, None

    laser_sensors=[item for key, item in laser_key().items()]
    DEM_sensors=np.array([key for key in sensor_dict.keys() if key not in laser_sensors ])
    if reference_epoch is None:
        reference_epoch=len(np.arange(t_span[0], t_span[1], spacing['dt']))

    if lagrangian_dict is not None:
        lagrangian_dict=update_data_for_lagrangian( data, mask_data=mask_data,
            **lagrangian_dict)

    # make every dataset a double
    for field in data.fields:
        setattr(data, field, getattr(data, field).astype(np.float64))

    if (firn_fixed or firn_rescale) and reread_dirs is None and \
        calc_error_file is None and reread_file is None:
        assign_firn_variable(data, firn_correction, firn_directory, hemisphere,
                         model_version=firn_version, subset_valid=True)
        if firn_fixed:
            data.z -= data.h_firn
    if firn_rescale:
        # the grid has one node at each domain corner.
        this_grid=fd_grid( [xy0[1]+np.array([-0.5, 0.5])*Wxy,
                xy0[0]+np.array([-0.5, 0.5])*Wxy], [Wxy, Wxy],\
             name=firn_correction+'_scale', srs_proj4=SRS_proj4)
        bias_model_args += [{'name': firn_correction+'_scale',  \
                            'param':'h_firn',\
                            'grid':this_grid,\
                            'expected_rms':0.25, 'expected_value':1.}]

    #report data counts
    for val, sensor in sensor_dict.items():
        if val < 5:
            print("for %s found %d data" %(sensor, np.sum(data.sensor==val)))
    print("for DEMs, found %d data" % np.sum(np.in1d(data.sensor, np.array(DEM_sensors))))

    sensor_grid_bias_params=None
    if DEM_grid_bias_params is not None:
        sensor_grid_bias_params=[]
        for sensor in DEM_sensors:
            sensor_grid_bias_params += [{'sensor':sensor, 'expected_val':0}]
            sensor_grid_bias_params[-1].update(DEM_grid_bias_params)
            sensor_grid_bias_params[-1].update({'filename':sensor_dict[sensor]})
    # not sure why this was ever needed.  Commenting until there's a reason for not doing so
    #if isinstance(data,pc.data):
    #    temp=pc.data().from_dict({item:data.__dict__[item] for item in data.fields})
    #    data=temp
    #    temp=None

    # apply any custom edits
    custom_edits(data)
    if data.size == 0:
        if verbose:
            print("fit_altimetry:No data found, returning")
        return None, None, None
    assign_sigma_corr(data)

    avg_masks=None
    if avg_mask_directory is not None:
        avg_masks = {os.path.basename(file).replace('.tif',''):pc.grid.data().from_geotif(file) for file in \
                     glob.glob(avg_mask_directory+'/*.tif')}

    if year_mask_dir is not None:
        mask_data_by_year(data, year_mask_dir);

    sigma_extra_masks = {'laser': np.in1d(data.sensor, laser_sensors),
                         'DEM': ~np.in1d(data.sensor, laser_sensors)}

    if reference_DEM_file is not None:
        # N.B.  Reference DEM is subtracted AFTER firn correction
        data.z -= pc.grid.data().from_file(reference_DEM_file,
                                    bounds=[xy0[0]+np.array([-0.5, 0.5])*Wxy,
                                            xy0[1]+np.array([-0.5, 0.5])*Wxy])\
                    .interp(data.x, data.y)
        E_RMS0['z0'] = reference_DEM_uncertainty
    # run the fit
    print("="*50)
    #N.B. Using smooth_fit instead of smooth_xytb_fit_aug
    print("about to run smooth_fit with bias params="+str(bias_params))
    S=smooth_fit(data=data, ctr=ctr, W=W, spacing=spacing, E_RMS=E_RMS0,
                     reference_epoch=reference_epoch, compute_E=compute_E,
                     bias_params=bias_params,
                     repeat_res=repeat_res, max_iterations=max_iterations,
                     srs_proj4=SRS_proj4, VERBOSE=True, Edit_only=Edit_only,
                     data_slope_sensors=DEM_sensors,\
                     E_slope_bias=E_slope_bias,\
                     mask_file=mask_file,\
                     mask_data=mask_data,\
                     dzdt_lags=dzdt_lags,\
                     bias_model_args = bias_model_args, \
                     bias_nsigma_edit=bias_nsigma_edit, \
                     bias_nsigma_iteration=bias_nsigma_iteration, \
                     error_res_scale=error_res_scale,\
                     mask_scale={0:10, 1:1}, \
                     avg_scales=avg_scales, \
                     avg_masks=avg_masks, \
                     sigma_extra_masks=sigma_extra_masks,\
                     sensor_grid_bias_params=sensor_grid_bias_params,\
                     lagrangian_coords=lagrangian_coords,\
                     converge_tol_frac_TSE=0.005)

    if lagrangian_dict is not None and S['data'] is not None and S['data'].size > 0:
        update_output_data_for_lagrangian(S, **lagrangian_dict)

    return S, data, sensor_dict

def main(argv):
    # account for a bug in argparse that misinterprets negative agruents
    for i, arg in enumerate(argv):
        if (arg[0] == '-') and arg[1].isdigit(): argv[i] = ' ' + arg

    def path(p):
        return os.path.abspath(os.path.expanduser(p))

    import argparse
    parser=argparse.ArgumentParser(description="function to fit icebridge data with a smooth elevation-change model", \
                                   fromfile_prefix_chars="@")
    parser.add_argument('--xy0', type=float, nargs=2, help="fit center location")
    parser.add_argument('--Width','-W',  type=float, help="fit width")
    parser.add_argument('--time_span','-t', type=str, help="time span, first year,last year AD (comma separated, no spaces)")
    parser.add_argument('--reference_epoch', type=int)
    parser.add_argument('--grid_spacing','-g', type=str, help='grid spacing:DEM (meters),dh maps xy (meters),dh_maps time (years): comma-separated, no spaces', default='250.,4000.,1.')
    parser.add_argument('--Hemisphere','-H', type=int, default=1, help='hemisphere: -1=Antarctica, 1=Greenland')
    parser.add_argument('--base_directory','-b', type=path, help='base directory')
    parser.add_argument('--GeoIndex_source_file', type=path, help='json file containing locations for geoIndex files')
    parser.add_argument('--reread_file', type=str, help='reread data from this file')
    parser.add_argument('--out_name', '-o', type=path, help="output file name")
    parser.add_argument('--dzdt_lags', type=str, default='1,2,4', help='lags for which to calculate dz/dt, comma-separated list, no spaces')
    parser.add_argument('--prelim', action="store_true")
    parser.add_argument('--E_d2zdt2', type=float, default=5000)
    parser.add_argument('--E_d2z0dx2', type=float, default=0.02)
    parser.add_argument('--E_d3zdx2dt', type=float, default=0.0003)
    parser.add_argument('--E_slope_bias', type=float, default=1.e-5)
    parser.add_argument('--data_gap_scale', type=float,  default=2500)
    parser.add_argument('--max_iterations', type=int, default=8)
    parser.add_argument('--reference_DEM_file', type=str, help='DEM that will be subtracted from data before fitting')
    parser.add_argument('--reference_DEM_uncertainty', type=float, help='Uncertainty in the reference DEM is correct')
    parser.add_argument('--bias_params', type=str, nargs='+', default=['time_corr','sensor','spot'])
    parser.add_argument('--DEM_grid_bias_params_file', type=path, help='file containing DEM grid bias params')
    parser.add_argument('--bias_nsigma_edit', type=int, default=6, help='edit points whose estimated bias is more than this value times the expected')
    parser.add_argument('--bias_nsigma_iteration', type=int, default=6, help='apply bias_nsigma_edit after this iteration')
    parser.add_argument('--bm_scale_laser', type=float, default=50, help="blockmedian scale to apply to laser data")
    parser.add_argument('--bm_scale_DEM', type=float, default=200, help='blockmedian scale to apply to DEM data')
    parser.add_argument('--N_target_laser', type=float, default=200000, help='target number of laser data')
    parser.add_argument('--N_target_DEM', type=float, default=800000, help='target number of DEM data')
    parser.add_argument('--extra_error', type=float)
    parser.add_argument('--DEM_file', type=path, help='DEM file to use with the DEM_tol parameter and in slope error calculations')
    parser.add_argument('--mask_floating', action="store_true")
    parser.add_argument('--map_dir','-m', type=path)
    parser.add_argument('--firn_directory', type=path, help='directory containing firn model')
    parser.add_argument('--firn_model', type=str, help='firn model name')
    parser.add_argument('--firn_version', type=str, help='firn version')
    parser.add_argument('--rerun_file_with_firn', type=str)
    parser.add_argument('--firn_rescale', action='store_true')
    parser.add_argument('--firn_fixed', action='store_true')
    parser.add_argument('--lagrangian', action='store_true')
    parser.add_argument('--velocity_files', type=path, nargs='+', help='lagrangian velocity files.  May contain multiple time values.')
    parser.add_argument('--lagrangian_epoch', type=float, help='time (decimal year) to which data will be advected')
    parser.add_argument('--lagrangian_mask_files', type=str, nargs='+', help='filenames for geotifs containing areas that have calved.  Filenames contain dates _YYYYMMDD_')
    parser.add_argument('--lagrangian_ref_dem', type=path, help='dem to be subtracted from data before advection')
    parser.add_argument('--lagrangian_dz_spacing', type=float, help='field to estimate moving topography, ignores lagrangian argument')
    parser.add_argument('--lagrangian_dz_E_RMS', type=float, help='Expected RMS for moving topography')
    parser.add_argument('--lagrangian_dz_E_RMS_grad', type=float, help='Expected RMS gradient for moving topography')
    parser.add_argument('--shelf_only', action='store_true', help='use only data points originally on the shelf')
    parser.add_argument('--mask_file', type=path)
    parser.add_argument('--and_mask_files', type=str, nargs='+')
    parser.add_argument('--or_mask_files', type=str, nargs='+')
    parser.add_argument('--geoid_file', type=path)
    parser.add_argument('--water_mask_threshold', type=float)
    parser.add_argument('--year_mask_dir', type=path)
    parser.add_argument('--tide_mask_file', type=path)
    parser.add_argument('--tide_directory', type=path)
    parser.add_argument('--tide_model', type=str, help='tide model name')
    parser.add_argument('--avg_mask_directory', type=path)
    parser.add_argument('--calc_error_file','-c', type=path)
    parser.add_argument('--calc_error_for_xy', action='store_true')
    parser.add_argument('--avg_scales', type=str, help='scales at which to report average errors, comma-separated list, no spaces')
    parser.add_argument('--error_res_scale','-s', type=float, nargs=2, default=[4, 2], help='if the errors are being calculated (see calc_error_file), scale the grid resolution in x and y to be coarser')
    parser.add_argument('--max_mem', type=float, default=15., help='maximum memory the program is allowed to use, in GB.')
    args, unk=parser.parse_known_args()

    if unk:
        print("unknown arguments:"+str([jj for jj in unk if (len(jj)>0) and not jj[0]=='#']))

    if args.max_mem is not None and args.max_mem > 0:
        set_memory_limit(int(args.max_mem*1024*1024*1024))

    if args.avg_scales is not None:
        args.avg_scales = [int(temp) for temp in args.avg_scales.split(',')]
    args.grid_spacing = [float(temp) for temp in args.grid_spacing.split(',')]
    args.time_span = [float(temp) for temp in args.time_span.split(',')]
    args.dzdt_lags = [int(temp) for temp in args.dzdt_lags.split(',')]

    spacing={'z0':args.grid_spacing[0], 'dz':args.grid_spacing[1], 'dt':args.grid_spacing[2]}
    E_RMS={'d2z0_dx2':args.E_d2z0dx2, 'd3z_dx2dt':args.E_d3zdx2dt, 'd2z_dxdt':args.E_d3zdx2dt*args.data_gap_scale,  'd2z_dt2':args.E_d2zdt2}

    # read in the geoIndex locations
    with open(args.GeoIndex_source_file) as fh:
        geoIndex_dict=json.load(fh)

    reread_dirs=None
    if args.prelim:
        dest_dir = args.base_directory+'/prelim'

    in_file=None
    if args.calc_error_file is not None:
        args.reread_file = args.calc_error_file
        args.out_name=args.calc_error_file
        in_file=args.calc_error_file
        dest_dir=os.path.dirname(args.reread_file)

    if args.reread_file is not None:
        # get xy0 from the filename
        in_file=args.reread_file

    if in_file is not None:
        re_match=re.compile('E(.*)_N(.*).h5').search(in_file)
        args.xy0=[float(re_match.group(ii))*1000 for ii in [1, 2]]

    if args.out_name is None:
        args.out_name=dest_dir + '/E%d_N%d.h5' % (args.xy0[0]/1e3, args.xy0[1]/1e3)

    if args.calc_error_for_xy:
        args.calc_error_file=args.out_name
        if not os.path.isfile(args.out_name):
            print(f"{args.out_name} not found, returning")
            return 1

    if args.error_res_scale is not None:
        if args.calc_error_file is not None:
            for ii, key in enumerate(['z0','dz']):
                spacing[key] *= args.error_res_scale[ii]

    if args.bias_params is not None and " " in args.bias_params[0]:
        args.bias_params=args.bias_params[0].split(" ")

    DEM_bias_params=None
    if args.DEM_grid_bias_params_file is not None:
        # DEM_bias_params file should include key=value lines, with keys:
        #spacing, expected_rms, expected_rms_grad
        DEM_bias_params={}
        with open(args.DEM_grid_bias_params_file, 'r') as fh:
            for line in fh:
                try:
                    if line[0]=="#":
                        continue
                    key, val = line.rstrip().split('=')
                    try:
                        # is val a float?
                        DEM_bias_params[key]=float(val.split('#')[0])
                    except ValueError:
                        # if not, assume str
                        DEM_bias_params[key]=val.split('#')[0]
                except Exception:
                    print("problem with DEM_bias_params line\n"+line)

    if not os.path.isdir(args.base_directory):
        os.mkdir(args.base_directory)
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

    if args.out_name is None:
        args.out_name=dest_dir + '/E%d_N%d.h5' % (args.xy0[0]/1e3, args.xy0[1]/1e3)

    lagrangian_dict=None
    if args.lagrangian:
        if args.lagrangian_epoch is None:
            args.lagrangian_epoch = np.arange(args.time_span[0], args.time_span[1], args.grid_spacing[2])[args.reference_epoch]
        lagrangian_dict = {field:getattr(args, field) for field in \
                           ['velocity_files', 'lagrangian_epoch', 'lagrangian_ref_dem',
                            'lagrangian_dz_spacing', 'lagrangian_dz_E_RMS',
                            'lagrangian_dz_E_RMS_grad', 'lagrangian_mask_files']
                           }

    S, data, sensor_dict = fit_altimetry(args.xy0, \
            Wxy=args.Width, E_RMS=E_RMS, \
            t_span=args.time_span, spacing=spacing, \
            reference_epoch=args.reference_epoch, \
            reread_file=args.reread_file,\
            calc_error_file=args.calc_error_file,\
            error_res_scale=args.error_res_scale,\
            max_iterations=args.max_iterations, \
            bias_nsigma_edit=args.bias_nsigma_edit,
            bias_nsigma_iteration=args.bias_nsigma_iteration,\
            bias_params=args.bias_params,\
            reference_DEM_file=args.reference_DEM_file,\
            reference_DEM_uncertainty=args.reference_DEM_uncertainty,\
            hemisphere=args.Hemisphere, reread_dirs=reread_dirs, \
            out_name=args.out_name, \
            GI_files=geoIndex_dict,\
            bm_scale={'laser':args.bm_scale_laser,\
                         'DEM':args.bm_scale_DEM},\
            N_target={'laser':args.N_target_laser,\
                         'DEM':args.N_target_DEM},
            firn_directory=args.firn_directory,\
            firn_version=args.firn_version,\
            firn_correction=args.firn_model,\
            firn_fixed=args.firn_fixed,\
            firn_rescale=args.firn_rescale,\
            lagrangian_dict=lagrangian_dict,\
            shelf_only=args.shelf_only,\
            mask_file=args.mask_file, \
            and_mask_files=args.and_mask_files,\
            or_mask_files=args.or_mask_files,\
            DEM_file=args.DEM_file,\
            geoid_file=args.geoid_file,\
            mask_floating=args.mask_floating,\
            extra_error=args.extra_error, \
            E_slope_bias=args.E_slope_bias, \
            water_mask_threshold=args.water_mask_threshold, \
            year_mask_dir=args.year_mask_dir, \
            tide_directory=args.tide_directory, \
            tide_mask_file=args.tide_mask_file, \
            tide_model=args.tide_model, \
            avg_mask_directory=args.avg_mask_directory, \
            dzdt_lags=args.dzdt_lags, \
            avg_scales=args.avg_scales,\
            DEM_grid_bias_params=DEM_bias_params)

    if S is None:
        return

    if args.calc_error_file is None:
        if 'm' in S and len(S['m']) > 0:
            save_fit_to_file(S, args.out_name, sensor_dict=sensor_dict,\
                             dzdt_lags=S['dzdt_lags'], \
                             reference_epoch=args.reference_epoch)
    else:
        S['E']['sigma_z0']=interp_ds(S['E']['sigma_z0'], args.error_res_scale[0])
        for field in S['E'].keys():
            if 'sigma_dz' in field: # ['sigma_dz', 'sigma_dzdt_lag1', 'sigma_dzdt_lag2', 'sigma_dzdt_lag4']:
                S['E'][field] = interp_ds( S['E'][field], args.error_res_scale[1] )
        save_errors_to_file(S, args.out_name)

    print("done with " + args.out_name)
    total_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss +\
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    print(f"peak memory usage (kB)={total_memory}")
if __name__=='__main__':
    main(sys.argv)
