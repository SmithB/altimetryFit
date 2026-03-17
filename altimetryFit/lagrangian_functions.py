#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 09:18:45 2026

@author: ben
"""

import numpy as np
import pointCollection as pc
from LSsurf import fd_grid
import os
import glob
import h5py
import pointAdvection

def get_xy01_grids(interpolator_save_files, bounds):
    SPY=365.25*24*3600
    xy0_obj=pc.grid.mosaic().from_list(interpolator_save_files, bounds=bounds, group='xy0')
    xy1_obj=pc.grid.mosaic().from_list(interpolator_save_files, bounds=bounds, group='xy1')
    xy0_obj.years_to_time=SPY
    with h5py.File(interpolator_save_files[0],'r') as h5f:
        try:
            if h5f['xy0']['t'].attrs['units']=='years':
                print("years!")
                xy0_obj.years_to_time = 1
                xy1_obj.years_to_time = 1
        except:
            pass
    return xy0_obj, xy1_obj

def setup_lagrangian(velocity_files=None, velocity_type = None, lagrangian_epoch=None, reference_epoch=None,
    SRS_proj4=None, xy0=None, Wxy=None, t_span=None, spacing=None, verbose=False,
    **kwargs):

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
    if 'xy01_grids' in velocity_files[0] or velocity_type=='xy01_grids':
        interpolator_save_files=velocity_files
        if not os.path.isfile(interpolator_save_files[0]):
            interpolator_save_files=glob.glob(interpolator_save_files[0])
            if len(interpolator_save_files)==0:
                raise(FileNotFoundError(f"Interpolator save files {interpolator_save_files} not found"))
    else:
        interpolator_save_files=[os.path.splitext(velocity_files[0])[0] +'_xy01_grids.h5']

    # read a larger extent from the interpolator save files than we will need
    bounds_pad = [xyi + np.array([-1, 1])*Wxy for xyi in xy0]
    xy0_obj, xy1_obj = get_xy01_grids(interpolator_save_files, bounds_pad)

    gridy, gridx, gridt = np.meshgrid(y, x,
                                      (np.array(t_span)-lagrangian_epoch)*xy0_obj.years_to_time, indexing='ij')
    x1=xy1_obj.interp(gridx.ravel(), gridy.ravel(), gridt.ravel(), field='x1')
    y1=xy1_obj.interp(gridx.ravel(), gridy.ravel(), gridt.ravel(), field='y1')
    x1 = np.append(x1, bounds[0])
    y1 = np.append(y1, bounds[1])
    bounds_full=[[np.nanmin(x1), np.nanmax(x1)],[np.nanmin(y1), np.nanmax(y1)]]
    reread = False
    for dim in [0, 1]:
        if bounds_full[dim][0] < bounds[dim][0] or bounds_full[dim][1] > bounds[dim][1] :
            reread=True
            break
    if reread:
        xy0_obj, xy1_obj = get_xy01_grids(interpolator_save_files, bounds_full)

    out_args.update({'xy1_obj': xy1_obj,
        'xy0_obj': xy0_obj,
        'bds':bounds_full})
    return out_args


def apply_lagrangian_masks(data, x, y, mask_files=None, **kwargs):
    for mask_file in mask_files:
        mask=pc.grid.data().from_geotif(mask_file, bounds=kwargs['xy0_obj'].bounds())
        if mask is None or np.any(mask.shape==0):
            continue
        year_mask = pc.grid.DEM_year(mask_file)
        t_mask=( year_mask - kwargs['lagrangian_epoch'])*kwargs['xy0_obj'].years_to_time
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
    kwargs.setdefault('reread_data', False)

    # get latitude and longitude of the original data
    data.get_latlon(proj4_string=kwargs['SRS_proj4'])

    if lagrangian_ref_dem is not None:
        print(f'Subtracting reference dem from {lagrangian_ref_dem}')
        data.assign(z0=data.z.copy())
        data.z -= pc.grid.data().from_geotif(lagrangian_ref_dem,
            bounds=data.bounds()).interp(data.x, data.y)
        data.index(np.isfinite(data.z))

    t_lag = (data.time - kwargs['lagrangian_epoch'])*kwargs['xy0_obj'].years_to_time

    if kwargs['reread_data'] and 'x_lag' in data.fields:
        # we're using data that has been reread from a previous run.
        data.assign(x_original = data.x.copy())
        data.assign(y_original = data.y.copy())
        data.x = data.x_lag.copy()
        data.y = data.y_lag.copy()
        data.fields.remove('x_lag')
        data.fields.remove('y_lag')
        out_args=kwargs
        return out_args

    if not kwargs['lagrangian_zero_velocity']:
        x0 = kwargs['xy0_obj'].interp(data.x, data.y, t_lag, field='x0')
        y0 = kwargs['xy0_obj'].interp(data.x, data.y, t_lag, field='y0')
    else:
        x0=data.x.copy()
        y0=data.y.copy()

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

    # if x_lag has been assigned, it means we're doing a hybrid calculation,
    # don't want to overwrite it
    if 'x_lag' not in S['data'].fields:
        # save the lagrangian displaced coordinates
        S['data'].assign(x_lag=S['data'].x.copy())
        S['data'].assign(y_lag=S['data'].y.copy())

    # reassign the unadvected locations to the data 'x' and 'y' fields
    if 'x_original' in S['data'].fields:
        S['data'].x=S['data'].x_original.copy()
        S['data'].y=S['data'].y_original.copy()
        for field in ['x_original','y_original']:
            S['data'].fields.remove(field)
