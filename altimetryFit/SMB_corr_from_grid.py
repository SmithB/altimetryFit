#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:15:45 2023

@author: ben
"""


import pointCollection as pc
import numpy as np
import os
import glob
import h5py
import sys

import scipy.ndimage as snd
def smooth_corrected(z, w_smooth, set_NaN=True, mask=None, return_mask=False):
    if mask is None:
        mask=np.isfinite(z)
    mask1=snd.gaussian_filter(np.float64(mask), w_smooth, mode="constant", cval=0)
    ztemp=np.nan_to_num(z)
    ztemp[mask==0]=0.0
    zs=snd.gaussian_filter(ztemp, w_smooth, mode="constant", cval=0)
    zs[mask1>0]=zs[mask1>0]/mask1[mask1>0]
    if set_NaN:
        zs[mask1==0]=np.NaN
    if return_mask:
        return zs, mask
    else:
        return zs

def fill_gaps(z, w_smooth, mask=None, set_NaN=True, return_mask=False):
    if return_mask:
        zs, mask1 = smooth_corrected(z, w_smooth, mask=mask, set_Nan=set_NaN)
        missing= mask1==0
        z[missing]=zs[missing]
        return z, mask1
    else:
        zs=smooth_corrected(z, w_smooth, mask=mask, set_NaN=set_NaN)
        missing=~np.isfinite(z)
        z[missing]=zs[missing]
        return z

def fill_SMB_gaps(z, w_smooth):
    if z.ndim==3:
        for ii in range(z.shape[2]):
            temp=z[:,:,ii]
            fill_gaps(temp, w_smooth)
            z[:,:,ii]=temp
    else:
        fill_gaps(z, w_smooth)
    return z

def SMB_corr_from_grid(data, model_file=None, time=None, var_mapping=None,
                       w_smooth=1, rho_water=1, rho_ice=0.917, gridded=False):
    """
    Interpolate a gridded SMB model to data points.

    Parameters
    ----------
    data : pointCollection.data
        Data structure into which the SMB model will be interpolated.  Must have x, y fields
    model_file : str, optional
        Model file from which to read the data.
    time : numpy.array, optional
        Time values at which to interpolate the model.  If None, the data.t field will be used. The default is None.
    var_mapping : str, optional
        Dictionary describing how data fields will be read from model fields. The default is None.
    w_smooth : int, optional
        Number of grid cells over which nodata values of the model will be smoothed at the model edges. The default is 1.
    rho_water : float, optional
        Density of water used in calculating SMB_a effect on elevation, kg/L. The default is 1.
    rho_ice : float, optional
        Density of ice used in calculating SMB_a effect on elevation, kg/L. The default is 0.917.
    gridded : bool, optional
        If set, the interpolation will treat x, y, t as grid coordinates and return gridded interpolated values.
    Returns
    -------
    None.

    """

    if var_mapping is None:
        var_mapping={'SMB_a':'SMB_a','FAC':'FAC','h_a':'h_a'}

    w_smooth=1
    pad_f=np.array([-2.6e4, 2.6e4])
    pad_c=pad_f*3

    if time is None:
        if hasattr(data,'t'):
            time=data.t.copy()
        else:
            time=data.time.copy()

    bounds=data.bounds()
    t_range = [np.nanmin(time), np.nanmax(time)]

    with h5py.File(model_file,'r') as h5f:
        if 't' in h5f:
            delta_t=np.diff(np.array(h5f['t'][0:2]))
        else:
            delta_t=np.diff(np.array(h5f['time'][0:2]))
    # first, try to read a narrow range of data
    smbfd=pc.grid.data().from_file(model_file, bounds=[np.array(jj)+pad_f for jj in bounds],
                                   t_range=[t_range[0]-delta_t, t_range[1]+delta_t],
                                   field_mapping=var_mapping)
    if not(np.all(np.isfinite(smbfd.SMB_a))):
        # if there are NaNs in the SMB fields, fill the gaps with smooth extrapoation
        smbfd=pc.grid.data().from_nc(model_file, bounds=[np.array(jj)+pad_c for jj in bounds],
                                     t_range=[t_range[0]-delta_t, t_range[1]+delta_t],
                                     field_mapping=var_mapping)
        for field in smbfd.fields:
            setattr(smbfd, field, fill_SMB_gaps(getattr(smbfd, field), w_smooth))

    SMB_data={}
    for field in var_mapping:
        SMB_data[field]=smbfd.interp(data.x, data.y, t=time, field=field, gridded=gridded)

    if 'floating' in data.fields and np.any(data.floating):
        float_scale = (data.floating==0) + (rho_water-.917)/rho_water*(data.floating==1)
        data.assign({'h_firn':SMB_data['FAC'] + float_scale*SMB_data['SMB_a']})
    else:
        data.assign({'h_firn':SMB_data['FAC'] + SMB_data['SMB_a']})
    return
