#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 09:36:54 2026

@author: ben
"""
import numpy as np
import glob
import re
import pointCollection as pc
import scipy.ndimage

# provides mask_data_by_year, update_mask_data_andor, setup_gzone_relax,
#   apply_tide_mask_keep, make_default_mask, read_named_masks, mask_is_empty

def mask_data_by_year(data, mask_dir):
    masks={}
    year_re=re.compile(r'(\d\d\d\d.+\d+).tif')
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

def setup_gzone_relax( gzone_relax_scale,
                      tide_mask_file,
                      bds,
                      spacing,
                      constraint_scaling_maps,
                      sigma=None ):
    if sigma is None:
        sigma = spacing['dz']*3


    dx=spacing['dz']
    mask = pc.grid.data().from_dict({
        'x':np.arange(bds['x'][0]-6*dx, bds['x'][1]+6*dx+1, dx),
        'y':np.arange(bds['y'][0]-6*dx, bds['y'][1]+6*dx+1, dx)})

    temp = pc.grid.data().from_file(tide_mask_file,
                                  bounds=mask.bounds())
    temp.z = np.nan_to_num(temp.z)
    mask.assign(z = np.nan_to_num( temp.interp( mask.x, mask.y, gridded=True) ))
    zx, zy = np.gradient( mask.z )
    mask.assign( grad_mag = np.abs(zx+1j*zy) )
    mask.crop((bds['x'][0]-3*dx, bds['x'][1]+3*dx+1),
              (bds['y'][0]-3*dx, bds['y'][1]+3*dx+1))
    mask.assign( z = scipy.ndimage.gaussian_filter(mask.grad_mag, sigma/dx ))
    mask.z = 1 + (gzone_relax_scale - 1) * mask.z
    if 'd3z_dx2dt' in constraint_scaling_maps:
        constraint_scaling_maps['d3z_dx2dt'].z *= \
            mask.interp(constraint_scaling_maps['d3z_dx2dt'].x, \
                        constraint_scaling_maps['d3z_dx2dt'].y, gridded=True)
    else:
        constraint_scaling_maps['d3z_dx2dt'] = mask

def apply_tide_mask_keep(tide_mask_keep_val,
                     buffer_W=None,
                     mask_data=None,
                     tide_mask_file=None):
    if tide_mask_keep_val is None:
        return
    # The tide_mask_value must match the tide_mask_only_value to within a tolerance of 0.1
    this_tide_mask = pc.grid.data().from_file(tide_mask_file,
                                             bounds=mask_data.bounds())
    this_tide_mask.z=np.nan_to_num(this_tide_mask.z)
    this_tide_mask.z = np.abs(this_tide_mask.z - tide_mask_keep_val) < 0.1
    if buffer_W is not None:
        N = int(buffer_W/(this_tide_mask.x[1]-this_tide_mask.x[0]))
        this_tide_mask.z = scipy.ndimage.binary_dilation(
                scipy.ndimage.binary_dilation(this_tide_mask.z, np.ones([N,1]))
                , np.ones([1,N]))

    if mask_data.z.ndim==2:
        mask_data.z *= (this_tide_mask.interp(mask_data.x, mask_data.y, gridded=True) > 0.1)
    else:
        mask_data.z *= np.tile(\
            (this_tide_mask.interp(mask_data.x, mask_data.y, gridded=True) > 0.1)[:,:,None],
                               [1, 1, mask_data.shape[2]])

def make_default_mask(bounds, mask_file=None, spacing=None,
                      tide_mask_keep_val=None, tide_mask_keep_buffer=None,
                      tide_mask_file=None, and_mask_files=None, or_mask_files=None):
    '''
    Build the default mask (the mask named 'default' in fit_altimetry)

    The mask is read from mask_file (or is all ones, at the specified spacing),
    restricted to the tide-mask 'keep' value, and combined with the and/or masks.

    Parameters
    ----------
    bounds : list
        [[xmin, xmax], [ymin, ymax]] extent of the mask
    mask_file : str, optional
        mask file (1 inside the mask).  The default is None.
    spacing : float, optional
        spacing of the all-ones mask used if mask_file is None.  The default is None.
    tide_mask_keep_val, tide_mask_keep_buffer, tide_mask_file : optional
        see apply_tide_mask_keep
    and_mask_files, or_mask_files : list of str, optional
        see update_mask_data_andor

    Returns
    -------
    mask_data : pointCollection.grid.data
        the mask
    '''
    if mask_file is not None:
        mask_data=pc.grid.data().from_file(mask_file,bounds=bounds)
    else:
        mask_data=pc.grid.data().from_dict({'x':np.arange(bounds[0][0], bounds[0][1]+.1, spacing),
                                            'y':np.arange(bounds[1][0], bounds[1][1]+.1, spacing)})
        mask_data.assign(z=np.ones((len(mask_data.y), len(mask_data.x))))

    apply_tide_mask_keep(tide_mask_keep_val,
                         buffer_W=tide_mask_keep_buffer,
                         mask_data=mask_data,
                         tide_mask_file=tide_mask_file)

    if and_mask_files is not None or or_mask_files is not None:
        update_mask_data_andor(mask_data, and_mask_files, or_mask_files)
    return mask_data

def read_named_masks(masks, bounds):
    '''
    Read a dictionary of named masks

    Parameters
    ----------
    masks : dict
        maps mask names to mask files (geotif or h5, with fields x, y, [t,] z),
        which are read over bounds, or to pointCollection.grid.data objects,
        which are used as they are.
    bounds : list
        [[xmin, xmax], [ymin, ymax]] extent to read

    Returns
    -------
    dict
        maps mask names to pointCollection.grid.data objects.  A mask file that
        does not overlap bounds gives an all-zero mask.
    '''
    out={}
    for name, mask in masks.items():
        if isinstance(mask, str):
            mask = pc.grid.data().from_file(mask, bounds=bounds)
        if mask is None or getattr(mask, 'z', None) is None or np.size(mask.z)==0:
            mask = pc.grid.data().from_dict({'x':np.array(bounds[0], dtype=float),
                                             'y':np.array(bounds[1], dtype=float),
                                             'z':np.zeros((2,2))})
        out[name]=mask
    return out

def mask_is_empty(mask_data):
    '''
    Return True if no part of a mask is greater than 0.5
    '''
    return not np.any(np.nan_to_num(mask_data.z) > 0.5)
