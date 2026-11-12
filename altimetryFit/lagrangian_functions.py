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
import pointAdvection
import LSsurf


def get_xy01_grids(interpolator_save_files, bounds):
    """
    Read the parcel-position grids from a list of interpolator (xy01_grids) files

    Returns
    -------
    pointAdvection.advectionGrid
        grid with fields x0, y0 (position at the lagrangian epoch of the parcel
        at x, y, t) and x1, y1 (position at t of the parcel at x, y at the
        epoch), with absolute times in decimal years and the epoch in time_0
    """
    return pointAdvection.advectionGrid().from_files(interpolator_save_files, bounds=bounds)

def setup_lagrangian_fit(data, grids, G_data, constraint_ops_list, args=None, **params):

    bds = {coord: args['ctr'][coord] +
           np.array([-0.5, 0.5])*args['W'][coord] for coord in ('x', 'y', 't')}

    grids['lagrangian_dz'] = fd_grid([bds['y'], bds['x']],
                                     [args['spacing']['lagrangian_dz'],
                                         args['spacing']['lagrangian_dz']],
                                     name='lagrangian_dz', col_0=G_data.col_N,
                                     srs_proj4=args['srs_proj4'])

    E_RMS = args['E_RMS']

    coords = np.c_[[getattr(data, field) for field in params['coords']]]
    G_data.add(
        LSsurf.lin_op(grids['lagrangian_dz'], name='interp_lagrangian_dz')
        .interp_mtx( coords, bounds_error=False))

    if 'lagrangian_dz' in E_RMS and E_RMS['lagrangian_dz'] is not None:
        root_A_lag = np.sqrt(
            grids['lagrangian_dz'].delta[0]*grids['lagrangian_dz'].delta[1])
        lag_dz = LSsurf.lin_op(
            grids['lagrangian_dz'],
            name='lagrangian_rms')\
            .one(DOF='lagrangian_dz')
        lag_dz.expected = np.zeros(lag_dz.N_eq) + \
            E_RMS['lagrangian_dz']/root_A_lag
        constraint_ops_list += [lag_dz]

    if 'lagrangian_dzdx' in E_RMS and E_RMS['lagrangian_dzdx'] is not None:
        root_A_lag = np.sqrt(
            grids['lagrangian_dz'].delta[0]*grids['lagrangian_dz'].delta[1])
        grad_lag_dz = LSsurf.lin_op(
            grids['lagrangian_dz'],
            name='lagrangian_rms_grad')\
            .grad(DOF='lagrangian_dz')
        grad_lag_dz.expected = np.zeros(
            grad_lag_dz.N_eq) + E_RMS['lagrangian_dzdx']/root_A_lag
        constraint_ops_list += [grad_lag_dz]

def parse_lagrangian_fit(m0, G_data, grids, sigma=False):
    m = {}
    shp = grids['lagrangian_dz'].shape

    if sigma:
        key = 'sigma_lagrangian_dz'
    else:
        key = 'lagrangian_dz'
    m[key] = pc.grid.data().from_dict({
        'x': grids['lagrangian_dz'].ctrs[1],
        'y': grids['lagrangian_dz'].ctrs[0]})
    if sigma:
        m[key].assign(
            sigma_dz = np.reshape(m0[G_data.TOC['cols']['lagrangian_dz']], shp))
    else:
        m[key].assign(
            dz = np.reshape(m0[G_data.TOC['cols']['lagrangian_dz']], shp),
            cell_area = grids['lagrangian_dz'].cell_area,
            mask = grids['lagrangian_dz'].mask)
    return m

# velocity types that mark velocity files as advectionGrid (interpolator) files.
# Files whose names contain one of these are advectionGrid files whatever the velocity type.
advection_grid_types = ('advectionGrid', 'xy01_grids')


def advection_grid_files(velocity_files, velocity_type=None):
    """
    Find the advectionGrid (xy01_grids) files for a lagrangian fit

    If velocity_type is 'advectionGrid' or 'xy01_grids', or if the name of the
    first velocity file contains one of these, the velocity files are
    advectionGrid files or glob patterns for them (e.g. a directory of tiles,
    which are mosaicked when they are read).  Otherwise, the grids are read
    from a file named like the first velocity file, with its extension
    replaced by '_xy01_grids.h5'.

    Returns
    -------
    list of str
        advectionGrid files
    """
    if velocity_files is None or len(velocity_files) == 0:
        raise ValueError('lagrangian fits need velocity files (--velocity_files)')
    if velocity_type in advection_grid_types or \
            any(name in velocity_files[0] for name in advection_grid_types):
        files = []
        for pattern in velocity_files:
            matches = [pattern] if os.path.isfile(pattern) else sorted(glob.glob(pattern))
            if len(matches) == 0:
                raise FileNotFoundError(f'no advectionGrid files match {pattern}')
            files += matches
        return files
    return [os.path.splitext(velocity_files[0])[0] + '_xy01_grids.h5']


def advection_grid_epoch(velocity_files, velocity_type=None):
    """
    The epoch (time_0) of the advectionGrid files for a lagrangian fit, in decimal years

    The files advect points to and from this epoch, so it is the lagrangian epoch
    of the fit.  See advection_grid_files for the arguments.
    """
    files = advection_grid_files(velocity_files, velocity_type)
    time_0, units = pointAdvection.advectionGrid.read_time_0(files)
    if units != 'years':
        raise ValueError(f'the times in {files[0]} are in {units}: lagrangian fits need advectionGrid '
                         'files with times in years (decimal years)')
    return time_0


def setup_lagrangian(velocity_files=None, velocity_type = None, lagrangian_epoch=None, reference_epoch=None,
    SRS_proj4=None, xy0=None, Wxy=None, t_span=None, spacing=None, verbose=False,
    **kwargs):
    """
    Read the advectionGrid for a lagrangian fit, and find the bounds of the
    area that the fit's parcels cover during the fit

    The advectionGrid files are found by advection_grid_files.  The data are
    advected to and from the epoch (time_0) of the files: lagrangian_epoch
    must match it, and if lagrangian_epoch is None, time_0 is used.
    reference_epoch is not used.
    """
    interpolator_save_files = advection_grid_files(velocity_files, velocity_type)

    bounds=[xyi + np.array([-0.5, 0.5])*Wxy for xyi in xy0]
    # create arrays of grid coordinates
    x = fd_grid([bounds[0]], [spacing['dz']], name='x').ctrs[0]
    y = fd_grid([bounds[1]], [spacing['dz']], name='y').ctrs[0]

    # read a larger extent from the interpolator save files than we will need
    bounds_pad = [xyi + np.array([-1, 1])*Wxy for xyi in xy0]
    advection_grid = get_xy01_grids(interpolator_save_files, bounds_pad)
    # the advectionGrid times are absolute: they must be in the same units as the data times
    if advection_grid.units != 'years':
        raise ValueError(f'the times in {interpolator_save_files[0]} are in {advection_grid.units}: '
                         'lagrangian fits need advectionGrid files with times in years (decimal years)')
    # the interpolators advect points to and from the epoch they were made for
    if lagrangian_epoch is None:
        lagrangian_epoch = advection_grid.time_0
    elif not np.isclose(advection_grid.time_0, lagrangian_epoch, rtol=0, atol=1.e-6):
        raise ValueError(f'lagrangian_epoch={lagrangian_epoch} does not match the epoch of the '
                         f'interpolator files ({advection_grid.time_0}): {interpolator_save_files[0]}')
    # data outside the times of the advectionGrid can't be advected (they are not used)
    t_grid = [np.nanmin(advection_grid.time), np.nanmax(advection_grid.time)]
    if np.min(t_span) < t_grid[0] - 1.e-6 or np.max(t_span) > t_grid[1] + 1.e-6:
        print(f'setup_lagrangian: WARNING: the time span of the fit ({t_span[0]} to {t_span[1]}) is not '
              f'inside the time range of the advectionGrid ({t_grid[0]} to {t_grid[1]}): '
              'data outside the advectionGrid time range will not be used')

    # verbose output of lagrangian parameters
    if verbose:
        print(f'Velocity File(s): {",".join(interpolator_save_files)}')
        print(f'Advect parcels to {lagrangian_epoch:0.2f}')

    # make the ouput arguments dictionary
    out_args=kwargs
    out_args.update({
            'lagrangian_epoch':lagrangian_epoch,
            'SRS_proj4':SRS_proj4,
            'xy0':xy0,
            'Wxy':Wxy})

    gridy, gridx, gridt = np.meshgrid(y, x, np.array(t_span), indexing='ij')
    x1=advection_grid.interp(gridx.ravel(), gridy.ravel(), gridt.ravel(), field='x1')
    y1=advection_grid.interp(gridx.ravel(), gridy.ravel(), gridt.ravel(), field='y1')
    x1 = np.append(x1, bounds[0])
    y1 = np.append(y1, bounds[1])
    bounds_full=[[np.nanmin(x1), np.nanmax(x1)],[np.nanmin(y1), np.nanmax(y1)]]
    reread = False
    for dim in [0, 1]:
        if bounds_full[dim][0] < bounds[dim][0] or bounds_full[dim][1] > bounds[dim][1] :
            reread=True
            break
    if reread:
        # pad the bounds by two grid cells: the grid read for bounds_full has
        # only the nodes inside it, so without the padding, data close to the
        # edges of bounds_full could not be interpolated
        dx = np.abs(advection_grid.x[1] - advection_grid.x[0])
        advection_grid = get_xy01_grids(interpolator_save_files,
                                        [np.array(bb) + np.array([-2, 2])*dx for bb in bounds_full])

    out_args.update({'advection_grid': advection_grid,
        'bds':bounds_full})
    return out_args


def apply_lagrangian_masks(data, x, y, mask_files=None, **kwargs):
    for mask_file in mask_files:
        mask=pc.grid.data().from_geotif(mask_file, bounds=kwargs['advection_grid'].bounds())
        if mask is None or np.any(mask.shape==0):
            continue
        year_mask = pc.grid.DEM_year(mask_file)
        # if we have moved the points, the location of the points at the time of the mask is
        # found by interpolating x1 and y1 (advection_grid times are absolute)
        xx = kwargs['advection_grid'].interp(x, y, year_mask, field='x1')
        yy = kwargs['advection_grid'].interp(x, y, year_mask, field='y1')
        mask_i = mask.interp(xx, yy)
        keep =  (~np.isfinite(mask_i)) | ( mask_i < 0.1 ) | (data.time < year_mask)
        data.index(keep)


def advect_mask(mask_data, t_span=None, spacing=None, mask_spacing=None, max_substeps=10,
                xy0=None, Wxy=None, lagrangian_epoch=None, advection_grid=None,
                lagrangian_zero_velocity=False, verbose=False, **kwargs):
    '''
    Map an eulerian mask into the lagrangian frame of a full-lagrangian fit

    The nodes of a grid in lagrangian coordinates (parcel positions at
    lagrangian_epoch) are advected through the time span of the fit, and the
    mask is sampled at each parcel's position (and, for a mask with a time
    dimension, at the time).  A parcel is inside the mask where the sampled
    value is greater than 0.5.

    Parameters
    ----------
    mask_data : pointCollection.grid.data
        eulerian mask, optionally with a time dimension (t, in decimal years)
    t_span : iterable
        start and end of the fit (decimal years)
    spacing : dict
        grid spacing of the fit, with entries 'dz' and 'dt'.  The time-varying
        mask is sampled at the epochs of the dz grid.
    mask_spacing : float, optional
        spacing of the lagrangian mask grid.  The default is spacing['dz']/2,
        which puts mask nodes on the dz nodes.
    max_substeps : int, optional
        Between epochs, parcels are also sampled at substeps short enough that
        they move less than mask_spacing between samples, up to this many per
        epoch.  The substeps only affect mask_any.  The default is 10.
    xy0, Wxy, lagrangian_epoch, advection_grid, lagrangian_zero_velocity :
        lagrangian parameters, as returned by setup_lagrangian
    verbose : bool, optional
        if True, report the number of substeps.  The default is False.

    Returns
    -------
    mask_epochs : pointCollection.grid.data
        (y, x, t) mask, true where the parcel is inside mask_data at each dz epoch
    mask_any : pointCollection.grid.data
        (y, x) mask, true where the parcel is inside mask_data at any time
        during the fit
    '''
    if lagrangian_epoch is None:
        raise ValueError('advect_mask: lagrangian_epoch must be specified')
    if mask_spacing is None:
        mask_spacing = spacing['dz']/2
    # lagrangian grid covering the domain, padded by two cells
    pad = 2*mask_spacing
    x = np.arange(xy0[0]-Wxy/2-pad, xy0[0]+Wxy/2+pad+mask_spacing/2, mask_spacing)
    y = np.arange(xy0[1]-Wxy/2-pad, xy0[1]+Wxy/2+pad+mask_spacing/2, mask_spacing)
    yg, xg = np.meshgrid(y, x, indexing='ij')
    xg, yg = xg.ravel(), yg.ravel()
    # the dz epochs, calculated as they are for the dz grid
    t_epochs = fd_grid([np.array(t_span)], [spacing['dt']], name='t').ctrs[0]
    mask_is_3d = len(mask_data.shape) == 3

    def positions(t):
        # positions of the parcels at time t
        if lagrangian_zero_velocity:
            return xg, yg
        tt = np.zeros_like(xg) + t
        return advection_grid.interp(xg, yg, tt, field='x1'), advection_grid.interp(xg, yg, tt, field='y1')

    def inside(xx, yy, t):
        if mask_is_3d:
            mask_i = mask_data.interp(xx, yy, np.zeros_like(xx)+t)
        else:
            mask_i = mask_data.interp(xx, yy)
        return np.nan_to_num(mask_i) > 0.5

    inside_epochs = np.zeros((len(y), len(x), len(t_epochs)), dtype=bool)
    max_step = 0
    last = None
    for k, t in enumerate(t_epochs):
        xx, yy = positions(t)
        inside_epochs[:, :, k] = inside(xx, yy, t).reshape(len(y), len(x))
        if last is not None:
            step = np.abs((xx-last[0]) + 1j*(yy-last[1]))
            if np.any(np.isfinite(step)):
                max_step = np.maximum(max_step, np.nanmax(step))
        last = (xx, yy)
    inside_any = np.any(inside_epochs, axis=2)

    # substeps between epochs, so that parcels that are inside the mask only
    # between epochs are included in mask_any
    n_sub = int(np.clip(np.ceil(max_step/mask_spacing), 1, max_substeps))
    if verbose:
        print(f'advect_mask: parcels move up to {max_step:0.0f} m between epochs, using {n_sub} substeps per epoch')
    for t0 in t_epochs[:-1]:
        for t in t0 + spacing['dt']*np.arange(1, n_sub)/n_sub:
            inside_any |= inside(*positions(t), t).reshape(len(y), len(x))

    mask_epochs = pc.grid.data().from_dict({'x':x, 'y':y, 't':t_epochs, 'z':inside_epochs})
    mask_any = pc.grid.data().from_dict({'x':x, 'y':y, 'z':inside_any})
    return mask_epochs, mask_any


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

    if kwargs['reread_data'] and 'x_lag' in data.fields:
        if 'move_points' in kwargs and kwargs['move_points']:
            # if we are using data that was read from a previous fit in full
            # lagrangian mode, copy the 'x_lag' and 'y_lag' fields to 'x' and 'y'
            data.assign(x_original = data.x.copy())
            data.assign(y_original = data.y.copy())
            data.x = data.x_lag.copy()
            data.y = data.y_lag.copy()
            data.fields.remove('x_lag')
            data.fields.remove('y_lag')
        out_args=kwargs
        # if we are not moving the points, do nothing.
        return out_args

    if not kwargs['lagrangian_zero_velocity']:
        # advection_grid times are absolute
        x0 = kwargs['advection_grid'].interp(data.x, data.y, data.time, field='x0')
        y0 = kwargs['advection_grid'].interp(data.x, data.y, data.time, field='y0')
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
