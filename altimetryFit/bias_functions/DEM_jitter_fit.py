#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 09:41:48 2023

@author: ben
"""

import numpy as np
import LSsurf as LS
import re
import pointCollection as pc
import geopandas as gpd
import altimetryFit

def get_ATC_xform(filename, url_list_file, params=None):
    '''
    Calculate the transform between projected and along-track coordinates for a DEM

    Parameters
    ----------
    filename : str
        DEM filename.
    url_list_file : str
        File containing a URL for each DEM filename.
    params : dict
        dict containing metadata parameters

    Returns
    -------
    xform : dict
        dictionary containing origin and basis vectors for the transform to ATC coordinates.
    poly : numpy.array
        Coordinates of the DEM bounding polygon.

    '''

    try:
        gdf = gpd.GeoDataFrame.from_features(params)
        assert(hasattr(gdf, 'geometry') and gdf.geometry[0] is not None)
    except Exception:
        # read the PGC metadata
        meta = altimetryFit.get_pgc(filename, url_list_file, targets=['meta'])['meta']
        gdf = gpd.GeoDataFrame.from_features([meta])

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

    return xform, poly

def parse_jitter_stripe_bias_grids(m0, G_data, grids):
    """
    Extract the jitter and stripe bias grids from a model vector

    Parameters
    ----------
    m0 : numpy array
        solution vector from LSsurf.smooth_fit.
    G_data : LSsurf lin_op
        fitting matrix LSsurf.from smooth_fit
    grids : dict
        dictionary from LSsurf.smooth_fit defining grids for degrees of freedom

    Returns
    -------
    m : dict
        dictionary containing jitter bias grids.

    This function is passed to smooth_fit as a model_parsing_function, and
    extracts the jitter and stripe bias models

    """
    sensor_bias_res=[re.compile('sensor_.*_jitter'),
                     re.compile('sensor_.*_stripe')]
    m={}
    for key, cols in G_data.TOC['cols'].items():
        skip=True
        for sensor_bias_re in sensor_bias_res:
            if sensor_bias_re.search(key) is not None:
                skip=False
        if skip:
            continue
        m[key]=pc.grid.data().from_dict({
            'x':grids[key].ctrs[0],\
            'y':np.array([0]),\
            'z':np.reshape(m0[cols], grids[key].shape)})
        m[key].xform=grids[key].xform
    return m

def setup_DEM_jitter_fit(data, filename, col_0=0,
                     url_list_file=None, res=500,
                     sensor=None,
                     expected_rms_grad=1.e-5, expected_rms_bias=2,
                     skip_plane=False,
                     local_coord_index=0,
                     coord_name='x_atc',
                     fit_name='jitter',
                     expected_plane_slope=0.02, expected_plane_bias=5,
                     params=None,
                     **kwargs):
    """
    Setup equations to fit the jitter variations in a DEM

    Parameters
    ----------
    data : pointCollection.data
        point data against which to compare the DEM.
    filename : str
        DEM filename.
    col_0 : int
        first column for the DOFs in the fit
    url_list_file : str, optional
        File matching PGC urls to DEM filenames. The default is None.
    res : float, optional
        Resolution of the jitter estimates. The default is 500.
    sensor : int, optional
        if specified, only points with data.sensor==sensor are used
    expected_rms_grad : float, optional
        Expected value for the RMS gradient of the solution. The default is 1.e-5.
    expected_rms_bias : float, optional
        Expected value for the RMS of the solution. The default is 2.
    skip_plane : float, optional
        If True, no plane is fit to the residuals.  The default is False.
    local_coord_index : int, optional
        Which local coordinate to use (0-> along-track->jitter, 1->xtrack->stripes).  The default is 0.
    coord_name : str, optional
        Name of the coordinate to be fit (use x_atc for the jitter fit).  The default is 'x_atc'
    fit_name : str, optional
        Name of the fit (use 'jitter' for a jitter fit, or 'stripe' for a stripe fit).  The default is 'jitter'
    expected_plane_slope : float, optional
        Expected slope of the mean tilt of the solution. The default is 0.02.
    expected_plane_bias : float, optional
        Expected mean bias of the solution. The default is 5.

    Returns
    -------
    Gd : LSsurf.lin_op
        Matrix mapping bias parameters to each data point.
    Gc : LSsurf.lin_op
        Matrix measuring the magnitude of model parameters
    xform : dict
        Parameters describing the origin and basis vectors of the transform between along-track and projected coordinates
    this_grid : LSsurf.fd_grid
        Grid object for the along-track fit.
    xy_atc : numpy.array
        along-track coordinates for the data points.
    poly : numpy.array
        Polygon for the DEM, in projected coordinates

    """

    if sensor is not None:
        sensor_rows=np.flatnonzero(data.sensor==sensor)
        D = data[sensor_rows]
        name=f'sensor_{sensor}_{fit_name}'
    else:
        D = data
        name='fit_name'

    if 'xform' in kwargs:
        xform=kwargs['xform']
        poly=None
    else:
        xform, poly = get_ATC_xform(filename, url_list_file, params)

    # params['coords'] can specify which data coordinates to project into the
    # local coordinate system.  For example, in the lagrangian case, the right
    # coordinates are ['x_original', 'y_original']
    if 'coords' not in params or params['coords'] is None:
        coords = [D.x, D.y]
    else:
        coords = [getattr(D, params['coords'][0]),
                              getattr(D, params['coords'][1])]
    xy_atc = ( np.c_[*coords] - xform['origin']) @ xform['basis_vectors']

    # Along-track nodes:
    local_coord=xy_atc[:, local_coord_index]
    XlR = np.round(np.array([ np.min(local_coord), np.max(local_coord) ])/res + np.array([-1, 1]))*res

    grid_name=name+'_'+coord_name

    grid = LS.fd_grid([XlR], [res], name=grid_name, xform=xform)
    grid.col_0 = col_0
    grid.col_N = grid.col_0 + grid.N_nodes

    # data fit
    Gd = LS.lin_op( grid = grid, name=name).interp_mtx(coords,
                                                       dims=[local_coord_index])
    if not skip_plane > 0:
        # plane fit
        G_plane = LS.lin_op( col_0=Gd.col_N, col_N = Gd.col_N+3 )
        rr=np.arange(Gd.shape[0])
        cc=np.zeros_like(rr)+Gd.col_N
        G_plane.r = np.c_[rr, rr, rr]
        G_plane.c = np.c_[cc+0, cc+1, cc+2]
        G_plane.v = np.c_[(coords[0]-xform['origin'][0])/1000,
                          (coords[1]-xform['origin'][1])/1000, np.ones_like(rr)]
        G_plane.N_eq=Gd.shape[0]
        G_plane.ravel()
        G_plane.__update_size_and_shape__()
        Gd.add(G_plane)

    # # if the sensor is specified, remap the rows in the fitting matrix
    if sensor is not None:
        Gd.r=sensor_rows[Gd.r.ravel()].reshape(Gd.r.shape)

    # constraint equations
    Gc_zero = LS.lin_op(grid=grid, name=name+'_bias_zero').one()
    Gc_zero.expected = np.zeros(Gc_zero.shape[0])+expected_rms_bias
    Gc_slope = LS.lin_op(grid=grid, name=name+'_bias_grad').grad()
    Gc_slope.expected= np.zeros(Gc_slope.shape[0])+expected_rms_grad
    constraint_list = [Gc_zero, Gc_slope]
    if not skip_plane > 0:
        Gc_plane = LS.lin_op(col_N=Gd.col_N, name=name+'_plane').data_bias(np.arange(3), val=np.ones(3), col=Gd.col_N+[-3, -2, -1])
        Gc_plane.expected = np.array([expected_plane_slope, expected_plane_slope, expected_plane_bias])
        constraint_list += [Gc_plane]


    return Gd, constraint_list, xform, grid, xy_atc, poly


def setup_jitter_stripe_fit(data, grids, G_data, constraint_op_list, args, **params):
    """
    setup the jitter and stripe fits for a DEM

    Parameters
    ----------
    data : pointCollection.data
        input data.
    grids : dict
        grids for different degrees of freedom in the fit.
    G_data : LSsurf.lin_op
        data-fitting matrix, as used in LSsurf.smooth_fit.
    constraint_op_list : iterable
        list of constraint-equation operators.
    args : dict, optional
        dictionary containing additional arguments

    Returns
    -------
    None.

    This is a setup function that should get passed to  LSsurf.smooth_fit to
    set up the jitter and stripe fitting for DEMs

    """

    jitter_params={param.replace('jitter_',''):val for param, val in params.items() if 'jitter' in param}
    stripe_params={param.replace('stripe_',''):val for param, val in params.items() if 'stripe' in param}
    if len(jitter_params):
        try:
            jitter_params['filename']=params['filename']
            jitter_params['sensor']=params['sensor']
            if 'xform' in params:
                jitter_params['xform']=params['xform']
            if 'sensor' in jitter_params and np.sum(data.sensor==jitter_params['sensor']) < 1:
                return
            Gd_jitter, jitter_constraint_list, xform, jitter_grid, xy_atc, poly =\
                setup_DEM_jitter_fit(data, col_0=G_data.col_N, params=params,
                                     **jitter_params)
            G_data.add(Gd_jitter)
            constraint_op_list += jitter_constraint_list
            grids[jitter_grid.name] = jitter_grid
        except ValueError as e:
            print(f"jitter fit failed for {params['filename']}")
            raise(e)
            pass
    if len(stripe_params):
        try:
            stripe_params['filename']=params['filename']
            stripe_params['sensor']=params['sensor']
            if 'xform' in params:
                stripe_params['xform']=params['xform']
            stripe_params.update({'local_coord_index':1,
                                  'coord_name':'y_atc',
                                  'fit_name':'stripe'})
            if 'sensor' in stripe_params and np.sum(data.sensor==stripe_params['sensor']) < 1:
                return
            Gd_stripe, stripe_constraint_list, xform, stripe_grid, xy_atc, poly =\
                setup_DEM_jitter_fit(data, col_0=G_data.col_N,  params=params,
                                     **stripe_params)
            G_data.add(Gd_stripe)
            constraint_op_list += stripe_constraint_list
            grids[stripe_grid.name] = stripe_grid
        except ValueError as e:
            print(f"stripe fit failed for {params['filename']}")
            raise(e)
            pass
