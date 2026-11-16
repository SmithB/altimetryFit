#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Horizontal-offset (slope-scaled) biases from a fitted surface

A horizontal error (dx, dy) in the position of a datum (geolocation error, or
in a lagrangian fit an error in its advected position) appears as a vertical
error of about dx*z_x + dy*z_y, where z_x, z_y are the surface slopes.
LSsurf.smooth_fit models this with data_scale_fields=['z_x', 'z_y']: for each
bias_ID it estimates the coefficients (offsets, in m) that multiply the data
fields z_x and z_y, with expected magnitudes E_scale_z_x and E_scale_z_y (the
median over the bias_ID's data).

The --DEM_slope_bias_scale option of fit_altimetry takes the slopes from a
reference DEM file.  slope_fields_from_fit takes them from the z0 (+ dz) of an
initial fit instead, which resolves the surface at the scale of the model
grids, and sets the expected offsets from a geolocation error for each type of
sensor plus, in a lagrangian fit, a term that grows linearly with the time from
the lagrangian epoch (errors in the velocity field).

UPDATE HISTORY:
    Written 11/2026
"""

import numpy as np
import pointCollection as pc

# sensors that are not DEMs (laser and radar altimeters; see read_optical.laser_key
# and read_CS2_data.radar_key)
ALTIMETER_SENSORS = (1, 2, 3, 3.5, 4, 5, -1, -2)
DEFAULT_E_GEO = {'ICESat2': 3.5, 'DEM': 5., 'other': 1.}


def parse_E_geo(arg):
    """
    Parse a string like 'ICESat2:3.5,DEM:5,other:1' into a dict, starting from the defaults
    """
    E_geo = dict(DEFAULT_E_GEO)
    if arg is None:
        return E_geo
    if isinstance(arg, dict):
        E_geo.update(arg)
        return E_geo
    for item in arg.split(','):
        key, val = item.split(':')
        if key not in E_geo:
            raise ValueError(f'slope bias geolocation error: unknown sensor type {key}; use one of {list(E_geo)}')
        E_geo[key] = float(val)
    return E_geo


def surface_slopes(z0, x, y, dz=None, t=None):
    """
    Surface slopes (z_x, z_y) at points x, y (and times t), from a z0 grid and,
    if given, a dz grid (whose slopes at time t are added)

    Slopes are central differences on the grid nodes, interpolated linearly to
    the points.  Points without a finite slope get zero (no offset parameter
    sensitivity).
    """
    gy, gx = np.gradient(np.asarray(z0.z0, dtype=float), z0.y, z0.x)
    G = pc.grid.data().from_dict({'x': z0.x, 'y': z0.y, 'z_x': gx, 'z_y': gy})
    z_x = G.interp(x, y, field='z_x')
    z_y = G.interp(x, y, field='z_y')
    if dz is not None and t is not None:
        dzz = np.asarray(dz.dz, dtype=float)
        gy_dz, gx_dz = np.gradient(dzz, dz.y, dz.x, axis=(0, 1))
        Gdz = pc.grid.data().from_dict({'x': dz.x, 'y': dz.y, 'time': dz.time,
                                        'z_x': gx_dz, 'z_y': gy_dz})
        z_x = z_x + np.nan_to_num(Gdz.interp(x, y, t, field='z_x'))
        z_y = z_y + np.nan_to_num(Gdz.interp(x, y, t, field='z_y'))
    return np.nan_to_num(z_x), np.nan_to_num(z_y)


def expected_offsets(sensor, time, E_geo=None, E_rate=0., t_epoch=None):
    """
    Expected horizontal offset (m) for each datum: the geolocation error of its
    sensor type, combined (in quadrature) with E_rate * |time - t_epoch| if
    E_rate and t_epoch are given
    """
    E_geo = parse_E_geo(E_geo)
    sensor = np.asarray(sensor)
    E = np.full(sensor.shape, E_geo['other'], dtype=float)
    E[sensor == 2] = E_geo['ICESat2']
    E[~np.isin(sensor, ALTIMETER_SENSORS)] = E_geo['DEM']
    if E_rate and t_epoch is not None:
        E = np.sqrt(E**2 + (E_rate*(np.asarray(time) - t_epoch))**2)
    return E


def slope_fields_from_fit(data, m, E_geo=None, E_rate=0., t_epoch=None):
    """
    Assign the fields smooth_fit needs for data_scale_fields=['z_x', 'z_y']
    (z_x, z_y, E_scale_z_x, E_scale_z_y) to data, from the model m of a fit
    (m['z0'] and m['dz'] as returned by LSsurf.smooth_fit).  data.x, data.y must
    be in the coordinates of the model grids (lagrangian coordinates in a
    lagrangian fit).

    Returns the list of data_scale_fields.
    """
    z_x, z_y = surface_slopes(m['z0'], data.x, data.y, dz=m.get('dz'), t=data.time)
    E = expected_offsets(data.sensor, data.time, E_geo=E_geo, E_rate=E_rate, t_epoch=t_epoch)
    data.assign({'z_x': z_x, 'z_y': z_y, 'E_scale_z_x': E, 'E_scale_z_y': E.copy()})
    return ['z_x', 'z_y']
