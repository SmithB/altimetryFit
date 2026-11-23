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
reference DEM file.  slope_fields_from_fit takes them from the surface of an
initial fit instead: z0 + dz, plus the fine grids (z0_fine, dz_fine) where the
fit has them, plus the slopes of any reference surface that was subtracted from
the data before the fit (a reference DEM), which resolves the surface at the
scale of the model grids.  The expected offsets are the geolocation error for
each type of sensor (see altimetryFit.sensors.geolocation_error) plus, in a
lagrangian fit, a term that grows linearly with the time from the lagrangian
epoch (errors in the velocity field).

A fit that makes slope-bias fields records how (slope_bias_meta: the source
of the slopes, the expected offsets, what the slopes include) in its output
file, as attributes of /meta/slope_bias.  A fit that rereads the data (a
matched fit, an error calculation) continues with the saved fields, and
check_slope_bias_fields compares the record with its options.

UPDATE HISTORY:
    Written 11/2026
    11/2026: fine grids and reference-surface slopes; sensor numbers and
        geolocation errors from altimetryFit.sensors; record of how the
        fields were made
"""

import json
import h5py
import numpy as np
import pointCollection as pc
# parse_E_geo is defined in altimetryFit.sensors (imported here for older code)
from altimetryFit.sensors import parse_E_geo, geolocation_error  # noqa: F401


def grid_slopes(grid, x, y, field='z', t=None, fill_value=None):
    """
    Slopes (z_x, z_y) of a grid field at points x, y (and times t, for a 3-D field)

    Slopes are central differences on the grid nodes, interpolated linearly to
    the points.  If fill_value is given, NaN nodes are set to it before
    differencing (e.g. 0 for a fine grid, which is zero outside its region);
    otherwise slopes next to NaN nodes are NaN.  Points without a finite slope
    (including points outside the grid) get zero.
    """
    zz = np.asarray(getattr(grid, field), dtype=float)
    if zz.shape[0] < 2 or zz.shape[1] < 2:
        return np.zeros_like(np.asarray(x, dtype=float)), np.zeros_like(np.asarray(x, dtype=float))
    if fill_value is not None:
        zz = np.where(np.isfinite(zz), zz, fill_value)
    gy, gx = np.gradient(zz, grid.y, grid.x, axis=(0, 1))
    coords = {'x': grid.x, 'y': grid.y}
    if zz.ndim == 3:
        coords['time'] = grid.time
    G = pc.grid.data().from_dict(coords | {'z_x': gx, 'z_y': gy})
    if zz.ndim == 3:
        z_x = G.interp(x, y, t, field='z_x')
        z_y = G.interp(x, y, t, field='z_y')
    else:
        z_x = G.interp(x, y, field='z_x')
        z_y = G.interp(x, y, field='z_y')
    return np.nan_to_num(z_x), np.nan_to_num(z_y)


def surface_slopes(z0, x, y, dz=None, t=None, z0_fine=None, dz_fine=None):
    """
    Surface slopes (z_x, z_y) of a model at points x, y (and times t)

    The model is z0 (a grid with field z0) plus, if given, dz (field dz) at
    times t, and the fine grids z0_fine (field z0_fine) and dz_fine (field
    dz_fine, at times t), which are NaN outside their region, where the model
    has no fine part.  Points without a finite slope get zero (no offset
    parameter sensitivity).
    """
    z_x, z_y = grid_slopes(z0, x, y, field='z0')
    if dz is not None and t is not None:
        dz_x, dz_y = grid_slopes(dz, x, y, field='dz', t=t)
        z_x, z_y = z_x + dz_x, z_y + dz_y
    if z0_fine is not None:
        f_x, f_y = grid_slopes(z0_fine, x, y, field='z0_fine', fill_value=0.)
        z_x, z_y = z_x + f_x, z_y + f_y
    if dz_fine is not None and t is not None:
        f_x, f_y = grid_slopes(dz_fine, x, y, field='dz_fine', t=t, fill_value=0.)
        z_x, z_y = z_x + f_x, z_y + f_y
    return z_x, z_y


def expected_offsets(sensor, time, E_geo=None, E_rate=0., t_epoch=None, sensor_dict=None):
    """
    Expected horizontal offset (m) for each datum: the geolocation error of its
    sensor type (altimetryFit.sensors.geolocation_error, with sensors classified
    by name from sensor_dict if it is given), combined (in quadrature) with
    E_rate * |time - t_epoch| if E_rate and t_epoch are given
    """
    E = geolocation_error(sensor, sensor_dict=sensor_dict, E_geo=E_geo)
    if E_rate and t_epoch is not None:
        E = np.sqrt(E**2 + (E_rate*(np.asarray(time) - t_epoch))**2)
    return E


def slope_fields_from_fit(data, m, E_geo=None, E_rate=0., t_epoch=None,
                          ref_slopes=None, sensor_dict=None):
    """
    Assign the fields smooth_fit needs for data_scale_fields=['z_x', 'z_y']
    (z_x, z_y, E_scale_z_x, E_scale_z_y) to data, from the model m of a fit
    (m['z0'], m['dz'] and, if present, m['z0_fine'] and m['dz_fine'], as
    returned by LSsurf.smooth_fit).  data.x, data.y must be in the
    coordinates of the model grids (lagrangian coordinates in a lagrangian
    fit).  ref_slopes, if given, are the slopes (z_x, z_y) at the data of a
    reference surface that was subtracted from the data before the fit, added
    to the slopes of the model.

    Returns the list of data_scale_fields.
    """
    z_x, z_y = surface_slopes(m['z0'], data.x, data.y, dz=m.get('dz'), t=data.time,
                              z0_fine=m.get('z0_fine'), dz_fine=m.get('dz_fine'))
    if ref_slopes is not None:
        z_x = z_x + np.nan_to_num(ref_slopes[0])
        z_y = z_y + np.nan_to_num(ref_slopes[1])
    E = expected_offsets(data.sensor, data.time, E_geo=E_geo, E_rate=E_rate, t_epoch=t_epoch,
                         sensor_dict=sensor_dict)
    data.assign({'z_x': z_x, 'z_y': z_y, 'E_scale_z_x': E, 'E_scale_z_y': E.copy()})
    return ['z_x', 'z_y']


SLOPE_BIAS_META_GROUP = 'meta/slope_bias'
SLOPE_BIAS_SOURCES = {'fit': 'the surface of the fit (--slope_bias_from_z0)',
                      'DEM': 'a DEM file (--DEM_slope_bias_scale)'}


def slope_bias_meta(source, E_geo=None, E_rate=0., t_epoch=None, slopes=''):
    """
    The record of how slope-bias fields were made (saved by
    fit_altimetry.save_fit_to_file as attributes of /meta/slope_bias)

    Parameters
    ----------
    source : str
        'fit' (slopes from the fit's surface, slope_fields_from_fit) or 'DEM'
        (slopes from a DEM file, read_optical_data's DEM_slope_bias_scale)
    E_geo : dict or str, optional
        geolocation errors; the full table (parse_E_geo(E_geo)) is recorded,
        as json
    E_rate : float, optional
        growth of the expected offset with time from t_epoch (m/yr); recorded
        as 0 if there is no t_epoch (it is used only in lagrangian fits)
    t_epoch : float, optional
        the lagrangian epoch (NaN if None)
    slopes : str, optional
        what the slopes include (e.g. 'z0,dz,z0_fine,reference_DEM', or
        'DEM_file:<file>')
    """
    if source not in SLOPE_BIAS_SOURCES:
        raise ValueError(f'slope_bias_meta: source must be one of {list(SLOPE_BIAS_SOURCES)}, not {source}')
    return {'source': source,
            'E_geo': json.dumps(parse_E_geo(E_geo), sort_keys=True),
            'E_rate': float(E_rate) if (E_rate and t_epoch is not None) else 0.,
            't_epoch': float(t_epoch) if t_epoch is not None else np.nan,
            'slopes': slopes}


def write_slope_bias_meta(h5f, meta):
    """write a slope_bias_meta record to an open h5py.File"""
    if SLOPE_BIAS_META_GROUP in h5f:
        del h5f[SLOPE_BIAS_META_GROUP]
    group = h5f.create_group(SLOPE_BIAS_META_GROUP)
    for key, val in meta.items():
        if val is not None:
            group.attrs[key] = val


def read_slope_bias_meta(filename):
    """
    The slope-bias record of a fit file (see slope_bias_meta), or None

    Files written before the record existed (Nov 2026) have none: if the
    fit's arguments (/meta/input_args) show DEM_slope_bias_scale or
    slope_bias_from_z0, the source is taken from them, with the other
    entries None and 'inferred' True.
    """
    with h5py.File(filename, 'r') as h5f:
        if SLOPE_BIAS_META_GROUP in h5f:
            meta = {}
            for key, val in h5f[SLOPE_BIAS_META_GROUP].attrs.items():
                if isinstance(val, bytes):
                    val = val.decode()
                elif isinstance(val, np.generic):
                    val = val.item()
                meta[key] = val
            return meta
        if 'meta/input_args' in h5f:
            args = h5f['meta/input_args'].attrs
            for arg, source in [('DEM_slope_bias_scale', 'DEM'), ('slope_bias_from_z0', 'fit')]:
                if arg in args:
                    return {'source': source, 'E_geo': None, 'E_rate': None, 't_epoch': None,
                            'slopes': None, 'inferred': True}
    return None


def _slope_bias_meta_differences(saved, current):
    # how the expected offsets of a saved record differ from current ones
    out = []
    E_saved = json.loads(saved['E_geo']) if saved.get('E_geo') else None
    E_current = json.loads(current['E_geo'])
    if E_saved is None:
        out += ['the geolocation errors (E_geo) are not recorded']
    elif set(E_saved) != set(E_current) or \
            any(not np.isclose(E_saved[key], E_current[key]) for key in E_current):
        out += [f'E_geo was {E_saved}; this fit has {E_current}']
    E_rate = saved.get('E_rate')
    if E_rate is None:
        if current['E_rate']:
            out += ['E_rate is not recorded']
    elif not np.isclose(E_rate, current['E_rate']):
        out += [f'E_rate was {E_rate}; this fit has {current["E_rate"]}']
    elif current['E_rate'] and not np.isclose(saved.get('t_epoch', np.nan), current['t_epoch']):
        out += [f't_epoch was {saved.get("t_epoch")}; this fit has {current["t_epoch"]}']
    return out


def recompute_expected_offsets(data, E_geo=None, E_rate=0., t_epoch=None, sensor_dict=None):
    """set E_scale_z_x and E_scale_z_y of data from expected_offsets (the slopes are not changed)"""
    E = expected_offsets(data.sensor, data.time, E_geo=E_geo, E_rate=E_rate, t_epoch=t_epoch,
                         sensor_dict=sensor_dict)
    data.assign({'E_scale_z_x': E, 'E_scale_z_y': E.copy()})


def check_slope_bias_fields(data, saved, source, E_geo=None, E_rate=0., t_epoch=None,
                            sensor_dict=None, recompute_E=False):
    """
    Check the slope-bias fields of reread data against this fit's options

    Data reread from a fit (e.g. for a matched fit or an error calculation)
    keep the slope-bias fields (z_x, z_y, E_scale_z_x, E_scale_z_y) of the
    fit that made them, so that the fit is continued.  This compares that
    fit's record (saved, from read_slope_bias_meta) with this fit's options.

    Parameters
    ----------
    data : pointCollection.data
        the reread data, with the slope-bias fields
    saved : dict or None
        the record of the fit that made the fields
    source : str
        where this fit takes its slopes from: 'fit' or 'DEM'
    E_geo, E_rate, t_epoch : optional
        this fit's expected offsets (see slope_bias_meta)
    sensor_dict : dict, optional
        sensor names by number, to classify the sensors
    recompute_E : bool, optional
        if True, and the expected offsets differ from this fit's (or are
        not recorded), E_scale_z_x and E_scale_z_y are recomputed from this
        fit's options (the slopes are kept).  Otherwise the saved values are
        used, and a message says so.

    Raises
    ------
    ValueError
        if the fields were made from the other source (from a DEM file when
        this fit takes slopes from the fit, or the reverse)

    Returns
    -------
    dict or None
        the record for this fit's output: saved (updated if the expected
        offsets were recomputed), or None if there is no record
    """
    current = slope_bias_meta(source, E_geo=E_geo, E_rate=E_rate, t_epoch=t_epoch)
    if saved is None or saved.get('source') is None:
        print("check_slope_bias_fields: the slope-bias fields (z_x, z_y) of the reread data have no "
              "record of how they were made (a file written before Nov 2026): using them as they are. "
              "Their slopes may not include the fine grids or the reference DEM, and may come from a "
              "DEM file (--DEM_slope_bias_scale) rather than from the fit.")
        if recompute_E:
            recompute_expected_offsets(data, E_geo=E_geo, E_rate=E_rate, t_epoch=t_epoch,
                                       sensor_dict=sensor_dict)
            print(f"check_slope_bias_fields: recomputed the expected offsets (E_scale_z_x, E_scale_z_y): "
                  f"E_geo={current['E_geo']}, E_rate={current['E_rate']}")
            return current | {'slopes': 'not recorded'}
        return None
    if saved['source'] != source:
        raise ValueError(f"the slope-bias fields of the reread data were made from "
                         f"{SLOPE_BIAS_SOURCES[saved['source']]}, but this fit takes them from "
                         f"{SLOPE_BIAS_SOURCES[source]}: rerun the fit that wrote the file, or use "
                         f"the option of that fit")
    if saved.get('inferred'):
        print(f"check_slope_bias_fields: the slope-bias fields of the reread data come from "
              f"{SLOPE_BIAS_SOURCES[source]} (from the arguments of the fit that wrote them; "
              f"the file was written before Nov 2026 and has no record of how they were made, so "
              f"their slopes may not include the fine grids or the reference DEM)")
    differences = _slope_bias_meta_differences(saved, current)
    if len(differences) == 0:
        return saved
    if recompute_E:
        recompute_expected_offsets(data, E_geo=E_geo, E_rate=E_rate, t_epoch=t_epoch,
                                   sensor_dict=sensor_dict)
        print("check_slope_bias_fields: recomputed the expected offsets (E_scale_z_x, E_scale_z_y) "
              "from this fit's options (" + '; '.join(differences) + ")")
        return saved | {key: current[key] for key in ['E_geo', 'E_rate', 't_epoch']}
    print("check_slope_bias_fields: using the expected offsets (E_scale_z_x, E_scale_z_y) of the "
          "reread data, not this fit's options (" + '; '.join(differences) + "); "
          "--slope_bias_recompute_E recomputes them")
    return saved
