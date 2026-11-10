#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_DEM_bias.py

Estimate jitter (along-track), stripe (cross-track), and planar bias
parameters for a single DEM by comparing it directly against overlapping
ICESat-2 elevations.

@author: ben
"""
import sys
import os

if __name__ == '__main__':
    # run as a script, set the thread counts before numpy loads.  (The
    # fit_DEM_bias.py command gets them from altimetryFit/__init__.py, and an
    # import of this module should not change them.)
    from altimetryFit.set_threads import set_threads
    set_threads()

import json
import numpy as np
import h5py
import scipy.sparse as sp
import sparseqr
import LSsurf as LS
import pointCollection as pc

from altimetryFit.read_optical import find_gI_files
from altimetryFit.read_ICESat2 import read_ICESat2
from altimetryFit.fit_altimetry import get_SRS_proj4
from altimetryFit.bias_functions.DEM_jitter_fit import (
    get_ATC_xform, setup_DEM_jitter_fit, parse_jitter_stripe_bias_grids,
    parse_DEM_bias_params_file)


# Defaults mirror default_args/DEM_jitter_bias_params.txt.  By default the
# jitter fit skips its own plane (jitter_skip_plane=1), so the plane is fit
# as part of the stripe term instead.
DEFAULT_JITTER_PARAMS = {
    'res': 500.,
    'expected_rms_bias': 5.,
    'expected_rms_grad': 2.5e-3,
    'skip_plane': 1.,
}
DEFAULT_STRIPE_PARAMS = {
    'res': 500.,
    'expected_rms_bias': 1.,
    'expected_rms_grad': 2e-4,
    'skip_plane': 0.,

}


def read_DEM(DEM_file):
    """
    Read a DEM geotiff, splitting a 2-band [z, sigma] raster if present.
    """
    DEM = pc.grid.data().from_geotif(DEM_file)
    DEM_sigma = None
    if np.ndim(DEM.z) == 3:
        DEM_sigma = DEM.z[:, :, 1]
        DEM.z = DEM.z[:, :, 0]
    return DEM, DEM_sigma


def read_overlapping_ICESat2(DEM, geoIndex_source_file, hemisphere=1,
                              max_dt=1.5, blockmedian_scale=100.,
                              DEM_time=None):
    """
    Read ICESat-2 data overlapping a DEM, using a geoIndex_source_file
    (the same JSON format used by fit_altimetry's GeoIndex_source_file).
    """
    XR = np.array([np.nanmin(DEM.x), np.nanmax(DEM.x)])
    YR = np.array([np.nanmin(DEM.y), np.nanmax(DEM.y)])
    xy0 = [np.mean(XR), np.mean(YR)]
    W = {'x': np.diff(XR)[0], 'y': np.diff(YR)[0]}

    with open(geoIndex_source_file, 'r') as fh:
        geoIndex_dict = json.load(fh)
    gI_files = find_gI_files(geoIndex_dict['ICESat2'])

    t_range = None
    if DEM_time is not None and max_dt is not None:
        t_range = [DEM_time - max_dt, DEM_time + max_dt]

    D_list = read_ICESat2(xy0, W, gI_files, sensor=2,
                           SRS_proj4=get_SRS_proj4(hemisphere),
                           cplx_accept_threshold=0.25,
                           blockmedian_scale=blockmedian_scale,
                           t_range=t_range)
    D_list = [Di for Di in D_list if Di is not None]
    if len(D_list) == 0:
        return None
    return pc.data().from_list(D_list)


def compute_residuals(D, DEM, DEM_sigma=None, sigma_corr=1.0, max_abs_r0=50.):
    """
    Compute elevation residuals (r0 = z_IS2 - z_DEM) and a combined sigma
    for each ICESat-2 point, dropping points outside the DEM or with
    unreasonably large residuals.
    """
    D.assign({'z_dem': DEM.interp(D.x, D.y)})
    D.index(np.isfinite(D.z_dem))

    if DEM_sigma is not None:
        sigma_dem_grid = pc.grid.data().from_dict({'x': DEM.x, 'y': DEM.y, 'z': DEM_sigma})
        sigma_dem = sigma_dem_grid.interp(D.x, D.y)
        sigma_dem[~np.isfinite(sigma_dem)] = 0.
    else:
        sigma_dem = np.zeros(D.size)

    D.assign({'r0': D.z - D.z_dem})
    D.assign({'sigma': np.sqrt(D.sigma**2 + D.sigma_corr**2 + sigma_dem**2 + sigma_corr**2)})
    D.index(np.isfinite(D.r0) & (np.abs(D.r0) < max_abs_r0))
    D.assign({'sensor': np.zeros(D.size)})
    return D


def setup_jitter_stripe(D, DEM_file, xform, jitter_params, stripe_params):
    """
    Set up the combined jitter (along-track) + stripe (cross-track) fitting
    matrix and constraints for a single DEM.
    """
    Gd, jitter_constraints, _, jitter_grid, xy_atc, poly = setup_DEM_jitter_fit(
        D, DEM_file, col_0=0, sensor=0, xform=xform, params={}, **jitter_params)

    Gd_stripe, stripe_constraints, _, stripe_grid, _, _ = setup_DEM_jitter_fit(
        D, DEM_file, col_0=Gd.col_N, sensor=0, xform=xform, params={},
        local_coord_index=1, coord_name='y_atc', fit_name='stripe',
        **stripe_params)

    Gd.add(Gd_stripe)

    constraint_list = jitter_constraints + stripe_constraints
    grids = {jitter_grid.name: jitter_grid, stripe_grid.name: stripe_grid}
    has_plane = {
        jitter_grid.name: not (jitter_params.get('skip_plane', 0) > 0),
        stripe_grid.name: not (stripe_params.get('skip_plane', 0) > 0),
    }
    return Gd, constraint_list, grids, has_plane, poly


def robust_solve(D, Gd, constraint_list, n_iter=3):
    """
    Iteratively solve for the bias model, downweighting outliers using
    LSsurf.calc_sigma_extra and 3-sigma editing (following
    est_DEM_jitter_AT.est_along_track_jitter).
    """
    Gc = LS.lin_op().vstack(constraint_list)
    G_full = LS.lin_op().vstack([Gd, Gc])
    Gd_full = Gd.toCSR()

    valid_data = np.ones(D.size, dtype=bool)
    sigma_extra = 0.
    m = None
    b_est = np.zeros(D.size)
    for _ in range(n_iter):
        N_data_rows = G_full.shape[0] - np.sum(valid_data == 0)
        TCinv = sp.diags(np.concatenate([1/D.sigma[valid_data], 1/Gc.expected]),
                          shape=[N_data_rows, N_data_rows], offsets=0)
        keep = np.ones(G_full.shape[0], dtype=bool)
        keep[0:len(valid_data)] = valid_data

        m = sparseqr.solve(TCinv @ G_full.toCSR()[keep, :],
                            TCinv @ np.concatenate([D.r0[valid_data], np.zeros(Gc.shape[0])]))

        b_est = Gd_full @ m
        sigma_extra = LS.calc_sigma_extra(D.r0 - b_est, D.sigma, valid_data)
        valid_data = np.abs(D.r0 - b_est) < 3 * np.sqrt(D.sigma**2 + sigma_extra**2)

    return m, valid_data, b_est, sigma_extra


def extract_plane_params(m, grids, has_plane):
    """
    Extract the planar bias terms (x-slope [m/km], y-slope [m/km], offset
    [m]) for each grid for which a plane was fit.
    """
    plane_params = {}
    for name, grid in grids.items():
        if has_plane.get(name, False):
            plane_params[name] = {
                'x_slope': float(m[grid.col_N]),
                'y_slope': float(m[grid.col_N + 1]),
                'offset': float(m[grid.col_N + 2]),
            }
    return plane_params


def assign_bias_fields(D, xform, bias_grids, plane_params):
    """
    Assign per-point jitter, stripe, and planar bias contributions
    (dz_jitter, dz_stripe, dz_plane) to D, evaluated from the fitted
    bias grids and plane parameters.
    """
    xy_atc = (np.c_[D.x, D.y] - xform['origin']) @ xform['basis_vectors']

    D.assign({'dz_jitter': np.zeros(D.size)})
    D.assign({'dz_stripe': np.zeros(D.size)})
    D.assign({'dz_plane': np.zeros(D.size)})

    for name, grid in bias_grids.items():
        if 'jitter' in name:
            D.dz_jitter[:] = np.interp(xy_atc[:, 0], grid.x, grid.z)
        elif 'stripe' in name:
            D.dz_stripe[:] = np.interp(xy_atc[:, 1], grid.x, grid.z)

    # the planes (e.g. one each for the jitter and stripe fits) add
    for params in plane_params.values():
        D.dz_plane[:] += (params['x_slope']*(D.x - xform['origin'][0])/1000.
                         + params['y_slope']*(D.y - xform['origin'][1])/1000.
                         + params['offset'])

    return D


def _jsonify(obj):
    if isinstance(obj, dict):
        return {key: _jsonify(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(val) for val in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def write_output(out_file, DEM_file, bias_grids, summary):
    """
    Write the jitter/stripe bias grids to out_file (h5), in the same
    /jitter_bias/sensor_<N>_<jitter|stripe>_<x|y>_atc format read by
    altimetryFit.eval_model.read_DEM_biases, and write a companion .json
    file with the fit summary.
    """
    if os.path.isfile(out_file):
        os.remove(out_file)

    for name, grid in bias_grids.items():
        group = f'/jitter_bias/{name}'
        attributes = {group: {
            'xform_basis_vectors': np.asarray(grid.xform['basis_vectors']),
            'xform_origin': np.asarray(grid.xform['origin']),
        }}
        grid.to_h5(out_file, group=group, attributes=attributes)

    with h5py.File(out_file, 'a') as h5f:
        h5f.require_group('meta/sensors')
        h5f['meta/sensors'].attrs['sensor_0'] = DEM_file
        # eval_model.read_DEM_biases unconditionally reads '/bias' and
        # '/slope_bias'; write empty groups so it can be called on this
        # file's output directly.
        h5f.create_dataset('/bias/ID', data=np.array([], dtype=int))
        h5f.create_dataset('/bias/val', data=np.array([], dtype=float))
        h5f.create_dataset('/slope_bias/sensors', data=np.array([], dtype=int))
        h5f.create_dataset('/slope_bias/x_slope', data=np.array([], dtype=float))
        h5f.create_dataset('/slope_bias/y_slope', data=np.array([], dtype=float))

    json_file = os.path.splitext(out_file)[0] + '.json'
    with open(json_file, 'w') as fh:
        json.dump(_jsonify(summary), fh, indent=2)


def fit_DEM_bias(DEM_file, geoIndex_source_file, DEM_bias_params_file=None,
                 hemisphere=1, max_dt=1.5, sigma_corr=1.0,
                 blockmedian_scale=100., url_list_file=None, out_file=None):
    """
    Estimate jitter, stripe, and planar bias parameters for a single DEM.

    Parameters
    ----------
    DEM_file : str
        DEM geotiff filename.
    geoIndex_source_file : str
        JSON file mapping data-type names to geoIndex glob patterns (must
        contain an 'ICESat2' entry), as used by fit_altimetry.py.
    DEM_bias_params_file : str, optional
        key=value file giving jitter_*/stripe_* parameters for
        setup_DEM_jitter_fit (see default_args/DEM_jitter_bias_params.txt).
        If not given, DEFAULT_JITTER_PARAMS/DEFAULT_STRIPE_PARAMS are used.
    hemisphere : int, optional
        1 for Greenland, -1 for Antarctica.  The default is 1.
    max_dt : float, optional
        ICESat-2 data are restricted to within this many years of the DEM's
        acquisition date.  The default is 1.5.
    sigma_corr : float, optional
        Additional uncertainty (m) added in quadrature to the ICESat-2/DEM
        residuals.  The default is 1.0.
    blockmedian_scale : float, optional
        Blockmedian scale (m) applied to the ICESat-2 data.  The default is 100.
    url_list_file : str, optional
        File giving the PGC url for each DEM filename, used to compute the
        along-track/cross-track transform.  If not given, falls back to the
        'jitter_url_list_file'/'stripe_url_list_file' entry in
        DEM_bias_params_file.
    out_file : str, optional
        If given, write the bias grids to this h5 file (in the format read
        by altimetryFit.eval_model.read_DEM_biases) and a companion .json
        file with the fit summary.

    Returns
    -------
    result : dict
        'bias_grids' : dict of pc.grid.data, keyed by
            'sensor_0_jitter_x_atc' / 'sensor_0_stripe_y_atc'.
        'summary' : dict of fit-quality statistics, plane parameters, and
            the along-track/cross-track transform.
        'data' : pc.data, the ICESat-2/DEM residual data used in the fit,
            with per-point bias contributions assigned to dz_jitter,
            dz_stripe, and dz_plane.
        'valid' : boolean array, the data used in the final iteration.
        'model' : numpy array, the full model-parameter vector.
    """
    if DEM_bias_params_file is not None:
        params = parse_DEM_bias_params_file(DEM_bias_params_file)
    else:
        params = {}

    jitter_params = {**DEFAULT_JITTER_PARAMS,
                      **{key.replace('jitter_', ''): val
                         for key, val in params.items() if 'jitter' in key}}
    parsed_stripe_params = {key.replace('stripe_', ''): val
                             for key, val in params.items() if 'stripe' in key}
    stripe_params = {**DEFAULT_STRIPE_PARAMS, **parsed_stripe_params}
    if not parsed_stripe_params and not (jitter_params.get('skip_plane', 0) > 0):
        # the jitter fit already includes a plane; don't fit a second one
        stripe_params['skip_plane'] = 1.

    if url_list_file is None:
        url_list_file = jitter_params.get('url_list_file', stripe_params.get('url_list_file'))
    jitter_params.pop('url_list_file', None)
    stripe_params.pop('url_list_file', None)

    DEM, DEM_sigma = read_DEM(DEM_file)
    DEM_time = pc.grid.DEM_year(DEM_file)

    D = read_overlapping_ICESat2(DEM, geoIndex_source_file, hemisphere=hemisphere,
                                  max_dt=max_dt, blockmedian_scale=blockmedian_scale,
                                  DEM_time=DEM_time)
    if D is None or D.size == 0:
        raise ValueError(f"fit_DEM_bias: no ICESat-2 data found overlapping {DEM_file}")

    D = compute_residuals(D, DEM, DEM_sigma=DEM_sigma, sigma_corr=sigma_corr)
    if D.size == 0:
        raise ValueError(f"fit_DEM_bias: no valid ICESat-2/DEM residuals for {DEM_file}")

    xform, poly = get_ATC_xform(DEM_file, url_list_file, params={})

    Gd, constraint_list, grids, has_plane, _ = setup_jitter_stripe(
        D, DEM_file, xform, jitter_params, stripe_params)

    m, valid_data, b_est, sigma_extra = robust_solve(D, Gd, constraint_list)

    bias_grids = parse_jitter_stripe_bias_grids(m, Gd, grids)
    plane_params = extract_plane_params(m, grids, has_plane)
    D = assign_bias_fields(D, xform, bias_grids, plane_params)

    summary = {
        'DEM_file': DEM_file,
        'DEM_time': DEM_time,
        'N_data': int(np.sum(valid_data)),
        'R0': float(pc.RDE(D.r0[valid_data])),
        'sigma_uncorr': float(np.std(D.r0[valid_data])),
        'R': float(pc.RDE((D.r0 - b_est)[valid_data])),
        'sigma_corr': float(np.std((D.r0 - b_est)[valid_data])),
        'sigma_extra': float(np.median(sigma_extra)),
        'plane_params': plane_params,
        'xform': {'origin': np.asarray(xform['origin']),
                  'basis_vectors': np.asarray(xform['basis_vectors'])},
        'poly': poly,
    }

    if out_file is not None:
        write_output(out_file, DEM_file, bias_grids, summary)

    return {'bias_grids': bias_grids,
            'summary': summary,
            'data': D,
            'valid': valid_data,
            'model': m}


def parse_inputs(argv):
    import argparse
    parser = argparse.ArgumentParser(
        description='Estimate jitter, stripe, and planar bias parameters for a single DEM.',
        fromfile_prefix_chars='@')
    parser.add_argument('filename', type=str, help='DEM filename')
    parser.add_argument('--geoIndex_source_file', '-g', type=str, required=True,
                         help='JSON file giving geoIndex glob patterns for each data type (must include an "ICESat2" entry)')
    parser.add_argument('--DEM_grid_bias_params_file', type=str,
                         help='file containing key=value DEM bias parameters (jitter_*/stripe_* prefixes)')
    parser.add_argument('--url_list_file', '-u', type=str,
                         help='File containing the PGC url for each DEM filename')
    parser.add_argument('--Hemisphere', '-H', type=int, default=1,
                         help='hemisphere: -1=Antarctica, 1=Greenland')
    parser.add_argument('--max_dt', type=float, default=1.5,
                         help='ICESat-2 data are restricted to within this many years of the DEM acquisition date')
    parser.add_argument('--sigma_corr', type=float, default=1.0,
                         help='additional uncertainty (m) added in quadrature to the ICESat-2/DEM residuals')
    parser.add_argument('--blockmedian_scale', type=float, default=100.,
                         help='blockmedian scale (m) for the ICESat-2 data')
    parser.add_argument('--out_file', '-o', type=str,
                         help='output h5 file (a companion .json file with the fit summary is also written)')
    parser.add_argument('--THREADS', type=int, default=1, help='number of threads to use')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_inputs(argv if argv is not None else sys.argv[1:])
    result = fit_DEM_bias(args.filename, args.geoIndex_source_file,
                           DEM_bias_params_file=args.DEM_grid_bias_params_file,
                           hemisphere=args.Hemisphere,
                           max_dt=args.max_dt,
                           sigma_corr=args.sigma_corr,
                           blockmedian_scale=args.blockmedian_scale,
                           url_list_file=args.url_list_file,
                           out_file=args.out_file)
    print(json.dumps(_jsonify(result['summary']), indent=2))


if __name__ == '__main__':
    main()
