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
January 2026: Lagrangian update functions moved to lagrangian_functions.py
"""

from altimetryFit.lagrangian_functions import setup_lagrangian, \
    update_data_for_lagrangian, \
    advect_mask, \
    update_output_data_for_lagrangian,\
    setup_lagrangian_fit,\
    parse_lagrangian_fit
import json
import glob
import h5py
import resource
import os
import sys
import re
import altimetryFit as aF

# The BLAS/OpenMP thread-count environment variables are set from --THREADS
# in altimetryFit/__init__.py (see set_threads.py), before numpy is imported.


#import warnings
#warnings.simplefilter('error')

import numpy as np
from LSsurf.smooth_fit import smooth_fit
from LSsurf import fd_grid
from LSsurf.grid_functions import resolve_mask_roles
from altimetryFit.reread_data_from_fits import reread_data_from_fits
import pointCollection as pc
try:
    from pyTMD import compute_tide_corrections
except Exception:
    pass
try:
    from SMBcorr import assign_firn_variable
except Exception:
    pass
from altimetryFit import SMB_corr_from_grid
from altimetryFit.read_optical import read_optical_data, laser_key
from altimetryFit.decimation_mask import read_no_decimation_mask, in_no_decimation_mask
from altimetryFit.read_DEM_data import regen_DEM_meta
from altimetryFit.read_CS2_data import read_CS2_data, radar_key
from altimetryFit.apply_tides import apply_tides
from altimetryFit.mask_functions import mask_data_by_year, setup_gzone_relax, \
    make_default_mask, read_named_masks, mask_is_empty


def set_memory_limit(max_bytes):
    '''
    Limit the amount of memory the program can use.
    '''
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, hard))


def make_sensor_dict(h5_file, data=None):
    '''
    Make a dictionary matching sensor number to bias id and source.

    Input: h5f: h5py file containing the fit output
    Output: dict giving the sensor for each number
    '''
    this_sensor_dict=dict()
    with h5py.File(h5_file,'r') as h5f:
        if '/meta/sensors/' not in h5f:
            return this_sensor_dict

        sensor_re=re.compile(r'sensor_(\d+)')
        for sensor_key, sensor in h5f['/meta/sensors/'].attrs.items():
            sensor_num=int(sensor_re.search(sensor_key).group(1))
            if data is not None and not np.isin(sensor_num, data.sensor):
                continue
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
    try:
        # day 731891 has bad ICESat elevations
        bad=data.day==731891
        # bad ICESat data for 2003.782
        bad=np.abs(data.time - 2003.7821) < 0.1/24/365.25
        data.index(bad==0)
    except AttributeError:
        print("fit_altimetry.custom_data_edits: Expected data fields not found, skipping")


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


def save_fit_to_file(S,  filename, sensor_dict = None, dzdt_lags = None,
                     reference_epoch = 0,
                     args = None):
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
            if S['E_RMS'][key] is None:
                val=np.array([np.nan])
            else:
                val=S['E_RMS'][key]
            h5f.create_dataset('E_RMS/'+key, data=val)
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
        if 'jitter_bias' in S['m'] and 'jitter_bias' not in h5f:
            h5f.create_group('/jitter_bias')

        def safer_JSON_encoder(obj):
            try:
                return json.JSONEncoder().default(obj)
            except TypeError:
                return(str(obj))
        if args is not None:
            if 'input_args' not in h5f['/meta/']:
                h5f.create_group('/meta/input_args')
                for name, val in args.__dict__.items():
                    if val is not None:
                        if isinstance(val, dict):
                            val=json.dumps(val, default= safer_JSON_encoder)
                        try:
                            h5f['/meta/input_args'].attrs[name]=val
                        except Exception as e:
                            print(f"failed to write argument {name}")
                            print(e)
                            pass
    if 'sensor_bias_grids' in S['m']:
        for name, ds in S['m']['sensor_bias_grids'].items():
                ds.to_h5(filename, group='/grid_bias/'+name)
    if 'jitter_bias' in S['m']:
        for name, ds in S['m']['jitter_bias'].items():
                ds.to_h5(filename, group='/jitter_bias/'+name)
    for key, ds in S['m'].items():
        if isinstance(ds, pc.grid.data):
                ds.to_h5(filename, group=key)

def assign_sigma_corr(D, orbital_sensors=[-2, -1, 1, 2], airborne_sensors=[3, 3.5, 4, 5]):

    delta_t_corr={'airborne':60*10/(24*3600*365.25),
                  'orbital':10/(24*3600*365.25)}
    time_corr=D.time.copy()
    these=np.flatnonzero(np.isin(D.sensor, orbital_sensors))
    time_corr[these]=np.floor(D.time[these]/delta_t_corr['orbital'])*delta_t_corr['orbital']
    these=np.flatnonzero(np.isin(D.sensor, airborne_sensors))
    time_corr[these]=np.floor(D.time[these]/delta_t_corr['airborne'])*delta_t_corr['airborne']
    D.assign({'time_corr':time_corr})
    _, sensor_dict=pc.unique_by_rows(np.c_[D.sensor, D.time_corr], return_dict=True)
    for key, ind in sensor_dict.items():
        #skip DEMs
        if key[0] > np.max(orbital_sensors+airborne_sensors) or key[0] < 0:
            continue
        if 'slope_mag' in D.fields and np.any(np.isfinite(D.slope_mag[ind])):
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

def no_fit_file_name(out_name):
    # written by the fit, in place of out_name, when the tile has nothing to fit
    return os.path.splitext(out_name)[0] + '.no_data'

def write_no_fit_file(no_fit_file, reason):
    """
    Record that the tile had nothing to fit, and why.

    Queues run the fit and its --calc_error_for_xy step joined by ';', so the
    job's exit code is the error step's.  This file tells the error step that
    a missing tile file means no data (exit 0) rather than a failed fit (exit 1).
    """
    if no_fit_file is None:
        return
    with open(no_fit_file, 'w') as fh:
        fh.write(reason + '\n')

def fit_altimetry(xy0=None, Width=4e4, \
            reread_file=None,\
            data=None,\
            E_RMS={},\
            return_fit_objects=False,\
            d2z_dt2_scale_file=None,\
            constraint_scaling_files=None,
            t_span=[2003, 2020],\
            spacing={'z0':2.5e2, 'dz':5.e2, 'dt':0.5}, \
            DEM_slope_bias_scale=None, \
            hemisphere=1, reference_epoch=None, \
            reread_dirs=None,\
            max_iterations=5, \
            sigma_extra_relax = False, \
            dzdt_lags=[1, 4], \
            E_slope_bias=1.e-5, \
            sensor_dict={}, \
            out_name=None, \
            no_fit_file=None, \
            bias_nsigma_edit=None, \
            bias_nsigma_iteration=3,\
            prior_edge_args=None,\
            replace=False, \
            xy_bias_file=None,\
            no_decimation_mask_file=None,\
            tide_mask_keep_val=None,\
            tide_mask_keep_buffer=None,\
            firn_fixed=False, firn_rescale=False, \
            firn_correction=None, firn_directory=None, firn_version=None,\
            firn_grid_file=None,\
            firn_mask_file=None,\
            calc_FAC_anomaly=True,\
            GI_files=None,\
            DEM_t_buffer=None,\
            geoid_file=None,\
            mask_file=None, \
            masks=None, \
            grid_masks=None, \
            data_mask=None, \
            lagrangian_grid_masks=False, \
            lagrangian_mask_spacing=None, \
            erode_source_mask=True,\
            mask_zero_scale=10,\
            and_mask_files=None, \
            or_mask_files=None, \
            DEM_file=None,\
            DEM_tol=None,\
            data_sigma_max=None,\
            problem_DEM_files=None,\
            water_mask_threshold=None, \
            bm_scale=None,
            N_target=None,\
            calc_error_file=None, \
            calc_error=False,\
            extra_error=None,\
            repeat_res=None,\
            tide_mask_file=None,\
            tide_mask_shelf_val=None,\
            gzone_relax_scale=None,\
            error_res_scale=None,\
            avg_mask_directory=None, \
            velocity_files=None, \
            lagrangian_dict=None, \
            tide_directory=None, \
            tide_model='CATS2008', \
            tide_adjustment_file=None,
            year_mask_dir=None, \
            avg_scales=None,\
            bias_params=['time_corr','sensor','spot'],\
            seg_diff_tol=4,\
            verbose=True,\
            reference_DEM_file = None,\
            reference_DEM_uncertainty = None,\
            reference_DEM_offset = 0, \
            reference_DEM_add_geoid=False,
            DEM_dt_min=None,\
            converge_tol_frac=0.005,\
            params=None,\
            DEM_grid_bias_params=None,\
            DEM_sigma_corr=20,\
            solver='cholmod',\
            cholmod_scale_columns=True,\
            cholmod_cache_ordering=False,\
            fine_grid_args=None,\
            THREADS=1):
    """
        Wrapper for smooth_fit that can find data and set the appropriate parameters

        Masks:
            The mask built from mask_file (or all ones), tide_mask_keep_val, and
            and_mask_files/or_mask_files is named 'default'.  Other masks can be
            given in masks, a dictionary mapping mask names to pointCollection.grid.data
            objects or to mask files (1 inside the mask).  Masks are assigned roles with:
                grid_masks: dictionary mapping grid names ('z0', 'dz') to mask names
                data_mask: name of the mask used to select data.  Data are used
                    where this mask is greater than 0.5 at the location (and, for a
                    mask with a time dimension, the time) of the measurement.  For
                    full-lagrangian fits, this is the location before advection.
            Roles that are not assigned use the default mask.  If no roles are
            assigned, the default mask is used for everything, as it was before
            roles were introduced.

            In a full-lagrangian fit, two more masks can be used for the grids,
            made by advecting the nodes of the (lagrangian) model grid through
            the time span of the fit and sampling the data mask at the parcel
            positions (see lagrangian_functions.advect_mask):
                'advected_any': parcels that are inside the data mask at any
                    time during the fit
                'advected': (time-varying) parcels that are inside the data
                    mask at each epoch
            lagrangian_grid_masks=True is short for
            grid_masks={'z0':'advected_any', 'dz':'advected'} (grids assigned
            in grid_masks keep their assignment).  For a floating-ice data mask,
            the model includes every parcel that floats at some time during the
            fit, and the dz mask shows when each parcel is grounded.
            lagrangian_mask_spacing sets the spacing of the advected masks
            (default: half the dz spacing).
    """
    print("fit_OIB: working on %s" % out_name)

    # default:
    lagrangian_coords = None
    sensor_grid_bias_params = []
    SRS_proj4=get_SRS_proj4(hemisphere)
    bias_model_args={}

    # set defaults for E_RMS, then update with input parameters
    E_RMS0={'d2z0_dx2':200000./3000/3000, 'd3z_dx2dt':3000./3000/3000, 'd2z_dxdt':3000/3000, 'd2z_dt2':5000}
    E_RMS0.update(E_RMS)

    W={'x':Width, 'y':Width,'t':np.diff(t_span)}
    ctr={'x':xy0[0], 'y':xy0[1], 't':np.mean(t_span)}

    bds={ dim: c_i+np.array([-0.5, 0.5])*W[dim]  for dim, c_i in ctr.items()}

    if bias_params is not None and len(bias_params)==1:
        if ',' in bias_params[0]:
            bias_params=bias_params[0].split(',')
    if out_name is not None:
        try:
            out_name=out_name %(xy0[0]/1000, xy0[1]/1000)
        except:
            pass

    if calc_error_file is not None:
        reread_file=calc_error_file
        calc_error=True
        max_iterations=0
        repeat_res=None

    pad=np.array([-1.e4, 1.e4])
    # the default mask
    default_mask_args = {'mask_file':mask_file,
                         'spacing':spacing['dz'],
                         'tide_mask_keep_val':tide_mask_keep_val,
                         'tide_mask_keep_buffer':tide_mask_keep_buffer,
                         'tide_mask_file':tide_mask_file,
                         'and_mask_files':and_mask_files,
                         'or_mask_files':or_mask_files}
    mask_data = make_default_mask([bds['x']+pad, bds['y']+pad], **default_mask_args)

    # masks can be assigned to the grids and to data selection (see the docstring)
    advected_mask_names = ('advected', 'advected_any')
    if lagrangian_grid_masks:
        grid_masks = {'z0':'advected_any', 'dz':'advected', **(grid_masks or {})}
    mask_roles_assigned = grid_masks is not None or data_mask is not None
    if masks is not None:
        reserved = [name for name in masks if name in ('default',) + advected_mask_names]
        if reserved:
            raise ValueError(f"the mask names {reserved} are reserved: 'default' is the mask built from mask_file, "
                             "tide_mask_keep_val, and and_mask_files/or_mask_files, and "
                             f"{advected_mask_names} are made by advecting the data mask")
        if not mask_roles_assigned:
            print("fit_altimetry: masks were given, but none is assigned to a grid or to data selection "
                  "(see grid_masks and data_mask), so the default mask will be used for everything")

    constraint_scaling_maps={}
    if d2z_dt2_scale_file is not None:
        constraint_scaling_maps['d2z_dt2']=pc.grid.data()\
            .from_file(d2z_dt2_scale_file, bounds=[bds['x']+pad, bds['y']+pad])
    if constraint_scaling_files is not None:
        constraint_scaling_files=constraint_scaling_files.split(',')
        for temp in constraint_scaling_files:
            temp=temp.split(':')
            constraint_scaling_maps[temp[0]]=pc.grid.data()\
                .from_file(temp[1], bounds=[bds['x']+pad, bds['y']+pad])
    if gzone_relax_scale is not None:
        setup_gzone_relax( gzone_relax_scale,
                              tide_mask_file,
                              bds,
                              spacing,
                              constraint_scaling_maps,
                              tide_mask_shelf_val = tide_mask_shelf_val)

    # if the masks have roles, the masks are checked after the lagrangian setup
    if not mask_roles_assigned and np.all(mask_data.z==0):
        if verbose:
            print("no points in mask, returning")
        write_no_fit_file(no_fit_file, "no points in mask")
        return None, None, None

    if lagrangian_dict is not None:
        lagrangian_dict = setup_lagrangian(
            SRS_proj4=SRS_proj4, xy0=xy0, Wxy=Width,
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
            sensor_grid_bias_params.append({
                'name' : 'lagrangian_dz',
                'sensor' : None,
                'coords' : lagrangian_coords,
                'setup_function' : setup_lagrangian_fit,
                'parsing_function' : parse_lagrangian_fit
                })
        else:
            # move points by default
            lagrangian_dict['move_points'] = True
            DEM_grid_bias_params['coords'] = ['x_original','y_original']
    full_lagrangian = lagrangian_dict is not None and lagrangian_dict['move_points']

    # the mask used to select data.  Without roles, this is the default mask
    data_selection_mask = mask_data
    if mask_roles_assigned:
        mask_bounds = [bds['x']+pad, bds['y']+pad]
        if full_lagrangian:
            # data are selected where they were measured, which can be anywhere
            # that the tile's parcels were during the fit.  The bounds grow by
            # whole multiples of the dz spacing, so that an all-ones default mask
            # has the same nodes as it does without roles
            dx = spacing['dz']
            mask_bounds = [mb + np.array([-1, 1])*dx*np.ceil(np.maximum(0, [mb[0]-(lb[0]+pad[0]), lb[1]+pad[1]-mb[1]])/dx)
                           for mb, lb in zip(mask_bounds, lagrangian_dict['bds'])]
            mask_data = make_default_mask(mask_bounds, **default_mask_args)
        mask_dict = {'default':mask_data}
        if masks is not None:
            mask_dict.update(read_named_masks(masks, mask_bounds))
        if any(name in advected_mask_names for name in (grid_masks or {}).values()):
            if not full_lagrangian:
                raise ValueError(f"the masks {advected_mask_names} can only be used in full-lagrangian fits")
            data_mask_name = 'default' if data_mask is None else data_mask
            if data_mask_name not in mask_dict:
                raise ValueError(f'data mask "{data_mask_name}" is not in the masks, which are {list(mask_dict)}')
            mask_dict['advected'], mask_dict['advected_any'] = advect_mask(
                mask_dict[data_mask_name], t_span=t_span, spacing=spacing,
                mask_spacing=lagrangian_mask_spacing, verbose=verbose, **lagrangian_dict)
        if data_mask in advected_mask_names:
            raise ValueError(f"the masks {advected_mask_names} cannot be used as the data mask")
        role_names = resolve_mask_roles(mask_dict, grid_masks=grid_masks, data_mask=data_mask)
        if verbose:
            print("fit_altimetry: masks: " + ', '.join([f'{role}:{name}' for role, name in role_names.items()]))
        data_selection_mask = mask_dict[role_names['data']]
        if mask_is_empty(data_selection_mask) or \
                (mask_is_empty(mask_dict[role_names['z0']]) and mask_is_empty(mask_dict[role_names['dz']])):
            if verbose:
                print("no points in mask, returning")
            write_no_fit_file(no_fit_file, "no points in mask")
            return None, None, None

    pgc_url_file=None
    default_data_fields=['x','y','z','time','sigma','sigma_corr',
                         'slope_mag', 'sensor','spot', 'rgt','cycle','BP']
    data_scale_fields = None
    if DEM_slope_bias_scale is not None:
        data_scale_fields=['z_x','z_y']
        default_data_fields += data_scale_fields
        default_data_fields += ['E_scale_'+jj for jj in data_scale_fields ]

    if data is not None:
        sensor_dict={}
    elif reread_file is not None:
        # get xy0 from the filename
        #re_match=re.compile('/E(.*)_N(.*).h5').search(reread_file)
        #xy0=[float(re_match.group(ii))*1000 for ii in [1, 2]]
        xy0=[ctr['x'], ctr['y']]
        data=pc.data().from_h5(reread_file, group='data')
        if lagrangian_dict is None or not lagrangian_dict['move_points']:
            #crop the data in case we're rereading subsets, skip if this is a full lagrangian fit.
            if verbose:
                print("fit_altimetry.py: cropping data to model domain")
            data.crop((xy0[0]+np.array([-1, 1])*W['x']/2, xy0[1]+np.array([-1, 1])*W['y']/2))
        sensor_dict=make_sensor_dict(reread_file, data)
        DEM_meta_dict={}
        if 'DEM' in GI_files:
            DEM_meta_dict = regen_DEM_meta(xy0, W, data, reread_file, GI_files['DEM'],
                        read_basis_vectors=True,
                        pgc_url_file = DEM_grid_bias_params['jitter_url_list_file'])
    elif reread_dirs is None:
        if lagrangian_dict is not None and lagrangian_dict['move_points']:
            this_xy0=[*map(np.mean, lagrangian_dict['bds'])]
            # scalar widths: numpy >= 2.5 won't convert the 1-element output of np.diff to a scalar
            this_W={'x':np.diff(lagrangian_dict['bds'][0])[0],
                           'y':np.diff(lagrangian_dict['bds'][1])[0],
                           't':W['t']}
        else:
            this_xy0, this_W = [xy0, W]

        if DEM_grid_bias_params is not None and 'jitter_url_list_file' in DEM_grid_bias_params:
            pgc_url_file=DEM_grid_bias_params['jitter_url_list_file']

        D, sensor_dict, DEM_meta_dict = read_optical_data(\
                                this_xy0, this_W,\
                                GI_files=GI_files, \
                                SRS_proj4=get_SRS_proj4(hemisphere),\
                                bm_scale=bm_scale,\
                                N_target=N_target,\
                                target_area=W['x']*W['y'],
                                mask_file=mask_file, \
                                geoid_file=geoid_file, \
                                water_mask_threshold=water_mask_threshold, \
                                time_range=t_span, \
                                DEM_file=DEM_file, \
                                DEM_dt_min=DEM_dt_min,\
                                DEM_t_buffer=DEM_t_buffer,\
                                DEM_slope_bias_scale=DEM_slope_bias_scale,\
                                data_sigma_max=data_sigma_max,\
                                pgc_url_file=pgc_url_file,\
                                DEM_sigma_corr=DEM_sigma_corr,\
                                hemisphere=hemisphere,
                                problem_DEM_files=problem_DEM_files,
                                seg_diff_tol=seg_diff_tol,
                                xy_bias_file=xy_bias_file,
                                no_decimation_mask=no_decimation_mask_file)

        if sensor_dict is None:
            sensor_dict={}

        if 'POCA' in GI_files:
            # read_optical_data returns D=None if it found no optical data
            if D is None:
                D=[]
            sensor_dict.update( {val:key for key,val in radar_key().items()})
            D += read_CS2_data(xy0, W, GI_files, apply_filters=True,
                DEM_file=DEM_file, dem_tol=DEM_tol, bias_lookup_dir=params['CS2']['bias_lookup_dir'],
                sensor_dict=radar_key(), POCA_sensor=radar_key()['CS2_POCA'])

        if D is None or all(Di is None for Di in D):
            if verbose:
                print("no data read, returning")
            write_no_fit_file(no_fit_file, "no data read")
            return None, None, None

        for ind, Di in enumerate(D):
            if Di is None:
                continue
            for field in ['rgt','cycle','spot']:
                if field not in Di.fields:
                    Di.assign({field:np.zeros_like(Di.x)+np.nan})

        data=pc.data(fields = default_data_fields).from_list(D)
        data.assign({'day':np.floor(data.time*365.25)})

        if no_decimation_mask_file is not None:
            # flag the data inside the no-decimation mask (evaluated, as in the readers, at the
            # measurement positions and times), so that they get their own sigma_extra
            no_decimation_mask = read_no_decimation_mask(no_decimation_mask_file, this_xy0, this_W)
            data.assign({'no_decimation':
                         in_no_decimation_mask(no_decimation_mask, data.x, data.y, data.time).astype(float)})

        if data.size==0:
            print("No data found, returning")
            write_no_fit_file(no_fit_file, "no data found")
            return None, None, None

        # apply the tides if a directory has been provided
        if tide_mask_file is not None:
            if lagrangian_dict is not None and lagrangian_dict['move_points']:
                this_Wxy = np.max([*map(np.diff, data.bounds())])+5000
                this_xy0 = [*map(np.mean, data.bounds())]
            else:
                this_Wxy, this_xy0 = [Width, xy0]

            apply_tides(data, this_xy0, this_Wxy,
                            tide_mask_file=tide_mask_file,
                            tide_mask_shelf_val = tide_mask_shelf_val,
                            tide_directory=tide_directory,
                            tide_model=tide_model,
                            tide_adjustment = (tide_adjustment_file is not None),
                            tide_adjustment_file = tide_adjustment_file,
                            extrapolate=True,
                            cutoff=200,
                            crs = SRS_proj4,
                            threads = THREADS)
    else:
        data, sensor_dict = reread_data_from_fits(xy0, Width, reread_dirs, template='E%d_N%d.h5')

    altimeter_sensors = [item for key, item in laser_key().items()]\
        + [item for key, item in radar_key().items()]

    DEM_sensors=np.array([key for key in sensor_dict.keys() if key not in altimeter_sensors ])

    if reference_epoch is None:
        reference_epoch=len(np.arange(t_span[0], t_span[1], spacing['dt']))

    if lagrangian_dict is not None:
        lagrangian_dict=update_data_for_lagrangian( data, mask_data=data_selection_mask,
            **lagrangian_dict)
        if data.size==0 or np.sum(np.isfinite(data.z))==0:
            if verbose:
                print("After lagrangian advection, no valid data remain")
            write_no_fit_file(no_fit_file, "no valid data after lagrangian advection")
            return None, None, None

    # make every dataset a double
    for field in data.fields:
        setattr(data, field, getattr(data, field).astype(np.float64))

    if (firn_fixed or firn_rescale) and reread_dirs is None and \
        calc_error_file is None and reread_file is None:
        if firn_grid_file is not None:
            if firn_correction=='MERRA2_hybrid':
                # defaults work here:
                SMB_corr_from_grid(data,
                                   model_file=os.path.join(firn_directory,firn_grid_file),
                                   calc_FAC_anomaly = calc_FAC_anomaly)
        else:
            assign_firn_variable(data, firn_correction, firn_directory, hemisphere,
                         model_version=firn_version, subset_valid=True)
        if firn_mask_file is not None:
            data.h_firn[pc.grid.data().from_geotif(firn_mask_file,
                                                   bounds=data.bounds(pad=5000))\
                                                    .interp(data.x, data.y)<0.01]=0
        if firn_fixed:
            data.z -= data.h_firn
    if firn_rescale:
        # the grid has one node at each domain corner.
        this_grid=fd_grid( [xy0[1]+np.array([-0.5, 0.5])*Width,
                xy0[0]+np.array([-0.5, 0.5])*Width], [Width, Width],\
             name=firn_correction+'_scale', srs_proj4=SRS_proj4)
        bias_model_args += [{'name': firn_correction+'_scale',  \
                            'param':'h_firn',\
                            'grid':this_grid,\
                            'expected_rms':0.25, 'expected_value':1.}]

    #report data counts
    for val, sensor in sensor_dict.items():
        if val < 5:
            print("for %s found %d data" %(sensor, np.sum(data.sensor==val)))
    print("for DEMs, found %d data" % np.sum(np.isin(data.sensor, np.array(DEM_sensors))))

    # Create a sensor_grid_bias_param structure for each sensor
    if DEM_grid_bias_params is not None:
        for sensor in DEM_sensors:
            sensor_grid_bias_params += [{'sensor':sensor, 'expected_val':0}]
            try:
                sensor_grid_bias_params[-1].update(DEM_grid_bias_params)
                sensor_grid_bias_params[-1].update(DEM_meta_dict[sensor])
                sensor_grid_bias_params[-1].update({'filename':sensor_dict[sensor]})
            except Exception as e:
                print(e)

    # apply any custom edits
    custom_edits(data)
    if data.size == 0:
        if verbose:
            print("fit_altimetry:No data found, returning")
        write_no_fit_file(no_fit_file, "no data found after custom edits")
        return None, None, None

    # assign time_corr and (for laser sensors) sigma_corr
    # (this is for ICESat and airborne sensors)
    if 'time_corr' in bias_params:
        assign_sigma_corr(data)

    avg_masks=None
    if avg_mask_directory is not None:
        avg_masks = {os.path.basename(file).replace('.tif',''):pc.grid.data().from_geotif(file) for file in \
                     glob.glob(avg_mask_directory+'/*.tif')}

    if year_mask_dir is not None:
        mask_data_by_year(data, year_mask_dir);

    #sigma_extra_masks = {'altimeter': np.isin(data.sensor, altimeter_sensors),
    #                     'DEM': ~np.isin(data.sensor, altimeter_sensors)}
    sigma_extra_keys={'altimeter':{'sensor':altimeter_sensors},
                      'DEM':{'sensor':DEM_sensors}}
    if 'no_decimation' in data.fields and np.any(data.no_decimation > 0):
        # data kept at full density (inside the no-decimation mask) have different
        # statistics from the decimated data, so they get their own sigma_extra:
        # sigma_extra_class is 0 for decimated altimetry, 1 for decimated DEMs, 2 and 3 at full density
        data.assign({'sigma_extra_class': (~np.isin(data.sensor, altimeter_sensors)
                                           + 2*(data.no_decimation > 0)).astype(float)})
        sigma_extra_keys={'altimeter':{'sigma_extra_class':[0]},
                          'DEM':{'sigma_extra_class':[1]},
                          'altimeter_no_decimation':{'sigma_extra_class':[2]},
                          'DEM_no_decimation':{'sigma_extra_class':[3]}}
    if reference_DEM_file is not None and reread_file is None:
        # N.B.  Reference DEM is subtracted AFTER firn correction
        reference_DEM = pc.grid.data().from_file(reference_DEM_file,
                                    bounds=[xy0[0]+np.array([-0.55, 0.55])*Width,
                                            xy0[1]+np.array([-0.55, 0.55])*Width])
        if reference_DEM_add_geoid:
            reference_DEM.z += pc.grid.data().from_file(geoid_file,
                                                        bounds=reference_DEM.bounds(pad=1.e4))\
                .interp(reference_DEM.x, reference_DEM.y, gridded=True)
        data.assign(z_ref = reference_DEM.interp(data.x, data.y) + reference_DEM_offset)
        data.z -= data.z_ref
        data.index(np.isfinite(data.z))
        E_RMS0['z0'] = reference_DEM_uncertainty

    data.index(np.isfinite(data.z) & (np.isfinite(data.sigma)))
    if data.size == 0:
        if verbose:
            print("fit_altimetry:No valid data found, returning")
        write_no_fit_file(no_fit_file, "no valid data")
        return None, None, None
    print("fit_altimetry:")
    print(data.summary())
    if mask_roles_assigned:
        # in a full-lagrangian fit, data are selected at the locations where they were measured
        mask_args = {'mask_file':None,
                     'mask_data':mask_dict,
                     'grid_masks':grid_masks,
                     'data_mask':data_mask,
                     'data_mask_coords':['x_original', 'y_original', 'time'] if full_lagrangian else None}
    else:
        mask_args = {'mask_file':mask_file, 'mask_data':mask_data}
    # run the fit
    if verbose:
        print("="*50)
        print("about to run smooth_fit with bias params="+str(bias_params))
    S=smooth_fit(data=data, ctr=ctr, W=W, spacing=spacing, E_RMS=E_RMS0,
                     return_fit_objects=return_fit_objects,
                     reference_epoch=reference_epoch,
                     compute_E=calc_error,
                     bias_params=bias_params,
                     repeat_res=repeat_res,
                     max_iterations=max_iterations,
                     srs_proj4=SRS_proj4,
                     VERBOSE=verbose,
                     data_slope_sensors=DEM_sensors,\
                     data_scale_fields=data_scale_fields,\
                     E_slope_bias=E_slope_bias,\
                     erode_source_mask=erode_source_mask,\
                     **mask_args,\
                     sigma_extra_relax=sigma_extra_relax,\
                     constraint_scaling_maps=constraint_scaling_maps,\
                     prior_edge_args=prior_edge_args,\
                     dzdt_lags=dzdt_lags,\
                     bias_model_args = bias_model_args, \
                     bias_nsigma_edit=bias_nsigma_edit, \
                     bias_nsigma_iteration=bias_nsigma_iteration, \
                     error_res_scale=error_res_scale,\
                     mask_scale={0:mask_zero_scale, 1:1}, \
                     avg_scales=avg_scales, \
                     avg_masks=avg_masks, \
                     sigma_extra_keys=sigma_extra_keys,\
                     sensor_grid_bias_params=sensor_grid_bias_params,\
                     converge_tol_frac_TSE=converge_tol_frac,
                     solver=solver,
                     cholmod_scale_columns=cholmod_scale_columns,
                     cholmod_cache_ordering=cholmod_cache_ordering,
                     fine_grid_args=fine_grid_args,
                     THREADS=THREADS)

    if return_fit_objects:
        return S

    if lagrangian_dict is not None and S['data'] is not None and S['data'].size > 0:
        update_output_data_for_lagrangian(S, **lagrangian_dict)
        data=S['data']

    return S, data, sensor_dict

def read_fine_grid_file(filename):
    '''
    Read the fine-grid parameters for smooth_fit from a json file

    The file contains a dict with the entries of the LSsurf.smooth_fit
    fine_grid_args argument (see LSsurf/fine_grid_functions.py), e.g.:
        {"region_file": "rift_region.geojson", "spacing": 100,
         "E_RMS": {"d2z0_dx2": 1.0, "d3z_dx2dt": 0.3, "d2z_dt2": 100}}
    A relative region_file path is interpreted relative to the json file's directory.
    '''
    with open(filename) as fh:
        fine_grid_args = json.load(fh)
    if fine_grid_args.get('region_file') is not None:
        fine_grid_args['region_file'] = os.path.join(
            os.path.dirname(os.path.abspath(filename)),
            os.path.expanduser(fine_grid_args['region_file']))
    return fine_grid_args

def parse_inputs(argv):
    # account for a bug in argparse that misinterprets negative agruents
    for i, arg in enumerate(argv):
        if (arg[0] == '-') and arg[1].isdigit(): argv[i] = ' ' + arg

    def path(p):
        return os.path.abspath(os.path.expanduser(p))

    import argparse
    parser=argparse.ArgumentParser(description="function to fit icebridge data with a smooth elevation-change model", \
                                   fromfile_prefix_chars="@")
    parser.add_argument('--xy0', type=float, nargs=2, help="fit center location")
    parser.add_argument('--km', action='store_true', help="if specified, coordinates for xy0 argument are interpreted as km.")
    parser.add_argument('--Width','-W',  type=float, help="fit width")
    parser.add_argument('--time_span','-t', type=str, help="time span, first year,last year AD (comma separated, no spaces)")
    parser.add_argument('--reference_epoch', type=int)
    parser.add_argument('--reference_time', type=float)
    parser.add_argument('--grid_spacing','-g', type=str, help='grid spacing:DEM (meters),dh maps xy (meters),dh_maps time (years): comma-separated, no spaces', default='250.,4000.,1.')
    parser.add_argument('--Hemisphere','-H', type=int, default=1, help='hemisphere: -1=Antarctica, 1=Greenland')
    parser.add_argument('--base_directory','-b', type=path, help='base directory')
    parser.add_argument('--GeoIndex_source_file', type=path, help='json file containing locations for geoIndex files')
    parser.add_argument('--reread_file', type=str, help='reread data from this file')
    parser.add_argument('--out_name', '-o', type=path, help="output file name")
    parser.add_argument('--tile_spacing', type=float, help="spacing between adjacent tile centers")
    parser.add_argument('--dzdt_lags', type=str, default='1,2,4', help='lags for which to calculate dz/dt, comma-separated list, no spaces')
    parser.add_argument('--prelim', action="store_true")
    parser.add_argument('--matched', action="store_true")
    parser.add_argument('--E_d2zdt2', type=float, default=5000)
    parser.add_argument('--E_d2z0dx2', type=float, default=0.02)
    parser.add_argument('--E_d3zdx2dt', type=float, default=0.0003)
    parser.add_argument('--E_slope_bias', type=float, default=1.e-5)
    parser.add_argument('--data_gap_scale', type=float,  default=2500)
    parser.add_argument('--d2z_dt2_scale_file', type=path, help='file that scales the d2z_dt2 field')
    parser.add_argument('--constraint_scaling_files', type=str, help='specify comma-separated list with component:file for each, no spaces')
    parser.add_argument('--max_iterations', type=int, default=8)
    parser.add_argument('--sigma_extra_relax', action='store_true', help='if set, the data errors will be relaxed by the calculated sigma_extra on the last iteration')
    parser.add_argument('--prior_edge_include', type=float, help="distance from tile edge over which adjacent tiles can constrain the fit")
    parser.add_argument('--prior_sigma_scale', type=float, default=1., help="error estimates in adjacent tiles will be scaled by this factor")
    parser.add_argument('--param_files', type=str, help='comma-separated list of parameter files, of form name1:file1,name2:file2')
    parser.add_argument('--xy_bias_file', type=str, help="CSV file containing fields delta_time, x_bias, and y_bias")
    parser.add_argument('--reference_DEM_file', type=str, help='DEM that will be subtracted from data before fitting')
    parser.add_argument('--reference_DEM_offset', type=float, default=0, help='Value that will be added to the reference DEM before it is subtracted from the data')
    parser.add_argument('--reference_DEM_uncertainty', type=float, help='Uncertainty in the reference DEM is correct')
    parser.add_argument('--reference_DEM_add_geoid', action='store_true', help="if set, add the geoid to the reference DEM before it is subtracted from the data")
    parser.add_argument('--bias_params', type=str, nargs='+', default=['time_corr','sensor','spot'])
    parser.add_argument('--DEM_grid_bias_params_file', type=path, help='file containing DEM grid bias params')
    parser.add_argument('--DEM_sigma_corr', type=float, default=20, help='tolerance for mean DEM bias, default=20')
    parser.add_argument('--DEM_t_buffer', type=float, default=0.5, help='do not use DEMs unless they are at least this time after the beginning/before the end of the time span')
    parser.add_argument('--bias_nsigma_edit', type=int, default=6, help='edit points whose estimated bias is more than this value times the expected')
    parser.add_argument('--bias_nsigma_iteration', type=int, default=6, help='apply bias_nsigma_edit after this iteration')
    parser.add_argument('--bm_scale_laser', type=float, default=50, help="blockmedian scale to apply to laser data")
    parser.add_argument('--bm_scale_DEM', type=float, default=200, help='blockmedian scale to apply to DEM data')
    parser.add_argument('--N_target_laser', type=float, default=200000, help='target number of laser data')
    parser.add_argument('--N_target_DEM', type=float, default=800000, help='target number of DEM data')
    parser.add_argument('--no_decimation_mask_file', type=path, help='raster file (geotif, or h5 with fields x, y, [t,] z) that is greater than 0.5 where data are not decimated.  The decimation (blockmedian) scales are calculated from all of the data, as without the mask, but are only applied outside it.  A mask with a time dimension is evaluated at the data times (e.g. to follow features that move with the ice).  The data inside the mask get their own sigma_extra (for altimetry and for DEMs), because their statistics differ from those of the decimated data.')
    parser.add_argument('--N_target', type=str, help='specify target data count for sensors as sensor:N, with multiple arguments separated by commas')
    parser.add_argument('--extra_error', type=float)
    parser.add_argument('--DEM_file', type=path, help='DEM file to use with the DEM_tol parameter and in slope error calculations')
    parser.add_argument('--DEM_tol', type=float, help='Tolerance for selecting points relative to the DEM file')
    parser.add_argument('--DEM_dt_min', type=float, help="if two DEMs are closer to time than this value, data from the one with the larger errors will be deleted")
    parser.add_argument('--DEM_slope_bias_scale', type=str, help='if specified, the bias the surface slope from the DEM times the geolocation error')
    parser.add_argument('--data_sigma_max', type=float, help="points with sigma estimates larger than this value will be ignored")
    parser.add_argument('--problem_DEM_files', type=str, help='files listing problematic DEM files, comma separated')
    parser.add_argument('--map_dir','-m', type=path)
    parser.add_argument('--firn_directory', type=path, help='directory containing firn model')
    parser.add_argument('--firn_model', type=str, help='firn model name')
    parser.add_argument('--firn_version', type=str, help='firn version')
    parser.add_argument('--firn_grid_file', type=str, help='gridded firn model file that can be interpolated directly.')
    parser.add_argument('--firn_mask_file', type=str, help='firn correction will only be nonzero where this mask is 1')
    parser.add_argument('--rerun_file_with_firn', type=str)
    parser.add_argument('--firn_rescale', action='store_true')
    parser.add_argument('--firn_fixed', action='store_true')
    parser.add_argument('--full_FAC_correction', action='store_true', help='if specified, subtract the full FAC from the data, rather than the anomaly.')
    parser.add_argument('--lagrangian', action='store_true')
    parser.add_argument('--velocity_files', type=path, nargs='+', help='lagrangian velocity files.  May contain multiple time values.')
    parser.add_argument('--velocity_type', type=str, help='velocity file type.  Specify xy01_grids if interpolators are used')
    parser.add_argument('--lagrangian_epoch', type=float, help='time (decimal year) to which data will be advected')
    parser.add_argument('--lagrangian_mask_files', type=str, nargs='+', help='filenames for geotifs containing areas that have calved.  Filenames contain dates _YYYYMMDD_')
    parser.add_argument('--lagrangian_ref_dem', type=path, help='dem to be subtracted from data before advection')
    parser.add_argument('--lagrangian_dz_spacing', type=float, help='field to estimate moving topography, ignores lagrangian argument')
    parser.add_argument('--lagrangian_dz_E_RMS', type=float, help='Expected RMS for moving topography')
    parser.add_argument('--lagrangian_dz_E_RMS_grad', type=float, help='Expected RMS gradient for moving topography')
    parser.add_argument('--lagrangian_zero_velocity', action='store_true', help='If set, all parameters will match the lagrangian parameters, but velocity will be set to zero')
    parser.add_argument('--fine_grid_file', type=path, help='json file containing parameters for fine-resolution z0 and dz grids that cover a region of the domain (see read_fine_grid_file).  The region must be in the coordinates of the fit (with --lagrangian, the advected coordinates at lagrangian_epoch).  Fine grids keep their resolution when errors are calculated.')
    parser.add_argument('--mask_file', type=path)
    parser.add_argument('--gzone_relax_scale', type=float, help='relax the d3z_dx2dt constraint at the grounding zone by this factor')
    parser.add_argument('--mask_zero_scale', type=float, default=10, help='constraint equations will be reduced by this factor where the mask is zero')
    parser.add_argument('--and_mask_files', type=str, nargs='+')
    parser.add_argument('--or_mask_files', type=str, nargs='+')
    parser.add_argument('--masks', type=str, nargs='+', metavar='NAME:FILE', help="named masks (geotif, or h5 with fields x, y, [t,] z; 1 inside the mask), assigned to grids and to data selection with --grid_masks and --data_mask.  The mask built from --mask_file (or all ones), --tide_mask_keep_val, and --and_mask_files/--or_mask_files is named 'default'")
    parser.add_argument('--grid_masks', type=str, nargs='+', metavar='GRID:NAME', help="masks for the model grids (GRID is z0 or dz).  Grids without a mask assigned use the default mask.  In full-lagrangian fits, NAME can also be advected_any or advected (see --lagrangian_grid_masks)")
    parser.add_argument('--lagrangian_grid_masks', action='store_true', help="in a full-lagrangian fit, make the grid masks by advecting the model nodes through the time span of the fit and sampling the data mask: the z0 grid covers parcels that are inside the data mask at any time during the fit, and the (time-varying) dz mask shows whether each parcel is inside the data mask at each epoch.  The same as '--grid_masks z0:advected_any dz:advected'; grids named in --grid_masks keep their assignment")
    parser.add_argument('--lagrangian_mask_spacing', type=float, help='spacing of the advected (lagrangian) masks.  Default: half the dz spacing')
    parser.add_argument('--data_mask', type=str, metavar='NAME', help="mask used to select data: data are used where this mask is greater than 0.5 at the location (and, for a mask with a time dimension, the time) of the measurement.  For full-lagrangian fits, this is the location before advection.  If not set, the default mask is used")
    parser.add_argument('--geoid_file', type=path)
    parser.add_argument('--water_mask_threshold', type=float)
    parser.add_argument('--skip_erode_source_mask', action='store_true', help='if set, do not erode the source mask before fitting')
    parser.add_argument('--year_mask_dir', type=path)
    parser.add_argument('--tide_mask_file', type=path)
    parser.add_argument('--tide_mask_shelf_val', type=int, default=1, help='this value marks shelf points')
    parser.add_argument('--tide_directory', type=path)
    parser.add_argument('--tide_model', type=str, help='tide model name')
    parser.add_argument('--tide_adjustment_file', type=str, help='h5 file_giving scaling for tide model')
    parser.add_argument('--tide_mask_keep_val', type=float, help="include in the mask only points for which the tide mask matches this value")
    parser.add_argument('--tide_mask_keep_buffer', type=float, help="buffer the mask associated with the tide_mask_keep value by this distance")
    parser.add_argument('--avg_mask_directory', type=path)
    parser.add_argument('--calc_error_file','-c', type=path)
    parser.add_argument('--calc_error_for_xy', action='store_true')
    parser.add_argument('--calc_error', action='store_true')
    parser.add_argument('--avg_scales', type=str, help='scales at which to report average errors, comma-separated list, no spaces')
    parser.add_argument('--error_res_scale','-s', type=float, nargs='+', default=[4, 2], help='if the errors are being calculated (see calc_error_file), scale the grid resolution in x and y to be coarser')
    parser.add_argument('--converge_tol_frac', type=float, default=0.005, help='if the fractional change between two smooth_fit iterations is less than this value, the iteration has converged')
    parser.add_argument('--sparseqr', action='store_true', help='solve the smooth_fit iterations with sparseqr (SPQR) instead of the default cholmod (normal equations + refinement, needs scikit-sparse)')
    parser.add_argument('--cholmod_scale_columns', action='store_true', default=True, help='scale the normal matrix to a unit diagonal before the cholmod factorization, so its rcond check measures column dependence rather than column scaling (the default)')
    parser.add_argument('--no_cholmod_scale_columns', dest='cholmod_scale_columns', action='store_false', help='factor the unscaled normal matrix in the cholmod solve (see --cholmod_scale_columns)')
    parser.add_argument('--cholmod_cache_ordering', action='store_true', help='compute the METIS fill-reducing ordering for the cholmod factorization in the first smooth_fit iteration only, and reuse it in the later iterations')
    parser.add_argument('--THREADS', type=int, default=1, help='number of threads to use in suitesparse calculations')
    parser.add_argument('--max_mem', type=float, default=15., help='maximum memory the program is allowed to use, in GB.')
    args, unk=parser.parse_known_args(argv)
    unk=[jj for jj in unk if (len(jj)>0) and not jj[0]=='#']
    if unk:
        print("unknown arguments:"+str(unk))

    args.erode_source_mask = not(args.skip_erode_source_mask)

    def colon_pairs(pairs, option):
        # parse a list of KEY:VALUE arguments into a dictionary
        out={}
        for pair in pairs:
            if ':' not in pair:
                raise ValueError(f'{option} arguments should be KEY:VALUE, got {pair}')
            key, val = pair.split(':', 1)
            out[key]=val
        return out
    if args.masks is not None:
        args.masks = {name:path(mask_file) for name, mask_file in colon_pairs(args.masks, '--masks').items()}
    if args.grid_masks is not None:
        args.grid_masks = colon_pairs(args.grid_masks, '--grid_masks')

    if args.km:
        args.xy0 = [jj*1000 for jj in args.xy0]

    if args.max_mem is not None and args.max_mem > 0:
        set_memory_limit(int(args.max_mem*1024*1024*1024))

    if args.avg_scales is not None:
        args.avg_scales = [int(temp) for temp in args.avg_scales.split(',')]

    args.spacing={}
    for dim, this_sp in zip(['z0','dz','dt'], args.grid_spacing.split(',')):
        if '/' in this_sp:
            # this is a fractional spacing (e.g. 1/12 year)
            this_sp = this_sp.split('/')
            this_sp = float(this_sp[0])/float(this_sp[1])
        else:
            this_sp = float(this_sp)
        args.spacing[dim] = this_sp
    #args.grid_spacing = [spacing['z0'], spacing['dz'], spacing['dt']]

    args.time_span = [float(temp) for temp in args.time_span.split(',')]
    args.dzdt_lags = [int(temp) for temp in args.dzdt_lags.split(',')]

    if args.reference_epoch is None:
        if args.reference_time is None:
            raise(ValueError("Need to specify either reference_epoch or reference_time"))
        args.reference_epoch = np.argmin(np.abs(
            np.arange(args.time_span[0], args.time_span[1]+0.01*args.spacing['dt'], args.spacing['dt'])-args.reference_time))

    args.E_RMS={'d2z0_dx2':args.E_d2z0dx2, 'd3z_dx2dt':args.E_d3zdx2dt, 'd2z_dxdt':args.E_d3zdx2dt*args.data_gap_scale,  'd2z_dt2':args.E_d2zdt2}

    # read in the geoIndex locations
    with open(args.GeoIndex_source_file) as fh:
        args.geoIndex_dict=json.load(fh)

    if args.full_FAC_correction:
        args.calc_FAC_anomaly=False
    else:
        args.calc_FAC_anomaly=True

    if args.base_directory is None:
        for temp in [args.reread_file, args.calc_error_file]:
            if temp is None:
                continue
            args.base_directory = os.path.dirname(os.path.dirname(temp))

    prior_dirs=None
    if args.matched:
        prior_dirs = [os.path.join(args.base_directory,'prelim')]
        args.max_iterations = 1

    args.prior_edge_args=None
    if args.prior_edge_include is not None:
        args.prior_edge_args={'prior_dir': prior_dirs,
                         'edge_include' : args.prior_edge_include,
                         'sigma_scale' : args.prior_sigma_scale,
                         'tile_spacing': args.tile_spacing}

    args.reread_dirs=None
    if args.out_name is not None:
        dest_dir = os.path.dirname(args.out_name)
    elif args.prelim:
        dest_dir = args.base_directory+'/prelim'
    elif args.matched:
        dest_dir = args.base_directory+'/matched'

    in_file=None
    if args.calc_error_file is not None:
        args.reread_file = args.calc_error_file
        if args.out_name is None:
            args.out_name=args.calc_error_file
        in_file=args.calc_error_file
        dest_dir=os.path.dirname(args.reread_file)

    if args.reread_file is not None:
        # get xy0 from the filename
        in_file=args.reread_file

    if in_file is not None:
        if args.xy0 is None:
            re_match=re.compile('E(.*)_N(.*).h5').search(os.path.basename(in_file))
            args.xy0=[float(re_match.group(ii))*1000 for ii in [1, 2]]

    if args.out_name is None:
        args.out_name=dest_dir + '/E%d_N%d.h5' % (args.xy0[0]/1e3, args.xy0[1]/1e3)

    if args.calc_error_for_xy:
        args.calc_error_file=args.out_name
        if not os.path.isfile(args.out_name):
            no_fit_file = no_fit_file_name(args.out_name)
            if os.path.isfile(no_fit_file):
                with open(no_fit_file) as fh:
                    reason = fh.read().strip()
                print(f"{args.out_name} not found: the fit had nothing to fit ({reason}), returning")
                return 0
            print(f"{args.out_name} not found, returning")
            return 1

    if args.error_res_scale is not None:
        if args.calc_error_file is not None or  args.calc_error:
            for ii, key in enumerate(['z0','dz']):
                args.spacing[key] *= args.error_res_scale[ii]
            if args.lagrangian_dz_spacing is not None:
                args.lagrangian_dz_spacing *= args.error_res_scale[2]

    if args.bias_params is not None and " " in args.bias_params[0]:
        args.bias_params=args.bias_params[0].split(" ")

    # read in the DEM grid bias params.  These will be applied to each DEM
    args.DEM_bias_params=None
    if args.DEM_grid_bias_params_file is not None:
        # DEM_bias_params file should include key=value lines, with keys:
        #spacing, expected_rms, expected_rms_grad
        args.DEM_bias_params=aF.bias_functions.parse_DEM_bias_params_file(args.DEM_grid_bias_params_file)

    args.fine_grid_args=None
    if args.fine_grid_file is not None:
        args.fine_grid_args=read_fine_grid_file(args.fine_grid_file)

    args.params={}
    if args.param_files is not None:
        for temp in args.param_files.split(','):
            name, file = temp.split(':')
            with open(file,'r') as fh:
                args.params[name]=json.load(fh)
                args.params[name]['param_file']=file

    for thedir in [args.base_directory, dest_dir]:
        try:
            os.mkdir(dest_dir)
        except (FileExistsError, TypeError):
            pass

    if args.problem_DEM_files is not None:
        # I haven't found a cleaner way to do this:
        temp =  args.problem_DEM_files.split(',')
        args.problem_DEM_files = []
        for jj in temp:
            args.problem_DEM_files += glob.glob(jj)
    else:
        args.problem_DEM_files = []

    if args.out_name is None:
        args.out_name=dest_dir + '/E%d_N%d.h5' % (args.xy0[0]/1e3, args.xy0[1]/1e3)

    args.lagrangian_dict=None
    if args.lagrangian:
        if args.lagrangian_epoch is None:
            args.lagrangian_epoch = np.arange(args.time_span[0], args.time_span[1], args.spacing['dt'])[args.reference_epoch]
        args.lagrangian_dict = {field:getattr(args, field) for field in \
                           ['velocity_files', 'velocity_type', 'lagrangian_epoch', 'lagrangian_ref_dem',
                            'lagrangian_dz_spacing', 'lagrangian_dz_E_RMS',
                            'lagrangian_dz_E_RMS_grad', 'lagrangian_mask_files',
                            'lagrangian_zero_velocity']
                           }
        if args.reread_file is not None or args.reread_dirs is not None:
            args.lagrangian_dict['reread_data']=True

    N_target={}
    if args.N_target is not None:
        N_target={}
        for arg in args.N_target.split(','):
            sensor, N = arg.split(':')
            N_target[sensor]=int(N)
        args.N_target=N_target
    else:
        args.N_target={}
        if args.N_target_laser is not None:
            args.N_target['laser']=args.N_target_laser
        if args.N_target_DEM is not None:
            args.N_target['DEM']=args.N_target_DEM

    args.bm_scale={'laser':args.bm_scale_laser,\
                 'DEM':args.bm_scale_DEM}
    return args

def main(argv=None):
    if argv is None:
        argv=sys.argv[1:]

    args=parse_inputs(argv)
    if isinstance(args, int):
        return args

    no_fit_file=None
    if args.calc_error_file is None and not args.calc_error:
        no_fit_file=no_fit_file_name(args.out_name)
        # one left by an earlier run would hide a failure of this one
        if os.path.isfile(no_fit_file):
            os.remove(no_fit_file)

    S, data, sensor_dict = fit_altimetry(args.xy0, \
            Width=args.Width,
            E_RMS=args.E_RMS, \
            d2z_dt2_scale_file=args.d2z_dt2_scale_file,\
            constraint_scaling_files=args.constraint_scaling_files,\
            t_span=args.time_span,\
            spacing=args.spacing, \
            reference_epoch=args.reference_epoch, \
            reread_file=args.reread_file,\
            calc_error_file=args.calc_error_file,\
            error_res_scale=args.error_res_scale,\
            calc_error=args.calc_error,\
            converge_tol_frac=args.converge_tol_frac,\
            max_iterations=args.max_iterations, \
            sigma_extra_relax=args.sigma_extra_relax, \
            bias_nsigma_edit=args.bias_nsigma_edit,
            bias_nsigma_iteration=args.bias_nsigma_iteration,\
            bias_params=args.bias_params,\
            prior_edge_args=args.prior_edge_args,\
            params=args.params,\
            reference_DEM_file=args.reference_DEM_file,\
            reference_DEM_offset=args.reference_DEM_offset,\
            reference_DEM_uncertainty=args.reference_DEM_uncertainty,\
            reference_DEM_add_geoid=args.reference_DEM_add_geoid,\
            problem_DEM_files=args.problem_DEM_files,\
            hemisphere=args.Hemisphere, \
            reread_dirs=args.reread_dirs, \
            out_name=args.out_name, \
            no_fit_file=no_fit_file, \
            GI_files=args.geoIndex_dict,\
            bm_scale=args.bm_scale,\
            N_target=args.N_target,\
            xy_bias_file=args.xy_bias_file,\
            no_decimation_mask_file=args.no_decimation_mask_file,\
            firn_directory=args.firn_directory,\
            firn_version=args.firn_version,\
            firn_correction=args.firn_model,\
            firn_grid_file=args.firn_grid_file,\
            firn_fixed=args.firn_fixed,\
            firn_rescale=args.firn_rescale,\
            firn_mask_file=args.firn_mask_file,\
            calc_FAC_anomaly=args.calc_FAC_anomaly,\
            lagrangian_dict=args.lagrangian_dict,\
            mask_file=args.mask_file, \
            masks=args.masks, \
            grid_masks=args.grid_masks, \
            data_mask=args.data_mask, \
            lagrangian_grid_masks=args.lagrangian_grid_masks, \
            lagrangian_mask_spacing=args.lagrangian_mask_spacing, \
            mask_zero_scale=args.mask_zero_scale, \
            and_mask_files=args.and_mask_files,\
            or_mask_files=args.or_mask_files,\
            DEM_file=args.DEM_file,\
            DEM_tol=args.DEM_tol,\
            DEM_t_buffer=args.DEM_t_buffer,\
            geoid_file=args.geoid_file,\
            gzone_relax_scale=args.gzone_relax_scale,\
            extra_error=args.extra_error, \
            E_slope_bias=args.E_slope_bias, \
            water_mask_threshold=args.water_mask_threshold, \
            erode_source_mask = args.erode_source_mask,\
            year_mask_dir=args.year_mask_dir, \
            tide_directory=args.tide_directory, \
            tide_mask_file=args.tide_mask_file, \
            tide_mask_shelf_val = args.tide_mask_shelf_val, \
            tide_model=args.tide_model, \
            tide_adjustment_file=args.tide_adjustment_file, \
            tide_mask_keep_val=args.tide_mask_keep_val,\
            tide_mask_keep_buffer=args.tide_mask_keep_buffer,\
            avg_mask_directory=args.avg_mask_directory, \
            dzdt_lags=args.dzdt_lags,\
            avg_scales=args.avg_scales,\
            DEM_dt_min=args.DEM_dt_min,\
            DEM_sigma_corr=args.DEM_sigma_corr,\
            DEM_grid_bias_params=args.DEM_bias_params,\
            DEM_slope_bias_scale=args.DEM_slope_bias_scale,\
            solver='spqr' if args.sparseqr else 'cholmod',\
            cholmod_scale_columns=args.cholmod_scale_columns,\
            cholmod_cache_ordering=args.cholmod_cache_ordering,\
            fine_grid_args=args.fine_grid_args,\
            THREADS=args.THREADS)

    if S is None:
        return

    if args.calc_error_file is None and not args.calc_error:
        if 'm' in S and len(S['m']) > 0:
            save_fit_to_file(S, args.out_name, sensor_dict=sensor_dict,\
                             dzdt_lags=S['dzdt_lags'], \
                             reference_epoch=args.reference_epoch,
                             args=args)
        else:
            write_no_fit_file(no_fit_file, "smooth_fit found no valid data")
    else:
        if 'sigma_z0' not in S['E']:
            print("fit_altimetry.py: no sigma_z0 calculated, returning")
            return

        S['E']['sigma_z0']=interp_ds(S['E']['sigma_z0'], args.error_res_scale[0])
        for field in S['E'].keys():
            if '_fine' in field:
                # the fine grids are not coarsened for the error calculation
                continue
            if 'sigma_dz' in field: # ['sigma_dz', 'sigma_dzdt_lag1', 'sigma_dzdt_lag2', 'sigma_dzdt_lag4']:
                S['E'][field] = interp_ds( S['E'][field], args.error_res_scale[1] )
            elif field == 'sigma_lagrangian_dz':
                S['E'][field] = interp_ds( S['E'][field], args.error_res_scale[2] )

        save_errors_to_file(S, args.out_name)

    print("done with " + args.out_name)
    total_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss +\
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    print(f"peak memory usage (kB)={total_memory}")

if __name__=='__main__':
    main(sys.argv[1:])
