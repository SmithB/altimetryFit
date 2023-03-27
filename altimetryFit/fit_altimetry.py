#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 17:12:48 2019

@author: ben
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
from LSsurf.smooth_xytb_fit_aug import smooth_xytb_fit_aug
from LSsurf import fd_grid
from altimetryFit.reread_data_from_fits import reread_data_from_fits
import pointCollection as pc
from pyTMD import compute_tide_corrections
from SMBcorr import assign_firn_variable
from altimetryFit.read_optical import read_optical_data, laser_key
from CS2_fit.read_CS2_data import read_CS2_data
import h5py
import sys
import glob
import json
import re

def make_sensor_dict(h5_file):
    '''
    make a dictionary of sensors a fit output file.
    
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

def apply_tides(D, xy0, W, tide_mask_file, tide_directory, tide_model):
    #read in the tide mask (for Antarctica) and apply dac and tide to ice-shelf elements
    # the tide mask should be 1 for non-grounded points (ice shelves?), zero otherwise
    tide_mask = pc.grid.data().from_geotif(tide_mask_file, bounds=[np.array([-0.6, 0.6])*W+xy0[0], np.array([-0.6, 0.6])*W+xy0[1]])
    is_els=tide_mask.interp(D.x, D.y) > 0.5
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
                EPOCH=(2000,1,1,0,0,0), TYPE='drift', TIME='utc', EPSG=3031)
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
    if 'sensor_bias_grids' in S['m']:
        for name, ds in S['m']['sensor_bias_grids'].items():
                ds.to_h5(filename, group='/grid_bias/'+name)

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
            tide_directory=None, \
            tide_model='CATS2008', \
            year_mask_dir=None, \
            avg_scales=None,\
            DEM_grid_bias_params=None):
    """
        Wrapper for smooth_xytb_fit_aug that can find data and set the appropriate parameters
    """
    print("fit_OIB: working on %s" % out_name)

    SRS_proj4=get_SRS_proj4(hemisphere)
    bias_model_args={}
    compute_E=False
    # set defaults for E_RMS, then update with input parameters
    E_RMS0={'d2z0_dx2':200000./3000/3000, 'd3z_dx2dt':3000./3000/3000, 'd2z_dxdt':3000/3000, 'd2z_dt2':5000}
    E_RMS0.update(E_RMS)

    W={'x':Wxy, 'y':Wxy,'t':np.diff(t_span)}
    ctr={'x':xy0[0], 'y':xy0[1], 't':np.mean(t_span)}
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

    if reread_file is not None:
        # get xy0 from the filename
        re_match=re.compile('E(.*)_N(.*).h5').search(reread_file)
        xy0=[float(re_match.group(ii))*1000 for ii in [1, 2]]
        data=pc.data().from_h5(reread_file, group='data')
        sensor_dict=make_sensor_dict(reread_file)
    elif reread_dirs is None:
        D, sensor_dict, DEM_meta_dict = read_optical_data(xy0, W, GI_files=GI_files, \
                                SRS_proj4=get_SRS_proj4(hemisphere),\
                                bm_scale=bm_scale,\
                                N_target=N_target,\
                                 mask_file=mask_file, geoid_file=geoid_file, \
                                 mask_floating=mask_floating,\
                                 water_mask_threshold=water_mask_threshold, \
                                 DEM_file=DEM_file, \
                                 hemisphere=hemisphere)
        for ind, Di in enumerate(D):
            if Di is None:
                continue
            for field in ['rgt','cycle','spot']:
                if field not in Di.fields:
                    Di.assign({field:np.zeros_like(Di.x)+np.NaN}) 
                
        data=pc.data(fields=['x','y','z','time','sigma','sigma_corr','slope_mag', 'sensor','spot', 'rgt','cycle']).from_list(D)
        data.assign({'day':np.floor(data.time*365.25)})
        if extra_error is not None:
            data.sigma[data.time < 2010] = np.sqrt(data.sigma[data.time<2010]**2 +extra_error**2)
        # apply the tides if a directory has been provided
        if tide_mask_file is not None:
            apply_tides(data, xy0, Wxy, tide_mask_file, tide_directory, tide_model)

    else:
        data, sensor_dict = reread_data_from_fits(xy0, Wxy, reread_dirs, template='E%d_N%d.h5')
    laser_sensors=[item for key, item in laser_key().items()]
    DEM_sensors=np.array([key for key in sensor_dict.keys() if key not in laser_sensors ])
    if reference_epoch is None:
        reference_epoch=len(np.arange(t_span[0], t_span[1], spacing['dt']))

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

    if DEM_grid_bias_params is not None:
        sensor_grid_bias_params=[]
        for sensor in DEM_sensors:
            sensor_grid_bias_params += [{'sensor':sensor, 'expected_val':0}]
            sensor_grid_bias_params[-1].update(DEM_grid_bias_params)

    if isinstance(data,pc.data):
        temp=pc.data().from_dict({item:data.__dict__[item] for item in data.fields})
        data=temp

    # apply any custom edits
    custom_edits(data)
    assign_sigma_corr(data)

    avg_masks=None
    if avg_mask_directory is not None:

        avg_masks = {os.path.basename(file).replace('.tif',''):pc.grid.data().from_geotif(file) for file in \
                     glob.glob(avg_mask_directory+'/*.tif')}

    if year_mask_dir is not None:
        mask_data_by_year(data, year_mask_dir);
    
    sigma_extra_masks = {'laser': np.in1d(data.sensor, laser_sensors), 
                         'DEM': ~np.in1d(data.sensor, laser_sensors)}
    # run the fit
    S=smooth_xytb_fit_aug(data=data, ctr=ctr, W=W, spacing=spacing, E_RMS=E_RMS0,
                     reference_epoch=reference_epoch, compute_E=compute_E,
                     bias_params=['time_corr','sensor','spot'],
                     repeat_res=repeat_res, max_iterations=max_iterations,
                     srs_proj4=SRS_proj4, VERBOSE=True, Edit_only=Edit_only, 
                     data_slope_sensors=DEM_sensors,\
                     E_slope_bias=E_slope_bias,\
                     mask_file=mask_file,\
                     dzdt_lags=dzdt_lags, \
                     bias_model_args = bias_model_args, \
                     bias_nsigma_edit=bias_nsigma_edit, \
                     bias_nsigma_iteration=bias_nsigma_iteration, \
                     error_res_scale=error_res_scale,\
                     mask_scale={0:10, 1:1}, \
                     avg_scales=avg_scales, \
                     avg_masks=avg_masks, \
                     sigma_extra_masks=sigma_extra_masks,\
                     sensor_grid_bias_params=sensor_grid_bias_params,\
                     converge_tol_frac_TSE=0.005)
    return S, data, sensor_dict

def main(argv):
    # account for a bug in argparse that misinterprets negative agruents
    for i, arg in enumerate(argv):
        if (arg[0] == '-') and arg[1].isdigit(): argv[i] = ' ' + arg

    import argparse
    parser=argparse.ArgumentParser(description="function to fit icebridge data with a smooth elevation-change model", \
                                   fromfile_prefix_chars="@")
    parser.add_argument('--xy0', type=float, nargs=2, help="fit center location")
    parser.add_argument('--Width','-W',  type=float, help="fit width")
    parser.add_argument('--time_span','-t', type=str, help="time span, first year,last year AD (comma separated, no spaces)")
    parser.add_argument('--reference_epoch', type=int)
    parser.add_argument('--grid_spacing','-g', type=str, help='grid spacing:DEM (meters),dh maps xy (meters),dh_maps time (years): comma-separated, no spaces', default='250.,4000.,1.')
    parser.add_argument('--Hemisphere','-H', type=int, default=1, help='hemisphere: -1=Antarctica, 1=Greenland')
    parser.add_argument('--base_directory','-b', type=str, help='base directory')
    parser.add_argument('--GeoIndex_source_file', type=str, help='json file containing locations for geoIndex files')
    parser.add_argument('--reread_file', type=str, help='reread data from this file')
    parser.add_argument('--out_name', '-o', type=str, help="output file name")
    parser.add_argument('--dzdt_lags', type=str, default='1,2,4', help='lags for which to calculate dz/dt, comma-separated list, no spaces')
    parser.add_argument('--prelim', action="store_true")
    parser.add_argument('--centers', action="store_true")
    parser.add_argument('--edges', action="store_true")
    parser.add_argument('--corners', action="store_true")
    parser.add_argument('--E_d2zdt2', type=float, default=5000)
    parser.add_argument('--E_d2z0dx2', type=float, default=0.02)
    parser.add_argument('--E_d3zdx2dt', type=float, default=0.0003)
    parser.add_argument('--E_slope_bias', type=float, default=1.e-5)
    parser.add_argument('--data_gap_scale', type=float,  default=2500)
    parser.add_argument('--max_iterations', type=int, default=8)
    parser.add_argument('--DEM_grid_bias_params_file', type=str, help='file containing DEM grid bias params')
    parser.add_argument('--bias_nsigma_edit', type=int, default=6, help='edit points whose estimated bias is more than this value times the expected')
    parser.add_argument('--bias_nsigma_iteration', type=int, default=6, help='apply bias_nsigma_edit after this iteration')
    parser.add_argument('--bm_scale_laser', type=float, default=50, help="blockmedian scale to apply to laser data")
    parser.add_argument('--bm_scale_DEM', type=float, default=200, help='blockmedian scale to apply to DEM data')
    parser.add_argument('--N_target_laser', type=float, default=200000, help='target number of laser data')
    parser.add_argument('--N_target_DEM', type=float, default=800000, help='target number of DEM data')    
    parser.add_argument('--extra_error', type=float)
    parser.add_argument('--DEM_file', type=str, help='DEM file to use with the DEM_tol parameter and in slope error calculations')
    parser.add_argument('--mask_floating', action="store_true")
    parser.add_argument('--map_dir','-m', type=str)
    parser.add_argument('--firn_directory', type=str, help='directory containing firn model')
    parser.add_argument('--firn_model', type=str, help='firn model name')
    parser.add_argument('--firn_version', type=str, help='firn version')
    parser.add_argument('--rerun_file_with_firn', type=str)
    parser.add_argument('--firn_rescale', action='store_true')
    parser.add_argument('--firn_fixed', action='store_true')
    parser.add_argument('--mask_file', type=str)
    parser.add_argument('--geoid_file', type=str)
    parser.add_argument('--water_mask_threshold', type=float)
    parser.add_argument('--year_mask_dir', type=str)
    parser.add_argument('--tide_mask_file', type=str)
    parser.add_argument('--tide_directory', type=str)
    parser.add_argument('--tide_model', type=str, help='tide model name')
    parser.add_argument('--avg_mask_directory', type=str)
    parser.add_argument('--calc_error_file','-c', type=str)
    parser.add_argument('--calc_error_for_xy', action='store_true')
    parser.add_argument('--avg_scales', type=str, help='scales at which to report average errors, comma-separated list, no spaces')
    parser.add_argument('--error_res_scale','-s', type=float, nargs=2, default=[4, 2], help='if the errors are being calculated (see calc_error_file), scale the grid resolution in x and y to be coarser')
    args, unk=parser.parse_known_args()
    print("unknown arguments:"+str(unk))

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
    if args.centers:
        dest_dir = args.base_directory+'/centers'
    if args.edges or args.corners:
        reread_dirs=[args.base_directory+'/centers']
        dest_dir = args.base_directory+'/edges'
    if args.corners:
        reread_dirs += [args.base_directory+'/edges']
        dest_dir = args.base_directory+'/corners'

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
                    DEM_bias_params[key]=float(val.split('#')[0])
                except Exception:
                    print("problem with DEM_bias_params line\n"+line)
                    
    if not os.path.isdir(args.base_directory):
        os.mkdir(args.base_directory)
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

    if args.out_name is None:
        args.out_name=dest_dir + '/E%d_N%d.h5' % (args.xy0[0]/1e3, args.xy0[1]/1e3)

    S, data, sensor_dict = fit_altimetry(args.xy0, Wxy=args.Width, E_RMS=E_RMS, \
            t_span=args.time_span, spacing=spacing, \
            reference_epoch=args.reference_epoch, \
            reread_file=args.reread_file,\
            calc_error_file=args.calc_error_file,\
            error_res_scale=args.error_res_scale,\
            max_iterations=args.max_iterations, \
            bias_nsigma_edit=args.bias_nsigma_edit,
            bias_nsigma_iteration=args.bias_nsigma_iteration,\
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
            mask_file=args.mask_file, \
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

    if args.calc_error_file is None:
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

# -264000 -636000 default_args_I1I2.txt

#-200000 -2520000 -g 500,4000,1 --calc_error_file /Volumes/ice2/ben/ATL14_test/d3zdxdt=0.00001_d2zdt2=4000_RACMO//centers/E-200_N-2520.h5  @/home/ben//git_repos/LSsurf/default_GL_args.txt

# 0 0 -m /Volumes/ice2/ben/ATL14_test/Jako_d2zdt2=5000_d3z=0.00001_d2zdt2=1500_RACMO -W 4e4
#-80000 -2320000 -W 40000 -t 2002.5 2019.5 -g 200 2000 1  -o /Volumes/ice2/ben/ATL14_test/Jako_d2zdt2=5000_d3z=0.00001_d2zdt2=1500_RACMO/E-80_N-2320.h5 --E_d3zdx2dt 0.00001 --E_d2zdt2 1500 -f RACMO
