#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 09:43:00 2022

@author: ben
"""

import numpy as np
#import matplotlib.pyplot as plt
from altimetryFit.read_ICESat2 import read_ICESat2
from LSsurf.matlab_to_year import matlab_to_year
from altimetryFit.read_DEM_data import read_DEM_data
import pointCollection as pc
from ATL11.calc_geoloc_bias import calc_geoloc_bias

import glob

def laser_key():
    return {'ICESat1':1, 'ICESat2':2, 'ATM':3, 'LVIS':4, 'riegl':5}

def find_gI_files(gI_files):

    out_list=[]
    if not isinstance(gI_files, (list, tuple)):
        if isinstance(gI_files, dict):
            gI_files = list(gI_files.items())
        if isinstance(gI_files, str):
            gI_files=[gI_files]
    for file in gI_files:
        out_list += glob.glob(file)
    return out_list


def read_ICESat(xy0, W, gI_files, sensor=1, hemisphere=-1, DEM=None):
    #fields=[ 'IceSVar', 'deltaEllip', 'numPk', 'ocElv', 'reflctUC', 'satElevCorr',  'time',  'x', 'y', 'z']
    fields=['x','y','z','time']
    D0=[]
    box=[xy0[0]+np.array([-W['x']/2, W['x']/2]), xy0[1]+np.array([-W['y']/2, W['y']/2])]
    for file in gI_files:
        temp = pc.geoIndex().from_file(file).query_xy_box(*box, fields=fields)
        if temp is not None:
            D0 += temp
    if D0 is None:
        return None
    for D in D0:
        if hemisphere==-1:
            # ICESat time is seconds after Noon, Jan 1 2000.
            # Convert to years after Midnight, Jan 1 2000
            # full calculation:
            #t_is1=np.datetime64('2000-01-01T12:00:00')\
            #    + D.time*np.timedelta64(1,'s')
            #D.time = 2000. + (t_is1 \
            #    - np.datetime64('2000-01-01T00:00:00')) \
            #    / np.timedelta64(1, 's') / (24*3600*365.25)
            #simplified version (checked that this gives the same results):
            D.time=2000+0.5/365.25+D.time/24./3600./365.25
        else:
            D.time=matlab_to_year(D.time)
        if DEM is not None:
            slope_mag=np.sqrt(DEM.interp(D.x, D.y, field='z_x')**2+
                              DEM.interp(D.x, D.y, field='z_y')**2)
        else:
            slope_mag=np.abs(np.diff(D.z)/(7000*np.diff(D.time*24*3600*365.25)))
        # note: changed sigma to 0.06 as a test, 1/20/2022
        # note: changed sigma to 0.25, sigma_corr to 0.2 to handle extrapolated errors, 8/15/2023.
        D.assign({'sigma':np.zeros_like(D.x) + 0.25, \
                  'sigma_corr':np.zeros_like(D.x) + 0.2,\
                  'sensor':np.zeros_like(D.x)+sensor, \
                  'slope_mag':slope_mag})
    return D0

def read_ATM(xy0, W, gI_files, sensor=3, blockmedian_scale=100.):
    fields=['x','y','z', 'time','bias_50m', 'noise_50m', 'N_50m','slope_x','slope_y']
    if W is not None:
        dx=1.e4
        bds={'x':np.r_[np.floor((xy0[0]-W['x']/2)/dx), np.ceil((xy0[0]+W['x']/2)/dx)]*dx, \
                     'y':np.r_[np.floor((xy0[1]-W['y']/2)/dx), np.ceil((xy0[1]+W['y']/2)/dx)]*dx}
        px, py=np.meshgrid(np.arange(bds['x'][0], bds['x'][1], dx),
                           np.arange(bds['y'][0], bds['y'][1], dx))
    else:
        px=np.array([xy0[0]])
        py=np.array([xy0[1]])

    D0=[]
    for file in gI_files:
        temp = pc.geoIndex().from_file(file).query_xy((px.ravel(), py.ravel()), fields=fields)
        if temp is not None:
            D0 += temp

    if D0 is None:
        return D0
    for ind, D in enumerate(D0):
        for field in D.fields:
            setattr(D, field, getattr(D, field).ravel())
        good=np.isfinite(D.bias_50m) & (D.N_50m > 20) & (np.abs(D.bias_50m) < 20)
        slope_mag=np.sqrt(D.slope_x**2 + D.slope_y**2)
        D.assign({'slope_mag':slope_mag})
        good=good & (D.bias_50m < 0.5) & (slope_mag < 6*np.pi/180) & (np.abs(D.bias_50m) < 10)
        D.assign({'sigma': np.sqrt((4*slope_mag)**2+D.noise_50m**2+0.05**2),
                     'sigma_corr':0.025+np.zeros_like(D.time)})
        good = good & np.isfinite(D.sigma+D.sigma_corr)
        good=good.ravel()
        D.assign({'sensor':np.zeros_like(D.time)+sensor})
        D.time=matlab_to_year(D.time)
        D0[ind]=D.copy_subset(good, datasets=['x','y','z','time',
                                              'sigma','sigma_corr','sensor',
                                              'slope_mag'])
        if blockmedian_scale is not None:
            D0[ind].blockmedian(blockmedian_scale)
    return D0

def read_LVIS(xy0, W, gI_files, sensor=4, blockmedian_scale=100):
    fields={'x','y','z','time','bias_50m', 'noise_50m','slope_x','slope_y'}
    D0=[]
    for gI_file in gI_files:
        temp = pc.geoIndex().from_file(gI_file).query_xy_box(xy0[0]+np.array([-W['x']/2, W['x']/2]), xy0[1]+np.array([-W['y']/2, W['y']/2]), fields=fields)
        if temp is not None:
            D0 += temp
    for ind, D in enumerate(D0):
        # LVIS data have the wrong sign on their 'bias' field
        D.bias_50m *=-1
        slope_mag=np.sqrt(D.slope_x**2 + D.slope_y**2)
        good = (D.bias_50m < 0.5) & (slope_mag < 6*np.pi/180) & (np.abs(D.bias_50m) < 10)
        good=good.ravel()
        if not np.any(good):
            continue
        D.assign({  'sigma': np.sqrt((4*slope_mag)**2+D.noise_50m**2+0.05**2),
                    'sigma_corr':0.025+np.zeros_like(D.time),
                    'slope_mag':slope_mag})
        #if D.size==D.zb.size:
        #    # some LVIS data have a zb field, which is a better estimator of surface elevation than the 'z' field
        #    D.z=D.zb
        D.assign({'sensor':np.zeros_like(D.time)+sensor})
        D.time=matlab_to_year(D.time)
        D0[ind]=D.copy_subset(good, datasets=['x','y','z','time','sigma',
                                              'sigma_corr','sensor',
                                              'slope_mag'])
        if D0[ind].size > 5:
            D0[ind].blockmedian(blockmedian_scale)
    return D0


def read_optical_data(xy0, W, hemisphere=1, GI_files=None, \
              bm_scale=None, N_target=None,\
              target_area=None,\
              SRS_proj4=None,\
              mask_file=None, DEM_file=None, \
              geoid_file=None, water_mask_threshold=None,
              mask_floating=False, time_range=None,
              dem_subset_TF=False, seg_diff_tol=4,
              pgc_url_file=None,
              DEM_dt_min=None,
              DEM_sigma_corr=20,
              problem_DEM_file=None,
              xy_bias_file=None):
    """
    Read laser-altimetry and DEM data from geoIndex files.

    Parameters
    ----------
    xy0 : iterable or list of two numbers
        center of region to read.
    W : numeric
        Width of region to read.
    hemisphere : numeric, optional
        Hemisphere. The default is 1.
    GI_files : dict, optional
        dictionary giving the location of geoindex files for each data type. The default is None.
    bm_scale : dict, optional
        dictionary giving the scale of the blockmedian to apply for laser and DEM data. The default is None.
    DEM_sigma_corr : float, optional
        tolerance on the constraint on the DEM bias.  The default is 20.
    N_target : dict, optional
        dictionary giving the maximum number of data from lasers and DEMs. The default is None.
    target_area : float, optional
        The expected area of the domain after data are selected. If None, will be calculated from W.  The default is None
    SRS_proj4 : str, optional
        projection specification. The default is None.
    mask_file : str, optional
        mask file to use in filtering data location. The default is None.
        DEM file to use in filtering data. The default is None.
    DEM_file : str, optional
        reference DEM against which DEM data will be tested based on DEM_tolerance
    geoid_file : str, optional
        geoid file to use in filtering data. The default is None.
    water_mask_threshold : float, optional
        data less than this elevation above the geoid are rejected. The default is None.
    mask_floating : str, optional
        mask file specifying floating data. The default is False.
    time_range : iterable of floats, optional
        time range within which are accepted.  The default is None
    dem_subset_TF : bool, optional
        If true, DEM data are subsetted to provide one value per year. The default is False.
    problem_DEM_file: str, optional
        File listing DEMs that should be excluded from the analysis
    xy_bias_file: str, optional
        file from which to read xy biases for ICESat-2 data.

    Returns
    -------
    D : pointCollection.data
         returned data.
    sensor_dict : dict
        Dictionary defining sensors associated with data 'sensor' values.
    DEM_meta_dict : dict
        Dictionary giving information about each DEM data file.

    """
    atc_shift_table=None
    laser_dict=laser_key()
    sensor_dict={laser_dict[key]:key for key in ['ICESat1', 'ICESat2', 'ATM','LVIS','riegl']}
    D=[]
    DEM=None
    if DEM_file is not None:
        DEM=pc.grid.data().from_geotif(DEM_file, \
                                       bounds=[ii+np.array([-1.1, 1.1])*W['x'] for ii in xy0])
        if 'z' in DEM.fields:
            DEM.calc_gradient()
    if bm_scale is None:
        bm_scale={'laser':100, 'DEM':200}

    if 'ICESat2' in GI_files:

        D = read_ICESat2(xy0, W, find_gI_files(GI_files['ICESat2']),
                    SRS_proj4=SRS_proj4,
                    sensor=laser_dict['ICESat2'],
                    cplx_accept_threshold=0.25,
                    blockmedian_scale=bm_scale['laser'],
                    N_target=N_target['laser'],
                    seg_diff_tol=seg_diff_tol)

        for Di in D:
            if not hasattr(Di,'x'):
                continue
            Di.assign({'slope_mag':np.sqrt(DEM.interp(Di.x, Di.y, field='z_x')**2+
                                           DEM.interp(Di.x, Di.y, field='z_y')**2)})

            if xy_bias_file is not None:
                if hemisphere==1:
                    EPSG=3413
                else:
                    EPSG=3031
                atc_shift_table = calc_geoloc_bias(Di,
                                        atc_shift_csv_file=xy_bias_file,
                                        atc_shift_table=atc_shift_table,
                                        EPSG=EPSG, move_points=True)
                Di.z += Di.dh_geoloc
                # cleanup additional fields:
                for field in ['h_li','seg_azimuth','ref_azimuth','ref_coelv']:
                    Di.fields.remove(field)

            #if hemisphere==-1:
                # remove cycle 1 (overconstrains 2018-2019 based on not enough data)
                #Di.index(Di.time > 2019.0)

    if 'ICESat1' in GI_files:
        D_IS = read_ICESat(xy0, W, find_gI_files(GI_files['ICESat1']),
                           sensor=laser_key()['ICESat1'],
                           hemisphere=hemisphere, DEM=DEM)
        if D_IS is not None:
            D += D_IS
    if 'LVIS' in GI_files:
        D_LVIS=read_LVIS(xy0, W,
                         find_gI_files(GI_files['LVIS']),
                         blockmedian_scale=bm_scale['laser'], sensor=laser_dict['LVIS'])
        if D_LVIS is not None:
            D += D_LVIS
    if 'ATM' in GI_files:
        D_ATM = read_ATM(xy0, W, find_gI_files(GI_files['ATM'])
                         , blockmedian_scale=bm_scale['laser'], sensor=laser_dict['ATM'])
        if D_ATM is not None:
            D += D_ATM
    DEM_meta_dict=None
    if 'DEM' in GI_files:
        if hemisphere==1:
            year_offset=0
        else:
            year_offset=0.5
        D_DEM, sensor_dict, DEM_meta_dict = read_DEM_data(xy0, W, sensor_dict, \
                            gI_files=find_gI_files(GI_files['DEM']), \
                            hemisphere=hemisphere,
                            blockmedian_scale=bm_scale['DEM'],
                            time_range=time_range,
                            target_area=target_area,
                            N_target=N_target['DEM'],
                            DEM_dt_min=DEM_dt_min,
                            sigma_corr=DEM_sigma_corr,
                            pgc_url_file=pgc_url_file,
                            problem_DEM_file=problem_DEM_file,
                            subset_stack=dem_subset_TF, year_offset=year_offset)
        if D_DEM is not None:
            D += D_DEM
    if len(D)==0:
        return None, None, None

    # two masking steps:
    # delete data over rock and ocean
    if mask_file is not None:
        mask=pc.grid.data().from_geotif(mask_file, bounds=[xy0[0]+np.array([-1, 1])*W['x']*1.1, xy0[1]+np.array([-1, 1])*W['y']*1.1])
        for Di in D:
            if Di is not None and mask_floating:
                Di.index(mask.interp(Di.x, Di.y) > 0.1)

    # if we have a geoid, delete data that are less than 10 m above it
    if geoid_file is not None:
        geoid=pc.grid.data().from_geotif(geoid_file, bounds=[xy0[0]+np.array([-1, 1])*W['x']*1.1, xy0[1]+np.array([-1, 1])*W['y']*1.1])
        for Di in D:
            if Di is not None:
                Di.assign({'geoid':geoid.interp(Di.x, Di.y)})
                if water_mask_threshold  is not None:
                    Di.index((Di.z-Di.geoid) > water_mask_threshold)

    return D, sensor_dict, DEM_meta_dict
