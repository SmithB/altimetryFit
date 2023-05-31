#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 17:53:26 2023

@author: ben
"""
import numpy as np
import pointCollection as pc
import os
import json
import glob
import h5py
from LSsurf.subset_DEM_stack import subset_DEM_stack


def read_problem_DEM_file(GI_files):
    # some DEMs come up as problematic in testing.  Here is where we remove them
    # need to make a 'problem_DEMs.txt' file in the same directory as the GI.
    # Lines should have a filename (no directory) and an optional comment
    problem_DEMs=[]
    for ff in GI_files:
        problem_DEMs_file=os.path.dirname(ff)+'/problem_DEMs.txt'

        if os.path.isfile(problem_DEMs_file):
            with open(problem_DEMs_file,'r') as fh:
                for line in fh:
                    # skip commented lines
                    if line[0] == '#':
                        continue
                    problem_DEMs += [line.split('#')[0].rstrip()]
    return problem_DEMs

def apply_DEM_metadata(D, DEM_meta_config):
    DEM_meta_config_default = {
        'max_offset':20,
        'min_f_valid':10,
        'sigma_percentile_max':99,
        'sigma_max':3,
        'sigma_percentile_for_fit':97.5,
        'sigma_improvement_for_shift':0.25,
        'sigma_f_improvement_for_shift':0.25
        }
    if isinstance(DEM_meta_config, str) and os.path.isfile(DEM_meta_config):
        # assume DEM_meta_config is a file
        with open(DEM_meta_config,'r') as fh:
            DEM_meta_config = json.loads(fh.read())
    if DEM_meta_config is None:
        DEM_meta_config={}
    for key, val in DEM_meta_config_default.items():
        if key not in DEM_meta_config:
            DEM_meta_config[key]=val

    for rep_str in ['_dem_filt.tif','.tif']:
        if rep_str in D.filename:
            break
    meta_file = D.filename.replace(rep_str,'_shift_est.h5')
    if not os.path.isfile(meta_file):
        return {'skip':False,'calc_fine_fit':True}, DEM_meta_config

    meta={}
    try:
        with h5py.File(meta_file,'r') as h5f:
            for key in h5f['meta'].keys():
                meta[key]=np.array(h5f['meta'][key])
        if ( (not np.isfinite(meta['sigma']))
            or meta['sigma'] > DEM_meta_config['sigma_max']
            or (meta['delta_x']**2 + meta['delta_y']**2) > DEM_meta_config['max_offset']**2
            or (('sigma_percentile' in meta) and (meta['sigma_percentile'] > DEM_meta_config['sigma_percentile_max']))
           ):
            return {'skip':True}, DEM_meta_config
        #if meta['sigma_percentile'] > DEM_meta_config['sigma_percentile_for_fit']:
        #    meta['calc_bias_grid']=True
        #else:
        #    meta['calc_bias_grid']=False
    except Exception as e:
        print(f"read_DEM_data:apply_DEM_metadata: reading {meta_file}, encountered error:")
        print(e)
        print('----')
    meta['calc_grid_bias']=True
    delta_var = meta['sigma_unshifted']**2-meta['sigma']**2
    if  delta_var > DEM_meta_config['sigma_improvement_for_shift']**2 and\
        delta_var/meta['sigma']**2 >  DEM_meta_config['sigma_f_improvement_for_shift']**2:
        D.x += meta['delta_x']
        D.y += meta['delta_y']
        meta['shift_applied']=True
    else:
        meta['shift_applied']=False
    dx=D.x-meta['x']
    dy=D.y-meta['y']
    D.z -= dx*meta['dh_dx'] + dy*meta['dh_dy']
    meta['tilt_applied']=True
    return meta, DEM_meta_config

def read_DEM_data(xy0, W, sensor_dict, gI_files=None, hemisphere=1, sigma_corr=20.,
                  blockmedian_scale=100., N_target=None, subset_stack=False, year_offset=0.5,
                  DEM_meta_config=None, DEM_res=32, DEBUG=False, target_area=None, time_range=None):

    if sensor_dict is None:
        sensor_dict={}

    if not isinstance(gI_files, list):
        if os.path.isfile(gI_files):
            gI_files=[gI_files]
        else:
            gI_files=glob.glob(gI_files)

    D=[]
    for this_file in gI_files:
        try:
            D += pc.geoIndex().from_file(this_file, read_file=False)\
                .query_xy_box(xy0[0]+np.array([-W['x']/2, W['x']/2]), xy0[1]+np.array([-W['y']/2, W['y']/2]), \
                              fields=['x','y','z','sigma','time','sensor'])
        except TypeError:
            if DEBUG:
                print(f"No data found for file: {this_file}")

    if time_range is not None:
        D = [Di  for Di in D if (Di.size > 0 ) and not (
             (np.max(Di.time) < time_range[0]) or \
                (np.min(Di.time) > time_range[1])) ]

    if target_area is None:
        target_area=W['x']*W['y']

    if D is None:
        return None, sensor_dict
    if len(sensor_dict) > 0:
        first_key_num=np.max([key for key in sensor_dict.keys()])+1
    else:
        first_key_num=0
    key_num=0
    temp_sensor_dict=dict()

    problem_DEMs = read_problem_DEM_file(gI_files)

    if DEM_meta_config is None:
        DEM_meta_config={}

    # check if there is a _plane_mask.tif file for each DEM, edit if needed.  mask_files indicate 1=discard / 0=keep
    for Di in D:
        mask_file = Di.filename.replace('.tif','_plane_mask.tif')
        if os.path.isfile(mask_file):
            keep = pc.grid.data().from_geotif(mask_file).interp(Di.x, Di.y) < 0.1
            Di.index(keep)

    if D is None or len(D)==0:
        return None, sensor_dict, None

    # if N_target is specified, adjust the blockmedian scale to avoid reading more than
    # N_target points
    if N_target is not None:
        N_total = np.sum([np.sum(np.isfinite(Di.z)) for Di in D])
        N_unique = len(np.unique(np.round(np.concatenate([Di.x+1j*Di.y for Di in D])/DEM_res)))
        overlap = N_total/N_unique # average number of repeats per cell
        if N_total > N_target:
            blockmedian_scale = np.maximum(blockmedian_scale, np.sqrt(target_area/(N_target/overlap)))
            print(f'\t read_DEM_data: DEM blockmedian scale = {np.round(blockmedian_scale)}')

    # copy valid DEMs into a temporary list
    D_temp=[]
    meta_list=[]
    this_sensor_number=-1
    for key_num, Di in enumerate(D):
        this_filename = os.path.basename(Di.filename)
        if this_filename in problem_DEMs:
            continue
        meta, DEM_meta_config = apply_DEM_metadata(Di, DEM_meta_config)
        if 'skip' in meta and meta['skip']:
            continue
        if blockmedian_scale is not None:
            Di.blockmedian(blockmedian_scale)
        if Di.size==0:
            continue
        this_sensor_number += 1
        temp_sensor_dict[this_sensor_number]=this_filename
        Di.assign({'sensor':np.zeros_like(Di.x)+this_sensor_number})
        Di.assign({'sigma_corr':np.zeros_like(Di.x)+sigma_corr})
        if 'sigma' not in Di.fields:
            Di.assign({'sigma':np.zeros_like(Di.x)+0.5})
        D_temp += [Di]
        meta['sensor'] = this_sensor_number
        meta['filename'] = Di.filename
        meta_list += [meta]

    if subset_stack:
        # subset the DEMs so that there is about one per year
        this_bin_width=np.maximum(400, 2*blockmedian_scale)
        DEM_number_list=subset_DEM_stack(D_temp, xy0, W['x'], \
                                         bin_width=this_bin_width,
                                         year_offset=year_offset)
    else:
        DEM_number_list=[int(Di.sensor[0]) for Di in D_temp]

    new_D=[]
    new_meta={}
    for count, num in enumerate(DEM_number_list):
        new_D += [D_temp[num]]
        new_D[-1].sensor[:]=count+first_key_num
        sensor_dict[count+first_key_num] = temp_sensor_dict[num]
        new_meta[count+first_key_num] = meta_list[num]
        new_meta[count+first_key_num]['sensor'] = count+first_key_num

    return new_D, sensor_dict, new_meta
