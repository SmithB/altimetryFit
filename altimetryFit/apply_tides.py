#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:19:34 2024

@author: mostly Tyler Sutterley, developed for ATL1415
"""

import pointCollection as pc
import numpy as np
import pyTMD

def apply_tides(D, xy0, W,
                tide_mask_file=None,
                tide_mask_data=None,
                tide_directory=None,
                tide_model=None,
                tide_adjustment=False,
                tide_adjustment_file=None,
                tide_adjustment_format='h5',
                extrapolate=True,
                cutoff=200,
                EPSG=None,
                verbose=False):

    '''
    read in the tide mask, calculate ocean tide elevations, and
    apply dynamic atmospheric correction (dac) and tide to ice-shelf elements

    inputs:
        D: data structure
        xy0: 2-element iterable specifying the domain center
        W: Width of the domain


    keyword arguments:
        tide_mask_file: geotiff file for masking to ice shelf elements
        tide_mask_data: pc.grid.data() object containing the tide mask (alternative to tide_mask_file)
        tide_directory: path to tide models
        tide_model: the name of the tide model to use
        tide_adjustment: adjust amplitudes of tide model amplitudes to account for ice flexure
        tide_adjustment_file: File for adjusting tide and dac values for ice shelf flexure
        tide_adjustment_format: file format of the scaling factor grid
        extrapolate: extrapolate outside tide model bounds with nearest-neighbors
        cutoff: extrapolation cutoff in kilometers

    output:
        D: data structure corrected for ocean tides and dac
    '''

    # the tide mask should be 1 for non-grounded points (ice shelves), zero otherwise
    if tide_mask_file is not None and tide_mask_data is None:
        try:
            tide_mask = pc.grid.data().from_geotif(tide_mask_file,
                        bounds=[np.array([-0.6, 0.6])*W+xy0[0], np.array([-0.6, 0.6])*W+xy0[1]])
        except IndexError:
            return None
        if tide_mask.shape is None:
            return
    # find ice shelf points
    is_els=tide_mask.interp(D.x, D.y) > 0.5
    # need to assign the 'floating' field in case the SMB routines need it
    D.assign(floating=is_els)
    if verbose:
        print(f"\t\t{np.mean(is_els)*100}% shelf data")
        print(f"\t\ttide model: {tide_model}")
        print(f"\t\ttide directory: {tide_directory}")
    # extrapolate tide estimate beyond model bounds
    # extrapolation cutoff is in kilometers
    if np.any(is_els.ravel()):

        D.assign(tide_ocean = pyTMD.compute_tide_corrections(\
                D.x, D.y, (D.time-2000)*24*3600*365.25,
                DIRECTORY=tide_directory, MODEL=tide_model,
                EPOCH=(2000,1,1,0,0,0), TYPE='drift', TIME='UTC',
                crop=True,
                EPSG=EPSG, EXTRAPOLATE=extrapolate, CUTOFF=cutoff))

    # use a flexure mask to adjust estimated tidal values
    if np.any(is_els.ravel()) and tide_adjustment:
        print(f"\t\t{tide_adjustment_file}") if verbose else None
        D.assign({'tide_adj_scale': np.ones_like(D.x)})
        # read adjustment grid and interpolate to values
        tide_adj_scale = pc.grid.data().from_file(tide_adjustment_file,
            file_format=tide_adjustment_format, field_mapping=dict(z='tide_adj_scale'),
            bounds=[np.array([-0.6, 0.6])*W+xy0[0], np.array([-0.6, 0.6])*W+xy0[1]])
        # interpolate tide adjustment to coordinate values
        D.tide_adj_scale[:] = tide_adj_scale.interp(D.x, D.y)
        # mask out scaling factors where grounded
        D.tide_adj_scale[is_els==0]=0.0
        # multiply tides and dynamic atmospheric correction by adjustments
        ii, = np.nonzero(D.tide_adj_scale != 0)
        D.tide_ocean[ii] *= D.tide_adj_scale[ii]
        if hasattr(D, 'dac'):
            D.dac[ii] *= D.tide_adj_scale[ii]
        if verbose:
            print(f'mean tide adjustment scale={np.nanmean(D.tide_adj_scale)}')

    # replace invalid tide and dac values
    if hasattr(D, 'tide_ocean'):
        D.tide_ocean[is_els==0] = 0
        D.tide_ocean[~np.isfinite(D.tide_ocean)] = 0
        D.z -= D.tide_ocean
    if hasattr(D, 'dac'):
        D.dac[is_els==0] = 0
        D.dac[~np.isfinite(D.dac)] = 0
        D.z -= D.dac

    return D
