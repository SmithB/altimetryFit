#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:19:34 2024

@author: mostly Tyler Sutterley, developed for ATL1415
"""

import contextlib
import pointCollection as pc
import numpy as np
try:
    import dask
except ImportError:
    dask = None


def check_tide_model(tide_model, tide_directory=None):
    '''
    check that the tides can be calculated, before any data are read

    apply_tides imports pyTMD and reads the tide model only when it reaches
    ice-shelf data.  This makes the same model lookup as
    pyTMD.compute.tide_elevations, so a missing pyTMD, a model name pyTMD
    doesn't know, or model files missing from tide_directory (e.g. an
    unmounted disk) raise an error at the start of a fit.

    inputs:
        tide_model: tide model name (e.g. 'CATS2008')
        tide_directory: directory holding the tide models
    '''
    if tide_model is None:
        raise ValueError('a tide mask file is set, but no tide model is')
    try:
        import pyTMD
    except ImportError as e:
        raise ImportError(f'tide model {tide_model} needs pyTMD, which could '
                          f'not be imported: {e}') from e
    try:
        pyTMD.io.model(tide_directory).from_database(tide_model)
    except ValueError as e:
        raise ValueError(f'tide model {tide_model}: {e}') from e
    except FileNotFoundError as e:
        raise FileNotFoundError(f'tide model {tide_model} is missing a file '
                                f'in {tide_directory}: {e}') from e

def apply_tides(D, xy0, W,
                tide_mask_file=None,
                tide_mask_shelf_val=1,
                tide_mask_data=None,
                tide_directory=None,
                tide_model=None,
                tide_adjustment=False,
                tide_adjustment_file=None,
                tide_adjustment_format='h5',
                extrapolate=True,
                cutoff=200,
                chunks='auto',
                crs = None,
                threads = None,
                verbose = False):

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
        chunks: kwarg passed to pyTMD
        crs : crs used by D.  proj4 format works
        threads: number of dask threads for pyTMD's chunked calculation
            (default: dask's default, one per core)
        verbose: control output verbosity

    output:
        D: data structure corrected for ocean tides and dac
    '''

    # the tide mask should be 1 for non-grounded points (ice shelves), zero otherwise
    if tide_mask_file is not None and tide_mask_data is None:
        try:
            tide_mask = pc.grid.data().from_geotif(tide_mask_file,
                        bounds=[np.array([-0.6, 0.6])*W+xy0[0], np.array([-0.6, 0.6])*W+xy0[1]])
            tide_mask.z = np.nan_to_num(tide_mask.z)
        except IndexError:
            return None
        if tide_mask.shape is None:
            return
    # find ice shelf points
    is_els = np.abs(tide_mask.interp(D.x, D.y) - tide_mask_shelf_val) < 0.5
    # need to assign the 'floating' field in case the SMB routines need it
    D.assign(floating=is_els)
    if verbose:
        print(f"\t\t{np.mean(is_els)*100}% shelf data")
        print(f"\t\ttide model: {tide_model}")
        print(f"\t\ttide directory: {tide_directory}")
    # extrapolate tide estimate beyond model bounds
    # extrapolation cutoff is in kilometers
    if np.any(is_els.ravel()):
        if np.any(is_els.ravel()):
            # imported here so that altimetryFit can be imported without pyTMD
            import pyTMD
            # cap the thread pool dask uses for pyTMD's chunked calculation
            if threads is not None and dask is not None:
                dask_config = dask.config.set(num_workers=threads)
            else:
                dask_config = contextlib.nullcontext()
            with dask_config:
                # D.time is years after Y2K + 2000 (see read_ICESat2), so the
                # epoch must be 2000: an epoch of 2018 puts the tides 12 h late
                D.tide_ocean = np.array(
                    pyTMD.compute.tide_elevations(
                    D.x, D.y, (D.time-2000)*24*3600*365.25,
                    directory=tide_directory,
                    model=tide_model,
                    infer_minor=True,
                    chunks=chunks,
                    crs=crs,
                    type="drift",
                    epoch=(2000, 1, 1, 0, 0, 0),
                    standard="GPS",
                    method="linear",
                    extrapolate=extrapolate,
                    cutoff=cutoff,
                ))

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
