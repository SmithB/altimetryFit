#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
No-decimation masks: areas in which data are kept at full density

A no-decimation mask is a raster (pointCollection.grid.data, or a file readable
by pointCollection.grid.data().from_file) that is greater than 0.5 where data
should not be decimated.  If the mask has a time dimension, it is evaluated at
each datum's time, so that it can follow features that move with the ice (e.g.
rifts on an ice shelf).  The data readers compute their decimation
(blockmedian) scale from all of the data, as they do without a mask, and apply
it only to the data outside the mask.  Data whose times are outside the time
range of a time-varying mask are evaluated at the mask's first or last epoch.
"""

import numpy as np
import pointCollection as pc


def read_no_decimation_mask(mask, xy0, W, pad=1.e4):
    '''
    Read the part of a no-decimation mask that covers a region

    Parameters
    ----------
    mask : str, pointCollection.grid.data, or None
        mask file (e.g. geotif, or h5 with fields x, y, [t,] z), or a mask that
        has already been read (returned unchanged)
    xy0 : iterable
        center of the region
    W : dict
        width of the region, with entries 'x' and 'y'
    pad : float, optional
        distance by which the region is padded. The default is 1.e4.

    Returns
    -------
    pointCollection.grid.data or None
        None if mask is None, or if the mask does not cover the region (e.g.
        a regional mask, read for a tile outside it)
    '''
    if mask is None or isinstance(mask, pc.grid.data):
        return mask
    bounds = [xy0[0]+np.array([-W['x']/2-pad, W['x']/2+pad]),
              xy0[1]+np.array([-W['y']/2-pad, W['y']/2+pad])]
    try:
        M = pc.grid.data().from_file(mask, bounds=bounds)
    except IndexError:
        # from_h5 raises IndexError for bounds that miss the grid
        return None
    if not _covers_region(M):
        return None
    return M


def _covers_region(mask):
    # True if the mask has a field, and at least two nodes in x and y (fewer
    # cannot be interpolated)
    if mask is None or len(mask.fields) == 0:
        return False
    return all(getattr(mask, coord) is not None and np.size(getattr(mask, coord)) >= 2
               for coord in ('x', 'y'))


def in_no_decimation_mask(mask, x, y, t=None):
    '''
    Find the points that are inside a no-decimation mask

    Parameters
    ----------
    mask : pointCollection.grid.data or None
        the mask.  A 3-D mask is interpolated in time at t, with times outside
        the mask's time range moved to its first or last epoch.
    x, y : numpy arrays
        point coordinates
    t : numpy array, optional
        point times, needed if the mask has a time dimension

    Returns
    -------
    numpy boolean array with the shape of x, True inside the mask
    '''
    if not _covers_region(mask):
        return np.zeros(np.shape(x), dtype=bool)
    field = 'z' if 'z' in mask.fields else mask.fields[0]
    shape = np.shape(x)
    xr, yr = [np.asarray(vals, dtype=float).ravel() for vals in (x, y)]
    if len(getattr(mask, field).shape) == 3:
        if t is None:
            raise ValueError('in_no_decimation_mask: the mask has a time dimension, but the data have no times')
        # the time coordinate that pointCollection.grid.data.interp uses
        band_name = mask._band_coord
        t_mask = np.asarray(getattr(mask, band_name) if band_name else mask.t, dtype=float)
        tr = np.clip(np.asarray(t, dtype=float).ravel(), np.nanmin(t_mask), np.nanmax(t_mask))
        vals = mask.interp(xr, yr, tr, field=field)
    else:
        vals = mask.interp(xr, yr, field=field)
    return (np.nan_to_num(vals) > 0.5).reshape(shape)


def blockmedian_outside_mask(D, scale, mask, field='z'):
    '''
    Blockmedian the points of D that are outside a no-decimation mask

    The points inside the mask are all kept.  D is modified in place (its
    fields become the decimated points outside the mask, followed by the points
    inside it), and is returned.  Without a mask, this is D.blockmedian(scale).
    '''
    if mask is None:
        return D.blockmedian(scale, field=field)
    inside = in_no_decimation_mask(mask, D.x, D.y, D.time if 'time' in D.fields else None)
    if not np.any(inside):
        return D.blockmedian(scale, field=field)
    if np.all(inside):
        return D
    D_out = D.copy_subset(np.flatnonzero(~inside)).blockmedian(scale, field=field)
    for name in D.fields:
        setattr(D, name, np.concatenate([getattr(D_out, name), getattr(D, name)[inside]], axis=0))
    D.__update_size_and_shape__()
    return D
