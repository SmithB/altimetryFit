#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:32:03 2024

@author: ben
"""

import geopandas as gpd
import numpy as np

def get_DEM_ATC_xform(meta):
    '''
    Calculate the transform between projected and along-track coordinates for a DEM

    Parameters
    ----------
    meta : dict
        PGC metadata dictionary
    Returns
    -------
    xform : dict
        dictionary containing origin and basis vectors for the transform to ATC coordinates.
    poly : numpy.array
        Coordinates of the DEM bounding polygon.

    '''

    # read the PGC metadata

    gdf=gpd.GeoDataFrame.from_features([meta])

    if 'bbox' in meta:
        # select the projection based on the first y coordinate of the bounding box
        if meta['bbox'][1] < 0:
            epsg_str='epsg:3031'
        else:
            epsg_str='epsg:3413'

        gdf=gdf.set_crs('epsg:4326').to_crs(epsg_str)

    # find the geometry with the largest area
    areas=[geom.area for geom in gdf.geometry[0].geoms]
    biggest=np.argmax(areas)
    poly = np.c_[[*gdf.geometry[0].geoms[biggest].exterior.coords]]
    ctrs=[tuple(geom.centroid.coords)[0] for geom in gdf.geometry[0].geoms]
    xy0=np.array(ctrs[biggest])

    # the along and across-track vectors are the eigenvectors of the polygon boundary segment differences
    dxy=np.diff( poly[:, 0:2], axis=0 )
    vals, vecs = np.linalg.eig(dxy.T@dxy)
    # numpy >= 2.5 returns complex eigenvectors; the matrix is symmetric, so they are real
    vals, vecs = np.real(vals), np.real(vecs)
    # along track vector is the eigenvector aligned with local north-south
    vec_order=np.argsort(np.abs(xy0@vecs))[::-1]
    xform = {'origin': xy0, 'basis_vectors':vecs[:,vec_order]}

    return xform, poly
