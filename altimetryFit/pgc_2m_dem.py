#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 16:38:11 2025

@author: ben
"""
import pointCollection as pc
import os
import numpy as np

class pgc_2m_dem(pc.grid.data):
    def from_geotif(self, file, **kwargs):
        self=pc.grid.data().from_geotif(file, **kwargs)
        matchtag_file=os.path.join(os.path.dirname(file), os.path.basename(file).replace('_dem.tif','_matchtag.tif'))
        bitmask_file=os.path.join(os.path.dirname(file), os.path.basename(file).replace('_dem.tif','_bitmask.tif'))
        if not (os.path.isfile(matchtag_file) and os.path.isfile(bitmask_file)):
            return
        # read the bitmask
        mask = pc.grid.data().from_geotif(bitmask_file, **kwargs)
        mask.z = np.squeeze((mask.z == 0) | (mask.z == 2))
        mask.z &= np.squeeze(pc.grid.data().from_geotif(matchtag_file, **kwargs).z)==1
        self.z[mask.z==0]=np.nan
        return self
