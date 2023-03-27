#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 15:54:48 2019

@author: ben
"""

import h5py
import re
import numpy as np
import pointCollection as pc
import os


def make_sensor_dict(h5f):
    '''
    make a dictionary of sensor numbers for a fit output file.
    
    Input: h5f: h5py file object for the fit.
    Output: dict giving the sensor number for each sensor
    '''
    this_sensor_dict=dict()
    if '/meta/sensors/' not in h5f:
        return this_sensor_dict
     
    sensor_re=re.compile('sensor_(\d+)')
    for sensor_key, sensor in h5f['/meta/sensors/'].attrs.items():
       sensor_num=int(sensor_re.search(sensor_key).group(1))            
       this_sensor_dict[sensor]=sensor_num
    return this_sensor_dict

def reconcile_sensors(sensor_key_list):
    """
    Make a list of the unique sensors from a set of fits
    """
    sensor_list=[]
    for this_dict in sensor_key_list:
        for key in this_dict.keys():
            if key not in sensor_list:
                sensor_list += [key]
    return sensor_list

def remap_sensors(sensor_nums, sensor_key, sensor_list):
    '''
    Assign reconciled sensor numbers 
    '''
    new_sensor_nums=np.zeros_like(sensor_nums, dtype=int)-1
    for key in sensor_key:
        new_sensor_nums[sensor_nums==sensor_key[key]] = sensor_list.index(key)+1
    return new_sensor_nums


def reread_data_from_fits(xy0, W, dir_list, single_file=False,  template='E%d_N%d.h5'):
    """
    Read data from a set of output (fit) files
    
    Inputs:
        xy0: 2-tuple box center
        W: box width
        dir_glob: glob pointing to directories to search
        template: template for file format
    returns:
        pointCollection.data object containing all data read
        list of tile center coordinates for files that were read
    For a directory of files, find the files overlapping the requested box, and 
    select the data closest to the nearest file's center
    """
    if single_file:
        d_i=np.array([0])
        d_j=np.array([0])
    else:
        d_i, d_j = np.meshgrid([-1., 0, 1.], [-1., 0., 1.])

    Lb=1.e4
    db0=(1+1j)*Lb/2
    data_list=[]
 
    index_list=[]
    tile_center_list=[]
    for thedir in dir_list:
        if not os.path.isdir(thedir):
            continue
        for ii, jj in zip(d_i.ravel(), d_j.ravel()):
            this_xy=[xy0[0]+W/2*ii, xy0[1]+W/2*jj]
            this_file=thedir+'/'+template % (this_xy[0]/1000, this_xy[1]/1000)
            if os.path.isfile(this_file):
                tile_center_list += [this_xy]
                with h5py.File(this_file,'r') as h5f:
                    this_data=dict()
                    for key in h5f['data'].keys():
                        this_data[key]=np.array(h5f['data'][key])
                this_data=pc.data(fields=this_data.keys()).from_dict(this_data) 
                # DEBUGGING PLOT
                #plt.plot(this_data.x, this_data.y,'.', markersize=1)
                these=(np.abs(this_data.x-xy0[0])<W/2) & \
                    (np.abs(this_data.y-xy0[1])<W/2)
                this_data.index(these) # & (this_data.three_sigma_edit))
                data_list.append(this_data)
                this_index={}
                # store the rounded coordinates of each point (offset by Lb/2 so that the 
                # geoindex boundary coincides with a bin boundary)
                this_index['xyb']=np.round((this_data.x+1j*this_data.y-db0)/Lb)*Lb+db0 
                # uniqur bins 
                this_index['xyb0']=np.unique(this_index['xyb'])
                #distance from unique bins to geoindex center
                this_index['dist0']=np.abs(this_index['xyb0']-(this_xy[0]+1j*this_xy[1]))
                #ID for geoindex bin
                this_index['N']=len(data_list)-1+np.zeros_like(this_index['xyb0'], dtype=int)
                index_list.append(this_index) 
    if len(index_list)==0:
        return None
    
    index={key:np.concatenate([item[key] for item in index_list]) for key in ['xyb0', 'dist0','N']}
    bins=np.unique(index['xyb0'])
    data_sub_list=[]
    for this_bin in bins:
        #find the bins that match this bin
        these=np.flatnonzero(index['xyb0']==this_bin)
        # find which geoindex's bin centers have the smallest dist0
        this=index['N'][these[np.argmin(index['dist0'][these])]]
        # get the data from that geoindex
        data_sub_list.append(data_list[this][index_list[this]['xyb']==this_bin])
    # return data and sensor key
    fields=data_list[0].fields
    return pc.data(fields=fields).from_list(data_sub_list), file_center_list

#def main():
#    import matplotlib.pyplot as plt
#    W=4e4
#    xy0=np.array([  360000., -2500000.])
#    thedir='/Volumes/ice2/ben/ATL14_test/Jako_d2zdt2=5000_d3z=0.00001_d2zdt2=1500_RACMO/'
#    
#    plt.figure()
#    D1, s1=reread_data_from_fits(xy0, W, thedir, template='E%d_N%d.h5')    
#    plt.scatter(D1.x, D1.y, c=D1.sensor)
    
#if __name__=='__main__':
#    main()
