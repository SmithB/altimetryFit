#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 21:52:23 2013

@author: ben

dem_cull.py
parameters:
    input_file, output_file, smooth_scale, smooth_tol, slope_tol, R_tol, dliate_by
"""

import argparse
from osgeo import gdal, gdalconst
from altimetryFit.im_subset import im_subset
import numpy as np
import scipy.ndimage as snd
import sys, os, time
blocksize=4096
#from altimetryFit import check_obj_memory
import gc

from osgeo import gdal
gdal.SetCacheMax(1024*1024*1024)

# def mem_by_class():
#     objs = gc.get_objects()
#     mems = [(obj.__class__, sys.getsizeof(obj)) for obj in objs]
#     class_mem_count = {}
#     for mem in mems:
#         if mem[0] not in class_mem_count:
#             class_mem_count[mem[0]] = [0, 0]
#         class_mem_count[mem[0]][0] += mem[1]
#         class_mem_count[mem[0]][1] += 1

#     mems = [ii[0] for ii in class_mem_count.values()]
#     classes = list(class_mem_count.keys())

#     ind=np.argsort(mems)[::-1]
#     print("memory usage by_class:")
#     for ii in ind:
#         if mems[ii] > 1.e6:
#             print(f'\t{classes[ii]}:, {mems[ii]/1.e6} M, N={class_mem_count[classes[ii]]} ')

def smooth_corrected(z, mask, w_smooth):
     mask1=snd.gaussian_filter(np.float32(mask), w_smooth, mode="constant", cval=0)
     ztemp=np.nan_to_num(z)
     ztemp[mask==0]=0.0
     zs=snd.gaussian_filter(ztemp, w_smooth, mode="constant", cval=0)
     zs[mask1>0]=zs[mask1>0]/mask1[mask1>0]
     return zs, mask1

def parse_input_args(args):
    '''
    parse_input_args: transform input argument string into a dataspace

    Parameters
    ----------
    args : iterable
        Input arguments.  Keywords should have two hyphens at the start

    Returns
    -------
    dataspace
        input arguments formatted as a namespace

    '''
    
    parser = argparse.ArgumentParser(description='cull out spurious values from a DEM', \
                                     fromfile_prefix_chars='@')
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('--smooth_scale','-s', type=float, default=16.)
    parser.add_argument('--smooth_tol','-t', type=float, default=2.)
    parser.add_argument('--slope_tol', '-m', type=float, default=10*np.pi/180.)
    parser.add_argument('--R_tol','-r', type=float, default=5.)
    parser.add_argument('--erode_By','-b', type=float, default=1.)
    parser.add_argument('--simplify_by', type=float, default=None) 
    parser.add_argument('--decimate_by','-d', type=int, default=1.)
    parser.add_argument('--target_resolution', '-R', type=float, default=None)
    parser.add_argument('--error_RMS_scale','-e', type=float, default=64)
    parser.add_argument('--geolocation_error','-g', type=float, default=5)
    parser.add_argument('--pgc_masks','-p', action='store_true')
    parser.add_argument('--facet_tol', '-f', type=float, default=None)
    return parser.parse_args()


def mask_pgc( this_bounds, mask, pgc_subs, dec):
    for key, sub in pgc_subs.items():
        sub.setBounds(*this_bounds, update=True)
    mask *= np.squeeze((pgc_subs['bitmask'].z == 0) | (pgc_subs['bitmask'].z == 2))
    if dec > 1:
        # skip erosion if the mask is all valid
        if not np.all(pgc_subs['matchtag'].z):
            mask *= snd.binary_erosion(
                snd.binary_erosion(np.squeeze(pgc_subs['matchtag'].z), np.ones((1, dec), dtype=bool)),
                np.ones((dec,1), dtype=bool))
    else:
        mask &= pgc_subs['bitmask'].z

def filter_dem(*args, **kwargs):

    if isinstance(args[0], argparse.Namespace):
        args=args[0]
    elif kwargs is not None:
        for key, value in kwargs.items():
            if not key.startswith('-'):
                key='-'+key
            args += [key, str(value)]
        args=parse_input_args(args)

    ds=gdal.Open(args.input_file);
    driver = ds.GetDriver()
    band=ds.GetRasterBand(1)
    noData=band.GetNoDataValue()
    if noData is None:
        noData=0.
    
    dec=int(args.decimate_by)
    xform_in=np.array(ds.GetGeoTransform())
    dx=xform_in[1]
    #ds=None
    
    if args.target_resolution is not None:
        dec_low=np.floor(args.target_resolution/dx)
        dec_high=np.ceil(args.target_resolution/dx)
        if np.abs(args.target_resolution - dec_low*dx) < np.abs(args.target_resolution - dec_high*dx):
            dec=int(dec_low)
        else:
            dec=int(dec_high)
        print("---chose decimation value based on target resolution: %d" % dec)
    nX=band.XSize;
    nY=band.YSize;
    
    xform_out=xform_in.copy()
    xform_out[1]=xform_in[1]*dec
    xform_out[5]=xform_in[5]*dec
    if np.mod(dec,2)==0:   # shift output origin by 1/2 pixel if dec is even
        xform_out[0]=xform_in[0]+xform_in[1]/2
        xform_out[3]=xform_in[3]+xform_in[5]/2
    
    nX_out=int(nX/dec)
    nY_out=int(nY/dec)
    if args.error_RMS_scale is not None:
        out_bands=[1,2]
        w_error=int(args.error_RMS_scale/dx/dec)
    else:
        out_bands=[1]
    
    if os.path.isfile(args.output_file):
        print("output_file %s exists, deleting" % args.output_file)
        os.remove(args.output_file)
    
    co=["COMPRESS=LZW", "TILED=YES", "PREDICTOR=3"]
    outDs = driver.Create(args.output_file, nX_out, nY_out, len(out_bands), gdalconst.GDT_Float32, options=co)
    
    argDict=vars(args)
    for key in argDict:
        if argDict[key] is not None:
            print("\t%s is %s" %(key, str(argDict[key])))
            outDs.SetMetadataItem("dem_filter_"+key, str(argDict[key]))
    
    if args.smooth_scale is not None:
        # smoothing kernel width, in pixels
        w_smooth=args.smooth_scale/dx
    
    if args.erode_By is not None:
        N_erode=np.ceil(args.erode_By/dx)
        xg,yg=np.meshgrid(np.arange(0, N_erode)-N_erode/2, np.arange(0, N_erode)-N_erode/2)
        k_erode=(xg**2 + yg**2) <= N_erode/2.
    
    if args.simplify_by is not None:
         N_simplify=np.ceil(args.simplify_by/dx)
         xg,yg=np.meshgrid(np.arange(0, N_simplify)-N_simplify/2, np.arange(0, N_simplify)-N_simplify/2)
         k_simplify=(xg**2 + yg**2) <= N_simplify/2.
        
    if args.facet_tol is not None:
        xxg, yyg=np.meshgrid(np.arange(-8., 9), np.arange(-8., 9.))
        opening_kernel=(xxg**2+yyg**2 <= 25)
        closing_kernel=(xxg**2+yyg**2 <= 64)
    
    pad=np.max([1, int(2.*(w_smooth+N_erode)/dec)]);
    if w_error is not None:
         pad=np.max([1, int(2.*w_error+2.*(w_smooth+N_erode)/dec)])
    
    stride=int(blocksize/dec)
    in_sub=im_subset(0, 0, nX, nY, ds, pad_val=0, Bands=[1])
    
    if args.pgc_masks:
        pgc_subs={}
        for key in ['matchtag','bitmask']:
            sub_ds=gdal.Open(args.input_file.replace('_dem.tif','_'+key+'.tif'))
            pgc_subs[key] = im_subset(0, 0, nX, nY, sub_ds, pad_val=0, Bands=[1])
    
    last_time=time.time()
    
    for sub_count, out_sub in enumerate(im_subset(0, 0,  nX_out,  nY_out, outDs, pad_val=0, Bands=out_bands, stride=stride, pad=pad)):
        this_bounds=[out_sub.c0*dec, out_sub.r0*dec, out_sub.Nc*dec, out_sub.Nr*dec]
        #ds=gdal.Open(args.input_file);
        #band=ds.GetRasterBand(1)
        #in_sub=im_subset(0, 0, nX, nY, ds, pad_val=0, Bands=[1])
        in_sub.setBounds(*this_bounds, update=True)
        #ds=None
        #band=None
        z=in_sub.z[0,:,:]
        mask=np.ones_like(in_sub.z[0,:,:])
        mask[np.isnan(in_sub.z[0,:,:])]=0
        mask[in_sub.z[0,:,:]==noData]=0
        
        if args.pgc_masks:
            mask_pgc( this_bounds, mask, pgc_subs, dec)

        out_temp=np.zeros([len(out_bands), stride, stride])
    
        if np.all(mask.ravel()==0):
            out_temp=out_temp+np.NaN
            out_sub.z=out_temp
            out_sub.setBounds(out_sub.c0+pad, out_sub.r0+pad, out_sub.Nc-2*pad, out_sub.Nr-2*pad)
            out_sub.writeSubsetTo(out_bands, out_sub)
            continue
    
        if (args.R_tol is not None) | (args.facet_tol is not None):
            lap=np.abs(snd.laplace(in_sub.z[0,:,:], mode='constant', cval=0.0))
    
        if args.R_tol is not None:
            mask[lap>args.R_tol]=0
    
        if args.facet_tol is not None:
            mask1=mask.copy()
            mask1[lap < args.facet_tol]=0
            mask1=snd.binary_closing(snd.binary_opening(mask1, structure=opening_kernel), structure=closing_kernel)
            #mask1=snd.binary_erosion(mask1, structure=simplify_kernel);
            mask[mask1==0]=0
    
        if args.smooth_scale is not None:
            zs, mask2 = smooth_corrected(z, mask, w_smooth)
            mask[np.abs(in_sub.z[0,:,:]-zs)>args.smooth_tol]=0.
    
        if args.slope_tol is not None:
             gx, gy=np.gradient(zs, dx, dx)
             mask[gx**2+gy**2 > args.slope_tol**2]=0
             z[mask==0]=0
    
        if args.simplify_by is not None:
             mask1=mask.copy()
             mask1=snd.binary_erosion(mask1, k_simplify)
             mask1=snd.binary_dilation(mask1, k_simplify)
             mask[mask1==0]=0
           
        if args.erode_By is not None:
            mask=snd.binary_erosion(mask, k_erode)
            z[mask==0]=0
    
        if args.decimate_by is not None:  # smooth again and decimate
            zs, mask1=smooth_corrected(z, mask, w_smooth)
            zs[mask1 < .25]=0
            mask[mask1 < .25]=0
            if args.error_RMS_scale is not None:
                r2=(z-zs)**2
                r2[mask==0]=0
                r2, dummy=smooth_corrected(r2, mask, w_smooth)
                r2[mask==0]=0
                r2=r2[int(dec/2.+0.5)::dec, int(dec/2.+0.5)::dec]
            zs=zs[int(dec/2.+0.5)::dec, int(dec/2.+0.5)::dec]  # this is now the same res as the output image, includes pad
            #print "decimate:r2.shape=%d %d "%r2.shape
            #print "decimate:zs.shape=%d %d "%zs.shape
            z=zs
            mask=mask[int(dec/2.+0.5)::dec, int(dec/2.+0.5)::dec]
            z[mask==0]=0
    
        if args.geolocation_error is not None:
            gx, gy=np.gradient(zs, dec*dx, dec*dx)
            mask1=snd.binary_erosion(mask, np.ones((3,3)))
            gxs, mask_x=smooth_corrected(gx, mask1, 4)
            gys, mask_y=smooth_corrected(gy, mask1, 4)
            edge_mask=(mask==1) & (mask1==0) & (mask_x >.25)
            gx[mask1==0]=0
            gy[mask1==0]=0
            gx[ edge_mask ] = gxs[edge_mask]
            gy[ edge_mask ] = gys[edge_mask ]
            e2_geo=(gx**2+gy**2)*args.geolocation_error**2
            e2_geo[mask==0]=0
        else:
            e2_geo=np.zeros_like(zs)
    
        if args.error_RMS_scale is not None:
            zss, mask2=smooth_corrected(z, mask, w_error)
            r2s, dummy=smooth_corrected(e2_geo+(zss-z)**2, mask, w_error)
            r2, dummy=smooth_corrected(r2, mask, w_error)
            error_est=np.sqrt(r2s+r2)
            error_est[mask==0]=np.NaN
            out_temp[1,:,:]=error_est[pad:-(pad), pad:-(pad)]
        else:
            out_temp[1,:,:]=e2_geo[pad:-pad, pad:-pad]
    
        z[mask==0]=np.NaN
        out_temp[0,:,:]=z[pad:-(pad), pad:-(pad)]

        out_sub.z=out_temp
        out_sub.setBounds(out_sub.c0+pad, out_sub.r0+pad, out_sub.Nc-2*pad, out_sub.Nr-2*pad)
        out_sub.writeSubsetTo(out_bands, out_sub)
        delta_time=time.time()-last_time
        sys.stdout.write("\r\b %d out of %d, last dt=%f" %(sub_count, out_sub.xy0.shape[0], delta_time))
        sys.stdout.flush()
        last_time=time.time()
        #print('\n')
        #mem_by_class()
        #print('----')
    outDs.SetGeoTransform(tuple(xform_out))
    for b in out_bands:
        outDs.GetRasterBand(b).SetNoDataValue(np.NaN)
    outDs.SetProjection(ds.GetProjection())
    outDs=None

def main(args=None):
    if args is None:
        args=sys.argv
    filter_dem(args)

if __name__=='__main__':
    main()
