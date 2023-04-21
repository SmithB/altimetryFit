#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 21:52:23 2013

@author: ben

dem_filter.py
"""

import argparse
from osgeo import gdal, gdalconst
from altimetryFit.im_subset import im_subset
import numpy as np
import scipy.ndimage as snd
import sys, os, time
blocksize=4096
#from altimetryFit import check_obj_memory
import re

gdal.SetCacheMax(1024*1024*256)

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
    parser.add_argument('--pgc_url_file',type=str)
    parser.add_argument('--facet_tol', '-f', type=float, default=None)
    parser.add_argument('--ref_dem', type=str)
    parser.add_argument('--ref_dem_tol', type=float, default=50)
    return parser.parse_args()

def get_pgc_masks(filename, pgc_url_file):
    import requests
    import shutil

    pgc_re=re.compile('(SETSM_.*_seg\d+)')
    pgc_base=pgc_re.search(filename).group(1)

    downscale_re=re.compile('(_(\d+)m).tif')
    m_down = downscale_re.search(filename)
    if m_down is None:
        resample=False
        resample_str=''
    else:
        resample_str = m_down.group(1)
        pgc_re=re.compile('(SETSM_.*_seg\d+)')
        print(filename)
        pgc_base=pgc_re.search(filename).group(1)

    out_files={}
    for extension in ['_matchtag','_bitmask']:
        out_files[extension] = os.path.join(os.path.dirname(filename), pgc_base+extension+resample_str+'.tif')

    if os.path.isfile(out_files['_matchtag']) and os.path.isfile(out_files['_bitmask']):
        return out_files

    pgc_url=None
    with open(pgc_url_file,'r') as fh:
        for line in fh:
            line = line.rstrip()
            if line.endswith(pgc_base):
                pgc_url=line
                break

    # complain if no entry is in the PGC url file
    if pgc_url is None:
        raise(IndexError(f'No PGC url found in {pgc_url_file} for {pgc_base}'))

    # Code for downloading the full-res masks(not necessary - see the option to read the url with gdal.Warp
    #for extension in ['_bitmask','_matchtag']:
    #    with requests.get(pgc_url+extension+'.tif', stream=True) as r:
    #        with open(os.path.join(os.path.dirname(filename), pgc_base+extension+'.tif'),'wb') as fh:
    #            shutil.copyfileobj(r.raw, fh)
    II=gdal.Info(filename, format='json')
    if resample_str is not None:
        for extension, resamp_alg in zip(['_matchtag','_bitmask'],['min','max']):
            warpoptions=gdal.WarpOptions(format="GTiff", outputBounds=II['cornerCoordinates']['lowerLeft']+II['cornerCoordinates']['upperRight'],
                 creationOptions = ['COMPRESS=LZW'],
                 resampleAlg='min', 
                 xRes=np.abs(II['geoTransform'][1]), 
                 yRes=np.abs(II['geoTransform'][-1]), srcNodata=255, dstNodata=255)
            gdal.Warp(out_files[extension], pgc_url+extension+'.tif',options=warpoptions)
            #plt.figure()
            #pc.grid.data().from_geotif(out_file).show()

    return out_files

def get_ref_dem(ref_dem_filename, in_filename):
    """Make the reference dem subset file."""

    pgc_re=re.compile('(SETSM_.*_seg\d+)')

    # Output / destination
    dst_filename = os.path.join(
          os.path.dirname(in_filename), pgc_re.search(in_filename).group(1)+'_refdem.tif')

    II=gdal.Info(in_filename, format='json')
    # Do the work
    warpoptions=gdal.WarpOptions(format="GTiff", outputBounds=II['cornerCoordinates']['lowerLeft']+II['cornerCoordinates']['upperRight'],
                 creationOptions = ['COMPRESS=LZW'],
                 resampleAlg='bilinear', 
                 xRes=np.abs(II['geoTransform'][1]), 
                 yRes=np.abs(II['geoTransform'][-1]))
    gdal.Warp(dst_filename, ref_dem_filename, options=warpoptions)


    return dst_filename

def mask_pgc( this_bounds, mask, pgc_subs, dec ):
    for key, sub in pgc_subs.items():
        sub.setBounds(*this_bounds, update=True)
    mask *= np.squeeze((pgc_subs['_bitmask'].z == 0) | (pgc_subs['_bitmask'].z == 2))
    if dec > 1:
        # skip erosion if the mask is all valid
        if not np.all(pgc_subs['_matchtag'].z):
            mask *= snd.binary_erosion(
                snd.binary_erosion(np.squeeze(pgc_subs['_matchtag'].z), np.ones((1, dec), dtype=bool)),
                np.ones((dec,1), dtype=bool))
    else:
        mask &= pgc_subs['_bitmask'].z

def mask_by_ref_dem( this_bounds, mask, z, ref_dem_sub, dec, tol=50):
    """Use a DEM to update the valid mask."""
    ref_dem_sub.setBounds(*this_bounds, update=True)
    keep = np.abs( z - ref_dem_sub.z ) < tol
    if not np.all(keep):
        keep = snd.binary_erosion(
                snd.binary_erosion(keep, np.ones((1, dec), dtype=bool), border_value=1),
                np.ones((dec,1), dtype=bool, border_value=1))
        mask &= keep

def filter_dem(*args, **kwargs):

    if isinstance(args[0], argparse.Namespace):
        args=args[0]
    elif kwargs is not None:
        for key, value in kwargs.items():
            if not key.startswith('-'):
                key='-'+key
            args += [key, str(value)]
        args=parse_input_args(args)

    in_ds=gdal.Open(args.input_file);
    driver = in_ds.GetDriver()
    band = in_ds.GetRasterBand(1)
    noData=band.GetNoDataValue()
    if noData is None:
        noData=0.

    dec=int(args.decimate_by)
    xform_in=np.array( in_ds.GetGeoTransform())
    dx=xform_in[1]

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
    in_sub=im_subset(0, 0, nX, nY, in_ds, pad_val=0, Bands=[1])

    ref_dem_sub=None
    pgc_subs=None
    if args.pgc_masks:
        try:
             pgc_subs={}
             pgc_files = get_pgc_masks(args.input_file, args.pgc_url_file)
             for key in ['_matchtag','_bitmask']:
                  sub_ds=gdal.Open(pgc_files[key])
                  pgc_subs[key] = im_subset(0, 0, nX, nY, sub_ds, pad_val=0, Bands=[1])
        except IndexError as e:
            print(f"failed to get pgc masks for input file: {args.input_file}")
            print('\t'+str(e))
            pgc_subs = None
    # use the reference DEM if the PGC subs failed
    if args.ref_dem is not None and pgc_subs is None:
        print("Using DEM for masking")
        ref_dem_file = get_ref_dem( args.ref_dem, args.input_file )
        ref_dem_ds = gdal.Open(ref_dem_file, gdalconst.GA_ReadOnly)
        ref_dem_sub = im_subset(0, 0, nX, nY, in_ds, pad_val=0, Bands=[1])
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

        if args.pgc_masks and pgc_subs is not None:
            mask_pgc( this_bounds, mask, pgc_subs, dec)

        if ref_dem_sub is not None:
            mask_by_ref_dem(this_bounds, mask, z, ref_dem_sub, dec, tol=args.ref_dem_tol)

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
    outDs.SetProjection(in_ds.GetProjection())
    outDs=None

def main(args=None):
    if args is None:
        args=sys.argv
    filter_dem(args)

if __name__=='__main__':
    main()
