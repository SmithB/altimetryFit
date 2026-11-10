
from osgeo import gdal
import numpy as np
import pointCollection as pc
import scipy.ndimage
import os
from LSsurf.smooth_fit import smooth_fit

gdal.SetConfigOption('AWS_NO_SIGN_REQUEST', 'YES')
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')

def pad_bds(bds, w_pad):
    return [np.array(jj) + np.array([-1, 1])*w_pad/2 for jj in bds]

def _derived_dir(out_dir, name):
    '''
    out_dir with z0 in its last component replaced by name (e.g. z0_filled ->
    sigma_z0_filled), or with _name appended if it does not contain z0
    '''
    head, tail = os.path.split(os.path.normpath(out_dir))
    tail = tail.replace('z0', name) if 'z0' in tail else tail + '_' + name
    return os.path.join(head, tail)

def apply_tide_mask(z0, args):
    mask = pc.grid.data().from_file( args.tide_mask_file,
                                    bounds=z0.bounds(pad=5.e3))
    mask.z=np.nan_to_num(mask.z).astype(int)
    mask.z = mask.z.astype(int) == args.tide_mask_keep_val
    if np.all(mask.z):
        return

    if not np.any(mask.z):
        z0.z_blend[:]=np.nan
        return

    if args.tide_mask_keep_buffer is not None:
        # a (2N+1)x(2N+1) structure dilates the mask by N pixels, about the buffer distance
        N_buffer = int(np.ceil(args.tide_mask_keep_buffer/np.abs(mask.spacing[0])))
        mask.z = scipy.ndimage.binary_dilation(mask.z, np.ones([2*N_buffer+1, 2*N_buffer+1]))
    mask.z = mask.z.astype(float)
    temp = mask.interp(z0.x, z0.y, gridded=True)>0.5
    z0.z_blend[temp==0] = np.nan

def inpaint_DEM_gaps(DEM, z0,
                  mask=None,
                  interp_scale=500,
                  smooth_scale=250):

    """
    Fill gaps in DEM with mosaic output

    The gaps in DEM where z0 is valid are filled with z0 plus a smoothed
    estimate of the DEM - z0 difference.  The other DEM values are unchanged.
    """
    bad_DEM = np.isnan(DEM.z)
    if not np.any(bad_DEM):
        return
    DEM.assign(z0 = z0.interp(DEM.x, DEM.y, gridded=True))
    DEM.assign(z0_masked = z0.interp(DEM.x, DEM.y, field='masked_out', gridded=True))

    # DEM gaps where z0 is valid
    fixable = bad_DEM & (DEM.z0_masked<0.5) & np.isfinite(DEM.z0)
    # Don't bother if there are no pixels that need to be fixed
    if not np.any(fixable):
        return
    N_fill = interp_scale/DEM.spacing[0]
    N_smooth = smooth_scale/DEM.spacing[0]

    DEM.assign(delta = DEM.z - DEM.z0)
    DEM.delta[DEM.z0_masked>0.5]=np.nan

    delta_s, _ = pc.grid.smooth_corrected(DEM.delta, np.isfinite(DEM.delta), N_smooth)
    delta_s[~np.isfinite(DEM.delta)] = np.nan
    delta_s = pc.grid.fill_edges(delta_s, N_fill)

    DEM.z[fixable] = DEM.z0[fixable] + delta_s[fixable]


def fit_delta_surface(z0, fit_spacing=250., E_RMS_dzdx=0.01,
                       sigma_delta=1.0, max_pts=1_000_000, max_iterations=5):
    """
    Fit a smooth surface to z0.delta (= z0.z - z0.DEM) with LSsurf,
    constrained only on the gradient of the surface (no curvature term),
    and return it resampled onto z0's grid.

    Requires z0.delta, z0.x, z0.y, z0.masked_out to already be set
    (as done before the inpainting step in fill_tile/fill_DEM_gaps).
    The input points are decimated so that no more than max_pts points 
    are passed to the fit.
    """
    valid_full = np.isfinite(z0.delta)
    if hasattr(z0, 'masked_out'):
        valid_full &= ~z0.masked_out
    n_valid = valid_full.sum()
    
    z0pt = z0.as_points()
    if hasattr(z0, 'masked_out'):
        z0pt = z0pt[~z0pt.masked_out]
        
    n_valid=z0pt.size
    if n_valid > max_pts:        
        domain_area = np.prod([*map(np.diff,z0.bounds())])
        blockmedian_scale = np.maximum(200, np.sqrt(domain_area/max_pts))
        z0pt=z0pt.blockmedian(blockmedian_scale, field='delta') 
    

    data = pc.data().from_dict({
        'x': z0pt.x, 'y': z0pt.y, 'z': z0pt.delta,
        'time': np.zeros_like(z0pt.delta),
        'sigma': np.zeros_like(z0pt.delta)+sigma_delta})

    x_bds, y_bds = z0.bounds()
    ctr = {'x': np.mean(x_bds), 'y': np.mean(y_bds), 't': 0.}
    # W['t']=0 collapses the dz grid to a single epoch located exactly at
    # ctr['t'], coincident with the data's time=0 values; combined with
    # reference_epoch=0 below, this removes all dz degrees of freedom, so
    # the fit reduces to a static surface on the z0 grid.  'd2z_dxdt' is
    # included only so the dz columns exist for smooth_fit to remove --
    # its value is irrelevant since those columns carry no data weight.
    W = {'x': np.diff(x_bds)[0], 'y': np.diff(y_bds)[0], 't': 0.}
    spacing = {'z0': fit_spacing, 'dz': fit_spacing, 'dt': 1.}
    E_RMS = {'dz0_dx': E_RMS_dzdx, 'd2z_dxdt': E_RMS_dzdx}

    S = smooth_fit(data=data, ctr=ctr, W=W, spacing=spacing, E_RMS=E_RMS,
                    reference_epoch=0, max_iterations=max_iterations,
                    compute_E=False, bias_params=None, dzdt_lags=None,
                    mask_data=None, srs_proj4=z0.srs_proj4)

    delta_s = S['m']['z0']
    delta_s.assign(z=delta_s.z0)
    return delta_s


def fill_tile(bds, z0_file, mos_file,
              cell_area_file=None,
              cell_area_min=None,
              count_file=None,
              sigma_z0_file=None,
              w_pad=1.e4,
              fill_sigma_val = None,
              cache_dir=None,
              epsg=None,
              gap_fill_method='inpaint',
              fit_spacing=250.,
              E_RMS_dzdx=0.01,
              fit_sigma_delta=1.0,
              fit_max_pts=1_000_000,
              fit_max_iterations=5):

    # read data
    bds_pad = pad_bds(bds, w_pad)
    if z0_file.endswith('.tif') or z0_file.endswith('.vrt'):
        # tifs and vrts can be read directly
        z0 = pc.grid.data().from_file(z0_file, bounds=bds_pad)
    elif z0_file.endswith('.json'):
        # if the source is a tilingSchema, need to query the files it contains
        ts = pc.tilingSchema().from_file(z0_file)
        # only the tiles that exist
        files = ts.resolve_files_for_box(bds_pad)
        z0 = pc.grid.mosaic().from_list(files,
                                        bounds=bds_pad,
                                        group='z0',
                                        fields=['z0', 'cell_area', 'count'])
        z0.z0[z0.cell_area < (0.01*np.prod(z0.spacing))] = np.nan
        z0.z = z0.z0

    if 'cell_area' in z0.fields or cell_area_file is not None:
        if cell_area_file is not None:
            z0.assign(cell_area = 
                      np.nan_to_num(
                          pc.grid.data().from_file(cell_area_file, bounds=bds_pad).z))

    if 'count' in z0.fields or count_file is not None:
        if count_file is not None:
            count = pc.grid.data().from_file(count_file, bounds=bds_pad)
            count.z = np.nan_to_num(count.z)
            z0.assign(masked_out = count.z < 0.5)
        else:
            z0.assign(masked_out = z0.count < 0.5)
    else:
        z0.assign(masked_out=~np.isfinite(z0.z))

    if 'cell_area' not in z0.fields:
        raise ValueError('fill_tile: no cell_area for z0: specify a cell_area_file, '
                         'or read z0 from a tilingSchema of tiles that have one')

    z0.z[z0.masked_out] = np.nan

    # read DEM
    DEM = pc.grid.data().from_file(mos_file, bounds=bds_pad)

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        ctr = [int(np.round(np.mean(jj))/1000) for jj in bds]
        dst = os.path.join(cache_dir, 'E%d_N%d.tif' % tuple(ctr))
        DEM.cropped(*bds).to_geotif(dst, srs_epsg=epsg)
    
    DEM_mask = np.isfinite(DEM.z)
    # avoid filling gaps in z0 with off-ice DEM data
    if 'cell_area' in z0.fields:
        z0.z[z0.cell_area < cell_area_min] = np.nan
        DEM.assign(cell_area = z0.interp(DEM.x, DEM.y, gridded=True, field='cell_area'))
        DEM.z[ DEM.cell_area < cell_area_min ] = np.nan

    z0.assign(delta = z0.z - DEM.interp(z0.x, z0.y, gridded=True))
    
    if gap_fill_method=='fit':
        delta_s = fit_delta_surface(z0, fit_spacing=fit_spacing,
                                     E_RMS_dzdx=E_RMS_dzdx,
                                     sigma_delta=fit_sigma_delta,
                                     max_pts=fit_max_pts,
                                     max_iterations=fit_max_iterations)
        z0.assign(delta_s=delta_s.interp(z0.x, z0.y, gridded=True))
    
    # check if there are cells in DEM that can be filled from z0
    if np.any(DEM_mask[DEM.cell_area < cell_area_min]):
        # fill the gaps in the DEM with z0 data (handles places where the DEM is 
        # undefined by z0 is defined)
        if gap_fill_method == 'inpaint':
            inpaint_DEM_gaps(DEM, z0, interp_scale=3.3e3/8)
        else:
            DEM.assign(delta_s = delta_s.interp(DEM.x, DEM.y, gridded=True),
                       z0 = z0.interp(DEM.x, DEM.y, gridded=True))
            to_fill =  (~np.isfinite(DEM.z)) & np.isfinite(DEM.z0)
            DEM.z[to_fill] = DEM.z0[to_fill] - DEM.delta_s[to_fill]
            

    # replace=True: DEM.z has been filled since its interpolator was made
    z0.assign(DEM = DEM.interp(z0.x, z0.y, gridded=True, replace=True))

    if gap_fill_method=='inpaint':
        z0.assign(delta = z0.z - z0.DEM)
        delta_res = z0.spacing[0]
        delta_kernel_w = np.maximum(1, 3.3e3 / (delta_res*8))
        z0.assign(delta_s = pc.grid.fill_edges(z0.delta, delta_kernel_w))

    if not hasattr(z0,'delta_s'):
        # if delta_s has not been defined, it's not needed, can be zero
        z0.assign(delta_s = np.zeros_like(z0.z))

    # fill in the gaps 
    empty_correction = ~np.isfinite(z0.DEM + z0.delta_s)
    z_blend = z0.z.copy()
    to_fill = np.isfinite(z0.DEM) & ~np.isfinite(z0.z)
    z_blend[to_fill] = z0.DEM[to_fill] + z0.delta_s[to_fill]
    if np.any(empty_correction & np.isfinite(z0.z)):
        z_blend[empty_correction] = z0.z[empty_correction]
    z0.assign(z_blend = z_blend)

    if False:
        # diagnostic plot.  Not invoked
        import matplotlib.pyplot as plt
        hf, hax=plt.subplots(2,2, sharex=True, sharey=True, figsize=[10, 10])
        hax=hax.ravel()
        plt.sca(hax[0]); plt.colorbar(z0.show(field='delta', cmap='RdBu', clim=[-2,2]))
        plt.sca(hax[1]); plt.colorbar(plt.imshow(z0.delta-z0.delta_s, **z0.im_kws, clim=[-2, 2], cmap='RdBu'))
        plt.sca(hax[2]); plt.colorbar(plt.imshow(z0.z_blend-z0.DEM, **z0.im_kws, clim=[-2, 2], cmap='RdBu'))
        plt.sca(hax[3]); z0.show(field='z_blend', gradient=True, clim=[-0.1, 0.1])

    return z0

def main():

    import argparse
    parser=argparse.ArgumentParser('fill gaps in one DEM with another')
    parser.add_argument('src_file', type=str,
                        help='file to read for DEM data')
    parser.add_argument('fillin_file', type=str,
                        help='file to read for fillin data')
    parser.add_argument('--EPSG', type=int, required=True,
                        help='output SRS EPSG')
    parser.add_argument('--out_dir', type=str, default='z0_filled',
                        help="output directory")
    parser.add_argument('--sigma_z0_out_dir', type=str,
                        help="sigma_z0 output directory, defaults to out_dir with z0 replaced by sigma_z0"
                        " (or with _sigma_z0 appended)")
    parser.add_argument('--ddx_z0_out_dir', type=str,
                        help="dzdx output directory, defaults to out_dir with z0 replaced by ddx_z0"
                        " (or with _ddx_z0 appended)")
    parser.add_argument('--out_filename', type=str,
                        help="output filename.  Defaults to basename of"
                        " tile_file or z0_ plus bounds")
    parser.add_argument('--base_dir', type=str,
                        help="base directory that will be added to all paths")
    parser.add_argument('--tile_file', type=str,
                        help="template tile to query for bounds")
    parser.add_argument('--bounds', type=float, nargs=4,
                        help='tile bounds, xmin, xmax, ymin, ymax')
    parser.add_argument('--count_file', type=str,
                        help="file to read to determine if DEM data are valid"
                        " (true for values > 0).  Default: count.vrt, if it exists")
    parser.add_argument('--cell_area_file', type=str,
                        help="file to read to determine if the DEM should be "
                        "output (true for values > cell_area_threshold).  Default: cell_area.vrt, if it exists")
    parser.add_argument('--sigma_z0_file', type=str,
                        help='file to read for DEM error estimates.  Default: sigma_z0.vrt, if it exists')
    parser.add_argument('--cell_area_threshold', type=float, default=1000,
                        help='threshold for cell_area below which the DEM will be masked')
    parser.add_argument('--tile_pad', type=float, default=5.e3,
                        help='pad the tile boundary by this amount for calculations')
    parser.add_argument('--fill_sigma_value', type=float, default=1, 
                        help='assign this value to sigma for the fillin DEM')
    parser.add_argument('--tide_mask_file', type=str,
                        help='file containing the tide mask')
    parser.add_argument('--tide_mask_keep_val', type=int, default=0,
                        help="include in the mask only points for which the"
                        " tide mask matches this value")
    parser.add_argument('--tide_mask_keep_buffer', type=float,
                        help="buffer the mask associated with the "
                        "tide_mask_keep value by this distance")
    parser.add_argument('--cache_dir', type=str,
                        help="if specified, DEM data will be saved to this directory")
    parser.add_argument('--gap_fill_method', type=str, default='fit',
                        choices=['inpaint', 'fit'],
                        help="method used to fill gaps in delta=z0-DEM: "
                             "'inpaint' smooths/extrapolates with "
                             "a Gaussian filter; 'fit' (default) fits a smooth surface "
                             "with LSsurf.")
    parser.add_argument('--fit_spacing', type=float, default=250.,
                        help="grid spacing for the LSsurf fit (--gap_fill_method fit)")
    parser.add_argument('--E_RMS_dzdx', type=float, default=0.05,
                        help="expected RMS gradient of the delta surface for the LSsurf fit")
    parser.add_argument('--fit_sigma_delta', type=float, default=0.1,
                        help="assumed sigma of delta=z0-DEM values for the LSsurf fit")
    parser.add_argument('--fit_max_pts', type=int, default=1_000_000,
                        help="maximum number of delta points used in the LSsurf fit "
                             "(input points are decimated to stay at or below this)")
    parser.add_argument('--fit_max_iterations', type=int, default=3,
                        help="max_iterations passed to LSsurf.smooth_fit for the gap-fill fit")
    parser.add_argument('--verbose','-v', action='store_true',
                        help="tell me more?")
    args=parser.parse_args()

    if args.bounds is None and args.tile_file is None:
        parser.error('one of --bounds or --tile_file must be specified')

    if args.verbose:
        print("-"*78)
        print("\tfill_z0.py")

    # the sigma and ddx outputs go to their own directories (they have the
    # z0 file's name, with z0 replaced, which may not change it)
    if args.sigma_z0_out_dir is None:
        args.sigma_z0_out_dir = _derived_dir(args.out_dir, 'sigma_z0')
    if args.ddx_z0_out_dir is None:
        args.ddx_z0_out_dir = _derived_dir(args.out_dir, 'ddx_z0')
    for attr in ['out_dir', 'sigma_z0_out_dir', 'ddx_z0_out_dir']:
        if args.base_dir is not None:
            setattr(args, attr, os.path.join(args.base_dir, getattr(args, attr)))
        os.makedirs(getattr(args, attr), exist_ok=True)
    if args.verbose:
        print(f"\t output to {args.out_dir}, {args.sigma_z0_out_dir}, {args.ddx_z0_out_dir}")

    if args.out_filename is not None:
        out_file = args.out_filename
    elif args.tile_file is not None:
        out_file = os.path.splitext(os.path.basename(args.tile_file))[0]+'.tif'
    else:
        bb = [*map(int, np.array(args.bounds).ravel()/1000)]
        out_file = f'z0_{bb[0]}_{bb[1]}_{bb[2]}_{bb[3]}.tif'
    out_file = os.path.join(args.out_dir, out_file)
    if args.verbose:
        print(f"\t\tworking on {out_file}")

    # input files: a file that is specified must exist (in base_dir, if it is
    # specified, or as given).  count, cell_area, and sigma_z0 files that are
    # not specified are read from their default vrts, if those exist
    default_files = {'count_file': 'count.vrt', 'cell_area_file': 'cell_area.vrt',
                     'sigma_z0_file': 'sigma_z0.vrt'}
    for attr in ['src_file','tile_file','count_file', 'cell_area_file',
                 'sigma_z0_file']:
        temp = getattr(args, attr)
        if temp is None:
            if attr not in default_files:
                continue
            temp = default_files[attr]
            if args.base_dir is not None:
                temp = os.path.join(args.base_dir, temp)
            setattr(args, attr, temp if os.path.exists(temp) else None)
        else:
            if args.base_dir is not None:
                temp_plus = os.path.join(args.base_dir, temp)
                if os.path.exists(temp_plus):
                    setattr(args, attr, temp_plus)
            if not os.path.exists(getattr(args, attr)):
                raise FileNotFoundError(f'missing {getattr(args, attr)}')
        if args.verbose:
            print(f"\t\tfor {attr} using {getattr(args, attr)}")

    if args.bounds is not None:
        args.bounds = [np.array(args.bounds[0:2]), np.array(args.bounds[2:4])]
    else:
        # if bounds are not specified, try to read them from the tile_file
        args.bounds = pc.grid.data().from_file(args.tile_file, meta_only=True, group='z0').bounds()

    z0 = fill_tile(args.bounds, args.src_file, args.fillin_file,
                count_file = args.count_file,
                cell_area_file = args.cell_area_file, 
                cell_area_min = args.cell_area_threshold,
                fill_sigma_val = args.fill_sigma_value,
                w_pad=2*args.tile_pad,
                cache_dir = args.cache_dir,
                epsg=args.EPSG,
                gap_fill_method = args.gap_fill_method,
                fit_spacing = args.fit_spacing,
                E_RMS_dzdx = args.E_RMS_dzdx,
                fit_sigma_delta = args.fit_sigma_delta,
                fit_max_pts = args.fit_max_pts,
                fit_max_iterations = args.fit_max_iterations)

    if args.tide_mask_file is not None:
        apply_tide_mask(z0, args)

    z0.calc_gradient(field='z_blend')


    if args.sigma_z0_file is not None:
        z0.assign(sigma_z0 = pc.grid.data().from_file(args.sigma_z0_file,
                                                     bounds=z0.bounds()).z)
        z0.sigma_z0[z0.masked_out] = args.fill_sigma_value
        z0.sigma_z0[z0.cell_area < args.cell_area_threshold] = np.nan

    z0.crop(args.bounds)
    if hasattr(z0, 'sigma_z0'):
        # no error estimate where there is no z0
        z0.sigma_z0[~np.isfinite(z0.z_blend)] = np.nan

    z0.to_geotif(out_file, field='z_blend', srs_epsg=args.EPSG)
    if hasattr(z0, 'sigma_z0'):
        sigma_z0_out_file = os.path.join(args.sigma_z0_out_dir, 
                                       os.path.basename(out_file).replace('z0','sigma_z0'))
        z0.to_geotif(sigma_z0_out_file, field='sigma_z0', srs_epsg=args.EPSG)
    
    dz0dx_file = os.path.join(args.ddx_z0_out_dir, 
                              os.path.basename(out_file).replace('z0','ddx_z0'))
    
    z0.z_blend_x[z0.cell_area < args.cell_area_threshold] = np.nan
    z0.to_geotif(dz0dx_file,
                 field='z_blend_x', srs_epsg=args.EPSG)
    
    if args.verbose:
        print("-"*78)

if __name__ == '__main__':
    main()
