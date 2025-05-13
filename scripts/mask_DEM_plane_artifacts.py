#! /usr/bin/env python
"""Makes masks that identify planar artifacts in DEMs."""

import numpy as np
import pointCollection as pc
import scipy.ndimage as snd
import sys
import scipy.stats as sps
import shutil
import os

def histogram_peak_stats(x, ctr=None, x_max=None, sided=1):
    """
    Find the cender and width of a histogram peak.

    Parameters
    ----------
    x : numpy array
        Data to histogram.
    ctr : float, optional
        Predicted peak location. The default is None.
    x_max : float, optional
        maximum data value to include in the histogram. The default is None.
    sided : int, optional
        calculate the one-sided or two-sided width of the histogram. The default is 1.

    Returns
    -------
    ctr : float
        Peak location.
    fwhm : TYPE
        Full width at half maximum of the peak.

    """
    if x_max is None:
        x_max=np.nanmax(x)
    n_bins=int(np.ceil(2*np.sqrt(np.sum(x<=x_max))))
    N, edges = np.histogram(x[x<=x_max], n_bins)
    bin_ctrs=(edges[0:-1]+edges[1:])/2

    if sided==1:
        ctr=0

    if ctr is None:
        # first true bin
        i_ctr=np.argmax(N==np.max(N))
        ctr=bin_ctrs[i_ctr]
    else:
        i_ctr = np.argmin(bin_ctrs-ctr)
    i_hm = [0, 0]
    if sided==2:
        i_hm[0] = np.argmax(N>N[i_ctr]/2)-1 # first true
        if (i_hm[0] < 0) or (i_hm[0] > i_ctr):
            i_hm[0]=0
    if i_ctr == len(bin_ctrs):
        i_hm[1] = i_ctr
    else:
        i_hm[1] = i_ctr + np.argmax(N[i_ctr:]> N[i_ctr]/2)
    fwhm = edges[i_hm[1]+1]-edges[i_hm[0]]
    return ctr, fwhm

def two_model_stats(D,  DEM_file, dx=500, N_min=100, v_ratio_max = [0.01, 0.1]):
    """
    Compare small regions of a DEM to another DEM and to a plane.

    Parameters
    ----------
    D : pointCollection.grid.data
        input DEM.  must have a 'z' field
    DEM_file : str
        DEM file against which to compare D
    v_ratio_max : 2-element list, optional
        ratio between plane misfit and DEM misfit.  labeled regions with ratio
        less than this are assumed to match the plane. The default is [0.01, 0.1].

    Returns
    -------
    l_stats : dict
        dict giving the statistics for labeled regions.
    xy0 : iterable (2)
        center of the DEM.
    slope_centroid : iterable (2)
        centroid of the plane slopes.
    slope_sigma : iterable (2)
        standard deviation of the plane slopes.

    """
    gx, gy=np.meshgrid(D.x, D.y)
    gx=gx.ravel()
    gy=gy.ravel()

    labels, label_dict=pc.unique_by_rows(np.round(np.c_[gx.ravel(), gy.ravel()]/dx)*dx,
                                         return_dict=True)

    xy0=np.array([np.mean(D.x), np.mean(D.y)])
    label_stats={}

    dz=D.z - pc.grid.data().from_geotif(DEM_file, bounds=D.bounds())\
                            .interp(D.x, D.y, gridded=True)
    dz=dz.ravel()
    z=D.z.ravel()
    label_stats={}
    for this_label, ii  in label_dict.items():
        ii=label_dict[this_label]
        ii=ii[np.isfinite(dz[ii])]
        if len(ii) < N_min:
            continue
        G=np.c_[np.ones(len(ii)), gx[ii]-xy0[0], gy[ii]-xy0[1]]
        m=np.linalg.solve(G.T.dot(G), G.T.dot(z[ii]))
        r_plane=z[ii]-G.dot(m)

        m_REF = np.linalg.solve(G.T.dot(G), G.T.dot(dz[ii]))
        r_REF = dz[ii]-G.dot(m_REF)
        #m_REF = np.mean(r_REF)
        #r_REF -= m_REF

        label_stats[this_label]= {
            'rms0':np.sqrt(np.mean(z[ii]**2)),
            'bar0':np.mean(z[ii]),
            'sigma_plane':np.std(r_plane),
            'z0_plane':m[0],
            'dzdx_plane':m[1],
            'dzdy_plane':m[2],
            'rms_dz':np.sqrt(np.nanmean(dz[ii]**2)),
            'sigma_dz':np.std(dz[ii]),
            'rms_REF':np.sqrt(np.mean(r_REF**2)),
            'sigma_REF':np.std(r_REF),
            'dz_REF': np.nanmean(dz[ii]),
            'N':len(ii)}
    labels=list(label_stats.keys())

    if len(labels) == 0:
        return None, None, xy0

    l_stats={field:[] for field in label_stats[labels[0]].keys()}
    l_stats['label']=np.array(np.c_[labels])
    for label, li in label_stats.items():
        for field in li.keys():
            l_stats[field]+= [li[field]]
    for field in l_stats.keys():
        l_stats[field]=np.array(l_stats[field])

    return l_stats, label_dict, xy0

def select_slope_mask_by_large_scale_fit(LS2, xy0, VERBOSE=False):
    """
    Identify regions that have a consistent slope and mask them.

    Parameters
    ----------
    LS2 : dict
        statistics from two_model_stats.
    xy0 : iterable
        Center of the DEM.
    VERBOSE : TYPE, optional
        Set to True to see intermediate steps The default is False.

    Returns
    -------
    numpy array
        best-fitting planar model for outliers.
    numpy array
        per-region residuals to the planar model.
    iterable
        centroid of the slopes of the outlier regions.
    iterable
        standard devaitions of slopes for the outlier regions.

    """
    N_min=50
    # initial guess of deviant fits
    ctr, whm_rms_dz= histogram_peak_stats(LS2['rms_dz'], sided=2)
    dz_bar = np.median(LS2['dz_REF'][LS2['rms_dz'] < ctr+3*whm_rms_dz])

    # recalculate rms_dz after shift for mean non-outlier offset
    rms_dz = np.sqrt((LS2['dz_REF']-dz_bar)**2 + LS2['sigma_dz']**2)
    if VERBOSE:
        print(f"dz_bar={dz_bar:2.3f}, p84_dz_ref={sps.scoreatpercentile(LS2['rms_dz'], 84)}, p84_dz_ref_corr={sps.scoreatpercentile(rms_dz, 84)}")

    ctr, whm_rms_dz= histogram_peak_stats(rms_dz, sided=2)
    outlier_rms_dz = np.maximum(10, ctr+3*whm_rms_dz)

    G=np.c_[np.ones_like(LS2['label'][:,0]), LS2['label'][:,0]-xy0[0], LS2['label'][:,1]-xy0[1]]
    these =  (LS2['sigma_plane']< 2)
    these &= (rms_dz > outlier_rms_dz)

    if VERBOSE:
        print(f"\twhm_rms_dz={whm_rms_dz}, n_outliers={these.sum()}")

    if np.sum(these) < N_min:
        if VERBOSE:
            print("\treturning because there are not enough outlier points.")
        return None, None, None, None

    for count in range(2):
        G1=G[these,:]
        # fit the pixel center elevation with a plane:
        m=np.linalg.inv(G1.T.dot(G1)).dot(G1.T.dot(LS2['bar0'][these]))
        r=LS2['bar0']-G.dot(m)
        sigma_hat = pc.RDE(r[these])
        these = (LS2['sigma_plane']< 2) & (rms_dz > outlier_rms_dz) & (np.abs(r) < np.maximum(10, 3*sigma_hat))

        if np.sum(these) < N_min:
            if VERBOSE:
                print("\treturning because there are not enough outlier points.")
            return None, None, None, None
    slope_ctr = [np.median(LS2['dzdx_plane'][these]), np.median(LS2['dzdy_plane'][these])]
    slope_sigma = [pc.RDE(LS2['dzdx_plane'][these]), pc.RDE(LS2['dzdy_plane'][these])]

    # reject if the plane fit slope is not inside the tolerance selected slope points
    if (m[1] < slope_ctr[0] - 3*slope_sigma[0]) | (m[1]>slope_ctr[0] + 3*slope_sigma[0]) \
        or (m[2] < slope_ctr[1] - 3*slope_sigma[1]) | (m[2]>slope_ctr[1] + 3*slope_sigma[1]):
        if VERBOSE:
            print("returning because the plane slope does not match the slopes of the facets")
            print([m, slope_ctr, slope_sigma])
        return None, None, None, None

    # reject if the selected slopes include zero
    if (0 > slope_ctr[0] - 3*slope_sigma[0]) & (0 < slope_ctr[0] + 3*slope_sigma[0]) \
        and (0 > slope_ctr[1] - 3*slope_sigma[1]) & (0 < slope_ctr[1] + 3*slope_sigma[1]):
        if VERBOSE:
            print("returning because the slope tolerances overlap zero")
            print([m, slope_ctr, slope_sigma])
        return None, None, None, None

    LS2['DEM_vs_slope_mask']=(np.abs(r)< np.maximum(10, 3*sigma_hat))\
        & (LS2['sigma_plane'] < 2) \
        & (np.abs(LS2['dzdx_plane']-slope_ctr[0]) < 3*slope_sigma[0]) \
        & (np.abs(LS2['dzdy_plane']-slope_ctr[1]) < 3*slope_sigma[1])
    return m, r, slope_ctr, slope_sigma


def map_stats(D, l_stats, label_dict, val):
    """
    Map a value into a field of a grid based on a label field.

    Parameters
    ----------
    D : pointCollection.grid.data
        input grid object
    label: numpy array
        field labeling the pixels in D
    l_stats : dict
        dictionary giving desciptive fields for each label value
    val : np.array
        label value to map (same size as stats['label'])

    Returns
    -------
    temp : pointCollection.grid.data
        grid object whose z value contains the mapped field

    """
    temp=pc.grid.data().from_dict({'x':D.x, 'y':D.y,'z':np.zeros([len(D.y), len(D.x)])+np.nan})
    #temp.z=temp.z.ravel()
    for ii, ll in enumerate(l_stats['label']):
        jj = label_dict[tuple(ll)]
        temp.z.ravel()[jj] = val[ii]
    return temp

def main(argv):
    """
    Make a planar artifact mask file for a DEM file.

    Parameters
    ----------
    argv : iterable
        arguments.

    Returns
    -------
    None.

    """
    import argparse
    parser=argparse.ArgumentParser(description="mask planar artifacts in a DEM", \
                                   fromfile_prefix_chars="@")
    parser.add_argument('dem_file', type=str, help="DEM file to test")
    parser.add_argument('ref_dem_file', type=str, help="reference DEM file")
    parser.add_argument('--out_file','-o', type=str, help="output file.  If none specified, will write to dem_file with _plane_mask.tif suffix")
    parser.add_argument('--EPSG', type=int, default=3413, help='output grid file EPSG')
    parser.add_argument('--dx', type=float, default=500, help='resolution at which to test the DEM')
    args=parser.parse_args()

    D=pc.grid.data().from_geotif(args.dem_file)
    if len(D.z.shape) > 2:
        D.z=D.z[:,:,0]

    if args.out_file is None:
        args.out_file=args.dem_file.replace('.tif','_plane_mask.tif')
    if args.dem_file==args.out_file:
        print(f"out_file matches dem_file for {args.dem_file}, returning.")
        return
    if os.path.isfile(args.out_file):
        os.remove(args.out_file)
    gx, gy=np.meshgrid(D.x, D.y)
    gx=gx.ravel()
    gy=gy.ravel()
    D.assign({'label': np.round((gx+1j*gy)/args.dx)*args.dx})

    LS2, label_dict, xy0 = two_model_stats(D,  args.ref_dem_file)

    if LS2 is None:
        print("mask_DEM_plane_artifacts.py: did not find enough data, returning")
        print("\t file="+args.dem_file)
        return

    m, r, slope_centroid, slope_sigma = select_slope_mask_by_large_scale_fit(LS2, xy0, VERBOSE=True)

    if slope_centroid is None:
        return

    mask = map_stats(D, LS2, label_dict, LS2['DEM_vs_slope_mask'])
    temp=np.zeros(mask.shape, dtype=bool)

    # mask marks the points that should be rejected
    temp[mask.z==1] = 1
    mask.z=temp

    Nk=np.ceil(args.dx/(D.x[1]-D.x[0]))
    kx, ky = np.meshgrid(np.arange(-Nk, Nk+0.1), np.arange(-Nk, Nk+1))
    K = (kx**2+ky**2) <= Nk**2
    mask.z = snd.binary_dilation(mask.z, K)


    mask.to_geotif(args.out_file, srs_epsg=args.EPSG)

    print(f"mask_DEM_plane_artifacts: file={args.dem_file}, N={np.sum(mask.z)},  slope_x={slope_centroid[0]}, slopey={slope_centroid[1]}, sigma_sx={slope_sigma[0]}, slope_sy={slope_sigma[1]}")
    return

if __name__=='__main__':
    main(sys.argv)


# mask_DEM_plane_artifacts.py /Volumes/insar10/ben//ArcticDEM/v2/2019_early/backups/SETSM_s2s041_WV02_20190326_103001008D53D400_103001008D70E500_32m_lsf_seg3_dem_filt.tif /Volumes/ice3/ben/COPdem/NPS_0240m_h_WGS84.tif --EPSG 3413
