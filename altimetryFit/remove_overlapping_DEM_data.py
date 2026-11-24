import numpy as np
import pointCollection as pc
import scipy.ndimage as snd


def make_coverage_grid(D, full_grid):
    if D.size == 0:
        return full_grid.copy()
    temp=pc.data().from_dict({'x':D.x,'y':D.y,'z':np.isfinite(D.z).astype(float)})
    return pc.points_to_grid(temp, None, full_grid.copy())

def remove_overlapping_DEM_data(Ds, bounds, res, dt_min=0.125, times=None):
    '''
    Thin the data where DEMs that are close in time overlap

    For each DEM, the DEMs whose times are within dt_min of its time are
    found, and among the pairs with the largest overlap, the data in the
    overlap of the DEM with the larger mean sigma (in the overlap) are
    removed.  This repeats until nothing changes.

    Parameters
    ----------
    Ds : list of pointCollection.data
        data for each DEM.  They are edited in place.
    bounds : list
        x and y bounds of the area
    res : float
        resolution of the grid on which the overlaps are found
    dt_min : float, optional
        DEMs closer in time than this overlap.  The default is 0.125 (years).
    times : iterable of floats, optional
        time of each DEM.  The default is the median of each DEM's time
        field, which read_DEM_data sets to the full acquisition time from
        the DEM's metadata.

    Returns
    -------
    list of pointCollection.data
        the DEMs with more than one datum left
    '''
    # DEMs are identified by their index in Ds: several can have the same time
    # (e.g. the segments of a strip, or DEMs whose times are their dates)
    if times is None:
        times=np.array([np.nanmedian(Di.time) for Di in Ds])
    times=np.asarray(times, dtype=float)
    DEM_masks = {}
    masking_changed=True
    xx=np.arange(bounds[0][0]-res, bounds[0][1]+res*1.1, res)
    yy=np.arange(bounds[1][0]-res,bounds[1][1]+res*1.1, res)
    full_grid = pc.grid.data().from_dict(
        {'x':xx,
        'y':yy,
        'z':np.zeros((len(yy), len(xx)))+np.nan})

    print(full_grid.shape)
    while masking_changed:
        masking_changed=False
        for i0, D0 in enumerate(Ds):
            i_overlaps = np.flatnonzero(np.abs(times-times[i0]) < dt_min)
            if len(i_overlaps)<2:
                continue
            i_pairs=[]
            N_overlap=[]
            sigma_overlap=[]
            overlap_grids=[]
            # find overlapping grids, make sure that data masks and data subsets exist for each
            for i1 in i_overlaps:
                if i1 not in DEM_masks:
                    DEM_masks[i1] = make_coverage_grid(Ds[i1], full_grid)
            # for each overlapping grid, count the overlapping points, calculate the mean sigma for the overlap
            for i1 in i_overlaps:
                if i1==i0:
                    continue
                D1 = Ds[i1]
                overlap_mask = snd.binary_dilation(np.isfinite(DEM_masks[i1].z + DEM_masks[i0].z), np.ones((3,3)))
                this_N=np.sum(overlap_mask)
                if this_N == 0:
                    continue
                N_overlap += [this_N]
                i_pairs += [[i0, i1]]
                overlap_grid = pc.grid.data().from_dict({'x':full_grid.x,'y':full_grid.y,'z':overlap_mask})
                overlap_grids += [overlap_grid]
                sigma_overlap += [[np.nanmean(Dsub.sigma[overlap_grid.interp(Dsub.x, Dsub.y) > 0.1]) for Dsub in [D0, D1]]]
            if len(N_overlap)==0:
                continue
            i_pairs=np.array(i_pairs)
            sigma_overlap=np.array(sigma_overlap)
            if not np.any(np.isfinite(sigma_overlap)):
                continue
            # find the largest overlaps, and the largest sigma among the largest overlaps
            largest=np.flatnonzero(N_overlap==np.max(N_overlap))
            if not np.any(np.isfinite(sigma_overlap[largest,:])):
                continue
            rmax,cmax=np.where(sigma_overlap[largest,:]==np.nanmax(sigma_overlap[largest,:]))
            rmax=rmax[0]
            cmax=cmax[0]
            # eliminate points from the largest sigma from among the largest overlaps
            i_eliminate = i_pairs[largest[rmax],cmax]
            # find the points to be eliminated:
            eliminate_points = overlap_grids[largest[rmax]].interp(Ds[i_eliminate].x, Ds[i_eliminate].y)>0.1
            if np.all(eliminate_points==0):
                continue
            Ds[i_eliminate].index(eliminate_points==0)
            DEM_masks[i_eliminate] = make_coverage_grid(Ds[i_eliminate], full_grid)
            masking_changed=True
    return [Di for Di in Ds if Di.size > 1]
