import numpy as np
import pointCollection as pc
import scipy.ndimage as snd


def make_coverage_grid(D, full_grid):
    temp=pc.data().from_dict({'x':D.x,'y':D.y,'z':np.isfinite(D.z).astype(float)})
    return pc.points_to_grid(temp, None, full_grid.copy())

def remove_overlapping_DEM_data(Ds, bounds, res, dt_min=0.125):

    times=np.array([np.nanmedian(Di.time) for Di in Ds])
    D_of_t={tt:DD for  DD, tt in zip(Ds, times)}
    dt_min=1/8
    DEM_masks = {}
    masking_changed=True
    xx=np.arange(bounds[0][0]-res, bounds[0][1]+res*1.1, res)
    yy=np.arange(bounds[1][0]-res,bounds[1][1]+res*1.1, res)
    full_grid = pc.grid.data().from_dict(
        {'x':xx,
        'y':yy,
        'z':np.zeros((len(yy), len(xx)))+np.NaN})

    print(full_grid.shape)
    while masking_changed:
        masking_changed=False
        for this_time, this_D in D_of_t.items():
            i_overlaps = np.flatnonzero(np.abs(times-this_time) < dt_min)
            if len(i_overlaps)<2:
                continue
            t_overlap=[]
            N_overlap=[]
            sigma_overlap=[]
            overlap_grids=[]
            # find overlapping grids, make sure that data masks and data subsets exist for each
            for i_overlap in i_overlaps:
                time_1=times[i_overlap]
                if time_1 not in DEM_masks:
                    DEM_masks[time_1] = make_coverage_grid(D_of_t[time_1], full_grid)
            # for each overlapping grid, count the overlapping points, calculate the mean sigma for the overlap
            for i_overlap in i_overlaps:
                time_1=times[i_overlap]
                if time_1==this_time:
                    continue
                D1 = D_of_t[time_1]
                D0 = D_of_t[this_time]
                overlap_mask = snd.binary_dilation(np.isfinite(DEM_masks[time_1].z + DEM_masks[this_time].z), np.ones((3,3)))
                this_N=np.sum(overlap_mask)
                if this_N == 0:
                    continue
                N_overlap += [this_N]
                t_overlap += [[this_time,time_1]]
                overlap_grid = pc.grid.data().from_dict({'x':full_grid.x,'y':full_grid.y,'z':overlap_mask})
                overlap_grids += [overlap_grid]
                sigma_overlap += [[np.nanmean(Dsub.sigma[overlap_grid.interp(Dsub.x, Dsub.y) > 0.1]) for Dsub in [D0, D1]]]
            if len(N_overlap)==0:
                continue
            t_overlap=np.array(t_overlap)
            sigma_overlap=np.array(sigma_overlap)
            if not np.any(np.isfinite(sigma_overlap)):
                continue
            # find the largest overlaps, and the largest sigma among the largest overlaps
            largest=np.flatnonzero(N_overlap==np.max(N_overlap))
            rmax,cmax=np.where(sigma_overlap[largest,:]==np.nanmax(sigma_overlap[largest,:]))
            rmax=rmax[0]
            cmax=cmax[0]
            # eliminate points from the largest sigma from among the largest overlaps
            t_eliminate = t_overlap[largest[rmax],cmax]
            # find the points to be eliminated:
            eliminate_points = overlap_grids[largest[rmax]].interp(D_of_t[t_eliminate].x, D_of_t[t_eliminate].y)>0.1
            if np.all(eliminate_points==0):
                continue
            D_of_t[t_eliminate].index(eliminate_points==0)
            DEM_masks[t_eliminate] = make_coverage_grid(D_of_t[t_eliminate], full_grid)
            masking_changed=True
    return [Di for Di in Ds if Di.size > 1]
