#! /usr/bin/env python


import os
import pointCollection as pc
import numpy as np
import glob
import re
import json
import sys

def out_filenames(file):
    out_files={}
    if '_dem_filt' in file:
        out_files['h5']=file.replace('_dem_filt.tif','_shift_est.h5')
    else:
        out_files['h5']=file.replace('.tiff','.tif').replace('.tif','_shift_est.h5')
    out_files['json'] = out_files['h5'].replace('.h5','.json')
    return out_files

def make_queue(args):
    files=glob.glob(args.queue_wc)
    arg_list=sys.argv.copy()
    queue_arg = arg_list.index('--queue_wc')
    if queue_arg is None:
        queue_arg = arg_list.index('-q')
    arg_list.pop(queue_arg+1)
    arg_list.pop(queue_arg)
    GI_arg = arg_list.index('--GeoIndex_wc')
    if GI_arg is None:
        GI_arg = arg_list.index('-g')
    arg_list[GI_arg+1] = '"'+arg_list[GI_arg+1]+'"'
    arg_list.pop(0)
    arg_string= ' '.join(arg_list)
    with open('register_queue.txt','w') as fh:
        for file in files:
            out_files=out_filenames(file)
            #if os.path.isfile(out_files['h5']):
            #    continue
            fh.write(f'register_WV_DEM_with_IS2.py --DEM_file {file} '+ arg_string + '\n')
    sys.exit(0)

def select_DEM_pts(D_pt, DEM, max_dist, max_dt):
    good = np.abs(D_pt.t - DEM.t) < max_dt

    deltas = np.meshgrid(*[np.array([-1, 0, 1]) for ii in [0, 1]])
    for dx, dy in zip(deltas[0].ravel(), deltas[1].ravel()):
        good[good] &= np.isfinite(DEM.interp(D_pt.x[good]+dx, D_pt.y[good]+dy, band=0))
    return good

def eval_DEM_shift(delta, D_pt, DEM, sigma_min=0, iterations=1, mask=None):

    dh = D_pt.z - DEM.interp(D_pt.x+delta[0], D_pt.y+delta[1], band=0)

    if 'DEM_tide' in D_pt.fields:
        dh += D_pt.DEM_tide

    G=np.c_[np.ones_like(D_pt.x).ravel(),\
            (D_pt.x.ravel()-np.nanmean(D_pt.x))/1000,\
            (D_pt.y.ravel() - np.nanmean(D_pt.y))/1000, \
            (D_pt.t.ravel() - DEM.t)]

    if mask is None:
        mask = np.ones_like(D_pt.x, dtype=bool)
    #try:
    mask &= np.isfinite(dh)
    #except Exception:
    #    print("HERE")
    last_mask = mask
    iteration=0
    while iteration==0 or (iteration < iterations and not np.all(last_mask==mask)):
        Cinv=(1/(np.maximum(sigma_min, D_pt.sigma[mask]))**2)
        try:
            m=np.linalg.solve(G[mask,:].transpose().dot(np.tile(Cinv[:, None], [1, G.shape[1]])*G[mask,:]), G[mask,:].transpose().dot(Cinv*dh[mask]))
        except np.linalg.LinAlgError:
            m = np.zeros(G.shape[1])
            m[0] = np.nanmean(dh)
            print(f"linalg error at iteration {iteration}")
        r=dh-G.dot(m)
        rs = r/np.maximum(sigma_min, D_pt.sigma)
        sigma_hat = pc.RDE(rs[mask])
        #print([iteration, sigma_hat, np.mean(mask), [np.min(rs[mask]), np.max(rs[mask])]])
        last_mask=mask
        mask &= (np.abs(rs) < 3*sigma_hat)
        iteration += 1
    mask=last_mask
    sigma_scaled = np.std(rs[mask])
    sigma = np.std(r[mask])
    return sigma, sigma_scaled, mask, m, r, dh

def edit_by_day(mask, D_pt, DEM, r0, dh_max=5, sigma_min=0.1):
    # Evaluate the median per-day bias, reject any with absolute values gt 0.5
    day=(D_pt.t*365.25).astype(int)
    DB=np.c_[[(uD, np.nanmedian(r0[mask & (day==uD)])) for uD in np.unique(day)]]
    bad_days = np.abs(DB[:,1]) > 5

    while np.any(bad_days):
        worst_day = np.argmax(np.abs(DB[:,1]))
        mask[np.in1d(day, DB[worst_day,0])] = False
        sigma0, sigma_scaled0, mask, m0, r0, dh0 = eval_DEM_shift([0,0], D_pt, DEM,
                                                                   sigma_min=sigma_min, iterations=50, mask=mask)
        DB=np.c_[[(uD, np.nanmedian(r0[mask & (day==uD)])) for uD in np.unique(day[mask])]]
        if DB.shape[0] > 0:
            bad_days = np.abs(DB[:,1]) > dh_max
        else:
            break
    return mask, r0

def search_offsets(D_pt, DEM, max_delta=10, delta_tol = 1, sigma_min=0.02):

    R_of_delta={}
    delta=max_delta/2
    best_offset = [0, 0]
    count=0
    while delta >= delta_tol:
        #print(delta)
        deltas = [ii.ravel() for ii in np.meshgrid(*[np.array([-1, 0, 1])*delta for jj in [0, 1]])]
        for dx, dy  in zip(deltas[0], deltas[1]):
            this_offset = (best_offset[0] + dx, best_offset[1]+dy)
            #print('\t'+str(this_offset))
            if np.any(np.abs(this_offset)> max_delta):
                continue
            if this_offset not in R_of_delta:
                R_of_delta[this_offset] = eval_DEM_shift(this_offset, D_pt, DEM, sigma_min=sigma_min, iterations=1)[1]
                #print('\t'+str([this_offset, R_of_delta[this_offset]]))
        searched=list(R_of_delta.keys())
        Rvals = [R_of_delta[ii] for ii in searched]
        best_offset = searched[np.argmin(Rvals)]
        #print('best_offset:'+str(best_offset))
        shrink = True
        for dx, dy in zip(deltas[0], deltas[1]):
            test_offset = (best_offset[0]+dx, best_offset[1]+dy)
            #print([test_offset, test_offset in R_of_delta])
            if test_offset not in R_of_delta and np.all(np.abs(test_offset) <= max_delta):
                #print("blocking shrink for "+str(test_offset))
                #print('\t not in R_of_delta: ' +str(test_offset not in R_of_delta))
                #print('\t LT max: ' +str(np.all(np.abs(test_offset) <= max_delta)))
                shrink=False
                break
        if shrink:
            delta /= 2
        count += 1
        if count > 20:
            print('quitting because count is too large\n')
            break

    return best_offset, R_of_delta

def calc_blockmedians(D_pt, mask, scale=2000):

    def med_spread(D, ind):
        return np.nanmedian(D.r[ind]), pc.RDE(D.r[ind])
    if np.sum(mask)>0:
        D_BM=pc.apply_bin_fn(D_pt[mask], scale, fn=med_spread, fields=['r_median','r_spread'])
        return D_BM
    else:
        return None

def write_json_output( D_out, json_file ):

    D_dict = {field:getattr(D_out, field).astype(float)[0] for field in D_out.fields}

    with open(json_file,'w') as fh:
        json.dump(D_dict, fh, indent=4)

def write_output(D_out, D_BM, filename, D_pt=None, DEBUG=False):

    # need every field in D_out to be a non-zero-dimension array
    D_out=pc.data().from_dict({key:np.array(item)[None] for key, item in D_out.items()})

    out_files=out_filenames(filename)

    D_out.to_h5(out_files['h5'], group='meta', extensible=False, replace=True)

    if D_pt is not None and DEBUG:
        D_pt.to_h5(out_files['h5'], group='data', extensible=False, replace=False)

    write_json_output(D_out, out_files['json'])

    if D_BM is not None:
        D_BM.to_h5(out_files['h5'], group='stats_2km', replace=False)

def register_one_DEM(DEM_file=None, GeoIndex_wc=None,
                     mask_file=None, max_dist=20,
                     sigma_min=0.1, sigma_max=5, 
                     save_data=True, min_data_points=50,
                     DEBUG=False, verbose=False):
    D_out={'x':np.NaN,
           'y':np.NaN,
           'N':0,
           'N_days': 0,
           'sigma':np.NaN,
           'sigma_unshifted':np.NaN,
           'sigma_scaled':np.NaN,
           'sigma_scaled_unshifted':np.NaN,
           'delta_x':np.NaN,
           'delta_y':np.NaN,
           'dh_dx':np.NaN,
           'dh_dy':np.NaN,
           'dh_dt':np.NaN,
           'delta_h':np.NaN,}

    out_files=out_filenames(DEM_file)
    for out_file in out_files.values():
        if os.path.isfile(out_file):
            os.remove(out_file)

    DEM=pc.grid.data().from_geotif(DEM_file)
    #if DEM.z.ndim==3:
    #    DEM=DEM[:,:,0]
    DEM.z[DEM.z==0]=np.NaN
    DEM.t=pc.grid.DEM_year(DEM.filename)
    XR, YR= [[np.floor(ii[0]/1.e4)*1.e4, np.ceil(ii[1]/1.e4)*1.e4] for ii in DEM.bounds()]
    for ii in [XR, YR]:
        if ii[0] == ii[1]:
            ii += np.array([-1.e4, 1.e4])

    DEM_cycle=np.floor((DEM.t-2018)*4-2)

    # read data from all cycles within 2 (0.5 yr) of the DEM
    cycle_re=re.compile('/cycle_(\d\d)/')
    D_pt=[]
    for GI_file in glob.glob(GeoIndex_wc):
        GI_cycle=int(cycle_re.search(GI_file).group(1))
        if np.abs(GI_cycle - DEM_cycle) > 2:
            continue
        try:
            D_pt += pc.geoIndex().from_file(GI_file).query_xy_box(XR, YR, fields=['x','y','delta_time', 'h_li', 'h_li_sigma','atl06_quality_summary'])
        except TypeError:
            # TypeError is thrown if the return from query_xy_box is None
            pass
    D_pt = pc.data().from_list(D_pt)

    mask_file = DEM_file.replace('.tif','_plane_mask.tif')
    if os.path.isfile(mask_file):
        keep = pc.grid.data().from_geotif(mask_file).interp(D_pt.x, D_pt.y) < 0.1
        D_pt.index(keep)

    if D_pt.size < 5:
        if 'x' in D_pt.fields:
            D_pt.assign({'r': np.zeros_like(D_pt.x)+np.NaN})
        print('register_WV_DEM_with_IS2.py: not enough valid points found for ' + DEM.filename)
        write_output(D_out, None, DEM.filename)
        return

    D_pt.assign({'t':D_pt.delta_time/24/3600/365.25+2018,
                 'z':D_pt.h_li,
                'sigma':D_pt.h_li_sigma})

    D_pt.sigma=np.sqrt(D_pt.sigma**2+pc.grid.data().from_dict({'x':DEM.x,'y':DEM.y,'z':np.squeeze(DEM.z[:,:,1])}).interp(D_pt.x, D_pt.y)**2)
    D_pt.index( (D_pt.sigma < sigma_max) & \
                (D_pt.atl06_quality_summary==0) & \
                np.isfinite(D_pt.sigma))

    mask0 = np.ones(D_pt.size, dtype=bool)
    if max_dist > 0:
        shift_list=[-max_dist, 0, max_dist]
    else:
        shift_list=[0]
    for dx in shift_list:
        for dy in shift_list:
            mask0 &= np.isfinite(DEM.interp(D_pt.x+dx, D_pt.y+dy, band=0))

    if np.sum(mask0) < 5:
        D_pt.assign({'r':np.zeros_like(D_pt.x)+np.NaN})
        print('register_WV_DEM_with_IS2.py: not enough valid points found for ' + DEM.filename)
        write_output(D_out, None, DEM.filename, D_pt=D_pt, DEBUG=DEBUG)
        return

    print(f'\tregister one DEM: after initial editing, found {np.sum(mask0)}')

    if DEBUG:
        D_pt.to_h5(out_filenames(DEM.filename)['h5'], group='input_data')

    sigma0, sigma_scaled0, mask0, m0, r0, dh0 = eval_DEM_shift([0,0], D_pt, DEM, sigma_min=sigma_min, iterations=50, mask=mask0)

    mask0, r0 = edit_by_day(mask0, D_pt, DEM, r0, dh_max=5, sigma_min=0.1)
    D_pt=D_pt[mask0]
    r0=r0[mask0]
    dh0=dh0[mask0]
    D_pt.assign({'r':r0})

    if np.sum(mask0) < min_data_points:
        print('register_WV_DEM_with_IS2.py: not enough valid points found for ' + DEM.filename)
        write_output(D_out, None, DEM.filename, D_pt=D_pt, DEBUG=DEBUG)
        return

    if max_dist > 0:
        delta_best, R_of_delta = search_offsets(D_pt, DEM, max_delta=20, delta_tol=1, sigma_min=sigma_min)
    else:
        delta_best = (0,0)
    sigma, sigma_scaled, mask, m, r, dh = eval_DEM_shift(delta_best, D_pt, DEM, sigma_min=sigma_min, iterations=1)
    sigma0, sigma_scaled0, mask0, m0, r0, dh0 = eval_DEM_shift([0,0], D_pt, DEM, sigma_min=sigma_min, iterations=1)
    if max_dist==0:
        R_of_delta = {delta_best : sigma0}

    D_out={'x':np.mean(D_pt.x),
           'y':np.mean(D_pt.y),
           'N':D_pt.size,
           'N_days': len(np.unique(np.round(D_pt.t*365.25))),
           'sigma':sigma,
           'sigma_unshifted':sigma0,
           'sigma_scaled':sigma_scaled,
           'sigma_scaled_unshifted':sigma_scaled0,
           'delta_x':-delta_best[0],
           'delta_y':-delta_best[1],
           'dh_dx':m[1]/1000,
           'dh_dy':m[2]/1000,
           'dh_dt':m[3],
           'delta_h':m[0]}

    D_pt.assign({'r':r,
                'r0':r0})
    D_BM=calc_blockmedians(D_pt, mask0, scale=2000)

    if save_data:
        write_output(D_out, D_BM, DEM.filename, D_pt=D_pt, DEBUG=DEBUG)

    return delta_best, m, D_pt, DEM, D_out, D_BM, R_of_delta, [sigma0, sigma], [sigma_scaled0, sigma_scaled]

def main():

    import argparse
    parser=argparse.ArgumentParser(description='Find the best offset for a DEM relative to a set of altimetry data', fromfile_prefix_chars="@")
    parser.add_argument("--DEM_file", '-d', type=str)
    parser.add_argument("--GeoIndex_wc", '-g', type=str)
    parser.add_argument("--mask_file", '-m', type=str)
    parser.add_argument("--max_dist", type=float, default=32)
    parser.add_argument("--sigma_min", type=float, default=0.1)
    parser.add_argument("--sigma_max", type=float, default=5)
    parser.add_argument("--queue_wc", '-q', type=str)
    parser.add_argument("--DEBUG", action='store_true')
    parser.add_argument("--verbose",'-v', action="store_true")
    args=parser.parse_args()

    if args.queue_wc is not None:
        make_queue(args)
        sys.exit(0)

    register_one_DEM(**{key:val for key, val in vars(args).items() if val is not None and key not in ['queue_wc']})

if __name__=='__main__':
    main()
