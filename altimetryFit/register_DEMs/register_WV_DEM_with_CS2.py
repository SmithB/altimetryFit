#! /usr/bin/env python


import os
import pointCollection as pc
import numpy as np
import scipy.ndimage as snd
import glob
import re
from datetime import date
import sys
from pyTMD import compute_tide_corrections
from dateutil import parser
import json

import altimetryFit.register_DEMs as rd

def matlab_to_year(t):
    # approximate conversion of matlab date to year.  Uses the matlab conversion
    # datestr('jan 1 2000') -> 730486

    return (t-730486.)/365.25+2000.


def read_poca_data(bounds, t_DEM, tile_dir, max_dt=1):
    fields=['x','y','time','h', 'power','coherence', \
            'abs_orbit']
    
    year_subs=glob.glob(tile_dir+'/2*')
    year_re=re.compile('.*/(2\d{3})')
    year_dir={float(year_re.match(sub).group(1)):sub for sub in year_subs}
    D=[]
    pad = np.array([-1.e4, 1.e4])
    for year, dirname in year_dir.items():
        if (np.abs(t_DEM-year) < max_dt) or (np.abs(t_DEM-(year+1)) < max_dt):
            GI_files=glob.glob(os.path.join(dirname, '*', 'POCA', 'GeoIndex.h5'))
            for index_file in GI_files:   
                D += [pc.data().from_list(pc.geoIndex().from_file(index_file).query_xy_box(bounds[0]+pad, \
                               bounds[1]+pad, fields=fields))]

    if D is not None:
        D=pc.data().from_list(D)
    if D is None:
        return D

    D.time = matlab_to_year(D.time)
    # remove suspect returns
    D.index((D.power > 5e-17) & 
            (D.power < 5e-13) & 
            (np.abs(t_DEM-D.time) < max_dt))
    
    # remap fieldnames, assign errors
    D.assign({
        'z':D.h,
        't':D.time,
        'sigma':50*0.01+ np.maximum(0, -0.64*(np.log10(D.power)+14)),
        'sigma_corr':0.13+np.zeros_like(D.x)})
    return D

def calc_tide(D_pt, t_DEM, directory=None, model=None, mask_file=None):
    
    mask_i=pc.grid.data().from_geotif(mask_file).interp(D_pt.x, D_pt.y)>0.01
    if not np.any(mask_i):
        return
    temp=np.zeros_like(D_pt.x) 
    temp[mask_i]=compute_tide_corrections(
                    D_pt.x[mask_i], D_pt.y[mask_i], (D_pt.time[mask_i]-2000)*365.25*24*3600,
                    DIRECTORY=directory, MODEL=model,
                    EPOCH=(2000,1,1,0,0,0), TYPE='drift', TIME='utc')
    
    
    D_pt.assign({'tide':np.array(temp)})
    temp=np.zeros_like(D_pt.x) 
    temp[mask_i]=compute_tide_corrections(
                    D_pt.x[mask_i], D_pt.y[mask_i], 
                    (t_DEM-2000+np.zeros(mask_i.sum()))*365.25*24*3600,
                    DIRECTORY=directory, MODEL=model,
                    EPOCH=(2000,1,1,0,0,0), TYPE='drift', TIME='utc')
    D_pt.assign({'DEM_tide':temp})

def get_DEM_date(filename):
    meta_file=filename.replace('_dem_40m_filt.tif','_meta.txt')

    with open(meta_file,'r') as fh:
        for line in fh:
            if "Image_1_Acquisition_time=" in line:

                timestr = line.split('=')[1]
                
                break
    return (parser.isoparse(timestr.rstrip()) - parser.isoparse('2000-01-01T00:00:00Z')).days/365.25+2000

def output_filenames(filename):
    file_re=re.compile('(.*)_dem(.*)filt.tif')
    out_base = file_re.search(filename).groups()[0]
    
    out_files={'h5': out_base+'_CS2_meta.h5',
               'json':out_base+'_CS2_meta.json'}
    return out_files
    
def make_CS2_queue(args):
    files=glob.glob(args.queue_wc)
    arg_list=sys.argv.copy()
    # remove the queue argument:
    queue_arg = arg_list.index('--queue_wc')
    if queue_arg is None:
        queue_arg = arg_list.index('-q')
    arg_list.pop(queue_arg+1)
    arg_list.pop(queue_arg)
    # remove the name of the script:
    prog = arg_list.pop(0)
    arg_string= ' '.join(arg_list)
    
    with open('register_queue.txt','w') as fh:
        for file in files:
            out_txt=output_filenames(file)['json']
            if os.path.isfile(out_txt):
                continue
            fh.write(f'{prog} --DEM_file {file} '+ arg_string + '\n')
    sys.exit(0)
    

def write_h5_output(D_out, D_BM, out_file):

    # need every field in D_out to be a non-zero-dimension array
    D_out=pc.data().from_dict({key:np.array(item)[None] for key, item in D_out.items()})

    if os.path.isfile(out_file):
        os.remove(out_file)
    D_out.to_h5(out_file, group='meta', extensible=False)
    if D_BM is not None:
        D_BM.to_h5(out_file, group='stats_2km', replace=False)
        
        
def write_json_output( D_out, json_file ):
    with open(json_file,'w') as fh:
        json.dump(D_out, fh, indent=4)

def register_one_DEM(DEM_file=None, tile_dir=None,
                     mask_file=None,
                     max_dt = 1,
                     tide_model=None, tide_directory=None,
                     tide_mask_file=None,
                     sigma_min=0.1, sigma_max=5, save_data=True, verbose=False):

    D_out={'x':np.NaN,
           'y':np.NaN,
           'N':0,
           'N_days': 0,
           'sigma':np.NaN,
           'sigma_scaled':np.NaN,
           'dh_dx':np.NaN,
           'dh_dy':np.NaN,
           'dh_dt':np.NaN,
           'delta_h':np.NaN}

    DEM=pc.grid.data().from_geotif(DEM_file)
    DEM.z[DEM.z==0]=np.NaN
    DEM.t=get_DEM_date(DEM.filename)

    out_files =  output_filenames(DEM.filename)
    
    XR, YR= [[np.floor(ii[0]/1.e4)*1.e4, np.ceil(ii[1]/1.e4)*1.e4] for ii in DEM.bounds()]
    for ii in [XR, YR]:
        if ii[0] == ii[1]:
            ii += np.array([-1.e4, 1.e4])
            
    D_pt = read_poca_data((XR, YR), DEM.t, tile_dir, max_dt=max_dt)
    
    calc_tide(D_pt, DEM.t, directory=tide_directory, model=tide_model, mask_file=tide_mask_file)
    D_pt.z -= D_pt.tide
       
    mask_file = DEM_file.replace('.tif','_plane_mask.tif')
    if os.path.isfile(mask_file):
        keep = pc.grid.data().from_geotif(mask_file).interp(D_pt.x, D_pt.y) < 0.1
        D_pt.index(keep)
    
    if D_pt.size < 5:
        if 'x' in D_pt.fields:
            D_pt.assign({'r': np.zeros_like(D_pt.x)+np.NaN})
        print('register_WV_DEM_with_IS2.py: not enough valid points found for ' + DEM.filename)
        write_json_output( D_out, out_files['json'])
        return
    
    D_pt.sigma = np.sqrt(D_pt.sigma**2+pc.grid.data()\
                     .from_dict({'x':DEM.x,'y':DEM.y,'z':np.squeeze(DEM.z[:,:,1])}).interp(D_pt.x, D_pt.y)**2)

    D_pt.index( (D_pt.sigma < sigma_max) & \
                    np.isfinite(D_pt.sigma))

    mask0 = np.ones(D_pt.size, dtype=bool)

    print(f'\tregister one DEM: after initial search, found {np.sum(mask0)}')

    sigma0, sigma_scaled0, mask0, m0, r0, dh0 = rd.eval_DEM_shift([0,0], D_pt, DEM, sigma_min=sigma_min, iterations=50, mask=mask0)

    mask0, r0 = rd.edit_by_day(mask0, D_pt, DEM, r0, dh_max=5, sigma_min=0.1)
    D_pt.assign({'r0':r0})
    D_pt=D_pt[mask0]
    mask0=mask0[mask0]

    print("starting second iteration")
    sigma, sigma_scaled, mask, m, r, dh  = rd.eval_DEM_shift([0,0], D_pt, DEM, sigma_min=sigma_min, iterations=50, mask=mask0)
    D_pt.assign({'r':r})


    if np.sum(mask) < 55:
        print('register_WV_DEM_with_IS2.py: not enough valid points found for ' + DEM.filename)
        write_json_output( D_out, out_files['json'])
        return
    
    D_out={'x':np.float64(np.mean(D_pt.x)),
           'y':np.float64(np.mean(D_pt.y)),
           'N':D_pt.size,
           'N_days': len(np.unique(np.round(D_pt.t*365.25))),
           'sigma':sigma,
           'sigma_scaled':sigma_scaled,
           'dh_dx':m[1]/1000,
           'dh_dy':m[2]/1000,
           'dh_dt':m[3],
           'delta_h':m[0]}

    
    D_BM = rd.calc_blockmedians(D_pt, mask, scale=2000)
    
    if save_data:
        print("writing h5")
        write_h5_output(D_out, D_BM, out_files['h5'])

    print("Writing json")
    write_json_output( D_out, out_files['json'])
    print("Done writing")
    return m, D_pt, DEM, D_out, D_BM,  sigma, sigma_scaled

def main():

    import argparse
    parser=argparse.ArgumentParser(description='Find the best vertical offset for a DEM relative to a set of altimetry data', fromfile_prefix_chars='@')
    parser.add_argument("--DEM_file", '-d', type=str)
    parser.add_argument("--tile_dir", type=str)
    parser.add_argument("--sigma_min", type=float, default=0.1)
    parser.add_argument("--sigma_max", type=float, default=5)
    parser.add_argument("--tide_directory", type=str)
    parser.add_argument("--tide_model", type=str)
    parser.add_argument("--tide_mask_file", type=str)
    parser.add_argument("--queue_wc", '-q', type=str)
    parser.add_argument("--verbose",'-v', action="store_true")
    args=parser.parse_args()

    if args.queue_wc is not None:
        make_CS2_queue(args)
        sys.exit(0)

    register_one_DEM(**{key:val for key, val in vars(args).items() if val is not None and key not in ['queue_wc']})

if __name__=='__main__':
    main()
