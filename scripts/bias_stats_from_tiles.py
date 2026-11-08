#! /usr/bin/env python


import pointCollection as pc
import numpy as np
import pandas as pd
import re
import os
import glob
import h5py
import sys

import argparse

def make_queue(args, xys):
    xy0 = pc.unique_by_rows(np.round(xys/args.width)*args.width)
    for xx in xy0:
        cmd = " ".join([f"bias_stats_from_tiles.py {args.tile_directory}",
            f"--width {args.width}",
            f"--xy0 {xx[0]} {xx[1]}",
            f"--out_dir {args.out_dir}"])
        print(cmd)

def main():
    parser = argparse.ArgumentParser("bias_stats_from_tiles.py")
    parser.add_argument("tile_directory", type=str, help="directory in which to find tiles")
    parser.add_argument("--width", "-w", type=float, help="width to process, in km")
    parser.add_argument("--xy0", type=float, nargs=2, help="center of area to process, in km")
    parser.add_argument("--out_dir",'-o', type=str)
    parser.add_argument("--queue", action="store_true", help="if flag set, make a queue of processing commands")
    args=parser.parse_args()

    xy_re=re.compile('/E(.*)_N(.*).h5')
    files=glob.glob(os.path.join(args.tile_directory+'/E*.h5'))
    xy_file={tuple([*map(int, xy_re.search(file).groups())]):file for file in files}
    xys=np.c_[list(xy_file.keys())]

    if args.out_dir is None:
        args.out_dir=os.path.join(args.tile_directory, 'bias_stats')
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    if args.queue:
        make_queue(args, xys)
        sys.exit(0)



    jj = (np.abs(xys[:,0]-args.xy0[0])<=args.width/2)\
            & (np.abs(xys[:,1]-args.xy0[1]) <= args.width/2)
    tiles=[xy_file[tuple(kk)] for kk in xys[jj,:]]

    all_sensors=['ICESat1','ICESat2','ATM','LVIS','riegl']

    not_DEM_sensors=[]

    collected_stats={}
    tile_re=re.compile('prelim/E(.*)_N(.*).h5')
    bias_fields=None
    not_bias_fields=['N_edited','N_raw','expected','rms_data_edited','rms_data_raw','val']
    for count, tile in enumerate(tiles):
        x0, y0 = [*map(int, tile_re.search(tile).groups())]
        B=pc.data().from_h5(tile, group='bias')
        if bias_fields is None:
            bias_fields=[field for field in B.fields if field not in not_bias_fields]
        sensor_dict={}
        with h5py.File(tile,'r') as h5f:
            for key, val in h5f['/meta/sensors'].attrs.items():
                sensor_dict[int(key.split('_')[1])] = val
                if 'SETSM' not in val and val not in not_DEM_sensors:
                    not_DEM_sensors += [val]
        for sensor in sensor_dict.values():
            if sensor not in all_sensors:
                all_sensors += [sensor]

        for count, sensor in enumerate(B.sensor):
            t_stats={}
            t_stats['sensor_ind'] = all_sensors.index(sensor_dict[int(sensor)])
            t_stats['x0'] = x0
            t_stats['y0'] = y0
            t_stats['bias'] = B.val[count]
            for field in bias_fields + ['rms_data_edited', 'rms_data_raw','N_edited','N_raw']:
                t_stats[field] = getattr(B, field)[count]
            collected_stats_key = sensor_dict[sensor]
            if 'SETSM' in collected_stats_key:
                collected_stats_key = 'DEM'
            if collected_stats_key not in collected_stats:
                collected_stats[collected_stats_key]=[]
            collected_stats[collected_stats_key]  += [t_stats]

    for stats_name, stats in collected_stats.items():
        this_df = pd.DataFrame(stats)
        if stats_name=='DEM':
            these_bias_fields=['sensor_ind']
        else:
            these_bias_fields=[field for field in bias_fields if field not in 'ID']

        u_bias_array=[]
        these_fields=[]
        for field in these_bias_fields:
            temp=np.array(this_df[field])
            if np.any(np.isfinite(temp)):
                u_bias_array += [temp]
                these_fields += [field]
        _, bf_dict = pc.unique_by_rows(np.c_[u_bias_array].T,
                                       return_dict=True)
        bf_stats = []
        for bf, i_bf in bf_dict.items():
            if not np.isfinite(bf[0]):
                continue
            #i_rc=i_rc[np.array(df.N)[i_rc]>20]
            #if len(i_rc)==0:
            #    continue
            temp={field:rci for field, rci in zip(these_fields, bf)}
            temp.update({
                'name':all_sensors[np.array(this_df.sensor_ind[i_bf])[0]],
                'mean_bias':np.nanmean(this_df.bias[i_bf]),
                'sigma_bias':np.nanstd(this_df.bias[i_bf]),
                'med_data_rms_raw': np.nanmedian(this_df.rms_data_raw[i_bf]),
                'med_data_rms_edited': np.nanmedian(this_df.rms_data_edited[i_bf]),
                'N_edited' : np.nansum(this_df.N_edited[i_bf]),
                'N_raw': np.nansum(this_df.N_raw[i_bf]),
                'N_tiles':len(i_bf)})
            bf_stats += [ temp ]
        bf_stats=pd.DataFrame(bf_stats)

        out_name=os.path.join(args.out_dir,f'E{int(args.xy0[0])}_N{int(args.xy0[1])}_{stats_name}.csv')
        bf_stats.to_csv(out_name, index=False)

if __name__=='__main__':
    main()
