#! /usr/bin/env python

# Retired 2026-11-09, replaced by make_mos_tiles.py.  Kept for reference (it has
# no main(), and runs when imported).  For a run like
#   make_200km_tiles.py DIR AA --step prelim --W 24000 --spacing 20000 \
#       --skip_sigma --max_res 0 --max_lag 4 --pboss -e devel
# use
#   make_mos_tiles.py -b DIR --prelim --calc_lags --max_lag 4 --max_res 0 \
#       -W 24000 --tile_spacing 20000 --pboss -e devel --region AA
# (--calc_sigma also mosaics the lag groups, with their sigma fields).
# make_mos_tiles chooses the pad and feather from each group's grid spacing;
# --pad and --feather give the overlap/4 and overlap/2 used here.

import glob
import numpy as np
import re
import sys
import pointCollection as pc
import os
import stat


def make_fields(max_lag=16, max_res=40000):
    fields={}
    fields['z0']="z0 sigma_z0 misfit_rms misfit_scaled_rms mask cell_area count".split(' ')

    fields['dz']="dz sigma_dz count misfit_rms misfit_scaled_rms mask cell_area".split(' ')

    lags=['_lag1'] + [f'_lag{lag}' for lag in np.arange(4, max_lag+1, 4, dtype=int)]
    res_vals=[ "_{res}m" for res in [40000, 20000, 10000] if res <= max_res ]
    
    for lag in lags:        
        fields['dzdt'+lag]=["dzdt"+lag, "sigma_dzdt"+lag, "cell_area"]

    for res in res_vals:
        fields['avg_dz'+res] = ["avg_dz"+res, "sigma_avg_dz"+res,'cell_area']
        for lag in lags:
            field_str='avg_dzdt'+res+lag
            fields[field_str]=[field_str, 'sigma_'+field_str, 'cell_area']

    #for key, item in fields.items():
    #print(key+" : "+str(item))
    #print(fields)
    return fields

def make_200km_tiles(region_dir, tile_W=200e3):
    print("looking for tiles for "+region_dir)
    tile_ctr_file=os.path.join(region_dir,'200km_tile_list.txt')

    if os.path.isfile(tile_ctr_file):
        with open(tile_ctr_file) as fh:
            xyc=[ [*map(float, line.rstrip().split(' '))] for line in fh]
        return xyc
                
    tile_files=[]
    for sub in ['prelim']:
        tile_files += glob.glob(os.path.join(region_dir, sub, 'E*N*.h5'))

    tile_list=[]
    for tile_name in tile_files:
        try:
            tile_list += [np.array([*map(int, tile_re.search(tile_name).groups())])]
        except Exception as e:
            print(tile_name)
            print(e)

    xy0=np.c_[tile_list]*1000
    xyc=pc.unique_by_rows(np.floor(xy0/tile_W)*tile_W+tile_W/2)
    with open(tile_ctr_file,'w') as fh:
        for line in xyc:
            fh.write(str(line[0])+' '+str(line[1])+'\n')
    return xyc


tile_W=2.e5

tile_re = re.compile('/E(.*)_N(.*).h5')
avg_re = re.compile('_(\d+)m')
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('region_dir', type=str)
parser.add_argument('region', type=str)
parser.add_argument('--step', type=str, default='matched')
parser.add_argument('--pad', type=float)
parser.add_argument('--feather', type=float)
parser.add_argument('--W', type=int, default=60000)
parser.add_argument('--spacing', type=int, default=40000)
parser.add_argument('--skip_sigma', action='store_true')
parser.add_argument('--max_res', type=int, default=40000)
parser.add_argument('--max_lag', type=int, default=16)
parser.add_argument('--pboss', action='store_true')
parser.add_argument('--name', type=str)
parser.add_argument('--environment','-e', type=str, default='IS2', help="environment that each job will activate")
args=parser.parse_args()

region_dir=args.region_dir
region=args.region

step=args.step

# make the pad and feather work for 44 km tiles:
overlap=args.W-args.spacing
if args.pad is None:
    args.pad = overlap/4
if args.feather is None:
    args.feather = overlap/2
print(f"***pad={args.pad}, feather={args.feather}, overlap={overlap}")

print(f"Skip sigma is {args.skip_sigma}, step is {step}")
print("region_dir is " +region_dir)

fields=make_fields(max_res=args.max_res, max_lag=args.max_lag)
xyc=make_200km_tiles(region_dir)

tile_dir_200km=os.path.join(region_dir,'200km_tiles')
if not os.path.isdir(tile_dir_200km):
    os.mkdir(tile_dir_200km)

if args.name is None:
    args.name=region
run_dir=f'tile_run_{args.name}'
if not os.path.isdir(run_dir):
    os.mkdir(run_dir)

if os.path.isdir(run_dir+'/logs'):
    N=len(glob.glob(run_dir+'/logs_round_*'))
    os.rename(run_dir+'/logs', run_dir+f'/logs_round_{N+1}')
    os.rename(run_dir+'/done', run_dir+f'/done_round_{N+1}')

for sub in ['queue','logs','done','running']:
    if not os.path.isdir(run_dir+'/'+sub):
        os.mkdir(run_dir+'/'+sub)

non_sigma_fields={}
sigma_fields={}
for group, field_list in fields.items():
    non_sigma_fields[group]=" ".join([field for field in field_list if 'sigma' not in field])
    sigma_fields[group]=" ".join([field for field in field_list if 'sigma' in field])

for count, xy in enumerate(xyc):
    search_bounds =[xy[0]-tile_W/2-1.e4, xy[0]+tile_W/2+1.e4, xy[1]-tile_W/2-1.e4, xy[1]+tile_W/2+1.e4]
    search_bounds_str = " ".join([str(ii) for ii in search_bounds])
    tile_bounds = [xy[0]-tile_W/2, xy[0]+tile_W/2, xy[1]-tile_W/2, xy[1]+tile_W/2]
    tile_bounds_1km = "_".join([str(int(ii/1000)) for ii in tile_bounds])
    tile_bounds_str = " ".join([str(ii) for ii in tile_bounds])
    
    task_file=f'{run_dir}/queue/task_{count+1}'
    with open(task_file,'w') as fh:
        fh.write("#! /usr/bin/env bash\n")
        fh.write(f"source activate {args.environment}\n") 
        for group in fields.keys():
            pad=args.pad
            feather=args.feather
            spacing_str=""
            avg_scale=avg_re.search(group)
            if avg_scale is not None:
                avg_scale=float(avg_scale.groups()[0])
                # NOTE - this is to deal with the truncated 20-km averages in 
                # release 003.  May need to be fixed in the future (avg_scale > overlap makes more sense)
                if avg_scale >= overlap:
                    pad=0
                    feather=0
                if "40000m" in group:
                    spacing_str="-S 40000 40000"
            out_dir = os.path.join(tile_dir_200km, group)
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            out_file = f"{group}{tile_bounds_1km}.h5"

            fh.write("#\n")
            fh.write(f"make_mosaic.py -w -R -d {region_dir} -g '{step}/E*.h5' -r {search_bounds_str} -f {feather} -p {pad} -c {tile_bounds_str} -G {group} -F {non_sigma_fields[group]} -O {out_file} {spacing_str}\n")
            if not args.skip_sigma:
                fh.write(f"make_mosaic.py -w  -d {region_dir} -g 'prelim/E*.h5' -r {search_bounds_str} -f {feather} -p {pad} -c {tile_bounds_str} -G {group} -F {sigma_fields[group]} -O {out_file} {spacing_str}\n")
    st=os.stat(task_file)
    os.chmod(task_file, st.st_mode | stat.S_IEXEC)   


if args.pboss:
    os.mkdir(run_dir+'/comms')
else:
    with open('slurm_scripts/slurm_mos_run','r') as fh_in:
        with open(run_dir+'/slurm_mos_run','w') as fh_out:
            for line in fh_in:
                fh_out.write(line.replace('LAST_TASK', str(count+1)).replace('_XX_', '_'+args.name+'_'))
