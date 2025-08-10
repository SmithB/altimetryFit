#! /usr/bin/env python


import glob
import numpy as np
import re
import sys
import pointCollection as pc
import os
import stat


def make_fields(max_coarse=40000, max_lag=1, compute_lags=False, \
                compute_sigma=False, compute_SMB=False, hybrid=False):

    fields={}
    fields['z0']="z0 sigma_z0 misfit_rms misfit_scaled_rms mask cell_area count".split(' ')
    if hybrid:
        fields['lagrangian_dz']=["dz"]

    fields['dz']="dz sigma_dz count misfit_rms misfit_scaled_rms mask cell_area".split(' ')
    if not compute_sigma:
        fields['z0']=[field for field in fields['z0'].copy() if 'sigma' not in field]
        fields['dz']=[field for field in fields['dz'].copy() if 'sigma' not in field]

    if compute_SMB:
        for ff in fields.values():
            ff += ['SMB_a','FAC']

    if not compute_lags:
        return fields

    lags=[]
    for lag in [1]+list(np.arange(4, max_lag+1, 4)):
        if lag > max_lag:
            continue
        this_lag=f'_lag{lag}'
        lags += [this_lag]
        fields['dzdt'+this_lag]=["dzdt"+this_lag, "sigma_dzdt"+this_lag]

    for res in ["_40000m", "_20000m", "_10000m"]:
        if max_coarse is None:
            continue
        this_res = int(res.replace('_','').replace('m',''))
        if this_res > max_coarse:
            continue
        fields['avg_dz'+res] = ["avg_dz"+res, "sigma_avg_dz"+res,'cell_area']
        for lag in lags:
            field_str='avg_dzdt'+res+lag
            fields[field_str]=[field_str, 'sigma_'+field_str]

    return fields

def make_tile_centers(region_dir, W):


    tile_ctr_file=os.path.join(region_dir,f'{int(W/1000)}km_tile_list.txt')

    if os.path.isfile(tile_ctr_file):
        with open(tile_ctr_file) as fh:
            xyc=[ [*map(float, line.rstrip().split(' '))] for line in fh]
        return xyc

    tile_files=[]
    for sub in ['prelim']:
        print(os.path.join(region_dir, sub, 'E*N*.h5'))
        tile_files += glob.glob(os.path.join(region_dir, sub, 'E*N*.h5'))

    tile_re=re.compile('/E(.*)_N(.*).h5')
    tile_list=[]
    for tile_name in tile_files:
        try:
            tile_list += [np.array([*map(int, tile_re.search(tile_name).groups())])]
        except Exception as e:
            print(tile_name)
            print(e)

    xy0=np.c_[tile_list]*1000
    xyc=pc.unique_by_rows(np.floor(xy0/W)*W + W/2)
    with open(tile_ctr_file,'w') as fh:
        for line in xyc:
            fh.write(str(line[0])+' '+str(line[1])+'\n')
    return xyc


import argparse
parser=argparse.ArgumentParser(description='generate tiled altimetryFit output.',\
                               fromfile_prefix_chars="@")
parser.add_argument('-W', '--tile_W', type=float, help="input tile width", default=40000.)
parser.add_argument('--width', type=float, help="output tile width", default=200000.)
parser.add_argument('--base_dir','-b', type=str, required=True)
parser.add_argument('--calc_sigma', action='store_true')
parser.add_argument('--calc_SMB', action='store_true')
parser.add_argument('--calc_averages', action='store_true')
parser.add_argument('--max_res', type=float)
parser.add_argument('--max_lag', type=int, default=1)
parser.add_argument('--hybrid', action='store_true')
parser.add_argument('--prelim', action='store_true')
parser.add_argument('--matched', action='store_true')
parser.add_argument('--tile_spacing', type=float, default=40000, help='distance between tile centers')
parser.add_argument('--pad', type=float, default=1000)
parser.add_argument('--region', type=str, default='GL')
parser.add_argument('--out_dir', type=str)
parser.add_argument('--environment','-e',  type=str)
parser.add_argument('--pboss', action='store_true')
parser.add_argument('--feather', type=float)
args, _=parser.parse_known_args()

out_W=args.width
tile_re = re.compile('E(.*)_N(.*).h5')
in_dir=args.base_dir

if not (args.prelim or args.matched):
    raise ValueError("Need to specify either prelim or matched argument")

if args.feather is None:
    # amount by which each tile overlaps its neighbors
    overlap=(args.tile_W-args.tile_spacing)/2
    args.feather=overlap-args.pad

if args.calc_averages and args.max_res is None:
    args.max_res = np.minimum(40000, args.tile_W)

fields=make_fields(max_coarse=args.max_res, \
                   max_lag=args.max_lag,\
                   compute_lags=args.calc_sigma, \
                   compute_sigma=args.calc_sigma, \
                   compute_SMB=args.calc_SMB,
                   hybrid=args.hybrid)
xyc=make_tile_centers(in_dir, args.width)

tile_sub=f'{int(args.width/1000)}km_tiles'

if args.out_dir is None:
    args.out_dir=in_dir
tile_dir_out=os.path.join(args.out_dir, tile_sub)
if not os.path.isdir(tile_dir_out):
    os.mkdir(tile_dir_out)

if args.pboss:
    run_dir='par_run'
else:
    run_dir=f'tile_run_{args.region}'
queue_dir=run_dir+'/queue'

if not os.path.isdir(run_dir):
    os.mkdir(run_dir)


for sub in ['queue','running','done','logs']:
    if not os.path.isdir(os.path.join(run_dir, sub)):
        os.mkdir(os.path.join(run_dir, sub))

non_sigma_fields={}
sigma_fields={}
for group, field_list in fields.items():
    non_sigma_fields[group]=" ".join([field for field in field_list if 'sigma' not in field])
    sigma_fields[group]=" ".join([field for field in field_list if 'sigma' in field])

for count, xy in enumerate(xyc):
    search_bounds =[xy[0]-out_W/2-1.e4, xy[0]+out_W/2+1.e4, xy[1]-out_W/2-1.e4, xy[1]+out_W/2+1.e4]
    search_bounds_str = " ".join([str(ii) for ii in search_bounds])
    tile_bounds = [xy[0]-out_W/2, xy[0]+out_W/2, xy[1]-out_W/2, xy[1]+out_W/2]
    tile_bounds_1km = "_".join([str(int(ii/1000)) for ii in tile_bounds])
    tile_bounds_str = " ".join([str(ii) for ii in tile_bounds])

    task_file=os.path.join(queue_dir, f'task_{count+1}')
    with open(task_file,'w') as fh:
        fh.write('#! /usr/bin/env bash\n')
        if args.environment is not None:
            fh.write(f'source activate {args.environment}\n')
        for group in fields.keys():
            group_spacing=re.compile('(\d+)m').search(group)
            if group_spacing is not None:
                group_spacing=group_spacing.group(1)
            pad=args.pad
            feather=args.feather
            spacing_str=""
            if group_spacing is not None:
                if int(group_spacing) > args.W/2:
                    spacing_str=f"-S {int(group_spacing)} {int(group_spacing)}"
                if group_spacing > args.tile_W-(2*pad+feather):
                    feather=0
                if group_spacing > args.tile_W-(2*pad):
                    pad=0
            out_sub=os.path.join(tile_sub, group)
            out_dir = os.path.join(tile_dir_out, group)
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            out_file = os.path.join(out_sub, f"{group}{tile_bounds_1km}.h5")
            fh.write("#\n")
            if args.prelim:
                fh.write(f"make_mosaic.py -v -w -d {in_dir} -g 'prelim/E*.h5' -r {search_bounds_str} -f {feather} -p {pad} -c {tile_bounds_str} -G {group} -F {sigma_fields[group]+non_sigma_fields[group]} -O {out_file} {spacing_str}\n")
            elif args.matched:
                fh.write(f"make_mosaic.py -v -w -R -d {in_dir} -g 'matched/E*.h5' -r {search_bounds_str} -f {feather} -p {pad} -c {tile_bounds_str} -G {group} -F {non_sigma_fields[group]} -O {out_file} {spacing_str}\n")
                if args.calc_sigma:
                    fh.write(f"make_mosaic.py -w -d {in_dir} -g 'prelim/E*.h5' -r {search_bounds_str} -f {feather} -p {pad} -c {tile_bounds_str} -G {group} -F {sigma_fields[group]} -O {out_file} {spacing_str}\n")
    st=os.stat(task_file)
    os.chmod(task_file, st.st_mode | stat.S_IEXEC)
