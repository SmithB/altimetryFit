#! /usr/bin/env python


import glob
import h5py
import numpy as np
import re
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

    if compute_SMB:
        for ff in fields.values():
            ff += ['SMB_a','FAC']

    if compute_lags:
        add_lag_fields(fields, max_coarse=max_coarse, max_lag=max_lag)

    if not compute_sigma:
        fields={group:[field for field in field_list if 'sigma' not in field]
                for group, field_list in fields.items()}
    return fields

def add_lag_fields(fields, max_coarse=40000, max_lag=1):
    '''
    Add the dzdt groups for lags 1, 4, 8, ... max_lag, and the averages (at
    resolutions up to max_coarse, none if max_coarse is None) of dz and dzdt.
    '''
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

def make_tile_centers(region_dir, W, R_max=None):


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

    if R_max is not None:
        import shapely
        pad_x = np.array([-1, -1, 1, 1, -1])*W/2
        pad_y = np.array([-1, 1, 1, -1, -1])*W/2
        theta = np.arange(0, 6.28, 0.1)
        circle = shapely.Polygon(np.c_[R_max*np.cos(theta), R_max*np.sin(theta)])
        keep = np.zeros(xyc.shape[0], dtype=bool)
        for row, xyi in enumerate(xyc):
            if shapely.Polygon(np.c_[xyi[0]+pad_x, xyi[1]+pad_y]).intersects(circle):
                keep[row]=True
        xyc=xyc[keep,:]

    with open(tile_ctr_file,'w') as fh:
        for line in xyc:
            fh.write(str(line[0])+' '+str(line[1])+'\n')
    return xyc

def read_group_spacing(tile_file, groups):
    '''
    Grid spacing of each group in a fit tile, from its x coordinates, or from
    the spacing in the group name (e.g. avg_dz_40000m) if the tile does not
    have the group.  None if neither gives it.
    '''
    spacing={}
    with h5py.File(tile_file,'r') as h5f:
        for group in groups:
            if group in h5f and 'x' in h5f[group] and h5f[group]['x'].size > 1:
                x=np.array(h5f[group]['x'][0:2])
                spacing[group]=float(np.abs(x[1]-x[0]))
            else:
                m=re.search(r'(\d+)m', group)
                spacing[group]=None if m is None else float(m.group(1))
    return spacing

def mosaic_weights(tile_W, tile_spacing, grid_spacing, pad=None, feather=None):
    '''
    pad and feather (make_mosaic.py -p and -f) for mosaicking a group of fit
    tiles whose grid spacing is grid_spacing.

    Each tile's weight tapers to zero across its taper region: the outer
    (tile_W - tile_spacing)/2 of the tile, which is the part of its overlap
    with each neighbor that is closer to the neighbor's center.  By default,
    the pad is two pixels (make_mosaic gives zero weight to points within pad
    of the tile edge), and the feather (a raised-cosine taper) covers the rest
    of the taper region.  If two pixels do not fit in the taper region (as for
    coarse grids), the pad is half of the taper region.  If the tiles do not
    overlap, pad and feather are zero, which make_mosaic treats as an
    unweighted mosaic.
    '''
    taper=np.maximum((tile_W-tile_spacing)/2, 0)
    if pad is None:
        if grid_spacing is None:
            raise ValueError('the grid spacing is needed to choose the pad')
        pad = 2*grid_spacing if 2*grid_spacing < taper else taper/2
    if feather is None:
        feather=np.maximum(taper-pad, 0)
    return float(pad), float(feather)

def main():
    import argparse
    parser=argparse.ArgumentParser(description='generate tiled altimetryFit output.',\
                                   fromfile_prefix_chars="@")
    parser.add_argument('-W', '--tile_W', type=float, help="input tile width, default is the Width of the fit tiles")
    parser.add_argument('--width', type=float, help="output tile width", default=200000.)
    parser.add_argument('--base_dir','-b', type=str, required=True)
    parser.add_argument('--calc_sigma', action='store_true')
    parser.add_argument('--calc_SMB', action='store_true')
    parser.add_argument('--calc_averages', action='store_true')
    parser.add_argument('--calc_lags', action='store_true', help='mosaic the dzdt lag groups (and, with --max_res or --calc_averages, the averages) without --calc_sigma, which also mosaics them')
    parser.add_argument('--max_res', type=float)
    parser.add_argument('--max_lag', type=int, default=1)
    parser.add_argument('--hybrid', action='store_true')
    parser.add_argument('--prelim', action='store_true')
    parser.add_argument('--matched', action='store_true')
    parser.add_argument('--tile_spacing', type=float, help='distance between tile centers, default is the tile_spacing of the fit tiles')
    parser.add_argument('--pad', type=float, help="mosaic pad width, default is two pixels of each group's grid (see mosaic_weights)")
    parser.add_argument('--region', type=str, default='GL')
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--environment','-e',  type=str)
    parser.add_argument('--pboss', action='store_true')
    parser.add_argument('--feather', type=float, help='mosaic feather width, default is the rest of the taper region, (tile_W-tile_spacing)/2 - pad')
    parser.add_argument('--xy0', type=float, nargs=2, help='if specified, provide the command for the tile that contains this point')
    parser.add_argument('--km', action='store_true', help='specifies that xy0 is in km')
    parser.add_argument('--R_max', type=float, help='make only tiles that intersect this radius')
    parser.add_argument('--mask_w', type=float, help="if specified, grid cells farther than this distance from a datapoint will be marked as invalid.")
    parser.add_argument('--EPSG', type=int, help="EPSG for output tifs")
    args, _=parser.parse_known_args()

    out_W=args.width
    in_dir=args.base_dir

    if not (args.prelim or args.matched):
        raise ValueError("Need to specify either prelim or matched argument")

    if args.xy0 is not None and args.km:
        args.xy0 = [float(jj)*1000 for jj in args.xy0]

    # the fit-tile geometry comes from the input arguments saved in the tiles,
    # unless it is specified (older tiles have no input_args)
    tile_files=sorted(glob.glob(os.path.join(in_dir, 'prelim', 'E*N*.h5')))
    if len(tile_files)==0:
        tile_files=sorted(glob.glob(os.path.join(in_dir, 'matched', 'E*N*.h5')))
    if len(tile_files)==0:
        raise ValueError(f"no fit tiles found in {in_dir}/prelim or {in_dir}/matched")
    with h5py.File(tile_files[0],'r') as h5f:
        input_args=dict(h5f['meta/input_args'].attrs) if 'meta/input_args' in h5f else {}
    for arg, key in [('tile_W', 'Width'), ('tile_spacing', 'tile_spacing')]:
        if getattr(args, arg) is None:
            if key not in input_args:
                raise ValueError(f"{key} is not in the input_args of {tile_files[0]}, specify --{arg}")
            setattr(args, arg, float(input_args[key]))

    if args.calc_averages and args.max_res is None:
        args.max_res = np.minimum(40000, args.tile_W)

    fields=make_fields(max_coarse=args.max_res, \
                       max_lag=args.max_lag,\
                       compute_lags=args.calc_sigma or args.calc_lags, \
                       compute_sigma=args.calc_sigma, \
                       compute_SMB=args.calc_SMB,
                       hybrid=args.hybrid)

    group_spacing=read_group_spacing(tile_files[0], fields.keys())
    weights={}
    for group in fields:
        weights[group]=mosaic_weights(args.tile_W, args.tile_spacing, group_spacing[group],
                                      pad=args.pad, feather=args.feather)
        print(f"{group}: grid spacing={group_spacing[group]}, pad={weights[group][0]:g}, feather={weights[group][1]:g}")

    if args.xy0 is not None:
        xyc = [tuple([(np.floor(jj/args.width)+0.5)
                      * args.width for jj in args.xy0])]
    else:
        xyc = make_tile_centers(in_dir, args.width, R_max=args.R_max)

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

    if args.xy0 is None:
        if not os.path.isdir(run_dir):
            os.mkdir(run_dir)

        for sub in ['queue','running','done','logs']:
            if not os.path.isdir(os.path.join(run_dir, sub)):
                os.mkdir(os.path.join(run_dir, sub))
        if args.environment is not None:
            with open("par_run/worker_env",'w') as fh:
                fh.writelines([f"CONDA_ENV={args.environment}"])

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

        if args.xy0 is None:
            task_file = os.path.join(queue_dir, f'task_{count+1}')
        else:
            task_file = 'mosaic_one_tile.sh'

        with open(task_file,'w') as fh:
            fh.write('#! /usr/bin/env bash\n')
            if args.environment is not None:
                fh.write(f'source activate {args.environment}\n')
            for group in fields.keys():
                out_sub=os.path.join(tile_sub, group)
                out_dir = os.path.join(tile_dir_out, group)
                if not os.path.isdir(out_dir):
                    os.mkdir(out_dir)
                out_file = os.path.join(out_sub, f"{group}{tile_bounds_1km}.h5")
                if group == "dz":
                    dz_file = out_file

                pad, feather=weights[group]

                spacing_str=""
                if group_spacing[group] is not None and group_spacing[group] > args.tile_W/2:
                    spacing_str=f"-S {int(group_spacing[group])} {int(group_spacing[group])}"

                fh.write("#\n")
                if args.prelim:
                    these_fields = ' '.join([sigma_fields[group], non_sigma_fields[group]])
                    fh.write(f"make_mosaic.py -v -w -d {in_dir} -g 'prelim/E*.h5' -r {search_bounds_str}" 
                             + f" -f {feather} -p {pad} -c {tile_bounds_str}" 
                             + f" -G {group} -F {these_fields} -O {out_file} {spacing_str}\n")
                elif args.matched:
                    fh.write(f"make_mosaic.py -v -w -R -d {in_dir} -g 'matched/E*.h5' -r {search_bounds_str} -f {feather} -p {pad} -c {tile_bounds_str} " 
                             + f"-G {group} -F {non_sigma_fields[group]} -O {out_file} {spacing_str}\n")
                    if args.calc_sigma:
                        fh.write(f"make_mosaic.py -w -d {in_dir} -g 'prelim/E*.h5' -r {search_bounds_str} -f {feather} -p {pad} -c {tile_bounds_str}"
                                 + f" -G {group} -F {sigma_fields[group]} -O {out_file} {spacing_str}\n")
            if args.EPSG is not None:
                cmd=f"make_tile_tifs.py {os.path.join(args.base_dir, dz_file)} {args.EPSG}"
                if args.calc_sigma:
                    cmd += " --read_sigma"
                if args.mask_w is not None:
                    cmd += f" --mask_w={args.mask_w}"
                fh.write(cmd+"\n")
        st=os.stat(task_file)
        os.chmod(task_file, st.st_mode | stat.S_IEXEC)

if __name__=='__main__':
    main()
