#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 14:12:41 2025

@author: ben
"""
import os
import sys
import numpy as np
from altimetryFit import surfaceModel

def parse_inputs(argv):
    # account for a bug in argparse that misinterprets negative agruents
    for i, arg in enumerate(argv):
        if (arg[0] == '-') and arg[1].isdigit(): argv[i] = ' ' + arg

    def path(p):
        return os.path.abspath(os.path.expanduser(p))

    import argparse
    parser=argparse.ArgumentParser(description="Calculate a surface for a location based on a surface model", \
                                   fromfile_prefix_chars="@")
    parser.add_argument('--xy0', type=float, nargs=2, help="fit center location")
    parser.add_argument('--Width','-W',  type=float, nargs='+', help="fit width")
    parser.add_argument('--spacing', type=float, help='if specified, gives the spacing of the output')
    parser.add_argument('--time','-t', type=float, help="time stamp for output")
    parser.add_argument('--output_file', type=path, help="output file, default is based on the output directory and field")
    parser.add_argument('--output_directory', type=path, help="directory in which to write the output file, used if output_file is not specified")
    parser.add_argument('--output_format', type=str, help='output format, can be h5, nc, or tif')
    parser.add_arugmnet('--EPSG', type=str, help='output EPSG, required for the tif format')
    parser.add_argument('--reference_time', type=float, help="reference time for the model")
    parser.add_argument('--base_directory','-b', type=path, help='base directory')
    parser.add_argumnet('--prelim', action='store_true', help='if specified, will look for tiles in [base_directory]/prelim')
    parser.add_argument('--firn_directory', type=path, help='directory in which to find the firn file')
    parser.add_argument('--firn_grid_file', type=str, help='gridded firn model file that can be interpolated directly.')
    parser.add_argument('--skip_lagrangian', type=str, help='skip calculation of advecting topography')
    parser.add_argument('--velocity_files', type=path, nargs='+', help='lagrangian velocity files.  May contain multiple time values.')
    parser.add_argument('--velocity_type', type=str, help='velocity file type.  Specify xy01_grids if interpolators are used')
    parser.add_argument('--lagrangian_epoch', type=float, help='time (decimal year) to which data will be advected')
    parser.add_argument('--tide_mask', type=path, help='mask file used to identify the floating part of the mask.  Synonym for floating_mask_file')
    parser.add_argument('--floating_mask', type=path, help='mask file used to identify the floating part of the mask.  Synonym for floating_mask_file')
    parser.add_argument('--floating_mask_file', type=path, help='mask file used to identify the floating part of the mask.')
    parser.add_argument('--floating_mask_value', type=float, default=1, help='this value in the floating_mask identifies floating ice')
    parser.add_argument('--mask_file', type=path)
    parser.add_argument('--fields', type=str, nargs='+', default='z_ice')
    parser.add_argument('--geoid_file', type=path, help='if provided, parts of the grid identified as floating but containing no data will be assigned values from this file')
    parser.add_argument('--format', type=str, help='')

    args, unk=parser.parse_known_args(argv)

    if args.time is None:
        args.time=args.reference_time

    if args.output_file is None and args.output_directory is None:
        args.output_diectory='.'
    # handle redundant arguments
    if args.floating_mask_file is None:
        if args.floating_mask is None and args.tide_mask is not None:
            args.floating_mask_file = args.tide_mask
        else:
            args.floating_mask_file = args.floating_mask

    if not os.path.isfile(args.firn_grid_file) and args.firn_grid_file is not None:
        args.firn_grid_file = os.path.join(args.firn_directory, args.firn_grid_file)
        if not os.path.isfile(args.firn_grid_file):
            raise FileNotFoundError(f'{args.firn_grid_file} not found')

    if args.prelim:
        args.base_directory = os.path.join(args.base_directory,'prelim')

    return args

def make_output_filename(args, field):
    output_file=args.output_file
    if args.output_file is None:
        output_file=os.path.join(args.output_directory, field+f'_{args.time:3.2f}.'+args.format)
    return output_file

def main():
    args=parse_inputs(sys.argv)

    include_lagrangian=True
    if args.skip_lagrangian:
        include_lagrangian=False

    model = surfaceModel()

    match_extent=True
    if args.W is not None and args.spacing is not None:
        if len(args.W)==1:
            args.W += args.W
        model.x = args.xy0[0] + np.arange(-0.5*args.W[0], 0.5*args.W[0]+0.1*args.spacing, args.spacing)
        model.y = args.xy0[1] + np.arange(-0.5*args.W[1], 0.5*args.W[0]+0.1*args.spacing, args.spacing)
        match_extent=False

    for field in ['velocity_files', 'velocity_type', 'firn_grid_file',
                  'floating mask_file', 'floating_mask_value']:
        this = getattr(args, field)
        if this is not None:
            setattr(model, field, this)

    model.from_h5_directory(args.base_directory, match_extent=match_extent)

    out_args={}
    if format in ['tif','tiff']:
        out_function = model.to_geotif
        out_args = {'srs_epsg':args.EPSG}

    model.get_z(include_lagrangian=include_lagrangian)

    if 'z' in args.fields:
        out_file=make_output_filename(args, 'z')
        out_function(out_file, field='z', **out_args)

    if 'z_ice' in args.fields or 'z_surf' in args.fields:
        model.get_z_ice()

    if 'z_ice' in args.fields:
        out_file=make_output_filename(args, 'z_ice')
        out_function(out_file, field='z_ice', **out_args)

    if 'z_surf' in args.fields:
        model.get_z_surf()
        out_file=make_output_filename(args, 'z_surf')
        out_function(out_file, field='z_surf', **out_args)

    for field in args.fields:
        if field in ['z','z_ice','z_surf']:
            continue
        out_file=make_output_filename(args, field)
        out_function(out_file, field=field, **out_args)
