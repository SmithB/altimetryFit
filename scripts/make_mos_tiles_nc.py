#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_mos_tiles_nc.py

Mosaic prelim or matched altimetryFit tiles (E*_N*.h5) in memory with
pointCollection.grid.mosaic, and write each output tile (200 km by default)
as a netCDF file formatted like ATL15.

The groups, variables and attributes of the output come from a data
dictionary (csv) with the columns:
    out_group, field, source_file, source_group, source_var, units,
    dimensions, datatype, coordinates, least_significant_digit,
    description, long_name
A row with an empty field gives the description of its group.

source_var is a field of the fit-tile group source_group, or a sum or
difference of fields (e.g. "dz + dh_SMBa + dh_FAC").  For a matched mosaic,
fields whose names contain 'sigma' are mosaicked from the prelim tiles, as in
make_mos_tiles.py, and the mosaic pad and feather are chosen as in
make_mos_tiles.mosaic_weights.

The MERRA-2 hybrid firn model (by default, the firn_grid_file used by the fit)
is read at its full time resolution for the model times that bracket the fit
epochs, and its gaps are filled as in SMB_corr_from_grid.  In the tile groups,
a term dh_F is F(t) - F(reference_time) for the firn-model field F (e.g.
dh_SMBa, dh_FAC), evaluated at the tile's grid points and epochs.  As in
fit_altimetry.py, dh_ for the SMB anomaly is scaled by
(rho_water - 0.917)/rho_water where the tide mask marks floating ice.
Groups whose rows have source_file 'M2H' are not written to the tiles: with
--make_M2H, they are written to one file, out_dir/M2H/[name_prefix]_M2H.nc,
which covers all the output tiles plus one model node in each direction (its
SMB anomaly is not scaled for floating ice).

With --pole_mask_lat, grid cells poleward of that latitude (e.g. -86 for
south of 86 S) are invalid in the tiles.

Times in decimal years are written as 'days since YYYY-...' when a field's
units say so, using 365.25-day years.

Without --xy0, the script writes a queue of commands, one per output tile
(see make_mos_tiles.make_tile_centers); with --make_vrt, it builds a vrt
mosaic of each map variable in the tiles.
"""

import argparse
import ast
import csv
import glob
import os
import re
import sys
import time

import h5py
import numpy as np
from netCDF4 import Dataset
import pointCollection as pc
from altimetryFit.SMB_corr_from_grid import fill_SMB_gaps
from scripts.make_mos_tiles import make_tile_centers, mosaic_weights, read_group_spacing

# model-file variable for each M2H source_var in the data dictionary
M2H_VARS = {'SMBa': 'SMB_a', 'FAC': 'FAC', 'h_a': 'h_a'}

# Polar_Stereographic variable attributes, as written by
# ATL1415.make_nc_projection_variable (ATL1415 is not an altimetryFit dependency)
CRS_ATTRS = {
    3031: {'straight_vertical_longitude_from_pole': 0.0,
           'latitude_of_projection_origin': -90.0,
           'standard_parallel': -71.0,
           'spatial_epsg': '3031',
           'spatial_ref': 'PROJCS["WGS 84 / Antarctic Polar Stereographic",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-71],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","3031"]]'},
    3413: {'straight_vertical_longitude_from_pole': -45.0,
           'latitude_of_projection_origin': 90.0,
           'standard_parallel': 70.0,
           'spatial_epsg': '3413',
           'spatial_ref': 'PROJCS["WGS 84 / NSIDC Sea Ice Polar Stereographic North",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",70],PARAMETER["central_meridian",-45],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],AUTHORITY["EPSG","3413"]]'}}

# words for the time spacing, as in ATL15_write2nc.py
LAG_ADJECTIVES = {1/12: 'monthly', 1/4: 'quarterly', 1/2: 'semiannual', 1: 'annual',
                  2: 'biennial', 3: 'triennial', 4: 'quadrennial', 5: 'pentennial',
                  6: 'hexennial', 7: 'heptennial', 8: 'octennial'}

NC_TYPES = {'float64': 'f8', 'float32': 'f4', 'int8': 'i1', 'int16': 'i2', 'int32': 'i4'}

# attributes copied from the data dictionary to each variable (as in ATL15_write2nc.make_dataset)
VAR_ATTRS = ['units', 'dimensions', 'datatype', 'coordinates', 'description', 'long_name', 'source']

TILE_RE = re.compile(r'E([-\d.]+)_N([-\d.]+)\.h5$')


def read_attrs_file(attrs_file):
    """
    Read the data dictionary.

    Returns
    -------
    groups : dict
        {out_group: {'description': str, 'fields': {field: row}}}, in file order
    """
    groups = {}
    with open(attrs_file, encoding='utf-8-sig') as fh:
        for row in csv.DictReader(fh):
            row = {key: ('' if val is None else val.strip()) for key, val in row.items()}
            if row['out_group'] == '':
                continue
            group = groups.setdefault(row['out_group'], {'description': '', 'fields': {}})
            if row['field'] == '':
                group['description'] = row['description']
            else:
                group['fields'][row['field']] = row
    return groups


def fill_templates(groups, replacements):
    """Replace {res}, {res_m} and {lag_name} in the data-dictionary text."""
    def fill(text):
        for key, val in replacements.items():
            text = text.replace('{'+key+'}', val)
        return text
    for group in groups.values():
        group['description'] = fill(group['description'])
        for row in group['fields'].values():
            for key in row:
                row[key] = fill(row[key])
    return groups


def source_terms(expression):
    """Parse a source_var like 'dz + dh_SMBa - x' into [(1, 'dz'), (1, 'dh_SMBa'), (-1, 'x')]."""
    if not re.fullmatch(r'\s*[A-Za-z_]\w*(\s*[+-]\s*[A-Za-z_]\w*)*\s*', expression):
        raise ValueError(f'cannot parse source_var "{expression}"')
    return [(-1 if sign == '-' else 1, name)
            for sign, name in re.findall(r'([+-]?)\s*([A-Za-z_]\w*)', expression)]


def evaluate(expression, sources):
    """Evaluate a source_var, given a dict of arrays."""
    result = None
    for sign, name in source_terms(expression):
        if name not in sources:
            raise KeyError(f'source_var "{expression}": {name} is not available')
        term = sign * np.asarray(sources[name], dtype=float)
        result = term if result is None else result + term
    return result


def tile_bounds(xy0, width):
    """Bounds of the output tile containing xy0, [[xmin, xmax], [ymin, ymax]]."""
    xyc = [(np.floor(jj/width)+0.5)*width for jj in xy0]
    return [[xyc[0]-width/2, xyc[0]+width/2], [xyc[1]-width/2, xyc[1]+width/2]]


def is_M2H_group(group):
    """True for a data-dictionary group made from the firn model."""
    return any(row['source_file'] == 'M2H' for row in group['fields'].values())


def map_terms(row):
    """The source_var terms of a data-dictionary row, other than coordinates."""
    return [name for _, name in source_terms(row['source_var']) if name not in ('x', 'y', 'time', 't')]


def poleward_mask(x, y, lat_limit, epsg):
    """True for grid nodes poleward of latitude lat_limit (south of it if lat_limit < 0)."""
    from pyproj import Transformer
    xg, yg = np.meshgrid(x, y)
    _, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(xg, yg)
    return lat < lat_limit if lat_limit < 0 else lat > lat_limit


def tile_file_name(bounds, prefix):
    """[prefix]_E[xc]_N[yc].nc, where xc and yc are the tile center in km."""
    xc, yc = [int(np.round(np.mean(jj)/1000)) for jj in bounds]
    return f'{prefix}_E{xc}_N{yc}.nc'


def read_dz_spacing(tile_file):
    """
    Horizontal and time spacing of the dz grid in a fit tile, from its
    coordinates (the grid_spacing argument is saved in several formats, e.g.
    '100,800,0.25', '800,1600,1/12', or an array)
    """
    with h5py.File(tile_file, 'r') as h5f:
        dz = h5f['dz']
        x = np.array(dz['x'][0:2])
        t = np.array(dz['time' if 'time' in dz else 't'][0:2])
    return float(np.abs(x[1]-x[0])), float(t[1]-t[0])


def read_input_args(tile_file):
    """Read the fit_altimetry arguments saved in a fit tile."""
    with h5py.File(tile_file, 'r') as h5f:
        return {key: (val.decode() if isinstance(val, bytes) else val)
                for key, val in h5f['meta/input_args'].attrs.items()}


def find_tiles(step_dir, bounds, half_width):
    """Find fit tiles whose centers are within half_width of the output bounds."""
    files = []
    for file in sorted(glob.glob(os.path.join(step_dir, 'E*_N*.h5'))):
        m = TILE_RE.search(file)
        if m is None:
            continue
        xc, yc = [float(jj)*1000 for jj in m.groups()]
        if (xc > bounds[0][0]-half_width) and (xc < bounds[0][1]+half_width) and \
           (yc > bounds[1][0]-half_width) and (yc < bounds[1][1]+half_width):
            files.append(file)
    return files


def mosaic_fields(files, group, fields, bounds, pad, feather, spacing=None):
    """
    Make a weighted mosaic (as make_mosaic.py -w) of fields in a group of the
    fit tiles, on the full grid of the output tile (NaN where no tile covers it).

    spacing is the group's grid spacing, needed (as make_mosaic.py -S) for
    groups that are too coarse for the mosaic to find it from the tiles.
    """
    mos = pc.grid.mosaic().from_list(files, bounds=bounds, fields=fields, group=group,
                                     pad=pad, feather=feather, by_band=False,
                                     spacing=[None, None] if spacing is None else [spacing, spacing])
    if isinstance(mos, str):
        raise RuntimeError(mos)
    dx, dy = mos.spacing
    x = np.arange(bounds[0][0], bounds[0][1]+dx/2, dx)
    y = np.arange(bounds[1][0], bounds[1][1]+dy/2, dy)
    c0 = int(np.round((mos.x[0]-x[0])/dx))
    r0 = int(np.round((mos.y[0]-y[0])/dy))
    if not (np.allclose(x[c0:c0+mos.x.size], mos.x) and np.allclose(y[r0:r0+mos.y.size], mos.y)):
        raise ValueError(f'the {group} grid is not aligned with the output bounds {bounds}')
    out = pc.grid.data().from_dict({'x': x, 'y': y, 'time': mos.t})
    for field in fields:
        z_mos = getattr(mos, field)
        z = np.full((y.size, x.size) + z_mos.shape[2:], np.nan)
        z[r0:r0+mos.y.size, c0:c0+mos.x.size, ...] = z_mos
        out.assign({field: z})
    return out


def read_M2H(M2H_file, bounds, t_range, fields, pad=7.8e4, w_smooth=1):
    """
    Read the firn model for the model times that bracket t_range, over bounds
    plus pad, and fill its gaps, as in SMB_corr_from_grid.

    fields are data-dictionary names (see M2H_VARS)
    """
    with h5py.File(M2H_file, 'r') as h5f:
        t = np.array(h5f['t' if 't' in h5f else 'time'])
    i0 = np.flatnonzero(t <= t_range[0])
    i0 = i0[-1] if i0.size else 0
    i1 = np.flatnonzero(t >= t_range[1])
    i1 = i1[0] if i1.size else t.size-1
    smbfd = pc.grid.data().from_nc(M2H_file, bounds=[np.array(jj)+np.array([-pad, pad]) for jj in bounds],
                                   t_range=[t[i0], t[i1]],
                                   field_mapping={field: M2H_VARS.get(field, field) for field in fields})
    for field in fields:
        setattr(smbfd, field, fill_SMB_gaps(getattr(smbfd, field), w_smooth))
    return smbfd


def crop_to_nodes(D, bounds, n_extra=1):
    """Crop a grid to the nodes that cover bounds, plus n_extra nodes on each side."""
    ranges = []
    for coord, (c0, c1) in zip([D.x, D.y], bounds):
        delta = coord[1]-coord[0]
        lo = coord[coord <= c0]
        hi = coord[coord >= c1]
        lo = (lo[-1] if lo.size else coord[0]) - n_extra*delta
        hi = (hi[0] if hi.size else coord[-1]) + n_extra*delta
        ranges.append([lo, hi])
    return D.cropped(*ranges)


def floating_scale(tide_mask_file, x, y, rho_water=1, shelf_val=1):
    """
    Scale for the SMB anomaly: 1 on grounded ice and (rho_water-0.917)/rho_water
    on floating ice, using the tide-mask test in apply_tides.py
    """
    if tide_mask_file is None:
        return np.ones((y.size, x.size))
    tide_mask = pc.grid.data().from_geotif(tide_mask_file,
                                           bounds=[[x[0]-1.e3, x[-1]+1.e3], [y[0]-1.e3, y[-1]+1.e3]])
    tide_mask.z = np.nan_to_num(tide_mask.z)
    floating = np.abs(tide_mask.interp(x, y, gridded=True) - shelf_val) < 0.5
    return (floating == 0) + (rho_water-.917)/rho_water*(floating == 1)


def make_projection_variable(nc_group, epsg):
    """Make a Polar_Stereographic variable like ATL1415.make_nc_projection_variable."""
    crs_var = nc_group.createVariable('Polar_Stereographic', np.byte, ())
    crs_var.standard_name = 'Polar_Stereographic'
    crs_var.grid_mapping_name = 'polar_stereographic'
    attrs = CRS_ATTRS[epsg]
    for key in ['straight_vertical_longitude_from_pole', 'latitude_of_projection_origin', 'standard_parallel']:
        crs_var.setncattr(key, attrs[key])
    crs_var.scale_factor_at_projection_origin = 1.
    crs_var.false_easting = 0.0
    crs_var.false_northing = 0.0
    crs_var.semi_major_axis = 6378.137
    crs_var.semi_minor_axis = 6356.752
    crs_var.inverse_flattening = 298.257223563
    crs_var.spatial_epsg = attrs['spatial_epsg']
    crs_var.spatial_ref = attrs['spatial_ref']
    crs_var.crs_wkt = attrs['spatial_ref']
    return crs_var


def write_variable(nc_group, field, data, attrs):
    """Write a variable as ATL15_write2nc.make_dataset does."""
    dimensions = tuple(dim.strip() for dim in attrs['dimensions'].split(','))
    if dimensions == (field,):
        nc_group.createDimension(field, data.shape[0])
    if attrs['datatype'].startswith('int'):
        fill_value = np.iinfo(np.dtype(attrs['datatype'])).max
    else:
        fill_value = np.finfo(np.dtype(attrs['datatype'])).max
    var = nc_group.createVariable(field, NC_TYPES[attrs['datatype']], dimensions,
                                  fill_value=fill_value, zlib=True,
                                  least_significant_digit=ast.literal_eval(attrs['least_significant_digit']))
    var[:] = np.ma.masked_invalid(data)
    for attr in VAR_ATTRS:
        if attr in attrs:
            var.setncattr(attr, attrs[attr])
    if not field.startswith('time'):
        var.setncattr('grid_mapping', 'Polar_Stereographic')
    if field == 'x':
        var.standard_name = 'projection_x_coordinate'
    if field == 'y':
        var.standard_name = 'projection_y_coordinate'
    return var


def days_since_year(units):
    """Year YYYY for units of 'days since YYYY-...', otherwise None."""
    m = re.match(r'days since (\d{4})', units)
    return None if m is None else float(m.group(1))


def to_output_array(field, data, attrs):
    """Convert times to the units of the field, and grids from (y, x, t) to the field's dimensions."""
    year0 = days_since_year(attrs['units'])
    if year0 is not None:
        data = (data - year0)*365.25
    if data.ndim == 3:
        dims = [dim.strip() for dim in attrs['dimensions'].split(',')]
        if dims[0] in ('time', 't'):
            data = np.moveaxis(data, 2, 0)
    return data


def M2H_file_name(args, fit_args):
    """The firn-model file: --M2H_file, or the firn_grid_file used by the fit."""
    if args.M2H_file is not None:
        return args.M2H_file
    return os.path.join(fit_args['firn_directory'], fit_args['firn_grid_file'])


def write_group(nc, group_name, group, sources, epsg, mask=None):
    """
    Write a data-dictionary group to a netCDF file.

    sources gives the grid for each row: sources['M2H'] for rows whose
    source_file is 'M2H', otherwise sources[source_group].  Map values are
    invalid where mask (a [y, x] array on the grid of the group) is True.
    """
    nc_group = nc.createGroup(group_name)
    nc_group.setncattr('description', group['description'])
    crs_var = make_projection_variable(nc_group, epsg)
    group_data = {}
    for field, row in group['fields'].items():
        D = sources['M2H' if row['source_file'] == 'M2H' else row['source_group']]
        these_sources = {'x': D.x, 'y': D.y, 'time': D.t, 't': D.t}
        these_sources.update({ff: getattr(D, ff) for ff in D.fields})
        group_data[field] = evaluate(row['source_var'], these_sources)
        if mask is not None and group_data[field].ndim >= 2:
            group_data[field][mask, ...] = np.nan
    # as in ATL15_write2nc.py: cells with no ice area are invalid, and so are their delta_h values
    if 'ice_area' in group_data:
        group_data['ice_area'][group_data['ice_area'] == 0] = np.nan
        for field in group_data:
            if 'delta_h' in field:
                group_data[field][~np.isfinite(group_data['ice_area'])] = np.nan
    for field, row in group['fields'].items():
        write_variable(nc_group, field, to_output_array(field, group_data[field], row), row)
    crs_var.GeoTransform = tuple(float(val) for val in
                                 (np.min(D.x), D.x[1]-D.x[0], 0, np.min(D.y), 0, D.y[1]-D.y[0]))
    return nc_group


def make_nc_tile(args, xy0, groups, fit_args, tile_file):
    """
    Mosaic the fit tiles for the output tile containing xy0 and write it.

    fit_args are the fit_altimetry arguments saved in the fit tiles, which give
    the defaults for the tile geometry, firn model, tide mask and reference time;
    tile_file is one of the fit tiles, which gives the grid spacing of each group
    """
    t_start = time.time()
    bounds = tile_bounds(xy0, args.width)
    out_file = os.path.join(args.out_dir, tile_file_name(bounds, args.name_prefix))
    # the firn-model groups go in a separate file (see make_M2H_file)
    tile_groups = {name: group for name, group in groups.items() if not is_M2H_group(group)}

    # fields to read from each fit-tile group, and from the firn model (for the dh_ terms)
    tile_fields, M2H_fields = {}, []
    for group in tile_groups.values():
        for field, row in group['fields'].items():
            terms = map_terms(row)
            M2H_fields += [name[3:] for name in terms if name.startswith('dh_')]
            tile_fields.setdefault(row['source_group'], [])
            tile_fields[row['source_group']] += [name for name in terms if not name.startswith('dh_')]
    M2H_fields = list(dict.fromkeys(M2H_fields))
    tile_fields = {group: list(dict.fromkeys(fields)) for group, fields in tile_fields.items()}

    step = 'matched' if args.matched else 'prelim'
    tile_W = args.tile_W if args.tile_W is not None else float(fit_args['Width'])
    tile_spacing = args.tile_spacing if args.tile_spacing is not None else float(fit_args['tile_spacing'])
    group_spacing = read_group_spacing(tile_file, tile_fields)
    weights = {group: mosaic_weights(tile_W, tile_spacing, group_spacing[group], pad=args.pad, feather=args.feather)
               for group in tile_fields}
    M2H_file = M2H_file_name(args, fit_args)
    tide_mask_file = args.tide_mask_file
    if tide_mask_file is None and fit_args.get('tide_mask_file', '') not in ('', 'None'):
        tide_mask_file = fit_args['tide_mask_file']
    reference_time = args.reference_time
    if reference_time is None:
        reference_time = float(fit_args['reference_time'])
    if args.verbose:
        print(f'{out_file}: bounds={bounds}, (pad, feather)={weights}, M2H_file={M2H_file}, '
              f'tide_mask_file={tide_mask_file}, reference_time={reference_time}')

    # mosaic the fit tiles
    sources = {}
    for group, fields in tile_fields.items():
        pad, feather = weights[group]
        D = None
        for this_step, these_fields in [(step, [ff for ff in fields if args.prelim or 'sigma' not in ff]),
                                        ('prelim', [ff for ff in fields if args.matched and 'sigma' in ff])]:
            if len(these_fields) == 0:
                continue
            files = find_tiles(os.path.join(args.base_dir, this_step), bounds, tile_W/2-pad)
            if len(files) == 0:
                print(f'no {this_step} tiles found for {bounds}')
                return None
            if args.verbose:
                print(f'\tmosaicking {group}: {these_fields} from {len(files)} {this_step} tiles')
            # coarse groups (at most two nodes per tile) need their spacing, as in make_mos_tiles
            coarse = group_spacing[group] is not None and group_spacing[group] > tile_W/2
            this_D = mosaic_fields(files, group, these_fields, bounds, pad, feather,
                                   spacing=group_spacing[group] if coarse else None)
            if D is None:
                D = this_D
            else:
                if not np.allclose(D.t, this_D.t):
                    raise ValueError(f'{group} epochs differ between the prelim and matched tiles')
                for field in these_fields:
                    D.assign({field: getattr(this_D, field)})
        sources[group] = D
    if args.verbose:
        print(f'\tmosaics done after {time.time()-t_start:0.1f} s')

    # firn model and dh_ terms
    tile_grid = next(iter(sources.values()))
    if len(M2H_fields) > 0:
        M2H = read_M2H(M2H_file, bounds, [np.min(tile_grid.t), np.max(tile_grid.t)], M2H_fields)
        t_eval = np.r_[tile_grid.t, reference_time]
        float_scale = None
        for field in M2H_fields:
            temp = M2H.interp(tile_grid.x, tile_grid.y, t_eval, field=field, gridded=True)
            dh = temp[:, :, :-1] - temp[:, :, -1:]
            if M2H_VARS.get(field, field) == 'SMB_a':
                if float_scale is None:
                    float_scale = floating_scale(tide_mask_file, tile_grid.x, tile_grid.y)
                dh *= float_scale[:, :, None]
            for D in sources.values():
                D.assign({'dh_'+field: dh})
        if args.verbose:
            print(f'\tM2H done after {time.time()-t_start:0.1f} s')

    mask = None
    if args.pole_mask_lat is not None:
        mask = poleward_mask(tile_grid.x, tile_grid.y, args.pole_mask_lat, args.EPSG)

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    tmp_file = out_file + '.tmp'
    with Dataset(tmp_file, 'w', clobber=True) as nc:
        nc.setncattr('GDAL_AREA_OR_POINT', 'Area')
        nc.setncattr('Conventions', 'CF-1.6')
        nc.setncattr('fit_directory', os.path.join(args.base_dir, step))
        for group, (pad, feather) in weights.items():
            suffix = '' if len(weights) == 1 else '_'+group
            nc.setncattr('mosaic_pad'+suffix, pad)
            nc.setncattr('mosaic_feather'+suffix, feather)
        if len(M2H_fields) > 0:
            nc.setncattr('M2H_file', M2H_file)
            nc.setncattr('tide_mask_file', str(tide_mask_file))
            nc.setncattr('reference_time', reference_time)
        if args.pole_mask_lat is not None:
            nc.setncattr('pole_mask_lat', args.pole_mask_lat)
        for group_name, group in tile_groups.items():
            write_group(nc, group_name, group, sources, args.EPSG, mask=mask)
    os.replace(tmp_file, out_file)
    if args.verbose:
        print(f'\twrote {out_file} after {time.time()-t_start:0.1f} s')
    return out_file


def make_M2H_file(args, groups, fit_args, tile_file):
    """
    Write the firn-model groups of the data dictionary to one file in
    out_dir/M2H, covering all the output tiles plus one model node in each
    direction, for the model times that bracket the fit epochs.  The model's
    gaps are filled before it is cropped, as for the dh_ terms of the tiles.
    """
    M2H_groups = {name: group for name, group in groups.items() if is_M2H_group(group)}
    if len(M2H_groups) == 0:
        print(f'no firn-model groups in {args.attrs_file}')
        return None
    fields = list(dict.fromkeys(name for group in M2H_groups.values()
                                for row in group['fields'].values() for name in map_terms(row)))
    xyc = np.array(make_tile_centers(args.base_dir, args.width))
    bounds = [[np.min(xyc[:, 0])-args.width/2, np.max(xyc[:, 0])+args.width/2],
              [np.min(xyc[:, 1])-args.width/2, np.max(xyc[:, 1])+args.width/2]]
    # the fit epochs, from the fit-tile groups used by the other groups of the data dictionary
    t_fit = []
    with h5py.File(tile_file, 'r') as h5f:
        for group in groups.values():
            for row in group['fields'].values():
                if row['source_file'] != 'M2H' and row['source_group'] in h5f:
                    h5g = h5f[row['source_group']]
                    t_fit += list(np.array(h5g['time' if 'time' in h5g else 't']))
    M2H_file = M2H_file_name(args, fit_args)
    M2H = read_M2H(M2H_file, bounds, [np.min(t_fit), np.max(t_fit)], fields)
    M2H = crop_to_nodes(M2H, bounds)

    out_dir = os.path.join(args.out_dir, 'M2H')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'{args.name_prefix}_M2H.nc')
    tmp_file = out_file + '.tmp'
    with Dataset(tmp_file, 'w', clobber=True) as nc:
        nc.setncattr('GDAL_AREA_OR_POINT', 'Area')
        nc.setncattr('Conventions', 'CF-1.6')
        nc.setncattr('M2H_file', M2H_file)
        nc.setncattr('gap_filling', 'gaps filled with altimetryFit.SMB_corr_from_grid.fill_SMB_gaps (w_smooth=1) before cropping')
        nc.setncattr('SMBa_floating_ice', 'SMBa is not scaled for floating ice; the dh_SMBa terms in the tiles '
                     'are scaled by (1 - 0.917) where tide_mask_file marks floating ice')
        for group_name, group in M2H_groups.items():
            write_group(nc, group_name, group, {'M2H': M2H}, args.EPSG)
    os.replace(tmp_file, out_file)
    if args.verbose:
        print(f'wrote {out_file}: x {M2H.x[0]}..{M2H.x[-1]}, y {M2H.y[0]}..{M2H.y[-1]}, '
              f't {M2H.t[0]:.4f}..{M2H.t[-1]:.4f} ({M2H.t.size} times)')
    return out_file


def make_vrts(args, groups):
    """Build a vrt mosaic for each map variable of the tile groups, in out_dir/vrt."""
    from osgeo import gdal
    gdal.UseExceptions()
    files = sorted(glob.glob(os.path.join(args.out_dir, f'{args.name_prefix}_E*_N*.nc')))
    if len(files) == 0:
        print(f'no tiles found in {args.out_dir}')
        return
    vrt_dir = os.path.join(args.out_dir, 'vrt')
    os.makedirs(vrt_dir, exist_ok=True)
    for group_name, group in groups.items():
        if is_M2H_group(group):
            continue
        # band descriptions and 'time' metadata give the epoch in decimal years
        # (pc.grid.data.from_gdal reads the 'time' metadata as t)
        times = None
        if 'time' in group['fields']:
            with Dataset(files[0], 'r') as nc:
                times = np.array(nc[group_name]['time'][:])
            year0 = days_since_year(group['fields']['time']['units'])
            if year0 is not None:
                times = year0 + times/365.25
        for field, row in group['fields'].items():
            dims = [dim.strip() for dim in row['dimensions'].split(',')]
            if 'x' not in dims or 'y' not in dims:
                continue
            vrt_file = os.path.join(vrt_dir, field+'.vrt')
            # subdataset names relative to the vrt, so that the directory can be moved
            sources = [f'NETCDF:"{os.path.relpath(file, vrt_dir)}":/{group_name}/{field}' for file in files]
            cwd = os.getcwd()
            try:
                os.chdir(vrt_dir)
                ds = gdal.BuildVRT(os.path.basename(vrt_file), sources)
                if times is not None and ds.RasterCount == len(times):
                    for band, this_time in enumerate(times):
                        ds.GetRasterBand(band+1).SetDescription(f'{this_time:.10g}')
                        ds.GetRasterBand(band+1).SetMetadataItem('time', f'{this_time:.10g}')
                ds = None
            finally:
                os.chdir(cwd)
            if args.verbose:
                print(f'wrote {vrt_file}')


def write_queue(args):
    xyc = make_tile_centers(args.base_dir, args.width)
    cmd = [sys.executable, os.path.abspath(__file__), '-b', args.base_dir,
           '--matched' if args.matched else '--prelim',
           '--attrs_file', args.attrs_file, '--out_dir', args.out_dir, '--width', str(args.width),
           '--name_prefix', args.name_prefix]
    for key in ['pad', 'feather', 'tile_W', 'tile_spacing', 'M2H_file', 'tide_mask_file',
                'reference_time', 'pole_mask_lat', 'EPSG']:
        if getattr(args, key) is not None:
            cmd += [f'--{key}', str(getattr(args, key))]
    if args.verbose:
        cmd += ['--verbose']
    lines = [' '.join(cmd + ['--xy0', str(xy[0]), str(xy[1])]) for xy in xyc]
    if args.queue_file is None:
        print('\n'.join(lines))
    else:
        with open(args.queue_file, 'w') as fh:
            fh.write('\n'.join(lines)+'\n')
        print(f'wrote {len(lines)} commands to {args.queue_file}')


def main():
    def path(p):
        return os.path.abspath(os.path.expanduser(p))

    parser = argparse.ArgumentParser(description='mosaic altimetryFit tiles into netCDF tiles formatted like ATL15',
                                     fromfile_prefix_chars='@')
    parser.add_argument('--base_dir', '-b', type=path, required=True, help='fit directory, containing prelim/ and matched/')
    parser.add_argument('--attrs_file', type=path, required=True, help='data-dictionary csv file')
    parser.add_argument('--prelim', action='store_true', help='mosaic the prelim tiles')
    parser.add_argument('--matched', action='store_true', help='mosaic the matched tiles (sigma fields from prelim)')
    parser.add_argument('--out_dir', type=path, help='output directory, default is [base_dir]/[width]km_tiles/nc')
    parser.add_argument('--width', type=float, default=200000., help='output tile width')
    parser.add_argument('--name_prefix', default='dh', help='output tiles are named [name_prefix]_E[xc]_N[yc].nc, with the tile center in km')
    parser.add_argument('--xy0', type=float, nargs=2, help='make the output tile that contains this point')
    parser.add_argument('--km', action='store_true', help='specifies that xy0 is in km')
    parser.add_argument('--pad', type=float, help='mosaic pad width, default is two pixels (see make_mos_tiles.mosaic_weights)')
    parser.add_argument('--feather', type=float, help='mosaic feather width, default is (tile_W-tile_spacing)/2-pad')
    parser.add_argument('--tile_W', '-W', type=float, help='fit tile width, default from the tiles')
    parser.add_argument('--tile_spacing', type=float, help='fit tile spacing, default from the tiles')
    parser.add_argument('--M2H_file', type=path, help='firn model file, default is the firn_grid_file used by the fit')
    parser.add_argument('--tide_mask_file', type=path, help='tide mask used to find floating ice, default from the fit')
    parser.add_argument('--reference_time', type=float, help='time at which dh_ terms are zero, default from the fit')
    parser.add_argument('--pole_mask_lat', type=float,
                        help='grid cells poleward of this latitude (e.g. -86 for south of 86S) are invalid in the tiles')
    parser.add_argument('--EPSG', type=int, default=3031, choices=list(CRS_ATTRS.keys()))
    parser.add_argument('--queue_file', type=path, help='without --xy0, write the per-tile commands to this file')
    parser.add_argument('--make_vrt', action='store_true', help='build vrt mosaics of the tiles in out_dir')
    parser.add_argument('--make_M2H', action='store_true',
                        help='write the firn-model groups for the area of all the tiles to out_dir/M2H')
    parser.add_argument('--verbose', '-v', action='store_true')
    args, _ = parser.parse_known_args()

    if args.prelim == args.matched and not (args.make_vrt or args.make_M2H):
        parser.error('specify one of --prelim or --matched')
    if args.out_dir is None:
        args.out_dir = os.path.join(args.base_dir, f'{int(args.width/1000)}km_tiles', 'nc')
    if args.xy0 is not None and args.km:
        args.xy0 = [jj*1000 for jj in args.xy0]

    groups = read_attrs_file(args.attrs_file)

    if args.make_vrt:
        make_vrts(args, groups)
    elif args.xy0 is not None or args.make_M2H:
        step_dir = os.path.join(args.base_dir, 'matched' if args.matched else 'prelim')
        tile_files = sorted(glob.glob(os.path.join(step_dir, 'E*_N*.h5')))
        if len(tile_files) == 0:
            parser.error(f'no tiles found in {step_dir}')
        fit_args = read_input_args(tile_files[0])
        # resolution and time spacing for the data-dictionary templates
        res, dt = read_dz_spacing(tile_files[0])
        lag_name = LAG_ADJECTIVES[min(LAG_ADJECTIVES, key=lambda lag: np.abs(lag-dt))]
        groups = fill_templates(groups, {'res': f'{res/1000:g}', 'res_m': f'{int(res)}', 'lag_name': lag_name})
        if args.make_M2H:
            make_M2H_file(args, groups, fit_args, tile_files[0])
        else:
            make_nc_tile(args, args.xy0, groups, fit_args, tile_files[0])
    else:
        write_queue(args)


if __name__ == '__main__':
    main()
