

import pointCollection as pc
import numpy as np


def _interp_to_grid(D, x, y, t=None, field='z', band=None):
    '''
    interpolate field of D at grid points defined by x, y (and optionally a
    scalar time t), returning an array of shape (len(y), len(x))
    '''
    if t is not None:
        t = np.atleast_1d(t)
    return D.interp(x, y, t, gridded=True, field=field, band=band).reshape(len(y), len(x))

def _bracketing_bands(t, ref_time, label=''):
    '''
    find the pair of bands in t that bracket ref_time
    '''
    t = np.asarray(t)
    if len(t) < 2:
        raise ValueError(f'{label}: need at least two time slices to interpolate, found {len(t)}')
    if ref_time < t[0] or ref_time > t[-1]:
        raise ValueError(f'{label}: requested time {ref_time} is outside the range of times [{t[0]}, {t[-1]}]')
    ii = np.clip(np.searchsorted(t, ref_time), 1, len(t)-1)
    return [ii-1, ii]

def correct_for_dz(z0, dz_files, ref_time, dz_group='dz', dz_field='dz'):
    meta = pc.grid.data().from_file(dz_files[0], group=dz_group, meta_only=True)
    x_pad = meta.x[1] - meta.x[0]
    bands = _bracketing_bands(meta.t, ref_time, label=f'correct_for_dz ({dz_files[0]})')

    dz = pc.grid.mosaic().from_list(list(dz_files),
                                    bounds=z0.bounds(pad=x_pad),
                                    bands=bands,
                                    group=dz_group,
                                    fields=[dz_field])
    z0.z += _interp_to_grid(dz, z0.x, z0.y, ref_time, field=dz_field)

def correct_for_SMB(z0, SMB_file, t_ref, FAC0_scale=1, apply_SMB=True, apply_FAC=True):
    if not (apply_SMB or apply_FAC):
        return
    meta = pc.grid.data().from_file(SMB_file, meta_only=True)
    dx = meta.x[1]-meta.x[0]
    dt = meta.t[1]-meta.t[0]
    # check that t_ref is within the SMB file's time range
    _bracketing_bands(meta.t, t_ref, label=f'correct_for_SMB ({SMB_file})')
    fields = [field for field, flag in zip(['SMB_a', 'FAC'], [apply_SMB, apply_FAC]) if flag]
    SMB = pc.grid.data().from_file(SMB_file,
                                   bounds=z0.bounds(pad=2*dx),
                                   t_range = [t_ref-dt, t_ref+dt],
                                   fields=fields)
    SMB.fill_smoothed(1)
    for field in fields:
        z0.z += _interp_to_grid(SMB, z0.x, z0.y, t_ref, field=field)

    if apply_FAC and FAC0_scale > 0:
        SMB0 = pc.grid.data().from_file(SMB_file, bounds=SMB.bounds(),
                                        t_range = [meta.t[0], meta.t[1]],
                                        fields=['FAC'])[:,:,0]
        SMB0.fill_smoothed(1)
        z0.z -= FAC0_scale*_interp_to_grid(SMB0, z0.x, z0.y, field='FAC')


def get_z0(bounds, z0_files, z0_group=None, z0_field=None):

    if z0_files[0].endswith('tif') or z0_files[0].endswith('tiff'):
        z0_group=None
        z0_field='z'
    else:
        if z0_group is None:
            z0_group='z0'
        if z0_field is None:
            z0_field='z0'

    z0 = pc.grid.mosaic().from_list(list(z0_files), group = z0_group,
                                    fields = [z0_field],
                                    bounds=bounds)
    if isinstance(z0, str):
        # mosaic.from_list returns an error message if it could not find the fields
        raise ValueError(z0)
    if not z0_field=='z':
        z0.assign(z=getattr(z0, z0_field))
    return z0

#TBD:
#def get_lag_dz():

def surf_for_time(bounds, ref_time,
                  SMB_file=None,
                  dz_files = None,
                  z0_files = None,
                  z0_field=None,
                  z0_group=None,
                  dz_field='dz',
                  dz_group='dz',
                  FAC0_scale=1,
                  corrections=('FAC', 'SMB', 'dz')):
    '''
    Calculate a surface for a specific time.

    The surface is z0 + dz(t) + FAC(t) - FAC0_scale*FAC(t_SMB_0) + SMB_a(t),
    where each correction is applied only if it is listed in corrections and
    the files it needs are provided.

    Parameters
    ----------
    bounds : iterable
        [[xmin, xmax], [ymin, ymax]] bounds of the output
    ref_time : float
        time (decimal year) for which the surface is calculated
    SMB_file : str, optional
        file containing SMB_a and FAC fields
    dz_files : list of str, optional
        dz tile files
    z0_files : list of str
        z0 files (tif or hdf5)
    z0_field, z0_group : str, optional
        field and group to read from hdf5 z0 files (default 'z0', 'z0')
    dz_field, dz_group : str, optional
        field and group to read from the dz files
    FAC0_scale : float, optional
        scaling for the FAC at the first time in the SMB file
    corrections : iterable of str, optional
        corrections to apply, any of 'FAC', 'SMB', 'dz'

    Returns
    -------
    pc.grid.mosaic
        the surface estimate in the 'z' field
    '''
    if z0_files is None or len(z0_files)==0:
        raise ValueError('surf_for_time: no z0 files specified')

    z0 = get_z0(bounds, z0_files, z0_field=z0_field, z0_group=z0_group)

    if 'dz' in corrections and dz_files is not None and len(dz_files) > 0:
        correct_for_dz(z0, dz_files, ref_time, dz_group=dz_group,
                            dz_field=dz_field)

    if SMB_file is not None:
        correct_for_SMB(z0, SMB_file, ref_time, FAC0_scale=FAC0_scale,
                        apply_SMB='SMB' in corrections,
                        apply_FAC='FAC' in corrections)

    return z0
