"""
DEM times: read_DEM_data uses the full acquisition time from each DEM's PGC
metadata (DEM_acquisition_time), and remove_overlapping_DEM_data identifies
DEMs by index, so DEMs with the same time are thinned too
"""
import os
import json
import numpy as np
import pointCollection as pc
from altimetryFit.read_DEM_data import DEM_acquisition_time, decimal_year
from altimetryFit.remove_overlapping_DEM_data import remove_overlapping_DEM_data
import datetime

PAIR = 'SETSM_s2s041_WV01_20200101_102001000000AA00_102001000000BB00'


def write_meta(meta_dir, seg, datetime_str):
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, f'{PAIR}_2m_lsf_seg{seg}.json'), 'w') as fh:
        json.dump({'properties': {'datetime': datetime_str}}, fh)


def test_DEM_acquisition_time(tmp_path):
    DEM_dir = tmp_path/'2020_early'/'32m'
    os.makedirs(DEM_dir)
    seg1 = str(DEM_dir/f'{PAIR}_32m_lsf_seg1_dem_filt.tif')
    seg2 = str(DEM_dir/f'{PAIR}_32m_lsf_seg2_dem_filt.tif')
    # no metadata: the filename date
    t, meta, source = DEM_acquisition_time(seg1)
    assert source == 'filename' and np.isclose(t, decimal_year(datetime.datetime(2020, 1, 1)))
    # metadata where get_pgc saves them (next to the DEM directory)
    write_meta(tmp_path/'2020_early'/'meta', 1, '2020-01-01T21:45:03Z')
    t, meta, source = DEM_acquisition_time(seg1)
    t_full = decimal_year(datetime.datetime(2020, 1, 1, 21, 45, 3))
    assert source == 'meta' and np.isclose(t, t_full) and meta['properties']['datetime'] == '2020-01-01T21:45:03Z'
    # another segment of the strip has the same images, so the same time
    t, meta, source = DEM_acquisition_time(seg2)
    assert source == 'segment_meta' and np.isclose(t, t_full) and meta is None


def DEM_points(x0, x1, t, sigma):
    x, y = np.meshgrid(np.arange(x0, x1, 100.), np.arange(0., 1000., 100.))
    return pc.data().from_dict({'x': x.ravel(), 'y': y.ravel(), 'z': np.zeros(x.size),
                                'sigma': np.full(x.size, sigma), 'time': np.full(x.size, t)})


def test_remove_overlapping_same_time():
    # two DEMs with the same time that overlap in 0 < x < 1000: the overlap is
    # removed from the one with the larger sigma.  A third, at the same time,
    # does not overlap them
    D = [DEM_points(-1000, 1000, 2020.5, 1.), DEM_points(0, 2000, 2020.5, 2.),
         DEM_points(5000, 6000, 2020.5, 1.)]
    N = [Di.size for Di in D]
    out = remove_overlapping_DEM_data(D, [[-2000, 7000], [-1000, 2000]], 100, dt_min=0.125)
    assert len(out) == 3
    assert D[0].size == N[0] and D[2].size == N[2]
    assert D[1].size < N[1] and np.all(D[1].x > 1000)


def test_sibling_meta_ignores_shift_files(tmp_path):
    # a _shift_est.json next to the DEM is not PGC metadata
    DEM_dir = tmp_path/'2020_early'
    os.makedirs(DEM_dir)
    seg1 = str(DEM_dir/f'{PAIR}_32m_lsf_seg1_dem_filt.tif')
    with open(DEM_dir/f'{PAIR}_32m_lsf_seg2_shift_est.json', 'w') as fh:
        json.dump({'delta_x': 1}, fh)
    t, meta, source = DEM_acquisition_time(seg1)
    assert source == 'filename'


def test_segment_meta_without_lsf(tmp_path):
    # newer strips are named ..._32m_seg1_dem_filt.tif (no lsf)
    DEM_dir = tmp_path/'2023_early'/'32m'
    os.makedirs(DEM_dir)
    meta_dir = tmp_path/'2023_early'/'meta'
    os.makedirs(meta_dir)
    with open(meta_dir/f'{PAIR}_2m_seg1.json', 'w') as fh:
        json.dump({'properties': {'datetime': '2020-01-01T12:00:00Z'}}, fh)
    t, meta, source = DEM_acquisition_time(str(DEM_dir/f'{PAIR}_32m_seg2_dem_filt.tif'))
    assert source == 'segment_meta' and np.isclose(t, decimal_year(datetime.datetime(2020, 1, 1, 12)))
    t, meta, source = DEM_acquisition_time(str(DEM_dir/f'{PAIR}_32m_seg1_dem_filt.tif'))
    assert source == 'meta'
