"""
--t_crop: the cropped grids (crop_grids_in_time), the reference epoch counted
from the start of --t_crop, and the checks on --t_crop and --slope_bias_from_z0
"""
import json
import numpy as np
import pytest
import pointCollection as pc
import importlib

fa = importlib.import_module('altimetryFit.fit_altimetry')


def grid(t, name):
    x = np.arange(3.)
    return pc.grid.data().from_dict({'x': x, 'y': x, 'time': t,
                                     name: np.zeros((3, 3, t.size)) + t[None, None, :]})


def test_crop_grids_in_time():
    dt = 0.25
    t = np.arange(2018., 2024.01, dt)
    t_fine = np.arange(2018., 2024.01, 0.125)
    grids = {'dz': grid(t, 'dz'),
             'dzdt_lag4': grid(t[:-4] + 2*dt, 'dzdt_lag4'),
             'dz_fine': grid(t_fine, 'dz_fine'),
             'dzdt_total_fine_lag4': grid(t_fine[:-4] + 2*0.125, 'dzdt_total_fine_lag4'),
             # one epoch: the spacing comes from dt_fine
             'dzdt_total_fine_lag8': grid(np.array([2019.5]), 'dzdt_total_fine_lag8'),
             'bias': {'val': [1]}}
    fa.crop_grids_in_time(grids, [2019., 2023.], dt, dt_fine=0.125)
    assert np.allclose(grids['dz'].time, np.arange(2019., 2023.01, dt))
    assert np.allclose(grids['dzdt_lag4'].time, np.arange(2019.5, 2022.51, dt))
    assert np.allclose(grids['dz_fine'].time, np.arange(2019., 2023.01, 0.125))
    assert np.allclose(grids['dzdt_total_fine_lag4'].time, np.arange(2019.25, 2022.751, 0.125))
    assert np.allclose(grids['dzdt_total_fine_lag8'].time, [2019.5])
    assert grids['dz'].dz.shape[2] == grids['dz'].time.size


def base_args(tmp_path):
    with open(tmp_path/'GI.json', 'w') as fh:
        json.dump({}, fh)
    return ['--xy0', '-1500000', '-450000', '--Width', '40000',
            '-g', '250,2000,0.25', '-H', '-1', '--max_mem', '0',
            '--GeoIndex_source_file', str(tmp_path/'GI.json'),
            '--out_name', str(tmp_path/'E-1500_N-450.h5')]


def test_reference_epoch_with_t_crop(tmp_path):
    argv = base_args(tmp_path)
    # without --t_crop, an index reference epoch counts from the start of --time_span
    args = fa.parse_inputs(argv + ['-t', '2018,2026', '--reference_epoch', '4'])
    assert args.reference_epoch == 4
    # with --t_crop, from the start of --t_crop, as in a fit with --time_span=2019,2025
    args = fa.parse_inputs(argv + ['-t', '2018,2026', '--t_crop', '2019,2025', '--reference_epoch', '4'])
    assert args.time_span[0] + args.reference_epoch*0.25 == 2020.
    # a reference time is a time
    args = fa.parse_inputs(argv + ['-t', '2018,2026', '--t_crop', '2019,2025', '--reference_time', '2020'])
    assert args.time_span[0] + args.reference_epoch*0.25 == 2020.
    # --t_crop must be on the time grid, and must keep the reference epoch
    with pytest.raises(ValueError):
        fa.parse_inputs(argv + ['-t', '2018,2026', '--t_crop', '2019.1,2025', '--reference_time', '2020'])
    with pytest.raises(ValueError):
        fa.parse_inputs(argv + ['-t', '2018,2026', '--t_crop', '2019,2025', '--reference_time', '2018.5'])


def test_slope_bias_from_z0_checks(tmp_path):
    argv = base_args(tmp_path) + ['-t', '2018,2026', '--reference_epoch', '4']
    args = fa.parse_inputs(argv + ['--slope_bias_from_z0', '2', '--max_iterations', '5'])
    assert args.slope_bias_from_z0 == 2
    for bad in [['--slope_bias_from_z0', '0'], ['--slope_bias_from_z0', '5', '--max_iterations', '5']]:
        with pytest.raises(ValueError):
            fa.parse_inputs(argv + bad)


def test_errors_to_fit_grids_with_t_crop(tmp_path):
    # an error run (full time span, coarser fine grid) for a fit whose output was
    # cropped to 2019-2023: the fine-grid errors are cropped before they are
    # interpolated onto the fit's (cropped) fine grid
    dt = 0.25
    t = np.arange(2018., 2024.01, dt)
    t_crop = [2019., 2023.]
    out_file = str(tmp_path/'E0_N0.h5')
    xf = np.arange(0., 1001., 50.)
    fit_fine = pc.grid.data().from_dict({'x': xf, 'y': xf, 'time': t,
                                         'dz_fine': np.zeros((xf.size, xf.size, t.size))})
    fa.crop_grids_in_time({'dz_fine': fit_fine}, t_crop, dt)
    fit_fine.to_h5(out_file, group='dz_fine')
    xe = np.arange(0., 1001., 100.)
    x = np.arange(0., 4001., 1000.)
    E = {'sigma_z0': pc.grid.data().from_dict({'x': x, 'y': x, 'sigma_z0': np.ones((5, 5))}),
         'sigma_dz': grid(t, 'sigma_dz'),
         'sigma_dz_fine': pc.grid.data().from_dict({'x': xe, 'y': xe, 'time': t,
                                                    'sigma_dz_fine': np.ones((xe.size, xe.size, t.size))
                                                    * t[None, None, :]}),
         'sigma_bias': {}}
    fa.errors_to_fit_grids(E, out_file, [4, 2], t_crop=t_crop, dt=dt)
    assert E['sigma_dz_fine'].sigma_dz_fine.shape == (xf.size, xf.size, fit_fine.time.size)
    assert np.allclose(E['sigma_dz_fine'].sigma_dz_fine[0, 0, :], fit_fine.time)
    assert np.allclose(E['sigma_dz'].time, np.arange(2019., 2023.01, dt))
