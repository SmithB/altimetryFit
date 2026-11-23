"""
slope_bias: slopes from a fitted z0 (+ dz) at the data, and the expected horizontal offsets
"""
import numpy as np
import pytest
import pointCollection as pc
from altimetryFit.slope_bias import surface_slopes, expected_offsets, slope_fields_from_fit, parse_E_geo


def plane_grids(a=0.02, b=-0.05, c=0.003):
    x = np.arange(-1000., 1001., 25.)
    y = np.arange(-800., 801., 25.)
    z0 = pc.grid.data().from_dict({'x': x, 'y': y, 'z0': a*x[None, :] + b*y[:, None] + 10.})
    xd = np.arange(-1000., 1001., 200.)
    yd = np.arange(-800., 801., 200.)
    t = np.arange(2019., 2025.01, 0.25)
    # dz has a slope c in x that grows linearly with time from 2022
    dz = pc.grid.data().from_dict({'x': xd, 'y': yd, 'time': t,
                                   'dz': c*xd[None, :, None]*(t[None, None, :]-2022.)*np.ones((yd.size, 1, 1))})
    return z0, dz, (a, b, c)


def test_slopes_of_a_plane():
    z0, dz, (a, b, c) = plane_grids()
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-900, 900, 500), rng.uniform(-700, 700, 500)
    zx, zy = surface_slopes(z0, x, y)
    assert np.allclose(zx, a) and np.allclose(zy, b)
    t = rng.uniform(2019, 2025, 500)
    zx, zy = surface_slopes(z0, x, y, dz=dz, t=t)
    assert np.allclose(zx, a + c*(t-2022.)) and np.allclose(zy, b)
    # outside the grid: no slope
    zx, zy = surface_slopes(z0, np.array([5000.]), np.array([0.]))
    assert zx[0] == 0 and zy[0] == 0


def test_expected_offsets():
    sensor = np.array([2, 2, 3, 7, 12])
    time = np.array([2022., 2025., 2022., 2019., 2022.])
    E = expected_offsets(sensor, time)
    assert np.allclose(E, [3.5, 3.5, 1., 5., 5.])
    E = expected_offsets(sensor, time, E_geo='ICESat2:4,DEM:6', E_rate=10., t_epoch=2022.)
    assert np.allclose(E, [4., np.hypot(4., 30.), 1., np.hypot(6., 30.), 6.])
    with pytest.raises(ValueError):
        parse_E_geo('GEDI:2')


def test_slope_fields_from_fit():
    z0, dz, (a, b, c) = plane_grids()
    D = pc.data().from_dict({'x': np.array([0., 100.]), 'y': np.array([0., -100.]),
                             'time': np.array([2022., 2024.]), 'sensor': np.array([2., 9.])})
    fields = slope_fields_from_fit(D, {'z0': z0, 'dz': dz}, E_rate=5., t_epoch=2022.)
    assert fields == ['z_x', 'z_y']
    assert np.allclose(D.z_x, [a, a + 2*c]) and np.allclose(D.z_y, b)
    assert np.allclose(D.E_scale_z_x, [3.5, np.hypot(5., 10.)]) and np.allclose(D.E_scale_z_y, D.E_scale_z_x)


def test_fine_grid_and_reference_slopes():
    z0, dz, (a, b, c) = plane_grids()
    # a fine z0 field with slope d in x, inside -200 < x < 200 (NaN outside,
    # where the model has no fine part)
    xf = np.arange(-400., 401., 10.)
    yf = np.arange(-400., 401., 10.)
    d = 0.3
    zf = d*xf[None, :]*np.ones((yf.size, 1))
    zf[:, np.abs(xf) > 200] = np.nan
    z0_fine = pc.grid.data().from_dict({'x': xf, 'y': yf, 'z0_fine': zf})
    x = np.array([0., 100., 300., 800.])
    y = np.zeros(4)
    zx, zy = surface_slopes(z0, x, y, z0_fine=z0_fine)
    assert np.allclose(zx, [a+d, a+d, a, a]) and np.allclose(zy, b)
    # a fine dz field, zero at 2022 and growing with time, everywhere in its grid
    t = np.arange(2019., 2025.01, 0.25)
    dzf = c*xf[None, :, None]*(t[None, None, :]-2022.)*np.ones((yf.size, 1, 1))
    dz_fine = pc.grid.data().from_dict({'x': xf, 'y': yf, 'time': t, 'dz_fine': dzf})
    zx, zy = surface_slopes(z0, x, y, t=np.full(4, 2024.), dz_fine=dz_fine)
    assert np.allclose(zx, [a+2*c, a+2*c, a+2*c, a])
    # the slopes of a reference surface subtracted from the data are added
    D = pc.data().from_dict({'x': x, 'y': y, 'time': np.full(4, 2022.), 'sensor': np.full(4, 2.)})
    slope_fields_from_fit(D, {'z0': z0, 'dz': dz, 'z0_fine': z0_fine},
                          ref_slopes=(np.full(4, 0.1), np.full(4, -0.2)))
    assert np.allclose(D.z_x, np.array([a+d, a+d, a, a]) + 0.1)
    assert np.allclose(D.z_y, b - 0.2)


def test_offsets_by_sensor_name():
    # sensors are classified by name when a sensor dict is given
    sensor_dict = {2: 'SETSM_a_32m_seg1_dem.tif', 9: 'ICESat2'}
    E = expected_offsets(np.array([2., 9.]), np.array([2022., 2022.]), sensor_dict=sensor_dict)
    assert np.allclose(E, [5., 3.5])


# the record of how slope-bias fields were made, and the checks of reruns

def slope_data():
    sensor = np.array([2., 2., 6., 7.])
    return pc.data().from_dict({'x': np.zeros(4), 'y': np.zeros(4), 'time': np.full(4, 2020.),
                                'sensor': sensor, 'z_x': np.full(4, 0.1), 'z_y': np.zeros(4),
                                'E_scale_z_x': np.array([3.5, 3.5, 5., 5.]),
                                'E_scale_z_y': np.array([3.5, 3.5, 5., 5.])})


def test_slope_bias_meta_roundtrip(tmp_path):
    import h5py
    from altimetryFit.slope_bias import slope_bias_meta, write_slope_bias_meta, read_slope_bias_meta
    meta = slope_bias_meta('fit', E_geo='DEM:4', slopes='z0,dz')
    # E_rate is used only with a lagrangian epoch
    assert meta['E_rate'] == 0. and np.isnan(meta['t_epoch'])
    filename = str(tmp_path/'fit.h5')
    with h5py.File(filename, 'w') as h5f:
        write_slope_bias_meta(h5f, meta)
    saved = read_slope_bias_meta(filename)
    assert saved['source'] == 'fit' and saved['slopes'] == 'z0,dz'
    assert saved['E_geo'] == meta['E_geo'] and np.isnan(saved['t_epoch'])
    # a file without the record: the source from the fit's arguments, or nothing
    with h5py.File(filename, 'w') as h5f:
        h5f.create_group('meta/input_args').attrs['DEM_slope_bias_scale'] = '1'
    assert read_slope_bias_meta(filename)['source'] == 'DEM'
    assert read_slope_bias_meta(filename)['inferred']
    with h5py.File(filename, 'w') as h5f:
        h5f.create_group('meta/input_args').attrs['max_iterations'] = 3
    assert read_slope_bias_meta(filename) is None


def test_check_slope_bias_fields(capsys):
    from altimetryFit.slope_bias import slope_bias_meta, check_slope_bias_fields
    sensor_dict = {2: 'ICESat2', 6: 'SETSM_a_32m_seg1_dem.tif', 7: 'SETSM_b_32m_seg1_dem.tif'}
    saved = slope_bias_meta('fit', slopes='z0,dz')
    # same options: the record is kept, the fields unchanged
    D = slope_data()
    assert check_slope_bias_fields(D, saved, 'fit', sensor_dict=sensor_dict) == saved
    # slopes from the other source: an error
    with pytest.raises(ValueError):
        check_slope_bias_fields(D, saved, 'DEM', sensor_dict=sensor_dict)
    # other expected offsets: the saved ones are used, unless recomputed
    capsys.readouterr()
    out = check_slope_bias_fields(D, saved, 'fit', E_geo='DEM:2', sensor_dict=sensor_dict)
    assert out == saved and np.array_equal(D.E_scale_z_x, [3.5, 3.5, 5., 5.])
    assert 'recompute' in capsys.readouterr().out
    out = check_slope_bias_fields(D, saved, 'fit', E_geo='DEM:2', sensor_dict=sensor_dict, recompute_E=True)
    assert np.array_equal(D.E_scale_z_x, [3.5, 3.5, 2., 2.]) and np.array_equal(D.E_scale_z_y, D.E_scale_z_x)
    assert out['slopes'] == 'z0,dz' and '"DEM": 2.0' in out['E_geo']
    # no record: a message, and the fields as they are (or recomputed)
    D = slope_data()
    assert check_slope_bias_fields(D, None, 'fit', sensor_dict=sensor_dict) is None
    assert 'no record' in capsys.readouterr().out
    out = check_slope_bias_fields(D, None, 'fit', E_geo='DEM:1', sensor_dict=sensor_dict, recompute_E=True)
    assert np.array_equal(D.E_scale_z_x, [3.5, 3.5, 1., 1.]) and out['source'] == 'fit'
    # a record inferred from the fit's arguments: the source is checked, the offsets are not recorded
    inferred = {'source': 'DEM', 'E_geo': None, 'E_rate': None, 't_epoch': None, 'slopes': None, 'inferred': True}
    with pytest.raises(ValueError):
        check_slope_bias_fields(slope_data(), inferred, 'fit', sensor_dict=sensor_dict)
    capsys.readouterr()
    assert check_slope_bias_fields(slope_data(), inferred, 'DEM', sensor_dict=sensor_dict) == inferred
    assert 'not recorded' in capsys.readouterr().out
