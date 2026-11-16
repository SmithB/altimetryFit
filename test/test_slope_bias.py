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
