"""
lagrangian_functions.lagrangianFrame: the data the readers select before
decimating in a full-lagrangian fit must be exactly the data that
update_data_for_lagrangian keeps, and the no-decimation mask is evaluated at
the positions at the lagrangian epoch.

The velocity is uniform, U m/yr in x, so the parcel measured at x at time t
was at x - U*(t - EPOCH) at the epoch.
"""
import numpy as np
import pointCollection as pc
from altimetryFit.lagrangian_functions import lagrangianFrame, update_data_for_lagrangian
from altimetryFit.decimation_mask import in_no_decimation_mask, blockmedian_outside_mask

U = 1000.
EPOCH = 2022.
SRS_PROJ4 = '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
XY0 = [-1520000., -590000.]
WXY = 10000.


def advection_grid():
    x = XY0[0] + np.arange(-20000., 20001., 500.)
    y = XY0[1] + np.arange(-12000., 12001., 500.)
    t = np.arange(2019., 2025.01, 0.5)
    X, _, T = np.meshgrid(x, y, t)
    x0 = X - U*(T - EPOCH)
    y0 = np.broadcast_to(y[:, None, None], x0.shape).copy()
    return pc.grid.data().from_dict({'x': x, 'y': y, 't': t, 'x0': x0, 'y0': y0})


def data_mask():
    # data are used where y < XY0[1] + 3 km where they were measured
    x = XY0[0] + np.arange(-20000., 20001., 500.)
    y = XY0[1] + np.arange(-12000., 12001., 500.)
    z = (y[:, None] < XY0[1] + 3000.)*np.ones((1, x.size))
    return pc.grid.data().from_dict({'x': x, 'y': y, 'z': z})


def make_data(N=20000, seed=2):
    rng = np.random.default_rng(seed)
    return pc.data().from_dict({'x': XY0[0] + rng.uniform(-14000., 14000., N),
                                'y': XY0[1] + rng.uniform(-8000., 8000., N),
                                'z': rng.normal(size=N),
                                'time': rng.uniform(2019., 2025., N),
                                'id': np.arange(N, dtype=float)})


def lagrangian_dict(grid, zero_velocity=False):
    return {'advection_grid': grid, 'xy0': XY0, 'Wxy': WXY, 'move_points': True,
            'lagrangian_zero_velocity': zero_velocity, 'SRS_proj4': SRS_PROJ4,
            'lagrangian_epoch': EPOCH}


def test_select_matches_update_data_for_lagrangian():
    grid, mask = advection_grid(), data_mask()
    D = make_data()
    frame = lagrangianFrame.from_lagrangian_dict(lagrangian_dict(grid), mask_data=mask)
    selected = frame.select(D.x, D.y, D.time)
    kept = D.copy()
    update_data_for_lagrangian(kept, mask_data=mask, **lagrangian_dict(grid))
    assert np.array_equal(np.sort(kept.id), D.id[selected])
    # most of the selected data were measured outside the tile
    assert 0 < selected.sum() < D.size
    measured_in_tile = (np.abs(D.x - XY0[0]) <= WXY/2) & (np.abs(D.y - XY0[1]) <= WXY/2)
    assert np.any(selected & ~measured_in_tile) and np.any(measured_in_tile & ~selected)


def test_select_with_two_dimensional_coordinates_and_nans():
    grid = advection_grid()
    D = make_data(N=1000)
    frame = lagrangianFrame.from_lagrangian_dict(lagrangian_dict(grid))
    x2, y2, t2 = [np.c_[getattr(D, f), getattr(D, f)] for f in ('x', 'y', 'time')]
    x2[::7, 1] = np.nan
    selected = frame.select(x2, y2, t2)
    assert selected.shape == x2.shape
    assert np.array_equal(selected[:, 0], frame.select(D.x, D.y, D.time))
    assert not np.any(selected[::7, 1])


def test_zero_velocity_frame_uses_the_measured_positions():
    grid = advection_grid()
    D = make_data(N=1000)
    frame = lagrangianFrame.from_lagrangian_dict(lagrangian_dict(grid, zero_velocity=True))
    x0, y0 = frame.epoch_xy(D.x, D.y, D.time)
    assert np.array_equal(x0, D.x) and np.array_equal(y0, D.y)


def lagrangian_no_decimation_mask():
    # 1 within 1 km of the tile center (in x) at the epoch
    x = XY0[0] + np.arange(-8000., 8001., 100.)
    y = XY0[1] + np.arange(-8000., 8001., 100.)
    z = (np.abs(x[None, :] - XY0[0]) < 1000.)*np.ones((y.size, 1))
    return pc.grid.data().from_dict({'x': x, 'y': y, 'z': z})


def test_no_decimation_mask_in_lagrangian_coordinates():
    frame = lagrangianFrame.from_lagrangian_dict(lagrangian_dict(advection_grid()))
    mask = lagrangian_no_decimation_mask()
    # measured 2 km east of the center: at the epoch, a parcel measured in 2024
    # was at the center, one measured in 2022 was still 2 km east
    x = np.array([XY0[0] + 2000., XY0[0] + 2000.])
    y = np.array([XY0[1], XY0[1]])
    t = np.array([2024., 2022.])
    assert np.array_equal(in_no_decimation_mask(mask, x, y, t, frame=frame), [True, False])
    assert np.array_equal(in_no_decimation_mask(mask, x, y, t), [False, False])


def test_blockmedian_outside_a_lagrangian_mask_keeps_the_points_inside_it():
    frame = lagrangianFrame.from_lagrangian_dict(lagrangian_dict(advection_grid()))
    mask = lagrangian_no_decimation_mask()
    D = make_data(N=20000)
    inside = in_no_decimation_mask(mask, D.x, D.y, D.time, frame=frame)
    out = blockmedian_outside_mask(D.copy(), 500., mask, frame=frame)
    assert set(D.id[inside]) <= set(out.id)
    assert out.size < D.size
    D_out = D.copy_subset(np.flatnonzero(~inside)).blockmedian(500.)
    assert out.size == inside.sum() + D_out.size
