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


# positions at the epoch, interpolated once (by the readers) and reused

def test_select_data_assigns_the_positions_at_the_epoch():
    grid, mask = advection_grid(), data_mask()
    D = make_data()
    frame = lagrangianFrame.from_lagrangian_dict(lagrangian_dict(grid), mask_data=mask)
    selected = frame.select(D.x, D.y, D.time)
    out = frame.select_data(D.copy())
    assert np.array_equal(out.id, D.id[selected])
    x0, y0 = frame.epoch_xy(out.x, out.y, out.time)
    assert np.array_equal(out.x_epoch, x0) and np.array_equal(out.y_epoch, y0)


def count_epoch_positions(monkeypatch):
    # count the points whose positions at the epoch are interpolated
    import altimetryFit.lagrangian_functions as lf
    counts = []
    epoch_positions = lf.epoch_positions

    def counting(advection_grid, x, y, t, **kwargs):
        counts.append(np.size(x))
        return epoch_positions(advection_grid, x, y, t, **kwargs)
    monkeypatch.setattr(lf, 'epoch_positions', counting)
    return counts


def test_update_data_for_lagrangian_uses_the_readers_positions(monkeypatch):
    grid, mask = advection_grid(), data_mask()
    frame = lagrangianFrame.from_lagrangian_dict(lagrangian_dict(grid), mask_data=mask)
    D = frame.select_data(make_data())
    # data from a reader without the frame (no positions at the epoch)
    D_other = make_data(N=500, seed=3)
    D_other.id += 1.e6
    D_all = pc.data().from_list([D, D_other])
    frame_counts = count_epoch_positions(monkeypatch)
    expected = D_all.copy()
    for field in ['x_epoch', 'y_epoch']:
        expected.fields.remove(field)
    update_data_for_lagrangian(expected, mask_data=mask, **lagrangian_dict(grid))
    assert frame_counts == [D_all.size]
    frame_counts.clear()
    update_data_for_lagrangian(D_all, mask_data=mask, **lagrangian_dict(grid))
    # only the data without positions are interpolated
    assert frame_counts == [D_other.size]
    assert 'x_epoch' not in D_all.fields and 'y_epoch' not in D_all.fields
    assert np.array_equal(D_all.id, expected.id)
    assert np.array_equal(D_all.x, expected.x) and np.array_equal(D_all.y, expected.y)
    assert np.array_equal(D_all.x_original, expected.x_original)


def test_complete_epoch_xy_fills_only_missing_positions(monkeypatch):
    frame = lagrangianFrame.from_lagrangian_dict(lagrangian_dict(advection_grid()))
    D = make_data(N=100)
    frame.assign_epoch_xy(D)
    reference = (D.x_epoch.copy(), D.y_epoch.copy())
    D.x_epoch[:10] = np.nan
    counts = count_epoch_positions(monkeypatch)
    frame.complete_epoch_xy(D)
    assert counts == [10]
    assert np.allclose(D.x_epoch, reference[0]) and np.allclose(D.y_epoch, reference[1])


def test_no_decimation_mask_with_known_positions():
    frame = lagrangianFrame.from_lagrangian_dict(lagrangian_dict(advection_grid()))
    mask = lagrangian_no_decimation_mask()
    D = make_data(N=5000)
    x0, y0 = frame.assign_epoch_xy(D)
    assert np.array_equal(in_no_decimation_mask(mask, D.x, D.y, D.time, xy_epoch=(x0, y0)),
                          in_no_decimation_mask(mask, D.x, D.y, D.time, frame=frame))


def sheared_advection_grid():
    # the velocity varies across flow, so the positions at the epoch are not
    # linear in the measured positions
    x = XY0[0] + np.arange(-20000., 20001., 500.)
    y = XY0[1] + np.arange(-12000., 12001., 500.)
    t = np.arange(2019., 2025.01, 0.5)
    X, Y, T = np.meshgrid(x, y, t)
    x0 = X - U*(1 + 0.5*np.sin(2*np.pi*(Y - XY0[1])/3000.))*(T - EPOCH)
    return pc.grid.data().from_dict({'x': x, 'y': y, 't': t, 'x0': x0, 'y0': Y.copy()})


def test_decimated_data_are_where_they_were_without_the_readers_positions(monkeypatch):
    # data selected, decimated outside a no-decimation mask, and moved to the
    # epoch: the same with the positions kept from the raw data as with
    # positions interpolated at each step
    grid, mask = sheared_advection_grid(), data_mask()
    frame = lagrangianFrame.from_lagrangian_dict(lagrangian_dict(grid), mask_data=mask)
    ND_mask = lagrangian_no_decimation_mask()
    raw = make_data(N=40000)
    # each step interpolates the positions
    D_old = raw.copy()
    D_old.index(frame.select(D_old.x, D_old.y, D_old.time))
    blockmedian_outside_mask(D_old, 300., ND_mask, frame=frame)
    update_data_for_lagrangian(D_old, mask_data=mask, **lagrangian_dict(grid))
    # the positions are interpolated for the raw data, and for the averaged values
    D_new = frame.select_data(raw.copy())
    x_raw = dict(zip(D_new.id, D_new.x_epoch))
    blockmedian_outside_mask(D_new, 300., ND_mask, frame=frame)
    picked = np.isfinite(D_new.x_epoch)
    assert 0 < picked.sum() < D_new.size
    # the values that are one of the raw points keep its position
    assert np.array_equal(D_new.x_epoch[picked], [x_raw[i] for i in D_new.id[picked]])
    counts = count_epoch_positions(monkeypatch)
    update_data_for_lagrangian(D_new, mask_data=mask, **lagrangian_dict(grid))
    assert counts == [np.sum(~picked)]
    assert np.array_equal(D_new.id, D_old.id)
    assert np.array_equal(D_new.x, D_old.x) and np.array_equal(D_new.y, D_old.y)


def test_blockmedian_outside_mask_uses_the_positions(monkeypatch):
    frame = lagrangianFrame.from_lagrangian_dict(lagrangian_dict(advection_grid()))
    mask = lagrangian_no_decimation_mask()
    D = make_data(N=20000)
    frame.assign_epoch_xy(D)
    expected = blockmedian_outside_mask(D.copy(), 500., mask, frame=frame)

    def fail(*args, **kwargs):
        raise AssertionError('positions at the epoch interpolated again')
    monkeypatch.setattr(frame, 'epoch_xy', fail)
    out = blockmedian_outside_mask(D.copy(), 500., mask, frame=frame)
    assert np.array_equal(out.id, expected.id)
    # without positions, the blockmedian is pointCollection's
    D2 = make_data(N=20000)
    assert np.array_equal(blockmedian_outside_mask(D2.copy(), 500., None).id, D2.blockmedian(500.).id)
    D2.assign(x_epoch=D2.x.copy(), y_epoch=D2.y.copy())
    assert np.array_equal(blockmedian_outside_mask(D2.copy(), 500., None).id, D2.blockmedian(500.).id)


# lagrangian_ref_dem is subtracted once: not again when the fit's data are reread

def test_lagrangian_ref_dem_is_not_subtracted_twice(tmp_path):
    from altimetryFit.lagrangian_functions import update_output_data_for_lagrangian
    grid = advection_grid()
    x = XY0[0] + np.arange(-20000., 20001., 250.)
    y = XY0[1] + np.arange(-12000., 12001., 250.)
    ref = pc.grid.data().from_dict({'x': x, 'y': y, 'z': 100. + 0.01*(x[None, :] - XY0[0])*np.ones((y.size, 1))})
    ref_file = str(tmp_path/'ref.tif')
    ref.to_geotif(ref_file, srs_epsg=3031)
    D = make_data(N=2000)
    z_measured = D.z.copy()
    ref_at_data = ref.interp(D.x, D.y)
    kwargs = lagrangian_dict(grid) | {'lagrangian_ref_dem': ref_file}
    update_data_for_lagrangian(D, **kwargs)
    kept = np.searchsorted(make_data(N=2000).id, D.id)
    assert np.allclose(D.z, z_measured[kept] - ref_at_data[kept], atol=1.e-3)
    assert np.allclose(D.z0, z_measured[kept])
    # the fit's output data (x, y are the measured positions again), reread
    S = {'data': D}
    update_output_data_for_lagrangian(S, **kwargs)
    z_fit, z0_fit = S['data'].z.copy(), S['data'].z0.copy()
    reread = S['data'].copy()
    update_data_for_lagrangian(reread, **(kwargs | {'reread_data': True}))
    assert np.array_equal(reread.z, z_fit) and np.array_equal(reread.z0, z0_fit)
    assert np.array_equal(reread.x, S['data'].x_lag)
