"""
fit_altimetry with slope_bias_from_z0: the two passes count iterations as one
fit, errors are calculated only after the second pass, the second pass's
biases use sigma_corr_base, the slopes include the reference DEM, and a fit
that rereads the output (e.g. --matched) continues it with its slope fields
and iteration count, after checking how they were made
"""
import json
import h5py
import numpy as np
import pointCollection as pc
import importlib
import pytest

fa = importlib.import_module('altimetryFit.fit_altimetry')

AMP = 50.
L = 2000.
DX = 5.


def surface(x, y):
    # slopes in x vary (up to 0.16), so an offset is not a vertical bias
    return AMP*np.sin(2*np.pi*x/L) + 5*np.sin(2*np.pi*y/4000)


def surface_x_slope(x):
    return AMP*2*np.pi/L*np.cos(2*np.pi*x/L)


def synthetic_data(seed=0, N=4000, sigma_corr_base=None, time_per_sensor=False):
    # a surface measured by 10 'DEMs' (sensors 6-15); sensor 9's positions
    # are off by DX in x, so its heights are off by about DX times the slope.
    # With time_per_sensor, each DEM has one time.
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1900, 1900, N)
    y = rng.uniform(-1900, 1900, N)
    sensor = rng.integers(6, 16, N).astype(float)
    x_true = x + np.where(sensor == 9, DX, 0)
    time = rng.uniform(2019.1, 2020.9, N)
    if time_per_sensor:
        time = 2019.1 + 0.18*(sensor - 6)
    fields = {'x': x, 'y': y, 'time': time,
              'z': surface(x_true, y) + rng.normal(0, 0.5, N),
              'sigma': np.full(N, 0.5), 'sigma_corr': np.full(N, 1.), 'sensor': sensor}
    if sigma_corr_base is not None:
        fields['sigma_corr_base'] = np.full(N, sigma_corr_base)
    return pc.data().from_dict(fields)


def run(data, **kwargs):
    args = dict(xy0=[0., 0.], Width=4000., t_span=[2019., 2021.],
                spacing={'z0': 200., 'dz': 1000., 'dt': 0.25},
                E_RMS={'d2z0_dx2': 0.02, 'd3z_dx2dt': 0.0003, 'd2z_dxdt': 0.0003*2500, 'd2z_dt2': 5000},
                reference_epoch=4, bias_params=['sensor'], bias_nsigma_edit=None,
                GI_files={}, verbose=False, data=data)
    return fa.fit_altimetry(**(args | kwargs))


def offsets(S):
    b = S['m']['bias']
    return dict(zip(np.array(b['sensor']), np.array(b['val_z_x'])))


def record_smooth_fit_calls(monkeypatch):
    calls = []
    smooth_fit = fa.smooth_fit

    def recording_smooth_fit(**kwargs):
        calls.append({key: kwargs.get(key) for key in ['max_iterations', 'first_iteration',
                                                       'compute_E', 'data_scale_fields']})
        D = kwargs['data']
        calls[-1]['sigma_corr'] = np.median(D.sigma_corr)
        calls[-1]['E_RMS_z0'] = kwargs['E_RMS'].get('z0')
        if 'E_scale_z_x' in D.fields:
            calls[-1]['E_scale'] = dict(zip(*np.unique(np.c_[D.sensor, D.E_scale_z_x], axis=0).T))
        return smooth_fit(**kwargs)
    monkeypatch.setattr(fa, 'smooth_fit', recording_smooth_fit)
    return calls


DEM_NAMES = {sensor: f'SETSM_{sensor}_32m_lsf_seg1_dem.tif' for sensor in range(6, 16)}


def test_two_passes_and_continuation(tmp_path, monkeypatch):
    calls = record_smooth_fit_calls(monkeypatch)

    S, data, sensor_dict = run(synthetic_data(sigma_corr_base=0.25), max_iterations=4,
                               slope_bias_from_z0=2, calc_error=True)
    # pass 1: 2 iterations, no slope biases, no errors; pass 2 continues the count
    assert {key: calls[0][key] for key in ['max_iterations', 'first_iteration', 'compute_E', 'data_scale_fields']} \
        == {'max_iterations': 2, 'first_iteration': 0, 'compute_E': False, 'data_scale_fields': None}
    assert calls[1]['first_iteration'] == 2 and calls[1]['max_iterations'] == 2
    assert calls[1]['compute_E'] and calls[1]['data_scale_fields'] == ['z_x', 'z_y']
    # pass 1 has no offsets: its biases use sigma_corr; pass 2's use sigma_corr_base
    assert calls[0]['sigma_corr'] == 1. and calls[1]['sigma_corr'] == 0.25
    assert S['n_iterations'] == 4
    assert 'sigma_z0' in S['E']
    # the offset of sensor 9 is found
    off = offsets(S)
    assert abs(off[9] - DX) < 1 and all(abs(val) < 1 for key, val in off.items() if key != 9)
    # the record of how the slope fields were made
    assert S['slope_bias_meta']['source'] == 'fit' and S['slope_bias_meta']['slopes'] == 'z0,dz'

    # save, and continue for one iteration (as a matched fit does)
    out_file = str(tmp_path/'E0_N0.h5')
    # the DEM names, as read_DEM_data records them
    fa.save_fit_to_file(S, out_file, sensor_dict=DEM_NAMES)
    with h5py.File(out_file, 'r') as h5f:
        record = dict(h5f['meta/slope_bias'].attrs)
        assert np.allclose(h5f['data/sigma_corr'][:], 0.25)
    assert record['source'] == 'fit' and json.loads(record['E_geo'])['DEM'] == 5.
    calls.clear()
    S2, _, _ = run(None, reread_file=out_file, max_iterations=1, slope_bias_from_z0=2)
    assert len(calls) == 1
    assert calls[0]['first_iteration'] == 4 and calls[0]['data_scale_fields'] == ['z_x', 'z_y']
    assert calls[0]['sigma_corr'] == 0.25
    assert S2['n_iterations'] == 5
    assert abs(offsets(S2)[9] - DX) < 1
    assert S2['slope_bias_meta']['source'] == 'fit'

    # a rerun with other expected offsets uses the saved ones, unless asked to recompute them
    calls.clear()
    S3, _, _ = run(None, reread_file=out_file, max_iterations=1, slope_bias_from_z0=2,
                   slope_bias_E_geo='DEM:2')
    assert set(calls[0]['E_scale'].values()) == {5.}
    assert json.loads(S3['slope_bias_meta']['E_geo'])['DEM'] == 5.
    calls.clear()
    S4, _, _ = run(None, reread_file=out_file, max_iterations=1, slope_bias_from_z0=2,
                   slope_bias_E_geo='DEM:2', slope_bias_recompute_E=True)
    assert set(calls[0]['E_scale'].values()) == {2.}
    assert json.loads(S4['slope_bias_meta']['E_geo'])['DEM'] == 2.
    assert S4['slope_bias_meta']['slopes'] == 'z0,dz'


def test_slope_fields_from_a_DEM_file_are_not_used_as_fit_slopes(tmp_path):
    S, _, _ = run(synthetic_data(), max_iterations=3, slope_bias_from_z0=1)
    out_file = str(tmp_path/'E0_N0.h5')
    fa.save_fit_to_file(S, out_file, sensor_dict=DEM_NAMES)
    with h5py.File(out_file, 'r+') as h5f:
        h5f['meta/slope_bias'].attrs['source'] = 'DEM'
    with pytest.raises(ValueError, match='DEM file'):
        run(None, reread_file=out_file, max_iterations=1, slope_bias_from_z0=1)


def test_reread_keeps_the_time_corr_sigma_corr(tmp_path):
    # assign_sigma_corr adds the slope term to sigma_corr once: not again
    # when the fit's data are reread
    data = synthetic_data(time_per_sensor=True)
    data.assign(slope_mag=np.full(data.size, 0.1))
    data.sensor[data.sensor == 6] = 2   # 'ICESat2', a laser altimeter
    S, _, _ = run(data, max_iterations=1, bias_params=['time_corr', 'sensor'])
    assert np.allclose(S['data'].sigma_corr[S['data'].sensor == 2], np.sqrt(1 + 0.5**2))
    assert np.all(S['data'].sigma_corr_base == 1.)
    out_file = str(tmp_path/'E0_N0.h5')
    fa.save_fit_to_file(S, out_file, sensor_dict={2: 'ICESat2'} | DEM_NAMES)
    S2, _, _ = run(None, reread_file=out_file, max_iterations=1, bias_params=['time_corr', 'sensor'])
    assert np.allclose(S2['data'].sigma_corr[S2['data'].sensor == 2], np.sqrt(1 + 0.5**2))


def test_reference_DEM_slopes(tmp_path, monkeypatch):
    # with a reference DEM (the surface's x variation) subtracted, the model is
    # nearly flat in x, and the slopes for the offsets come from the reference DEM
    x = np.arange(-3000., 3001., 25.)
    ref = pc.grid.data().from_dict({'x': x, 'y': x, 'z': surface(x[None, :], 0)*np.ones((x.size, 1))})
    ref_file = str(tmp_path/'ref.tif')
    ref.to_geotif(ref_file, srs_epsg=3031)
    S, _, _ = run(synthetic_data(seed=1), max_iterations=4, slope_bias_from_z0=2,
                  reference_DEM_file=ref_file, reference_DEM_uncertainty=10.)
    D = S['data']
    assert np.sqrt(np.mean((D.z_x - surface_x_slope(D.x))**2)) < 0.01
    assert abs(offsets(S)[9] - DX) < 1
    assert S['slope_bias_meta']['slopes'] == 'z0,dz,reference_DEM'
    # a rerun (e.g. matched) keeps the reference DEM's constraint on z0, and
    # does not subtract the DEM again
    out_file = str(tmp_path/'E0_N0.h5')
    fa.save_fit_to_file(S, out_file, sensor_dict=DEM_NAMES)
    calls = record_smooth_fit_calls(monkeypatch)
    S2, _, _ = run(None, reread_file=out_file, max_iterations=1, slope_bias_from_z0=2,
                   reference_DEM_file=ref_file, reference_DEM_uncertainty=10.)
    assert calls[0]['E_RMS_z0'] == 10.
    assert np.allclose(S2['data'].z, D.z)
