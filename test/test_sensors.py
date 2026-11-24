"""
altimetryFit.sensors: the sensor numbers and geolocation errors shared by all
of altimetryFit, their configuration, and the classification of sensors by name
"""
import os
import json
import h5py
import numpy as np
import pytest
import pointCollection as pc
from altimetryFit import sensors
from altimetryFit.read_optical import laser_key
from altimetryFit.read_CS2_data import radar_key
from altimetryFit.fit_altimetry import assign_sigma_corr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def default_config(monkeypatch):
    monkeypatch.delenv(sensors.CONFIG_ENV_VAR, raising=False)
    sensors.set_sensor_config()
    yield
    sensors.set_sensor_config()


def test_default_numbers():
    # the numbers used before the sensors module
    assert laser_key() == {'ICESat1': 1, 'ICESat2': 2, 'ATM': 3, 'ATM_ICESSN': 3.5, 'LVIS': 4, 'riegl': 5}
    assert radar_key() == {'CS2_POCA': -2, 'CS2_swath': -1}
    assert sensors.first_DEM_sensor_number() == 6


def test_example_config_file_matches_defaults():
    with open(os.path.join(REPO, 'default_args', 'sensor_config.json')) as fh:
        config = json.load(fh)
    config.pop('_comment')
    assert config == sensors.DEFAULT_SENSOR_CONFIG


def test_config_file(tmp_path, monkeypatch):
    config_file = tmp_path/'sensors.json'
    with open(config_file, 'w') as fh:
        json.dump({'geolocation_error': {'ICESat2': 4.}}, fh)
    sensors.set_sensor_config(str(config_file))
    assert sensors.parse_E_geo() == {'ICESat2': 4., 'DEM': 5., 'other': 1.}
    assert laser_key()['ICESat2'] == 2
    # the environment variable sets the default configuration
    monkeypatch.setenv(sensors.CONFIG_ENV_VAR, str(config_file))
    sensors.set_sensor_config()
    assert sensors.parse_E_geo()['ICESat2'] == 4.
    monkeypatch.delenv(sensors.CONFIG_ENV_VAR)
    sensors.set_sensor_config()
    assert sensors.parse_E_geo()['ICESat2'] == 3.5
    # a new set of altimeters
    sensors.set_sensor_config({'altimeters': {'ICESat2': {'number': 12, 'type': 'laser', 'platform': 'orbital'}}})
    assert laser_key() == {'ICESat2': 12} and radar_key() == {}
    assert sensors.first_DEM_sensor_number() == 13
    with pytest.raises(ValueError):
        sensors.set_sensor_config({'altimeters': {'A': {'number': 1, 'type': 'laser', 'platform': 'orbital'},
                                                  'B': {'number': 1, 'type': 'laser', 'platform': 'orbital'}}})


def test_classify_by_name():
    # a file that numbered a DEM 2 and ICESat-2 7: the names decide
    altimeters, DEMs = sensors.classify_sensors({2: 'SETSM_a_32m_seg1_dem.tif', 7: 'ICESat2'})
    assert 7 in altimeters and 2 not in altimeters
    assert DEMs.tolist() == [2]
    E = sensors.geolocation_error(np.array([2., 7., 1.]), {2: 'SETSM_a_32m_seg1_dem.tif', 7: 'ICESat2'})
    assert np.allclose(E, [5., 3.5, 1.])
    # without a sensor dict, the configured numbers
    assert np.allclose(sensors.geolocation_error(np.array([2., 3., 6.])), [3.5, 1., 5.])


def test_sensor_attribute_names(tmp_path):
    for num in [1, 2, 3.5, -2, -1, 6, 120]:
        assert sensors.sensor_from_attr_name(sensors.sensor_attr_name(num)) == num
    assert sensors.sensor_attr_name(6.0) == 'sensor_6'
    filename = str(tmp_path/'fit.h5')
    with h5py.File(filename, 'w') as h5f:
        h5f.create_group('meta/sensors')
        for num, name in {2: 'ICESat2', 3: 'ATM', 3.5: 'ATM_ICESSN', -2: 'CS2_POCA', 6: 'dem.tif'}.items():
            h5f['meta/sensors'].attrs[sensors.sensor_attr_name(num)] = name
    D = pc.data().from_dict({'sensor': np.array([2., 3.5, -2.])})
    assert sensors.read_sensor_dict(filename) == {2: 'ICESat2', 3: 'ATM', 3.5: 'ATM_ICESSN',
                                                  -2: 'CS2_POCA', 6: 'dem.tif'}
    assert sensors.read_sensor_dict(filename, data=D) == {2: 'ICESat2', 3.5: 'ATM_ICESSN', -2: 'CS2_POCA'}


def test_assign_sigma_corr():
    # time_corr is rounded for orbital and airborne altimeters; the slope term
    # is added for laser altimeters only
    sensor = np.array([2., 3.5, -2., 6.])
    D = pc.data().from_dict({'sensor': sensor, 'time': np.full(4, 2020.123456789),
                             'sigma_corr': np.full(4, 0.1), 'slope_mag': np.full(4, 0.02)})
    assign_sigma_corr(D)
    assert np.allclose(D.sigma_corr, [np.hypot(0.1, 0.1), np.hypot(0.1, 0.1), 0.1, 0.1])
    assert D.time_corr[3] == D.time[3] and np.all(D.time_corr[:3] < D.time[:3])
