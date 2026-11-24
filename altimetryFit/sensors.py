#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensor numbers and sensor properties, shared by all of altimetryFit

Each datum has a numeric 'sensor' value.  The laser and radar altimeters have
fixed numbers, set here; DEMs are numbered from one more than the largest
altimeter number upward, in the order they are read (see
read_DEM_data.read_DEM_data), and are identified by their filenames.  Fit
output files record the name of each sensor number in the attributes of
/meta/sensors (sensor_{number}: name), so data reread from a fit are
classified by name (see classify_sensors), whatever numbering the fit used.

The default configuration is DEFAULT_SENSOR_CONFIG.  It can be replaced (or
partly replaced: entries given override the defaults) with a json file, given
by the environment variable ALTIMETRYFIT_SENSOR_CONFIG or passed to
set_sensor_config (fit_altimetry.py --sensor_config_file), containing a dict
with entries:
    altimeters: {name: {"number": n, "type": "laser" or "radar",
                        "platform": "orbital" or "airborne"}}
        CS2_swath must be numbered CS2_POCA+1 (read_CS2_data numbers swath
        data as CS2_POCA + the swath flag).
    geolocation_error: {name: E} expected horizontal geolocation errors (m),
        by altimeter name, with 'DEM' for DEMs and 'other' for the remaining
        sensors.  Used for horizontal-offset (slope) biases.
See default_args/sensor_config.json for the defaults in this format.

UPDATE HISTORY:
    Written 11/2026, from read_optical.laser_key, read_CS2_data.radar_key,
        and the sensor numbers in fit_altimetry and slope_bias
"""

import copy
import json
import os
import re
import numpy as np

DEFAULT_SENSOR_CONFIG = {
    'altimeters': {
        'ICESat1':    {'number': 1,   'type': 'laser', 'platform': 'orbital'},
        'ICESat2':    {'number': 2,   'type': 'laser', 'platform': 'orbital'},
        'ATM':        {'number': 3,   'type': 'laser', 'platform': 'airborne'},
        'ATM_ICESSN': {'number': 3.5, 'type': 'laser', 'platform': 'airborne'},
        'LVIS':       {'number': 4,   'type': 'laser', 'platform': 'airborne'},
        'riegl':      {'number': 5,   'type': 'laser', 'platform': 'airborne'},
        'CS2_POCA':   {'number': -2,  'type': 'radar', 'platform': 'orbital'},
        'CS2_swath':  {'number': -1,  'type': 'radar', 'platform': 'orbital'},
    },
    'geolocation_error': {'ICESat2': 3.5, 'DEM': 5., 'other': 1.},
}

CONFIG_ENV_VAR = 'ALTIMETRYFIT_SENSOR_CONFIG'

_config = None


def _merge_config(update):
    config = copy.deepcopy(DEFAULT_SENSOR_CONFIG)
    if update is None:
        return config
    for key, val in update.items():
        if key.startswith('_'):
            # comments
            continue
        if key not in config:
            raise ValueError(f'sensor config: unknown entry {key}; expected one of {list(config)}')
        if key == 'altimeters':
            # a given set of altimeters replaces the default set
            config[key] = copy.deepcopy(val)
        else:
            config[key].update(val)
    _check_config(config)
    return config


def _check_config(config):
    numbers = [item['number'] for item in config['altimeters'].values()]
    if len(set(numbers)) != len(numbers):
        raise ValueError(f'sensor config: altimeter numbers are not unique: {numbers}')
    for name, item in config['altimeters'].items():
        if item.get('type') not in ('laser', 'radar') or item.get('platform') not in ('orbital', 'airborne'):
            raise ValueError(f'sensor config: altimeter {name} needs a type (laser or radar) and a platform '
                             f'(orbital or airborne), got {item}')
    alt = config['altimeters']
    if 'CS2_POCA' in alt and 'CS2_swath' in alt and alt['CS2_swath']['number'] != alt['CS2_POCA']['number']+1:
        raise ValueError('sensor config: CS2_swath must be numbered CS2_POCA+1')


def set_sensor_config(config=None):
    '''
    Set the sensor configuration

    config: None (the defaults, updated from the file named by the environment
        variable ALTIMETRYFIT_SENSOR_CONFIG if it is set), a json filename, or
        a dict.  Entries given replace those of DEFAULT_SENSOR_CONFIG.
    '''
    global _config
    if config is None:
        config = os.environ.get(CONFIG_ENV_VAR)
    if isinstance(config, str):
        with open(os.path.expanduser(config)) as fh:
            config = json.load(fh)
    _config = _merge_config(config)
    return get_sensor_config()


def get_sensor_config():
    '''the current sensor configuration (a copy)'''
    if _config is None:
        set_sensor_config()
    return copy.deepcopy(_config)


def _altimeters():
    if _config is None:
        set_sensor_config()
    return _config['altimeters']


def laser_key():
    '''sensor numbers of the laser altimeters, by name'''
    return {name: item['number'] for name, item in _altimeters().items() if item['type'] == 'laser'}


def radar_key():
    '''sensor numbers of the radar altimeters, by name'''
    return {name: item['number'] for name, item in _altimeters().items() if item['type'] == 'radar'}


def altimeter_key():
    '''sensor numbers of all the altimeters (laser and radar), by name'''
    return {name: item['number'] for name, item in _altimeters().items()}


def altimeter_names(sensor_type=None, platform=None):
    '''names of the altimeters, optionally only those of a type (laser, radar) or platform (orbital, airborne)'''
    return [name for name, item in _altimeters().items()
            if (sensor_type is None or item['type'] == sensor_type)
            and (platform is None or item['platform'] == platform)]


def first_DEM_sensor_number():
    '''DEMs are numbered from this value upward'''
    return int(np.floor(np.max(list(altimeter_key().values()))))+1


def sensor_numbers(names, sensor_dict=None):
    '''
    Sensor numbers for a list of altimeter names

    The numbers in sensor_dict (number: name, as in a fit file's /meta/sensors)
    for these names, plus the configured numbers for these names that
    sensor_dict does not use for another sensor.
    '''
    names = list(names)
    key = altimeter_key()
    numbers = []
    if sensor_dict is not None:
        numbers = [num for num, name in sensor_dict.items() if name in names]
    for name in names:
        if name not in key:
            continue
        num = key[name]
        if num in numbers:
            continue
        if sensor_dict is not None and num in sensor_dict and sensor_dict[num] != name:
            continue
        numbers.append(num)
    return numbers


def classify_sensors(sensor_dict=None):
    '''
    Altimeter and DEM sensor numbers, from a dict of sensor names by number

    Sensors are classified by name: those named as altimeters in the
    configuration are altimeters, the others (named by DEM filenames) are
    DEMs.  The configured altimeter numbers that sensor_dict does not use for
    other sensors are included in the altimeter numbers.

    Returns
    -------
    altimeter_sensors : list
    DEM_sensors : numpy array
    '''
    if sensor_dict is None:
        sensor_dict = {}
    altimeter_sensors = sensor_numbers(altimeter_names(), sensor_dict)
    DEM_sensors = np.array([num for num, name in sensor_dict.items() if name not in altimeter_names()])
    return altimeter_sensors, DEM_sensors


def parse_E_geo(arg=None):
    '''
    Expected horizontal geolocation errors (m), by sensor type

    The configured geolocation_error values, updated with arg: a dict, or a
    string like 'ICESat2:3.5,DEM:5,other:1'.  Keys are altimeter names, 'DEM',
    or 'other'.
    '''
    if _config is None:
        set_sensor_config()
    E_geo = dict(_config['geolocation_error'])
    if arg is None:
        return E_geo
    if isinstance(arg, str):
        arg_dict = {}
        for item in arg.split(','):
            key, val = item.split(':')
            arg_dict[key] = float(val)
        arg = arg_dict
    valid_keys = set(altimeter_names()) | {'DEM', 'other'}
    for key, val in arg.items():
        if key not in valid_keys:
            raise ValueError(f'geolocation error: unknown sensor type {key}; use one of {sorted(valid_keys)}')
        E_geo[key] = float(val)
    return E_geo


def geolocation_error(sensor, sensor_dict=None, E_geo=None):
    '''
    Expected horizontal geolocation error (m) for each value in sensor

    Altimeters with an entry in E_geo (see parse_E_geo) get that value, DEMs
    get E_geo['DEM'], and the other sensors get E_geo['other'].  Sensors are
    classified with classify_sensors(sensor_dict).
    '''
    E_geo = parse_E_geo(E_geo)
    sensor = np.asarray(sensor)
    altimeter_sensors, _ = classify_sensors(sensor_dict)
    E = np.full(sensor.shape, E_geo['other'], dtype=float)
    E[~np.isin(sensor, altimeter_sensors)] = E_geo['DEM']
    for name, val in E_geo.items():
        if name in ('DEM', 'other'):
            continue
        E[np.isin(sensor, sensor_numbers([name], sensor_dict))] = val
    return E


def sensor_attr_name(number):
    '''the attribute name for a sensor number in a fit file's /meta/sensors group'''
    number = float(number)
    if number == int(number):
        return f'sensor_{int(number)}'
    return f'sensor_{number:g}'


_sensor_attr_re = re.compile(r'sensor_(-?\d+(?:\.\d+)?)$')


def sensor_from_attr_name(attr_name):
    '''the sensor number for an attribute name written by sensor_attr_name (int if it is a whole number)'''
    number = float(_sensor_attr_re.search(attr_name).group(1))
    if number == int(number):
        return int(number)
    return number


def read_sensor_dict(h5f, data=None):
    '''
    The sensor names by number, from the /meta/sensors attributes of a fit file

    h5f: h5py.File or filename.  If data is given, only the sensors in
    data.sensor are included.  Files written before 11/2026 named sensor 3.5
    (ATM_ICESSN) sensor_3, so for those files 3.5 is not in the output.
    '''
    if isinstance(h5f, str):
        import h5py
        with h5py.File(h5f, 'r') as fh:
            return read_sensor_dict(fh, data=data)
    sensor_dict = {}
    if '/meta/sensors/' not in h5f:
        return sensor_dict
    for attr_name, name in h5f['/meta/sensors/'].attrs.items():
        number = sensor_from_attr_name(attr_name)
        if data is not None and not np.isin(number, data.sensor):
            continue
        if isinstance(name, bytes):
            name = name.decode()
        sensor_dict[number] = name
    return sensor_dict
