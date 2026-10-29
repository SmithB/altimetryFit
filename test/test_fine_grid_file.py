import os
import json
import importlib

fa = importlib.import_module('altimetryFit.fit_altimetry')


def test_fine_grid_file(tmp_path):
    # the json file sets fine_grid_args; a relative region_file is found next to it
    os.makedirs(tmp_path/'cfg')
    fine_args = {'region_file': 'rifts.geojson', 'spacing': 100, 'dt': 0.25,
                 'E_RMS': {'d2z0_dx2': 1.0, 'd3z_dx2dt': 0.3, 'd2z_dt2': 100}}
    with open(tmp_path/'cfg'/'fine.json', 'w') as fh:
        json.dump(fine_args, fh)
    with open(tmp_path/'GI.json', 'w') as fh:
        json.dump({}, fh)
    args = fa.parse_inputs(['--xy0', '-1500000', '-450000', '--Width', '40000', '-t', '2015,2025',
                            '-g', '250,2000,0.25', '--reference_epoch', '20', '-H', '-1',
                            '--GeoIndex_source_file', str(tmp_path/'GI.json'),
                            '--out_name', str(tmp_path/'E-1500_N-450.h5'),
                            '--fine_grid_file', str(tmp_path/'cfg'/'fine.json')])
    assert args.fine_grid_args == fine_args | {'region_file': str(tmp_path/'cfg'/'rifts.geojson')}

    # without the argument, there are no fine grids
    args = fa.parse_inputs(['--xy0', '-1500000', '-450000', '--Width', '40000', '-t', '2015,2025',
                            '-g', '250,2000,0.25', '--reference_epoch', '20', '-H', '-1',
                            '--GeoIndex_source_file', str(tmp_path/'GI.json'),
                            '--out_name', str(tmp_path/'E-1500_N-450.h5')])
    assert args.fine_grid_args is None
