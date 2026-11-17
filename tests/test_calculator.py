###############################################################################
"""
End-to-end tests of Calculator.calculate on small synthetic clouds.
"""
import json
import asyncio
import pytest
from os.path import dirname, join, abspath

from Calculator import calculate

IN_JSON = join(dirname(abspath(__file__)), 'in_jsons', 'testCase01.json')


def loadInJson(overrides={}):
    with open(IN_JSON) as f: inJson = json.load(f)
    params = {p['name']: p for p in inJson['data']['params']}
    for name, value in overrides.items():
        if name in params: params[name]['value'] = value
        else: inJson['data']['params'].append({'name': name, 'value': value, 'valueType': 2})
    return inJson


@pytest.fixture(autouse=True)
def mlflowStore(tmp_path, monkeypatch):
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'sqlite:///' + str(tmp_path / 'mlflow.db'))


###############################################################################
@pytest.mark.parametrize('geometry', ['terrain', 'torus'])
def test_synthetic_run(tmp_path, geometry):
    out = asyncio.run(calculate(inJson=loadInJson({'output_dir': str(tmp_path), 'n_points': 2000, 'geometry': geometry})))
    assert not isinstance(out, str), out
    result = out.flatten()
    assert result['data_check_passed'] is True and result['n_points'] == 2000


def test_existing_csv_run(tmp_path):
    first = asyncio.run(calculate(inJson=loadInJson({'output_dir': str(tmp_path), 'n_points': 1000}))).flatten()
    out = asyncio.run(calculate(inJson=loadInJson({'dataset_mode': 'existing', 'csv_path': first['csv_path']})))
    assert not isinstance(out, str), out
    assert out.flatten()['n_points'] == 1000


@pytest.mark.parametrize('overrides, message', [
    ({'problemType': 'PROB07'}, 'problemType'),
    ({'dataset_mode': 'cloud'}, 'dataset_mode'),
    ({'dataset_mode': 'existing'}, 'csv_path'),
    ({'geometry': 'cube'}, 'geometry'),
])
def test_bad_inputs_return_message(overrides, message):
    out = asyncio.run(calculate(inJson=loadInJson(overrides)))
    assert isinstance(out, str) and message in out
