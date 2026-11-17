###############################################################################
"""
API contract tests: the same checks run in every calculator repo so the UI can rely on them.
"""
import time
import pytest
from fastapi.testclient import TestClient

from api import app, CALCULATOR_NAME

client = TestClient(app)
SMALL_RUN = {'n_points':1500}
BAD_RUN = {'geometry':'cube'}


@pytest.fixture(autouse=True)
def mlflowStore(tmp_path, monkeypatch):
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'sqlite:///' + str(tmp_path / 'mlflow.db'))


def defaultsInJson(schema={}, overrides={}):
    """Build an input JSON from the schema defaults, the same way the UI does."""
    params, tables = [], []
    for group in schema['groups']:
        for f in group['fields']:
            value = overrides.get(f['name'], f['default'])
            if value in ('', None): continue
            (tables if f['kind'] == 'table' else params).append(
                {'name': f['name'], 'values': value} if f['kind'] == 'table' else {'name': f['name'], 'value': value})
    return {'data': {'params': params, 'tables': tables}}


def waitFor(job_id, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        status = client.get(f'/runs/{job_id}/status').json()
        if status['status'] in ('complete', 'failed'): return status
        time.sleep(0.2)
    raise TimeoutError(job_id)


###############################################################################
def test_health_and_ready():
    assert client.get('/health').json()['calculator'] == CALCULATOR_NAME
    assert client.get('/ready').json()['ready'] is True


def test_schema_shape():
    s = client.get('/schema').json()
    assert s['name'] == CALCULATOR_NAME and s['groups']
    for g in s['groups']:
        for f in g['fields']:
            assert {'name', 'label', 'type', 'default', 'kind'} <= set(f)
            if f['type'] == 'select': assert f['default'] in f['options']


def test_calculate_with_schema_defaults(tmp_path):
    inJson = defaultsInJson(client.get('/schema').json(), SMALL_RUN)
    inJson['data']['params'].append({'name': 'output_dir', 'value': str(tmp_path / 'out')})
    job = client.post('/calculate', json={'inJson': inJson}).json()
    assert job['status'] == 'pending'
    status = waitFor(job['job_id'])
    out = client.get(f"/jobs/{job['job_id']}/output").json()
    assert status['status'] == 'complete', status['error']
    assert out['output']['data_check_passed'] is True
    assert any(j['job_id'] == job['job_id'] for j in client.get('/jobs').json())


def test_bad_input_fails_cleanly():
    inJson = defaultsInJson(client.get('/schema').json(), {**SMALL_RUN, **BAD_RUN})
    job = client.post('/calculate', json={'inJson': inJson}).json()
    status = waitFor(job['job_id'])
    assert status['status'] == 'failed' and status['error']


def test_missing_data_rejected():
    assert client.post('/calculate', json={'inJson': {}}).status_code == 422
    assert client.get('/runs/nope/status').status_code == 404
