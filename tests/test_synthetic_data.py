"""
Tests for the synthetic point cloud generator and point cloud data checks.
"""
import os
import numpy as np
import pandas as pd
import pytest

from dataPrep.dataChecks import checkSyntheticInputs, checkPointCloud, POINT_COLUMNS, GEOMETRIES
from dataPrep.syntheticData import generateSyntheticPointCloud


def make(tmp_path, **kw):
    kw = {'output_dir': str(tmp_path), 'n_points': 3000, **kw}
    return generateSyntheticPointCloud(kw)


# Input checks ##############################################################################
def test_defaults_pass():
    assert checkSyntheticInputs({})['geometry'] == 'terrain'


@pytest.mark.parametrize('overrides, message', [
    ({'geometry': 'cube'}, 'geometry must be one of'),
    ({'n_points': 10}, 'n_points must be >= 100'),
    ({'target_min': 1, 'target_max': 0}, 'target_min'),
    ({'outlier_fraction': 0.7}, 'outlier_fraction'),
    ({'anomaly_radius': 50}, 'anomaly_radius'),
])
def test_bad_inputs_raise(overrides, message):
    with pytest.raises(ValueError, match=message):
        checkSyntheticInputs(overrides)


# Generated data ###########################################################################
@pytest.mark.parametrize('geometry', GEOMETRIES)
def test_each_geometry_passes_checks(tmp_path, geometry):
    manifest, path = make(tmp_path, geometry=geometry)
    report = checkPointCloud(path)
    assert report['passed'], report['errors']
    df = pd.read_csv(path)
    assert list(df.columns) == POINT_COLUMNS and len(df) == 3000


def test_truth_lines_up(tmp_path):
    manifest, path = make(tmp_path, outlier_fraction=0.01, n_anomaly_patches=2)
    df, truth = pd.read_csv(path), pd.read_csv(os.path.join(os.path.dirname(path), 'truth.csv'))
    assert (df['index'] == truth['index']).all()
    assert (truth['anomaly_type'] == 2).sum() == 30
    assert manifest['n_anomaly_points'] == (truth['anomaly_type'] == 1).sum() > 0


def test_anomaly_patches_raise_target(tmp_path):
    _, path = make(tmp_path, n_anomaly_patches=3, anomaly_target_shift=0.5, outlier_fraction=0, target_noise=0)
    df, truth = pd.read_csv(path), pd.read_csv(os.path.join(os.path.dirname(path), 'truth.csv'))
    patch = truth['anomaly_type'] == 1
    assert (df.loc[patch, 't'] - truth.loc[patch, 't_clean']).mean() > 0.05


def test_sphere_points_on_surface(tmp_path):
    _, path = make(tmp_path, geometry='sphere', size=10, n_anomaly_patches=0, outlier_fraction=0, position_noise=0)
    r = np.linalg.norm(pd.read_csv(path)[['x', 'y', 'z']].to_numpy(), axis=1)
    assert np.allclose(r, 5.0, atol=1e-4)


def test_same_seed_same_cloud(tmp_path):
    _, a = make(tmp_path / 'a')
    _, b = make(tmp_path / 'b')
    assert pd.read_csv(a).equals(pd.read_csv(b))


# Data checks on bad files ##################################################################
def test_check_catches_problems(tmp_path):
    bad = pd.DataFrame({'index': [0, 0, 1], 'x': [0, 0, 0], 'y': [0, 0, 0], 'z': [1, 1, 'a'], 't': [1, 2, 3]})
    p = tmp_path / 'bad.csv'; bad.to_csv(p, index=False)
    report = checkPointCloud(str(p), min_points=10)
    errs = ' | '.join(report['errors'])
    assert not report['passed']
    for msg in ['non-numeric', 'only 3 points', 'duplicates', 'degenerate']:
        assert msg in errs


def test_check_missing_columns(tmp_path):
    p = tmp_path / 'bad.csv'; pd.DataFrame({'x': [1], 'y': [2]}).to_csv(p, index=False)
    assert 'missing columns' in checkPointCloud(str(p))['errors'][0]
