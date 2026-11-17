############################################################################################
"""
Data checks for the point cloud calculator.

checkSyntheticInputs: fills defaults, casts types and validates inputs for the synthetic
                      point cloud generator (same idea as checkData in Production-Prediction-ML).
checkPointCloud:      validates any point cloud CSV with columns [index, x, y, z, t] before
                      it goes into a model, synthetic or not.
"""
############################################################################################
import os
import numpy as np
import pandas as pd

############################################################################################
POINT_COLUMNS = ['index', 'x', 'y', 'z', 't']
GEOMETRIES    = ['terrain', 'sphere', 'torus', 'box']
TARGET_FIELDS = ['smooth_field', 'height', 'radial']

SYNTHETIC_DEFAULTS = {
    'name':                  'Synthetic_PointCloud',
    'random_seed':           5195,
    'output_dir':            'Data/Synthetic',
    'geometry':              'terrain',
    'n_points':              20000,
    'size':                  10.0,      # terrain side length, sphere radius x2, box edge, torus major diameter
    'relief':                1.5,       # terrain only: vertical relief
    'minor_radius_ratio':    0.3,       # torus only: tube radius / major radius
    'position_noise':        0.01,      # Gaussian sensor noise on xyz (same units as size)
    'target_field':          'smooth_field',
    'target_min':            0.0,
    'target_max':            1.0,
    'target_noise':          0.01,
    'n_field_centers':       6,
    'n_anomaly_patches':     3,
    'anomaly_radius':        0.6,
    'anomaly_amplitude':     0.25,      # bump height along the surface normal
    'anomaly_target_shift':  0.5,       # added to t at the bump peak
    'outlier_fraction':      0.005,     # stray points off the surface
    'max_points':            5000000,
}

############################################################################################
def checkSyntheticInputs(inputData={}):
    """
    Fill defaults, cast types and validate synthetic point cloud inputs.
    Collects every problem and raises once so the UI can show them together.
    """
    ########################################################################################
    data = dict(SYNTHETIC_DEFAULTS)
    data.update({k: v for k, v in inputData.items() if v is not None})
    errors = []
    for key in ['random_seed', 'n_points', 'n_field_centers', 'n_anomaly_patches', 'max_points']:
        try: data[key] = int(data[key])
        except (TypeError, ValueError): errors.append(f"{key} must be an integer, got {data[key]!r}")
    for key, default in SYNTHETIC_DEFAULTS.items():
        if isinstance(default, float):
            try: data[key] = float(data[key])
            except (TypeError, ValueError): errors.append(f"{key} must be a number, got {data[key]!r}")
    if errors: raise ValueError("Input check failed:\n  - " + "\n  - ".join(errors))
    ########################################################################################
    data['geometry'] = str(data['geometry']).lower()
    data['target_field'] = str(data['target_field']).lower()
    if data['geometry'] not in GEOMETRIES:          errors.append(f"geometry must be one of {GEOMETRIES}")
    if data['target_field'] not in TARGET_FIELDS:   errors.append(f"target_field must be one of {TARGET_FIELDS}")
    if data['n_points'] < 100:                      errors.append("n_points must be >= 100")
    if data['n_points'] > data['max_points']:       errors.append(f"n_points must be <= {data['max_points']:,}")
    if data['size'] <= 0:                           errors.append("size must be > 0")
    if not (0 < data['minor_radius_ratio'] < 1):    errors.append("minor_radius_ratio must be in (0, 1)")
    if data['target_min'] >= data['target_max']:    errors.append("target_min must be less than target_max")
    if data['n_field_centers'] < 1:                 errors.append("n_field_centers must be >= 1")
    if data['n_anomaly_patches'] < 0:               errors.append("n_anomaly_patches must be >= 0")
    if not (0 <= data['outlier_fraction'] < 0.5):   errors.append("outlier_fraction must be in [0, 0.5)")
    for key in ['position_noise', 'target_noise', 'relief', 'anomaly_amplitude']:
        if data[key] < 0: errors.append(f"{key} must be >= 0")
    if data['n_anomaly_patches'] > 0 and not (0 < data['anomaly_radius'] < data['size'] / 2):
        errors.append("anomaly_radius must be > 0 and smaller than size / 2")
    ########################################################################################
    if errors: raise ValueError("Input check failed:\n  - " + "\n  - ".join(errors))
    return data

############################################################################################
def checkPointCloud(csvPath='', min_points=100, max_duplicate_fraction=0.01, extra_columns_ok=True):
    """
    Validate a point cloud CSV before modeling.
    :param csvPath:                Path to the CSV
    :param min_points:             Minimum number of points
    :param max_duplicate_fraction: Max share of exactly repeated xyz positions
    :param extra_columns_ok:       Allow columns beyond [index, x, y, z, t]
    :return:                       dict with passed, errors, warnings, summary
    """
    ########################################################################################
    errors, warnings, summary = [], [], {}
    if not os.path.exists(csvPath):
        return {'passed': False, 'errors': [f"missing file: {csvPath}"], 'warnings': warnings, 'summary': summary}
    df = pd.read_csv(csvPath)
    ########################################################################################
    missing = [c for c in POINT_COLUMNS if c not in df.columns]
    if missing:
        return {'passed': False, 'errors': [f"missing columns: {missing} (need {POINT_COLUMNS})"], 'warnings': warnings, 'summary': summary}
    extra = [c for c in df.columns if c not in POINT_COLUMNS]
    if extra: (warnings if extra_columns_ok else errors).append(f"extra columns: {extra}")
    ########################################################################################
    vals = df[POINT_COLUMNS].apply(pd.to_numeric, errors='coerce')
    if vals.isna().any().any():
        bad = vals.columns[vals.isna().any()].tolist()
        errors.append(f"non-numeric or empty values in {bad}")
    xyz = vals[['x', 'y', 'z']].to_numpy(dtype=float)
    if not np.all(np.isfinite(vals.to_numpy(dtype=float)[~vals.isna().any(axis=1).to_numpy()])):
        errors.append("infinite values present")
    if len(df) < min_points: errors.append(f"only {len(df)} points, need at least {min_points}")
    if vals['index'].duplicated().any(): errors.append("index column has duplicates")
    ########################################################################################
    finite = np.isfinite(xyz).all(axis=1)
    if finite.any():
        extent = np.nanmax(xyz[finite], axis=0) - np.nanmin(xyz[finite], axis=0)
        if np.sum(extent > 0) < 2: errors.append("points are degenerate (spread in fewer than 2 dimensions)")
        dup = pd.DataFrame(xyz[finite]).duplicated().mean()
        if dup > max_duplicate_fraction: errors.append(f"{dup:.1%} of points are exact duplicates")
        summary['bbox_min'] = np.nanmin(xyz[finite], axis=0).round(4).tolist()
        summary['bbox_max'] = np.nanmax(xyz[finite], axis=0).round(4).tolist()
    t = vals['t'].to_numpy(dtype=float)
    if np.nanstd(t) == 0: warnings.append("target t is constant")
    summary.update({'n_points': int(len(df)), 't_min': float(np.nanmin(t)), 't_max': float(np.nanmax(t))})
    ########################################################################################
    return {'passed': len(errors) == 0, 'errors': errors, 'warnings': warnings, 'summary': summary}
############################################################################################
