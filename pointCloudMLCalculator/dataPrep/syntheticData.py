############################################################################################
"""
Synthetic point cloud generator.

Samples points on a chosen surface (terrain, sphere, torus or box), puts a smooth target
field t on it, then adds the things the PROB01-PROB04 workflows care about:
    sensor noise        Gaussian jitter on xyz and t
    anomaly patches     local bumps along the surface normal with a matching jump in t
    outliers            stray points off the surface (what DBSCAN-style detection should flag)

Writes:
    pointcloud.csv   [index, x, y, z, t]          the format the calculator reads
    truth.csv        [index, is_anomaly, anomaly_type, t_clean, nx, ny, nz]
                     anomaly_type: 0 none, 1 surface patch, 2 off-surface outlier
    manifest.json    inputs and file names

Output goes to a gitignored folder and stays local.
"""
############################################################################################
import os
import sys
import json
import numpy as np
import pandas as pd
from os.path import dirname, join, isabs, abspath

try:
    from dataPrep.dataChecks import checkSyntheticInputs, POINT_COLUMNS
    from dataPrep.inputWrapper import standardInputWrapper
except ImportError:
    from .dataChecks import checkSyntheticInputs, POINT_COLUMNS
    from .inputWrapper import standardInputWrapper

REPO_ROOT = dirname(dirname(dirname(abspath(__file__))))

############################################################################################
def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-12)

############################################################################################
def sampleSurface(data={}, n=0, rng=None):
    """
    Points and unit normals on the chosen surface, roughly uniform by area.
    :return: (points n x 3, normals n x 3)
    """
    ########################################################################################
    s, g = data['size'], data['geometry']
    if g == 'terrain':
        # Height field z = relief * sum of Gaussian hills/valleys over a square
        k = 8
        cx, cy = rng.uniform(0, s, k), rng.uniform(0, s, k)
        amp, sig = rng.uniform(-1, 1, k), rng.uniform(0.12, 0.3, k) * s
        def height(x, y):
            return data['relief'] * np.sum(amp * np.exp(-((x[:, None] - cx) ** 2 + (y[:, None] - cy) ** 2) / (2 * sig ** 2)), axis=1)
        x, y = rng.uniform(0, s, n), rng.uniform(0, s, n)
        z = height(x, y)
        e = 1e-4 * s
        dzdx = (height(x + e, y) - height(x - e, y)) / (2 * e)
        dzdy = (height(x, y + e) - height(x, y - e)) / (2 * e)
        return np.c_[x, y, z], unit(np.c_[-dzdx, -dzdy, np.ones(n)])
    ########################################################################################
    if g == 'sphere':
        nrm = unit(rng.standard_normal((n, 3)))
        return nrm * (s / 2), nrm
    ########################################################################################
    if g == 'torus':
        R = (s / 2) / (1 + data['minor_radius_ratio']); r = R * data['minor_radius_ratio']
        u, v = np.empty(0), np.empty(0)
        while len(u) < n:  # rejection sampling keeps density uniform by area
            uu, vv = rng.uniform(0, 2 * np.pi, 2 * n), rng.uniform(0, 2 * np.pi, 2 * n)
            keep = rng.uniform(0, R + r, 2 * n) < (R + r * np.cos(vv))
            u, v = np.r_[u, uu[keep]], np.r_[v, vv[keep]]
        u, v = u[:n], v[:n]
        nrm = np.c_[np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)]
        ring = np.c_[R * np.cos(u), R * np.sin(u), np.zeros(n)]
        return ring + r * nrm, nrm
    ########################################################################################
    # box: pick a face, then a point on it
    face = rng.integers(0, 6, n)
    axis, sign = face // 2, np.where(face % 2 == 0, -1.0, 1.0)
    pts = rng.uniform(-s / 2, s / 2, (n, 3))
    pts[np.arange(n), axis] = sign * s / 2
    nrm = np.zeros((n, 3)); nrm[np.arange(n), axis] = sign
    return pts, nrm

############################################################################################
def targetField(data={}, pts=None, rng=None):
    """Smooth target on the clean surface, scaled to [target_min, target_max]."""
    ########################################################################################
    if data['target_field'] == 'height':
        f = pts[:, 2].copy()
    elif data['target_field'] == 'radial':
        f = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
    else:
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        k = data['n_field_centers']
        c = rng.uniform(lo, hi, (k, 3))
        w, sig = rng.uniform(-1, 1, k), rng.uniform(0.15, 0.35, k) * data['size']
        f = np.zeros(len(pts))
        for ck, wk, sk in zip(c, w, sig):  # loop over centers keeps memory flat for big clouds
            f += wk * np.exp(-((pts - ck) ** 2).sum(axis=1) / (2 * sk ** 2))
    f = (f - f.min()) / max(f.max() - f.min(), 1e-12)
    return data['target_min'] + f * (data['target_max'] - data['target_min'])

############################################################################################
def generateSyntheticPointCloud(inputData={}):
    """
    Build and write a synthetic point cloud.
    :param inputData: Flat dict (from standardInputWrapper). Missing keys fall back to generic defaults.
    :return:          (manifest dict, path to pointcloud.csv)
    """
    ########################################################################################
    data = checkSyntheticInputs(inputData=inputData)
    rng = np.random.default_rng(data['random_seed'])
    out = data['output_dir'] if isabs(data['output_dir']) else join(REPO_ROOT, data['output_dir'])
    os.makedirs(out, exist_ok=True)
    ########################################################################################
    n_out = int(round(data['outlier_fraction'] * data['n_points']))
    n_surf = data['n_points'] - n_out
    pts, nrm = sampleSurface(data=data, n=n_surf, rng=rng)
    t_clean = targetField(data=data, pts=pts, rng=rng)
    t = t_clean.copy()
    anomaly_type = np.zeros(n_surf, dtype=int)
    ########################################################################################
    # Anomaly patches: smooth bump along the normal, matching bump in t
    for c in pts[rng.choice(n_surf, size=min(data['n_anomaly_patches'], n_surf), replace=False)]:
        d = np.linalg.norm(pts - c, axis=1)
        w = np.where(d < data['anomaly_radius'], (1 - (d / data['anomaly_radius']) ** 2) ** 2, 0.0)
        pts = pts + (data['anomaly_amplitude'] * w)[:, None] * nrm
        t = t + data['anomaly_target_shift'] * w
        anomaly_type[w > 0.05] = 1
    ########################################################################################
    pts = pts + rng.normal(0, data['position_noise'], pts.shape)
    t = t + rng.normal(0, data['target_noise'], n_surf)
    ########################################################################################
    # Outliers: uniform in a slightly padded bounding box
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.15 * (hi - lo)
    out_pts = rng.uniform(lo - pad, hi + pad, (n_out, 3))
    out_t = rng.uniform(data['target_min'], data['target_max'], n_out)
    ########################################################################################
    all_pts = np.r_[pts, out_pts]
    all_t = np.r_[t, out_t]
    all_type = np.r_[anomaly_type, np.full(n_out, 2)]
    all_clean = np.r_[t_clean, np.full(n_out, np.nan)]
    all_nrm = np.r_[nrm, np.full((n_out, 3), np.nan)]
    order = rng.permutation(len(all_t))          # shuffle so outliers are not all at the end
    ########################################################################################
    df = pd.DataFrame({'index': np.arange(len(order)), 'x': all_pts[order, 0], 'y': all_pts[order, 1],
                       'z': all_pts[order, 2], 't': all_t[order]})[POINT_COLUMNS].round(6)
    truth = pd.DataFrame({'index': np.arange(len(order)), 'is_anomaly': (all_type[order] > 0).astype(int),
                          'anomaly_type': all_type[order], 't_clean': all_clean[order],
                          'nx': all_nrm[order, 0], 'ny': all_nrm[order, 1], 'nz': all_nrm[order, 2]}).round(6)
    csv_path = join(out, 'pointcloud.csv')
    df.to_csv(csv_path, index=False)
    truth.to_csv(join(out, 'truth.csv'), index=False)
    ########################################################################################
    manifest = {'name': data['name'], 'synthetic': True, 'random_seed': data['random_seed'],
                'files': {'points': 'pointcloud.csv', 'truth': 'truth.csv'},
                'n_points': int(len(df)), 'n_anomaly_points': int((all_type == 1).sum()), 'n_outliers': n_out,
                'inputs': data}
    with open(join(out, 'manifest.json'), 'w') as f: json.dump(manifest, f, indent=2)
    print(f"Synthetic {data['geometry']} point cloud: {len(df):,} points "
          f"({manifest['n_anomaly_points']} in anomaly patches, {n_out} outliers) written to {out}")
    return manifest, csv_path

############################################################################################
def main():
    thisDirName = dirname(abspath(__file__))
    fileName = sys.argv[1] if len(sys.argv) > 1 else join(thisDirName, '..', '..', 'tests', 'in_jsons', 'testCase01.json')
    with open(fileName) as json_file: inJson = json.load(json_file)
    generateSyntheticPointCloud(inputData=standardInputWrapper(inJson=inJson))

############################################################################################
if __name__ == "__main__":
    main()
############################################################################################
