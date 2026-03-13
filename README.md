# Point-Cloud-ML

**By: Jackson R. Tomski**

An end-to-end ML workflow for analyzing 3D point cloud data from industrial surface scans — covering mesh reconstruction, anomaly detection, and deep learning-based regression. The user selects a geometry and analysis type, and the workflow routes to the appropriate ML pipeline automatically.

---

## What This Does

Starting from raw XYZ scan data, the notebook reconstructs a 3D surface mesh and applies a selected ML approach to extract insight from the point cloud. The core application is industrial inspection — identifying imperfections and thickness variations across pressure vessel geometries.

**Supported Geometries**
- Cylinder
- Spherical Head
- Circular Flat Plate
- Pipe Elbow

**Analysis Types**
- `IMPERFECTION` — surface deviation from nominal geometry
- `THICKNESS` — wall thickness variation across the surface

**Problem Types**
```python
PROB01 = 'ANOMALY_DETECTION'    # DBSCAN — unsupervised spatial anomaly detection
PROB02 = 'REGRESSION'           # PointNet (TensorFlow) — point-wise t prediction
PROB03 = 'SEGMENTATION'         # PointNet++ — defect region segmentation (planned)
PROB04 = 'UNCERTAINTY'          # MC Dropout / BNN — prediction with uncertainty (planned)
```

**Mesh Reconstruction** — Ball-Pivoting algorithm via Open3D. Normals estimated and oriented per geometry type, with automated mesh cleanup (degenerate triangles, duplicates, non-manifold edges, Taubin smoothing).

**Visualization** — Interactive Plotly Mesh3d plots with per-face colormap and lighting, plus matplotlib trisurf for quick inspection.

---

## Project Structure

```
Point-Cloud-ML/
├── .devcontainer/
│   └── devcontainer.json
├── data/
│   └── README.md               # Dataset placement instructions
├── notebooks/
│   └── PointCloud_ML_Workflow.ipynb
├── pointCloudCalculator/
│   ├── config.py               # Geometry, analysis, and problem type selection
│   ├── utils/
│   │   ├── preprocessing.py    # Open3D pipeline and mesh reconstruction
│   │   └── visualization.py    # Plotly and matplotlib rendering
│   └── models/
│       ├── anomaly.py          # DBSCAN anomaly detection
│       ├── regression.py       # PointNet TF regression
│       ├── segmentation.py     # PointNet++ segmentation (planned)
│       └── uncertainty.py      # MC Dropout / BNN (planned)
├── results/                    # Figures and outputs generated at runtime
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .gitignore
```

---

## Getting Started

**Option 1 — VS Code Dev Container (recommended)**

```bash
git clone https://github.com/tomskija/Point-Cloud-ML.git
cd Point-Cloud-ML
```

Open in VS Code → `Reopen in Container`. Place your scan CSVs in `data/` before running the notebook.

**Option 2 — Docker Compose**

```bash
docker-compose up --build
```

Navigate to `http://localhost:8888` and open the notebook from `notebooks/`.

**Option 3 — Local**

```bash
pip install -r requirements.txt
jupyter lab
```

---

## Selecting a Workflow

At the top of the notebook, set your geometry, analysis type, and problem type:

```python
geometry     = 'CYLINDER'           # 'CYLINDER', 'SPHERICAL_HEAD', 'CIRCULAR_FLAT_PLATE', 'PIPE_ELBOW'
Analysis     = 'IMPERFECTION'       # 'IMPERFECTION' or 'THICKNESS'
problem_type = 'ANOMALY_DETECTION'  # 'ANOMALY_DETECTION', 'REGRESSION', 'SEGMENTATION', 'UNCERTAINTY'
```

The rest of the workflow routes automatically from there.

---

## Dependencies

Python 3.11 · open3d · tensorflow · scikit-learn · numpy · pandas · plotly · matplotlib · scipy · JupyterLab

---

## Data

Place scan CSV files in `data/` before running. Each CSV is expected to have columns `[index, x, y, z, t]` where `t` is the target measurement (imperfection or thickness). Not included in this repo — see `data/README.md`.