# Point-Cloud-ML

**By: Jackson R. Tomski**

A machine learning workflow for 3D point cloud data — covering surface mesh reconstruction, anomaly detection, and deep learning-based regression. The user selects a geometry and problem type, and the workflow routes to the appropriate ML pipeline automatically.

---

## What This Does

Starting from raw XYZ point cloud data, the notebook reconstructs a 3D surface mesh and applies a selected ML approach to extract insight from the spatial data. The workflow is designed to be geometry-agnostic and extensible to new problem types as they are added.

**Problem Types**
```python
PROB01 = 'ANOMALY_DETECTION'    # DBSCAN — unsupervised spatial anomaly detection
PROB02 = 'REGRESSION'           # PointNet (TensorFlow) — point-wise target prediction
PROB03 = 'SEGMENTATION'         # PointNet++ — region segmentation (planned)
PROB04 = 'UNCERTAINTY'          # MC Dropout / BNN — prediction with uncertainty (planned)
```

**Mesh Reconstruction** — Ball-Pivoting algorithm via Open3D with automated normal estimation and mesh cleanup.

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
│   ├── config.py               # Geometry and problem type selection
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

Open in VS Code → `Reopen in Container`. Place your CSVs in `data/` before running the notebook.

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

At the top of the notebook:

```python
problem_type = 'ANOMALY_DETECTION'  # 'ANOMALY_DETECTION', 'REGRESSION', 'SEGMENTATION', 'UNCERTAINTY'
```

The rest of the workflow routes automatically from there.

---

## Dependencies

Python 3.11 · open3d · tensorflow · scikit-learn · numpy · pandas · plotly · matplotlib · scipy · JupyterLab

---

## Data

Synthetic point cloud data in CSV format with columns `[index, x, y, z, t]` where `t` is the target variable. Not included in this repo — see `data/README.md`.