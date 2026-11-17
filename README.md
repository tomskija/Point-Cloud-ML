# Point-Cloud-ML

**By: Jackson R. Tomski**

This project is a machine learning workflow for 3D point clouds. You pick a problem type and the calculator routes to the right pipeline: anomaly detection, point-wise regression, segmentation, or prediction with uncertainty.

Everything runs on a synthetic point cloud generator, data checks, and the calculator, all in Docker the same way as my other repos. The generator comes with a truth file that labels every anomaly and outlier, so each model can be scored against known answers instead of just eyeballed.

---

## What's Inside

### Synthetic Point Cloud Generator

`dataPrep/syntheticData.py` builds a point cloud from an input JSON:

1. **Surface:** points sampled evenly by area on one of four shapes.

   | `geometry` | What it is |
   |------------|------------|
   | `terrain` | Height field made of smooth hills and valleys |
   | `sphere` | Sphere |
   | `torus` | Donut shape, sampled so the inner ring isn't over-dense |
   | `box` | Six faces of a cube |

2. **Target `t`:** a smooth value on the surface. `smooth_field` is a sum of Gaussian blobs, `height` follows z, and `radial` follows distance from the center. It gets scaled to `[target_min, target_max]`.
3. **Anomaly patches:** smooth bumps pushed out along the surface normal, with a matching jump in `t`.
4. **Outliers:** stray points scattered off the surface.
5. **Noise** on both the positions and `t`.

| File | What it is |
|------|------------|
| `pointcloud.csv` | `[index, x, y, z, t]`, the format the calculator reads |
| `truth.csv` | Per point: `is_anomaly`, `anomaly_type` (0 none, 1 patch, 2 outlier), clean `t`, and the surface normal |
| `manifest.json` | Inputs and file names |

As a quick sanity check, plain DBSCAN catches about 90% of outliers on the terrain and sphere but only 50 to 65% on the torus and box, where stray points can land close to another part of the surface. That spread is useful for testing.

### Data Checks

- **`checkPointCloud`** runs on any CSV with `[index, x, y, z, t]`, synthetic or not. It checks columns, numeric and finite values, a unique index, a minimum number of points, that the cloud isn't flat in all but one direction, and duplicate points.
- **`checkSyntheticInputs`** validates generator inputs and reports every problem at once.

### Calculator

`Calculator.py` follows the same flow as Production-Prediction-ML:

```
input JSON → checkData → build or load point cloud → checkPointCloud → MLflow logging → output wrapper
```

`dataset_mode` is either `synthetic` or `existing` (point `csv_path` at your own CSV).

---

## Problem Types

| Type | Name | Approach |
|------|------|----------|
| `PROB01` | `ANOMALY_DETECTION` | DBSCAN, unsupervised spatial anomaly detection |
| `PROB02` | `REGRESSION` | PointNet (TensorFlow), point-wise prediction of `t` |
| `PROB03` | `SEGMENTATION` | PointNet++ region segmentation |
| `PROB04` | `UNCERTAINTY` | MC Dropout / BNN, prediction with uncertainty |

Set `problemType` in the input JSON to pick one.

---

## Getting Started

**You need:** Docker Desktop and VS Code with the Dev Containers extension.

```bash
git clone https://github.com/tomskija/Point-Cloud-ML.git
cd Point-Cloud-ML
```

Open the folder in VS Code and select **Reopen in Container**. Docker Compose starts three services: the calculator, its FastAPI server, and an MLflow tracking server.

**Run the calculator:**
```bash
python pointCloudMLCalculator/Calculator.py
```

**Run the test suite:**
```bash
pytest tests -v
```

**Without VS Code:**
```bash
docker compose up -d --build
docker compose exec calculator pytest tests -v
docker compose exec calculator python pointCloudMLCalculator/Calculator.py
```

| Service   | URL                   |
|-----------|-----------------------|
| API       | http://localhost:8003  |
| API Docs  | http://localhost:8003/docs |
| MLflow UI | http://localhost:5003 |

MLflow is on 5003 so it can run next to my other repos.

---

## API Endpoints

| Method | Endpoint                | Description                                        |
|--------|-------------------------|----------------------------------------------------|
| `GET`  | `/health`               | Check if the API is running                        |
| `GET`  | `/ready`                | Check the calculator and its schema load           |
| `GET`  | `/schema`               | Input form definition (fields, types, defaults, groups) used by the UI |
| `POST` | `/calculate`            | Submit a run with `{"inJson": {...}}`, returns `job_id` immediately |
| `GET`  | `/runs/{job_id}/status` | Poll run status: pending / queued / running / complete / failed |
| `GET`  | `/jobs/{job_id}/output` | Output params and tables once the run is complete |
| `GET`  | `/jobs`                 | List every job the API has seen                    |

Runs go one at a time per calculator in a background thread, and each API run writes to its own folder under `Data/Synthetic/` so runs don't overwrite each other. The full interactive docs are at `http://localhost:8003/docs`. This is the same contract as Production-Prediction-ML, which is what lets [ML-Platform-UI](https://github.com/tomskija/ML-Platform-UI) drive every calculator the same way.

---

## Configuration

Inputs use the same `params` / `tables` JSON format as Production-Prediction-ML. Anything left out falls back to a default.

| Parameter | What it does |
|-----------|--------------|
| `problemType` | `PROB01` to `PROB04` |
| `dataset_mode` | `synthetic` or `existing` |
| `csv_path` | Your point cloud CSV (for `existing`) |
| `geometry` | `terrain`, `sphere`, `torus` or `box` |
| `n_points` | Total points, outliers included |
| `size` | Overall size of the shape |
| `target_field` | `smooth_field`, `height` or `radial` |
| `n_anomaly_patches`, `anomaly_radius`, `anomaly_amplitude`, `anomaly_target_shift` | How many bumps, how wide and tall, and how much `t` jumps |
| `outlier_fraction` | Share of stray points |
| `position_noise`, `target_noise` | Noise on xyz and `t` |
| `random_seed` | Same seed, same point cloud |

See `tests/in_jsons/testCase01.json` for an example. Generated data goes to `Data/Synthetic/`, which is gitignored.

---

## Project Structure

```
Point-Cloud-ML/
├── .devcontainer/devcontainer.json     # Dev container config
├── .github/workflows/ci.yml            # Builds the Docker stack, runs pytest and the calculator
├── pointCloudMLCalculator/
│   ├── Calculator.py                   # Orchestrator, routes PROB01 to PROB04
│   ├── dataPrep/
│   │   ├── syntheticData.py            # Synthetic point cloud generator
│   │   ├── schema.py                   # Input form definition served at /schema
│   │   ├── dataChecks.py               # Input and point cloud checks
│   │   ├── inputWrapper.py             # JSON input parser
│   │   ├── outputWrapper.py            # Output wrapper
│   │   └── mlFlowConfig.py             # MLflow (Docker server or local SQLite)
│   └── utils/utils.py                  # JSON helpers
├── tests/
│   ├── in_jsons/                       # pytest input configs
│   ├── test_synthetic_data.py
│   ├── test_api.py
│   └── test_calculator.py
├── api.py                              # FastAPI, same calculator contract as PPML
├── conftest.py                         # pytest path config
├── Dockerfile                          # Python 3.11
├── Dockerfile.mlflow                   # MLflow server
├── docker-compose.yml                  # calculator, api, mlflow
└── requirements.txt
```

---

## Dependencies

Python 3.11 · numpy · pandas · scipy · scikit-learn · matplotlib · seaborn · openpyxl · mlflow · pytest · fastapi · uvicorn · pydantic
