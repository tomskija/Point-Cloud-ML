############################################################################################
import os
import mlflow

############################################################################################
def setupMLFlow(experiment_name='point-cloud-ml'):
    """
    Configure MLflow tracking URI and experiment.
    Priority:
        1. MLFLOW_TRACKING_URI env var (http://mlflow:5000 inside Docker Compose, sqlite in tests)
        2. Local fallback: ./mlflow.db SQLite store, so the calculator still runs without a server

    :param experiment_name: MLflow experiment name to log runs under
    """
    ########################################################################################
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', '') or 'sqlite:///' + os.path.abspath('mlflow.db')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    ########################################################################################
    backend = 'Docker' if 'mlflow:' in tracking_uri else ('SQLite' if tracking_uri.startswith('sqlite:') else 'Remote')
    print(f"MLflow tracking URI : {tracking_uri}")
    print(f"MLflow experiment   : {experiment_name}")
    print(f"MLflow backend      : {backend}")
############################################################################################
