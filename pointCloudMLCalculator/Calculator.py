##########################################################
import os
import json
import time
import asyncio
import numpy as np
import pandas as pd
import mlflow
from os.path import dirname, join, isabs, abspath
from dataPrep.inputWrapper import standardInputWrapper
from dataPrep.outputWrapper import standardOutputWrapper
from dataPrep.mlFlowConfig import setupMLFlow
from dataPrep.dataChecks import checkPointCloud
from dataPrep.syntheticData import generateSyntheticPointCloud, REPO_ROOT

PROBLEM_TYPES = {
    'PROB01': 'ANOMALY_DETECTION',   # DBSCAN, unsupervised spatial anomaly detection
    'PROB02': 'REGRESSION',          # PointNet (TensorFlow), point-wise target prediction
    'PROB03': 'SEGMENTATION',        # PointNet++, region segmentation (planned)
    'PROB04': 'UNCERTAINTY',         # MC Dropout / BNN, prediction with uncertainty (planned)
}


##########################################################
def checkData(inputData={}):
    """
    Calculator-level input checks. Synthetic generator inputs are checked inside
    generateSyntheticPointCloud (checkSyntheticInputs).
    """
    ######################################################
    inputData["name"] = inputData.get("name", "PointCloud_Results")
    inputData["problemType"] = str(inputData.get("problemType", "PROB01")).upper()
    inputData["dataset_mode"] = str(inputData.get("dataset_mode", "synthetic")).lower()
    inputData["min_points"] = int(inputData.get("min_points", 100))
    ######################################################
    if inputData["problemType"] not in PROBLEM_TYPES:
        raise ValueError(f"problemType must be one of {list(PROBLEM_TYPES)}, got {inputData['problemType']!r}")
    if inputData["dataset_mode"] not in ("synthetic", "existing"):
        raise ValueError(f"dataset_mode must be 'synthetic' or 'existing', got {inputData['dataset_mode']!r}")
    if inputData["dataset_mode"] == "existing":
        path = inputData.get("csv_path", "")
        if not path: raise ValueError("dataset_mode 'existing' needs csv_path")
        inputData["csv_path"] = path if isabs(path) else join(REPO_ROOT, path)
        if not os.path.exists(inputData["csv_path"]):
            raise ValueError(f"point cloud CSV not found: {inputData['csv_path']}")
    ######################################################
    return inputData


##########################################################
async def calculate(inJson={}):
    ######################################################
    setupMLFlow()
    ######################################################
    try:
        inputData = standardInputWrapper(inJson=inJson)
        output_wrapper = standardOutputWrapper()
        inputData = checkData(inputData=inputData)
        startTime = time.time()
        ##################################################
        with mlflow.start_run(run_name=f"pipeline_{inputData['name']}") as run:
            mlflow.set_tags({"dataset_mode": inputData["dataset_mode"],
                             "problemType": PROBLEM_TYPES[inputData["problemType"]]})
            if inputData["dataset_mode"] == "synthetic":
                _, csv_path = generateSyntheticPointCloud(inputData=inputData)
                mlflow.log_param("geometry", str(inputData.get("geometry", "terrain")))
            else:
                csv_path = inputData["csv_path"]
            ##############################################
            report = checkPointCloud(csvPath=csv_path, min_points=inputData["min_points"])
            mlflow.log_metrics({"data_check_passed": float(report["passed"]),
                                "n_points": report["summary"].get("n_points", 0),
                                "n_errors": len(report["errors"]), "n_warnings": len(report["warnings"])})
            if not report["passed"]:
                raise ValueError("Data check failed:\n  - " + "\n  - ".join(report["errors"]))
            df = pd.read_csv(csv_path)
            ##############################################
            if inputData["problemType"] == "PROB01":
                # DBSCAN anomaly detection plugs in here
                pass
            if inputData["problemType"] == "PROB02":
                # PointNet regression on t plugs in here
                pass
            if inputData["problemType"] == "PROB03":
                # PointNet++ segmentation (planned)
                pass
            if inputData["problemType"] == "PROB04":
                # MC Dropout / BNN uncertainty (planned)
                pass
            ##############################################
            output_wrapper.add_param(name="data_check_passed", value=report["passed"], value_type=1)
            output_wrapper.add_param(name="n_points", value=int(len(df)))
            output_wrapper.add_param(name="csv_path", value=csv_path, value_type=3)
            output_wrapper.add_param(name="mlflow_run_id", value=run.info.run_id, value_type=3)
            output_wrapper.add_table(name="bbox_min", array=report["summary"]["bbox_min"])
            output_wrapper.add_table(name="bbox_max", array=report["summary"]["bbox_max"])
            output_wrapper.add_table(name="warnings", array=report["warnings"])
        ##################################################
        print("The total numerical time is " + str(np.round(time.time() - startTime, 5)) + " seconds")
        print("Completed Calculation")
        return output_wrapper
    except Exception as e:
        print("\nError Message: " + str(e) + "\n")
        return str(e)


##########################################################
async def main():
    ######################################################
    fileDirName = join(dirname(abspath(__file__)), "..", "tests", "in_jsons", "testCase01.json")
    with open(fileDirName) as json_file:
        inJson = json.load(json_file)
    output_wrapper = await calculate(inJson=inJson)
    if not isinstance(output_wrapper, str):
        print(json.dumps(output_wrapper.flatten(), indent=2))


##########################################################
if __name__ == "__main__":
    asyncio.run(main())
##########################################################
