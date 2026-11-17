############################################################################################
"""
REST API for the Point Cloud ML calculator.

Same contract as every calculator in my workspace, so the UI can talk to all of them the same way:
    GET  /health                 is the API up
    GET  /ready                  can it run a calculation (calculator imports cleanly)
    GET  /schema                 input form definition (fields, types, defaults, groups)
    POST /calculate              submit a run, returns a job_id right away
    GET  /runs/{job_id}/status   pending / queued / running / complete / failed
    GET  /jobs/{job_id}/output   output wrapper (params and tables) once complete
    GET  /jobs                   all jobs this API has seen, newest first

Runs execute one at a time per calculator in a background thread so the API stays responsive.
"""
############################################################################################
import sys
import uuid
import asyncio
import threading
from datetime import datetime
from os.path import dirname, abspath, join
sys.path.insert(0, join(dirname(abspath(__file__)), 'pointCloudMLCalculator'))
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from Calculator import calculate
from dataPrep.schema import getSchema

CALCULATOR_NAME = 'pointcloud'

############################################################################################
app = FastAPI(
    title       = "Point Cloud ML API",
    description = "REST API for the Point Cloud ML calculator: schema, async job submission, status and output.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

############################################################################################
# In-memory job store, keyed by job_id. Runs are serialized with a lock (one at a time).
job_store = {}
run_lock  = threading.Lock()

############################################################################################
class CalculateRequest(BaseModel):
    inJson: dict

############################################################################################
class JobStatusResponse(BaseModel):
    job_id:        str
    calculator:    str
    status:        str
    submitted:     str
    started:       Optional[str] = None
    completed:     Optional[str] = None
    error:         Optional[str] = None
    mlflow_run_id: Optional[str] = None

############################################################################################
def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

############################################################################################
def withJobOutputDir(inJson={}, job_id=''):
    """Give each API run its own output folder unless the caller set one, so runs don't overwrite each other."""
    ########################################################################################
    data = dict(inJson.get('data', {}))
    params = list(data.get('params', []))
    if not any(p.get('name') == 'output_dir' for p in params):
        params.append({'name': 'output_dir', 'value': f"Data/Synthetic/api_{job_id[:8]}", 'valueType': 0})
    data['params'] = params
    data.setdefault('tables', [])
    return {**inJson, 'data': data}

############################################################################################
def runJob(job_id='', inJson={}):
    """Background task: waits its turn, runs calculate(), stores status and output."""
    ########################################################################################
    job = job_store[job_id]
    job['status'] = 'queued'
    with run_lock:
        job['status'], job['started'] = 'running', now()
        try:
            output_wrapper = asyncio.run(calculate(inJson=withJobOutputDir(inJson, job_id)))
            if isinstance(output_wrapper, str):
                job['status'], job['error'] = 'failed', output_wrapper
            else:
                job['output'] = output_wrapper.flatten()
                job['mlflow_run_id'] = job['output'].get('mlflow_run_id')
                job['status'] = 'complete'
        except Exception as e:
            job['status'], job['error'] = 'failed', str(e)
        job['completed'] = now()

############################################################################################
@app.get("/health")
def health():
    """Health check, confirms the API is running."""
    return {"status": "ok", "calculator": CALCULATOR_NAME, "timestamp": now()}

############################################################################################
@app.get("/ready")
def ready():
    """Readiness check, confirms the calculator and schema load."""
    schema = getSchema()
    return {"ready": True, "calculator": CALCULATOR_NAME, "n_fields": sum(len(g['fields']) for g in schema['groups'])}

############################################################################################
@app.get("/schema")
def schema():
    """Input form definition the UI uses to build the calculator page."""
    return getSchema()

############################################################################################
@app.post("/calculate", response_model=JobStatusResponse)
def submit_job(request: CalculateRequest, background_tasks: BackgroundTasks):
    """Submit a run, returns a job_id immediately. Poll /runs/{job_id}/status for progress."""
    ########################################################################################
    if 'data' not in request.inJson:
        raise HTTPException(status_code=422, detail="inJson needs a 'data' object with 'params' and 'tables'")
    job_id = str(uuid.uuid4())
    job_store[job_id] = {'job_id': job_id, 'calculator': CALCULATOR_NAME, 'status': 'pending', 'submitted': now(),
                         'started': None, 'completed': None, 'error': None, 'mlflow_run_id': None, 'output': None}
    background_tasks.add_task(runJob, job_id=job_id, inJson=request.inJson)
    return job_store[job_id]

############################################################################################
@app.get("/runs/{job_id}/status", response_model=JobStatusResponse)
def get_run_status(job_id: str):
    """Status of a submitted job."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job_store[job_id]

############################################################################################
@app.get("/jobs/{job_id}/output")
def get_job_output(job_id: str):
    """Output wrapper (params and tables flattened) for a finished job."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job = job_store[job_id]
    return {"job_id": job_id, "calculator": CALCULATOR_NAME, "status": job['status'], "error": job['error'], "output": job['output']}

############################################################################################
@app.get("/jobs", response_model=list[JobStatusResponse])
def list_jobs(limit: int = 50):
    """All jobs this API has seen, newest first."""
    return sorted(job_store.values(), key=lambda j: j['submitted'], reverse=True)[:limit]

############################################################################################
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8003, reload=True)
############################################################################################
