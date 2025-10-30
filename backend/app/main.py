# app/main.py
from fastapi import FastAPI, UploadFile, File, Depends, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from sqlalchemy.orm import Session
import uuid
import os

from .db import Base, engine, get_db
from .models import Dataset, TrainingRun, ForecastPlot
from .schemas import TrainRequest, JobStatus
from .services import save_csv
from .jobs import train_job
from .supa import SUPABASE_URL  # per costruire URL se bucket è pubblico

app = FastAPI(title="TS WebApp Backend (MVP)")

# CORS per sviluppo (React su 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/datasets/upload")
def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    ds_id = save_csv(db, file)
    return {"dataset_id": ds_id}


@app.post("/train")
def start_train(req: TrainRequest, bg: BackgroundTasks, db: Session = Depends(get_db)):
    ds = db.get(Dataset, req.dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset non trovato")

    run_id = str(uuid.uuid4())
    db.add(TrainingRun(id=run_id, dataset_id=req.dataset_id, status="PENDING"))
    db.commit()

    # ✅ il job creerà la sessione DB da sé (non passiamo db)
    bg.add_task(train_job, run_id, req.dataset_id, req.horizon)

    return {"job_id": run_id}


@app.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str, db: Session = Depends(get_db)):
    run = db.get(TrainingRun, job_id)
    if not run:
        raise HTTPException(404, "Job non trovato")

    result = None
    if run.status == "SUCCESS":
        plot = (
            db.query(ForecastPlot)
            .filter(ForecastPlot.training_run_id == job_id)
            .order_by(ForecastPlot.created_at.desc())
            .first()
        )
        if plot:
            bucket = os.getenv("SUPABASE_BUCKET_PLOTS", "plots")
            url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{plot.path}"
            result = {
                "metrics": run.metrics_json,
                "forecast_csv_url": url,
                "file_name": plot.name,
            }

    return {"job_id": job_id, "status": run.status, "result": result}


@app.get("/plots/recent")
def recent_plots(limit: int = 10, db: Session = Depends(get_db)):
    rows = (
        db.query(ForecastPlot)
        .order_by(ForecastPlot.created_at.desc())
        .limit(limit)
        .all()
    )
    bucket = os.getenv("SUPABASE_BUCKET_PLOTS", "plots")
    return [
        {
            "id": r.id,
            "training_run_id": r.training_run_id,
            "name": r.name,
            "path": r.path,
            "url": f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{r.path}",
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
