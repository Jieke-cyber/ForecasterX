# app/main.py
from fastapi import FastAPI, UploadFile, File, Depends, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from sqlalchemy.orm import Session
import uuid

from .db import Base, engine, get_db
from .models import Dataset, TrainingRun, ForecastPlot
from .schemas import TrainRequest, JobStatus, RecentPlot
from .services import save_csv            # <-- nuova save_csv(db, file) che scrive nel DB
from .jobs import train_job               # <-- train_job(db, run_id, dataset_id, horizon)

app = FastAPI(title="TS WebApp Backend (MVP)")

# CORS per sviluppo (React su 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# crea tabelle
Base.metadata.create_all(bind=engine)

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/datasets/upload")
def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    ds_id = save_csv(db, file)   # ← passa anche db qui
    return {"dataset_id": ds_id}


@app.post("/train")
def start_train(req: TrainRequest, bg: BackgroundTasks, db: Session = Depends(get_db)):
    ds = db.get(Dataset, req.dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset non trovato")
    run_id = str(uuid.uuid4())
    db.add(TrainingRun(id=run_id, dataset_id=req.dataset_id, status="PENDING"))
    db.commit()
    # ⬇️ Avvia il job passando il dataset_id (non più il path del file)
    bg.add_task(train_job, db, run_id, req.dataset_id, req.horizon)
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
        result = {"metrics": run.metrics_json, "plot": plot.plot_json if plot else None}
    return {"job_id": job_id, "status": run.status, "result": result}

@app.get("/plots/recent", response_model=list[RecentPlot])
def recent_plots(limit: int = 10, db: Session = Depends(get_db)):
    q = (
        db.query(ForecastPlot)
        .order_by(ForecastPlot.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {"id": p.id, "plot_json": p.plot_json, "created_at": p.created_at.isoformat()}
        for p in q
    ]
