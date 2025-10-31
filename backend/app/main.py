# app/main.py
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Depends, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from sqlalchemy.orm import Session
import uuid
import os
import io

from .db import Base, engine, get_db
from .models import Dataset, TrainingRun, ForecastPlot
from .schemas import TrainRequest, JobStatus
from .services import save_csv
from .jobs import train_job
from .supa import SUPABASE_URL, SUPABASE_BUCKET, supa  # per costruire URL se bucket è pubblico

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

@app.get("/datasets")
def list_datasets(db: Session = Depends(get_db)):
    """
    Restituisce l'elenco dei dataset caricati (metadati, non i dati).
    Serve al frontend per riempire una tabella/select.
    """
    rows = (
        db.query(Dataset)
        .order_by(Dataset.created_at.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "name": r.name,
            "path": r.path,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@app.get("/datasets/{dataset_id}/data")
def get_dataset_data(dataset_id: str, db: Session = Depends(get_db)):
    """
    Scarica il CSV dal bucket dei dataset, lo normalizza a ds,value
    e lo restituisce pronto per il grafico.
    """
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset non trovato")

    client = supa()
    try:
        # scarico il file dallo storage
        file_bytes = client.storage.from_(os.getenv("SUPABASE_BUCKET")).download(ds.path)
    except Exception as e:
        raise HTTPException(500, f"Errore download CSV da storage: {e}")

    # trasformo in DataFrame
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(400, f"CSV non leggibile: {e}")

    # normalizza a ds,value
    if {"ds", "value"}.issubset(df.columns):
        df = df[["ds", "value"]].copy()
    else:
        if df.shape[1] < 2:
            raise HTTPException(400, "CSV deve avere almeno due colonne (data, valore)")
        df = df.iloc[:, :2].copy()
        df.columns = ["ds", "value"]

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"]).sort_values("ds")

    # restituisco in forma comoda per il grafico
    return [
        {"ds": row.ds.isoformat(), "value": float(row.value)}
        for _, row in df.iterrows()
    ]


@app.get("/plots/recent")
def list_recent_plots(limit: int = 10, db: Session = Depends(get_db)):
    """
    Restituisce gli ultimi forecast salvati (non i dati, solo i metadati).
    """
    rows = (
        db.query(ForecastPlot)
        .order_by(ForecastPlot.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "training_run_id": r.training_run_id,
            "name": r.name,
            "path": r.path,
            "url": f"{os.getenv('SUPABASE_URL')}/storage/v1/object/public/{os.getenv("SUPABASE_BUCKET_PLOTS")}/{r.path}"
            if os.getenv("SUPABASE_URL")
            else None,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@app.get("/plots/{plot_id}/data")
def get_plot_data(plot_id: str, db: Session = Depends(get_db)):
    """
    Scarica il CSV del forecast (quello con ds,value,kind) e lo restituisce al frontend.
    """
    plot = db.get(ForecastPlot, plot_id)
    if not plot:
        raise HTTPException(404, "Plot non trovato")

    client = supa()
    try:
        file_bytes = client.storage.from_(os.getenv('SUPABASE_BUCKET_PLOTS')).download(plot.path)
    except Exception as e:
        raise HTTPException(500, f"Errore download forecast da storage: {e}")

    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(400, f"CSV forecast non leggibile: {e}")

    # ci aspettiamo ds,value,kind
    if "ds" not in df.columns or "value" not in df.columns:
        raise HTTPException(400, "CSV forecast non ha le colonne attese")

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"]).sort_values("ds")

    # se manca kind, lo aggiungo di default a "history"
    if "kind" not in df.columns:
        df["kind"] = "history"

    return [
        {
            "ds": row.ds.isoformat(),
            "value": float(row.value),
            "kind": row.kind,
        }
        for _, row in df.iterrows()
    ]

from fastapi import Query

@app.get("/plots")
def list_plots(
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """
    Restituisce la lista dei forecast salvati (tabella forecast_plots),
    ordinati dal più recente.
    """
    rows = (
        db.query(ForecastPlot)
        .order_by(ForecastPlot.created_at.desc())
        .limit(limit)
        .all()
    )

    bucket = os.getenv("SUPABASE_BUCKET_PLOTS", "plots")
    base_url = os.getenv("SUPABASE_URL")

    result = []
    for r in rows:
        # se il bucket è pubblico e hai SUPABASE_URL possiamo costruire la URL
        public_url = (
            f"{base_url}/storage/v1/object/public/{bucket}/{r.path}"
            if base_url
            else None
        )
        result.append({
            "id": r.id,
            "training_run_id": r.training_run_id,
            "name": r.name,
            "path": r.path,
            "url": public_url,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })
    return result
@app.get("/plots/recent")
def list_recent_plots(db: Session = Depends(get_db)):
    return list_plots(limit=10, db=db)
