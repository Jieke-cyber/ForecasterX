import logging
from pathlib import Path
import sys
from dotenv import load_dotenv
from jose import jwt, ExpiredSignatureError, JWTError, JOSEError

from .utils import upload_to_bucket
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, status, Response, Security, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text, or_
import uuid
from .tasks import run_autots_training_task, run_lagllama_finetuning_task
import os, re
import io
from .db import Base, engine, get_db, SessionLocal
from .models import Dataset, TrainingRun, ForecastPlot, Model, User
from .schemas import JobStatus, ZeroShotPredictIn, FinetuneIn, PredictFTSaveIn, TrainRequest, CleanOutliersBody, \
    ImputeBody, RegisterBody, LoginBody
from .services import save_csv, delete_csv, delete_single_plot, delete_training_run, _load_dataset_as_df, \
    _run_lagllama_ft_forecast_and_save, _create_new_dataset_row
from .supa import SUPABASE_URL, SUPABASE_BUCKET, supa
from .models import User
from .auth_utils import hash_password, verify_password, create_access_token, SECRET_KEY, ALGORITHM
from FoundationModel.lagllama import predict_series
from .models_runtime.pypots_runtime import (
    preload_pypots_models,
    PYPOTS_MODELS, get_pypots_model, predict_future,
)
load_dotenv()
@asynccontextmanager
async def lifespan(app: FastAPI):
    db = SessionLocal()
    try:
        fm = (
            db.query(Model)
            .filter(Model.name == "Lag-Llama", Model.kind == "foundation")
            .first()
        )
        if not fm:
            fm = Model(
                name="Lag-Llama",
                kind="foundation",
                base_model="lag-llama",
                storage_path=None,
                params_json={},
                metrics_json={},
                owner_email=None,
                status="AVAILABLE",
            )
            db.add(fm)
            db.commit()
        preload_pypots_models()

        app.state.pypots_models = PYPOTS_MODELS

        for key, bundle in PYPOTS_MODELS.items():
            artifact = bundle["artifact"]
            pattern = artifact.get("pattern", key)
            model_type = artifact.get("model_type", "unknown")
            path = bundle["path"]
            L = bundle["L"]
            H = bundle["H"]
            metrics = artifact.get("metrics", {})
            init_kwargs = artifact.get("init_kwargs", {})

            name = key
            kind = "pypots"
            base_model = model_type

            params_json = {
                "pattern": pattern,
                "model_type": model_type,
                "L": L,
                "H": H,
                "init_kwargs": init_kwargs,
            }

            metrics_json = metrics or {}

            db_model = (
                db.query(Model)
                .filter(Model.name == name, Model.kind == kind)
                .first()
            )

            if not db_model:
                db_model = Model(
                    name=name,
                    kind=kind,
                    base_model=base_model,
                    storage_path=path,
                    params_json=params_json,
                    metrics_json=metrics_json,
                    owner_email=None,
                    status="AVAILABLE",
                )
                db.add(db_model)
                print(f"[Startup] Creato record models per '{name}'")
            else:
                db_model.base_model = base_model
                db_model.storage_path = path
                db_model.params_json = params_json
                db_model.metrics_json = metrics_json
                db_model.status = "AVAILABLE"
                print(f"[Startup] Aggiornato record models per '{name}'")

        db.commit()

    finally:
        db.close()

    yield

    PYPOTS_MODELS.clear()
    if hasattr(app.state, "pypots_models"):
        app.state.pypots_models.clear()

app = FastAPI(title="TS WebApp Backend (MVP)", lifespan=lifespan)

security = HTTPBearer()

origins = [
    "http://localhost:5173",
    "http://localhost:8080"
]

client_origin_url = os.getenv("CLIENT_ORIGIN")

if client_origin_url:
    origins.append(client_origin_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: Session = Depends(get_db),
):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (JWTError, JOSEError, ValueError):
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(401, "User not found", headers={"WWW-Authenticate": "Bearer"})
    return user


@app.post("/datasets/upload")
def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db), current_user = Depends(
    get_current_user), ):
    owner_email = str(current_user.email).strip().lower()
    existing = db.query(Dataset).filter(Dataset.name == file.filename).first()
    if existing:
        raise HTTPException(status_code=400, detail="File name already exists, please rename your file before uploading.")
    ds_id = save_csv(db, file, owner_email=owner_email)
    return {"dataset_id": ds_id}


@app.post("/train")
def start_train(
        req: TrainRequest,
        db: Session = Depends(get_db)
):
    if not req.dataset_ids:
        raise HTTPException(400, "Richiesto almeno un dataset ID")

    main_ds_id = req.main_dataset_id if req.main_dataset_id else req.dataset_ids[0]

    for ds_id in req.dataset_ids:
        if not db.get(Dataset, ds_id):
            raise HTTPException(404, f"Dataset {ds_id} non trovato")

    run_id = str(uuid.uuid4())
    run = TrainingRun(
        id=run_id,
        dataset_id=main_ds_id,
        status="PENDING"
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    task = run_autots_training_task.delay(
        run_id=run.id,
        dataset_ids=req.dataset_ids,
        horizon=req.horizon
    )

    run.celery_task_id = task.id
    db.commit()

    return {"job_id": run.id}

@app.get("/jobs/{job_id}/status", response_model=JobStatus)
def job_status(job_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    run = db.get(TrainingRun, job_id)
    if not run:
        raise HTTPException(404, "Job non trovato")
    ds = db.get(Dataset, run.dataset_id)
    owner_email = str(current_user.email).strip().lower()
    if ds.owner_email != owner_email:
        raise HTTPException(403, "Forbidden")

    result = None
    if run.status == "SUCCESS":
        plot = (
            db.query(ForecastPlot)
            .filter((ForecastPlot.training_run_id == job_id),
                    ForecastPlot.owner_email == owner_email)
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



@app.get("/datasets")
def list_datasets(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """
    Restituisce SOLO i dataset dell'utente loggato.
    """
    owner_email = str(current_user.email).strip().lower()
    rows = (
        db.query(Dataset)
        .filter(Dataset.owner_email == owner_email)
        .order_by(Dataset.created_at.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "name": r.name,
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
        file_bytes = client.storage.from_(os.getenv("SUPABASE_BUCKET")).download(ds.path)
    except Exception as e:
        raise HTTPException(500, f"Errore download CSV da storage: {e}")

    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(400, f"CSV non leggibile: {e}")

    if {"ds", "value"}.issubset(df.columns):
        df = df[["ds", "value"]].copy()
    else:
        if df.shape[1] < 2:
            raise HTTPException(400, "CSV deve avere almeno due colonne (data, valore)")
        df = df.iloc[:, :2].copy()
        df.columns = ["ds", "value"]

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"]).sort_values("ds")

    return [
        {"ds": row.ds.isoformat(), "value": float(row.value)}
        for _, row in df.iterrows()
    ]




@app.get("/plots/{plot_id}/data")
def get_plot_data(plot_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    """
    Scarica il CSV del forecast (quello con ds,value,kind) e lo restituisce al frontend.
    """
    plot = db.get(ForecastPlot, plot_id)
    if not plot:
        raise HTTPException(404, "Plot non trovato")

    owner_email = str(current_user.email).strip().lower()
    if plot.owner_email != owner_email:
        raise HTTPException(403, "Forbidden")

    client = supa()
    try:
        file_bytes = client.storage.from_(os.getenv('SUPABASE_BUCKET_PLOTS')).download(plot.path)
    except Exception as e:
        raise HTTPException(500, f"Errore download forecast da storage: {e}")

    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(400, f"CSV forecast non leggibile: {e}")

    if "ds" not in df.columns or "value" not in df.columns:
        raise HTTPException(400, "CSV forecast non ha le colonne attese")

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"]).sort_values("ds")

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
    limit: int = Query(15, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Restituisce la lista dei forecast salvati (tabella forecast_plots),
    ordinati dal più recente.
    """
    owner_email = str(current_user.email).strip().lower()
    rows = (
        db.query(ForecastPlot)
        .filter(ForecastPlot.owner_email == owner_email)
        .order_by(ForecastPlot.created_at.desc())
        .limit(limit)
        .all()
    )

    bucket = os.getenv("SUPABASE_BUCKET_PLOTS", "plots")
    base_url = os.getenv("SUPABASE_URL")

    result = []
    for r in rows:
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

from cleanlab.outlier import OutOfDistribution

@app.post("/datasets/{dataset_id}/clean-outliers")
def clean_outliers_on_dataset(
    dataset_id: str,
    body: CleanOutliersBody,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    email = str(current_user.email).strip().lower()
    df, dataset = _load_dataset_as_df(db, dataset_id)

    n = len(df)
    y = df["value"].astype(float).to_numpy()
    outlier_mask_all = np.zeros(n, dtype=bool)

    for start in range(0, n, body.chunk):
        end = min(start + body.chunk, n)
        seg = y[start:end].reshape(-1, 1)
        if (end - start) < 3:
            continue

        ood = OutOfDistribution()
        scores_seg = ood.fit_score(features=seg)

        mask_seg = scores_seg < body.threshold
        outlier_mask_all[start:end] = mask_seg

    df.loc[outlier_mask_all, "value"] = np.nan

    cleaned_key = f"{dataset.id}_cleaned.csv"
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    upload_to_bucket(cleaned_key, buf.getvalue().encode("utf-8"), bucket=SUPABASE_BUCKET)

    cleaned_name = body.new_name or f"{dataset.name} (cleaned)"
    new_ds = _create_new_dataset_row(db, cleaned_name, cleaned_key, owner_email=email)

    return {
        "original_dataset_id": str(dataset.id),
        "cleaned_dataset_id": str(new_ds.id),
        "outliers_found": int(outlier_mask_all.sum()),
    }

from pypots.imputation.lerp.model import Lerp

@app.post("/datasets/{dataset_id}/impute-linear")
def impute_dataset_with_lerp(
    dataset_id: str,
    body: ImputeBody | None = None,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    email = str(current_user.email).strip().lower()
    df, dataset = _load_dataset_as_df(db, dataset_id)

    values = df["value"].to_numpy(dtype=float)
    X = values.reshape(1, -1, 1)
    data_dict = {"X": X}

    imputer = Lerp()
    result = imputer.predict(data_dict)
    imputed = result["imputation"].reshape(-1)

    df["value"] = imputed

    imputed_key = f"{dataset.id}_imputed.csv"
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    upload_to_bucket(imputed_key, buf.getvalue().encode("utf-8"), bucket=SUPABASE_BUCKET)

    imputed_name = body.new_name if body and body.new_name else f"{dataset.name} (imputed)"
    new_ds = _create_new_dataset_row(
        db, imputed_name, imputed_key, owner_email=email
    )

    return {
        "original_dataset_id": str(dataset.id),
        "imputed_dataset_id": str(new_ds.id),
    }

@app.post("/auth/register")
def register(body: RegisterBody, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == body.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        id=str(uuid.uuid4()),
        email=body.email,
        password_hash=hash_password(body.password),
    )
    db.add(user)
    db.commit()
    return {"message": "registered"}

@app.post("/auth/login")
def login(body: LoginBody, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user.id, "email": user.email})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/datasets/{dataset_id}/deleter", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    owner_email = str(current_user.email).strip().lower()
    delete_csv(db, dataset_id, owner_email)
    return

def _split_bucket_path(path: str, default_bucket: str | None = None):
    """
    Se path è 'bucket/object.csv' → separa.
    Se path è solo 'folder/object.csv' → usa default_bucket se fornito.
    """
    m = re.match(r"([^/]+)/(.+)$", path)
    if m:
        return m.group(1), m.group(2)
    if default_bucket:
        return default_bucket, path
    raise HTTPException(500, detail="Path non valido (atteso 'bucket/object' o setta default_bucket)")


@app.get("/public/plots/forecast/{plot_id}/csv")
def public_forecast_csv(plot_id: str, db=Depends(get_db)):
    row = db.execute(
        text("""
            SELECT id, name, path, owner_email
            FROM public.forecast_plots
            WHERE id = :id
        """),
        {"id": plot_id}
    ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Plot non trovato")

    sb=supa()

    file_path = row.path

    try:
        csv_bytes = sb.storage.from_(SUPABASE_BUCKET).download(file_path)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"CSV non trovato nello storage: {e}")

    if not csv_bytes:
        raise HTTPException(status_code=404, detail="CSV vuoto o non trovato")

    return Response(content=csv_bytes, media_type="text/csv; charset=utf-8")

@app.get("/public/datasets/{dataset_id}/csv")
def public_dataset_csv(dataset_id: str, db=Depends(get_db)):
    row = db.execute(
        text("""
            SELECT id, name, path, owner_email
            FROM public.datasets
            WHERE id = :id
        """),
        {"id": dataset_id}
    ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Dataset non trovato")

    sb = supa()

    file_path = row.path

    try:
        csv_bytes = sb.storage.from_(SUPABASE_BUCKET).download(file_path)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"CSV non trovato nello storage: {e}")

    if not csv_bytes:
        raise HTTPException(status_code=404, detail="CSV vuoto o non trovato")

    return Response(content=csv_bytes, media_type="text/csv; charset=utf-8")
@app.get("/train/{run_id}")
def get_training_status(run_id: str, db: Session = Depends(get_db)):
    tr = db.get(TrainingRun, run_id)
    if not tr:
        raise HTTPException(404, "TrainingRun non trovato")

    row = db.execute(text("""
        SELECT id AS plot_id
        FROM public.forecast_plots
        WHERE training_run_id = :rid
        LIMIT 1
    """), {"rid": run_id}).mappings().first()

    return {
        "run_id": run_id,
        "status": tr.status,
        "error": tr.error,
        "plot_id": row["plot_id"] if row else None,
        "metrics": tr.metrics_json,
    }

@app.get("/jobs")
def list_jobs(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    owner_email = str(current_user.email).strip().lower()
    rows = (
        db.query(TrainingRun)
        .join(Dataset, TrainingRun.dataset_id == Dataset.id)
        .filter(Dataset.owner_email == owner_email)
        .order_by(TrainingRun.created_at.desc())
        .all()
    )

    def metric_from(m: dict | None) -> str:
        m = m or {}
        return m.get("model") or m.get("best_model") or "-"

    return [
        {
            "id": r.id,
            "dataset_name": r.dataset.name if r.dataset else None,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "status": r.status,
            "metric": metric_from(r.metrics_json),
        }
        for r in rows
    ]

@app.post("/plots/{plot_id}/delete", status_code=status.HTTP_204_NO_CONTENT)
def api_delete_plot(plot_id: str, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    delete_single_plot(db, plot_id, current_user.email)
    return

@app.post("/train/{run_id}/delete", status_code=204)
def api_delete_training_run(run_id: str, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    delete_training_run(db, run_id, current_user.email)
    return


def _load_df(db: Session, dataset_id: str):
    ds = db.get(Dataset, dataset_id)
    if not ds: raise HTTPException(404, "Dataset non trovato")
    df = pd.read_csv(ds.path)
    df["ds"] = pd.to_datetime(df["ds"])
    return df[["ds","value"]], ds

def _infer_freq(ds_col: pd.Series) -> str:
    try: return pd.infer_freq(ds_col.sort_values()) or "D"
    except Exception: return "D"


@app.post("/lag-llama/predict/save")
def lag_llama_predict_and_save(payload: ZeroShotPredictIn,
                               db: Session = Depends(get_db)):

    try:

        df, ds_row = _load_dataset_as_df(db, str(payload.dataset_id))
        owner_email = str(ds_row.owner_email).strip().lower()

        s = pd.Series(df["value"].values, index=df["ds"])
        yhat = predict_series(s, int(payload.horizon), int(payload.context_len))
        future_index = pd.date_range(df["ds"].max() + pd.tseries.frequencies.to_offset(_infer_freq(df["ds"])),
                                     periods=int(payload.horizon))
        fut = pd.DataFrame({"ds": future_index, "value": yhat[:int(payload.horizon)], "kind": "forecast"})
        hist = df.assign(kind="history")
        combined = pd.concat([hist, fut], ignore_index=True)

        csv_bytes = combined.to_csv(index=False).encode("utf-8")
        client = supa()
        plot_id = str(uuid.uuid4())
        safe_owner = owner_email.replace("@", "_at_")
        object_key = f"{safe_owner}/{str(payload.dataset_id)}/{plot_id}.csv"
        client.storage.from_(SUPABASE_BUCKET).upload(
            path=object_key, file=csv_bytes,
            file_options={"content-type": "text/csv", "upsert": "true"}
        )

        db.add(ForecastPlot(id=plot_id, training_run_id=None,
                            name=f"{ds_row.name} (Lag-Llama forecast)",
                            path=object_key, owner_email=owner_email))
        db.commit()

        return {"plot_id": plot_id, "rows": len(combined)}

    except Exception as e:
        logging.error(f"Errore in lag_llama_predict_and_save: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/lag-llama/{model_id}/finetune")
def lag_llama_finetune(
    model_id: str,
    payload: FinetuneIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    owner_email = str(current_user.email).strip().lower()

    if not payload.dataset_ids:
        raise HTTPException(400, "Richiesta almeno un ID per il fine-tuning.")

    for ds_id in payload.dataset_ids:
        if not db.get(Dataset, ds_id):
            raise HTTPException(400, f"Dataset {ds_id} non trovato.")

    main_ds_id = payload.dataset_ids[0]

    run_id = str(uuid.uuid4())
    run = TrainingRun(
        id=run_id,
        dataset_id=main_ds_id,
        status="PENDING",
        model_id_used=model_id,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    task = run_lagllama_finetuning_task.delay(
        run_id=run.id,
        dataset_ids=payload.dataset_ids,
        owner_email=owner_email,
        base_model_id=model_id,
        payload_dict=payload.dict()
    )

    run.celery_task_id = task.id
    db.commit()

    return {"job_id": run.id, "status": run.status}



@app.post("/lag-llama-ft/{model_id}/predict/save")
def lag_llama_ft_predict_and_save(
        model_id: str,
        payload: PredictFTSaveIn,
        background: BackgroundTasks,
        db: Session = Depends(get_db),
):
    m = db.query(Model).filter(Model.id == model_id).first()
    if not m:
        raise HTTPException(404, "Modello non trovato")
    if (m.kind or "").lower() != "fine_tuned" or (m.base_model or "").lower() != "lag-llama":
        raise HTTPException(400, "Il modello non è un Lag-Llama fine-tuned")

    background.add_task(
        _run_lagllama_ft_forecast_and_save,
        SessionLocal,

        model_id,
        str(payload.dataset_id),
        int(payload.horizon),
        int(payload.context_len),
    )

    return {"message": "Previsione avviata e salvataggio in corso."}
@app.get("/models")
def list_models(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """
    Restituisce SOLO i dataset dell'utente loggato.
    """
    owner_email = str(current_user.email).strip().lower()
    rows = (
        db.query(Model)
        .filter(
            or_(
                Model.owner_email == owner_email,
                Model.owner_email.is_(None),
            )
        )
        .order_by(Model.created_at.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "name": r.name,
            "kind": r.kind,
            "base_model": r.base_model,
            "storage_path": r.storage_path,
            "params_json": r.params_json,
            "metrics_json": r.metrics_json,
            "owner_email": r.owner_email,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]

class PyPotsForecastBody(BaseModel):
    dataset_id: str
    horizon: int

@app.post("/pypots/{model_id}/forecast-csv")
def pypots_forecast_csv(
    model_id : str,
    payload: PyPotsForecastBody,
    db: Session = Depends(get_db),

):
    """
    Usa un modello PyPOTS scelto tramite id (tabella models.id)
    per fare forecast su un CSV (Date,Value), salva il CSV su Supabase
    e crea un record in forecast_plots. Ritorna anche i dati in JSON.
    """
    db_model = (
        db.query(Model)
        .filter(Model.id == model_id, Model.kind == "pypots")
        .first()
    )

    if not db_model:
        raise HTTPException(
            status_code=404,
            detail=f"Modello con id={model_id} non trovato o non è di tipo 'pypots'",
        )

    model_key = str(db_model.name).strip()

    try:
        bundle = get_pypots_model(model_key)
    except KeyError:
        raise HTTPException(
            status_code=500,
            detail=f"Modello runtime '{model_key}' non presente in cache (problema di startup?)",
        )

    model = bundle["model"]
    scaler = bundle["scaler"]
    L = bundle["L"]
    H = min(bundle["H"], payload.horizon)

    df, ds_row = _load_dataset_as_df(db, payload.dataset_id)
    if df is None or df.empty:
        raise RuntimeError("Dataset vuoto o non trovato")

    df = df.sort_values("ds")
    owner_email = (str(ds_row.owner_email) if ds_row and ds_row.owner_email else "").strip()
    df = (
        df.dropna(subset=["ds", "value"])
          .drop_duplicates(subset=["ds"], keep="last")
          .sort_values("ds")
          .reset_index(drop=True)
    )
    if len(df) < L:
        raise HTTPException(
            status_code=400,
            detail=f"Serie troppo corta: servono almeno L={L} punti, ne hai {len(df)}",
        )

    y = df["value"].astype("float32").values.reshape(-1, 1)
    ys = scaler.transform(y)
    x_last = ys[-L:, :][None, ...]

    pred_scaled = predict_future(model, x_last, H)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)

    history_df = df[["ds", "value"]].copy()
    history_df["kind"] = "history"

    last_date = df["ds"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=H,
        freq="D",
    )
    future_df = pd.DataFrame({
        "ds": future_dates,
        "value": pred,
    })
    future_df["kind"] = "future"

    combined_df = pd.concat([history_df, future_df], ignore_index=True)

    csv_bytes = combined_df.to_csv(index=False).encode("utf-8")
    client = supa()

    plot_id = str(uuid.uuid4())
    object_key = f"pypots/{model_id}/{plot_id}.csv"

    client.storage.from_(SUPABASE_BUCKET).upload(
        path=object_key,
        file=csv_bytes,
        file_options={"content-type": "text/csv", "upsert": "true"},
    )

    db.add(ForecastPlot(
        id=plot_id,
        training_run_id=None,
        name=f"{ds_row.name} ({db_model.base_model})",
        path=object_key,
        owner_email= owner_email ,
    ))
    db.commit()

    return {
        "model_id": model_id,
        "model_key": model_key,
        "L": L,
        "H": H,
        "plot_id": plot_id,
        "path": object_key,
        "rows": combined_df.to_dict(orient="records"),
    }
