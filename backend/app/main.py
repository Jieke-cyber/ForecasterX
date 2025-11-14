# app/main.py
# --- add at top of app/main.py before other imports ---
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]  # .../backend
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------

import tempfile

from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Security, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt, ExpiredSignatureError, JOSEError
from starlette.responses import RedirectResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from sqlalchemy import text, or_
import uuid
import os, re
import io

from .db import Base, engine, get_db, SessionLocal
from .models import Dataset, TrainingRun, ForecastPlot, Model
from .schemas import JobStatus, ZeroShotPredictIn, FinetuneIn, PredictFTSaveIn
from .services import save_csv, delete_csv, delete_single_plot, delete_training_run
from .jobs import train_job, _run_lagllama_forecast, _run_lagllama_ft_forecast
from .supa import SUPABASE_URL, SUPABASE_BUCKET, supa  # per costruire URL se bucket è pubblico


from .models import User
from .auth_utils import hash_password, verify_password, create_access_token

from .auth_utils import SECRET_KEY, ALGORITHM
from FoundationModel.lagllama import predict_series, predict_series_with_predictor, \
    load_predictor_from_ckpt, finetune_and_dump_ckpt


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- STARTUP ----
    db: Session = SessionLocal()
    try:
        fm = db.query(Model).filter(Model.name=="Lag-Llama", Model.kind=="foundation").first()
        if not fm:
            db.add(Model(
                name="Lag-Llama",
                kind="foundation",
                base_model="lag-llama",
                storage_path=None,
                params_json={},
                metrics_json={},
                owner_email=None,
                status="AVAILABLE",
            ))
            db.commit()
    finally:
        db.close()

    yield
app = FastAPI(title="TS WebApp Backend (MVP)", lifespan=lifespan)

security = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: Session = Depends(get_db),
):
    token = credentials.credentials
    try:
        # leeway (facoltativo) per tollerare piccoli sfasamenti di orologio
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

def upload_to_bucket(key: str, data: bytes, bucket: str):
    client = supa()
    client.storage.from_(bucket).upload(
        path=key,
        file=data,
        file_options={
            "content-type": "text/csv",
            "upsert": "true",   # 👈 deve essere STRINGA, non bool
        },
    )

def download_from_bucket(key: str, bucket: str) -> bytes:
    client = supa()
    # supabase-py qui ti restituisce direttamente i bytes
    return client.storage.from_(bucket).download(key)

# CORS per sviluppo (React su 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",  "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

class TrainRequest(BaseModel):
    dataset_id: str
    horizon: int | None = None
    model_name: str | None = None

# 👇 NEW: schema per l’outliegetr detector
class CleanOutliersBody(BaseModel):
    chunk: int = 100
    threshold: float = 0.000001
    new_name: str | None = None

# 👇 NEW: schema per l’imputazione
class ImputeBody(BaseModel):
    new_name: str | None = None

class RegisterBody(BaseModel):
    email: EmailStr
    password: str

class LoginBody(BaseModel):
    email: EmailStr
    password: str

# ============================================================
# 👇 NEW: helper riutilizzabili
# ============================================================
def _load_dataset_as_df(db: Session, dataset_id: str) -> tuple[pd.DataFrame, Dataset]:
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id)
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    csv_bytes = download_from_bucket(dataset.path, bucket=SUPABASE_BUCKET)
    df = pd.read_csv(io.BytesIO(csv_bytes))

    # normalizza colonne → ds,value
    rename_map = {}
    for cand in ["ds", "date", "Date", "timestamp", "Timestamp"]:
        if cand in df.columns:
            rename_map[cand] = "ds"
            break
    for cand in ["value", "Value", "y"]:
        if cand in df.columns:
            rename_map[cand] = "value"
            break

    df = df.rename(columns=rename_map)

    if "ds" not in df.columns or "value" not in df.columns:
        raise HTTPException(400, "CSV must contain time and value columns")

    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    return df, dataset


def _create_new_dataset_row(db: Session, name: str, path: str, owner_email: str) -> Dataset:
    new_id = str(uuid.uuid4()) # 👈 lo facciamo stringa noi
    new_ds = Dataset(
        id=new_id,
        name=name,
        path=path,
        owner_email=owner_email,
    )
    db.add(new_ds)
    db.commit()
    # 👇 NON facciamo db.refresh(new_ds) perché SQLAlchemy lo rileggerebbe come UUID
    return new_ds

def _run_lagllama_ft_forecast_and_save(
    db_maker,
    run_id: str,
    model_id: str,
    dataset_id: str,
    horizon: int,
    context_len: int,
):
    db: Session = db_maker()
    try:
        # mark RUNNING
        run = db.get(TrainingRun, run_id)
        if not run:
            # run mancante: niente da fare
            return
        run.status = "RUNNING"
        db.commit()

        # --- carica dataset (come nel tuo zero-shot) ---
        df, ds_row = _load_dataset_as_df(db, dataset_id)
        if df is None or df.empty:
            raise RuntimeError("Dataset vuoto o non trovato")

        df = df.sort_values("ds")
        owner_email = (str(ds_row.owner_email) if ds_row and ds_row.owner_email else "").strip().lower()
        values = df["value"].to_numpy(dtype=float)
        start_ts = df["ds"].iloc[0]

        # inferisci frequenza (riusa il tuo helper se presente)
        try:
            # se hai _infer_freq(df["ds"]), puoi usare quello
            freq = pd.infer_freq(pd.to_datetime(df["ds"]).sort_values())
        except Exception:
            freq = None
        freq = freq or "D"

        # --- scarica ckpt FT dallo storage ---
        m = db.query(Model).filter(Model.id == model_id).first()
        if not m or not m.storage_path:
            raise RuntimeError("Modello FT senza storage_path")

        blob: bytes = supa().storage.from_(SUPABASE_BUCKET).download(m.storage_path)
        tmp_ckpt = tempfile.NamedTemporaryFile(delete=False, suffix=".ckpt")
        tmp_ckpt.write(blob); tmp_ckpt.flush(); tmp_ckpt.close()
        ckpt_local_path = tmp_ckpt.name

        # --- costruisci predictor FT dal ckpt ---
        predictor = load_predictor_from_ckpt(
            weights_ckpt_path=ckpt_local_path,
            horizon=horizon,
            context_len=context_len,
            freq=freq,
        )

        # --- predizione ---
        yhat = predict_series_with_predictor(
            predictor,
            series=values,
            horizon=horizon,
            freq=freq,
            start=start_ts,
        )

        # --- costruisci future index & CSV combinato (history + forecast) ---
        # offset dal freq (stesso pattern del tuo zero-shot)
        try:
            offset = pd.tseries.frequencies.to_offset(freq)
        except Exception:
            offset = pd.tseries.frequencies.to_offset("D")

        last_ts = df["ds"].max()
        future_index = pd.date_range(last_ts + offset, periods=horizon, freq=freq)

        fut = pd.DataFrame({"ds": future_index, "value": yhat[:horizon], "kind": "forecast"})
        hist = df.assign(kind="history")
        combined = pd.concat([hist, fut], ignore_index=True)

        # --- upload CSV su Supabase Storage ---
        csv_bytes = combined.to_csv(index=False).encode("utf-8")
        client = supa()
        plot_id = str(uuid.uuid4())
        safe_owner = (owner_email or "public").replace("@", "_at_")
        object_key = f"{safe_owner}/{dataset_id}/{plot_id}.csv"  # es. plots/<user>/<dataset>/<plot>.csv, se il bucket è quello dei plots

        client.storage.from_(SUPABASE_BUCKET).upload(
            path=object_key,
            file=csv_bytes,
            file_options={"content-type": "text/csv", "upsert": "true"},
        )

        # --- registra ForecastPlot e finalizza run ---
        db.add(ForecastPlot(
            id=plot_id,
            training_run_id=run_id,
            name=f"{getattr(ds_row, 'name', dataset_id)} (Lag-Llama FT forecast)",
            path=object_key,
            owner_email=owner_email,
        ))

        run.status = "SUCCESS"
        meta = (run.metrics_json or {})
        meta.update({
            "note": "Lag-Llama fine-tuned",
            "horizon": horizon,
            "context_len": context_len,
            "freq": freq,
            "model_id_used": model_id,
        })
        run.metrics_json = meta
        db.commit()

    except Exception as e:
        run = db.get(TrainingRun, run_id)
        if run:
            run.status = "FAILURE"
            run.error = str(e)
            db.commit()
        # facoltativo: loggare l'eccezione
        # print("FT predict/save error:", e)
        raise
    finally:
        db.close()


@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/datasets/upload")
def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db), current_user = Depends(get_current_user),):
    owner_email = str(current_user.email).strip().lower()
    existing = db.query(Dataset).filter(Dataset.name == file.filename).first()
    if existing:
        raise HTTPException(status_code=400, detail="File name already exists, please rename your file before uploading.")
    ds_id = save_csv(db, file, owner_email=owner_email)  # <-- passa l’owner
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


@app.get("/jobs/{job_id}/status", response_model=JobStatus)
def job_status(job_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    run = db.get(TrainingRun, job_id)
    if not run:
        raise HTTPException(404, "Job non trovato")
    ds = db.get(Dataset, run.dataset_id)
    owner_email = str(current_user.email).strip().lower()
    if ds.owner_email != owner_email:  # 👈 NEW
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
        .filter(Dataset.owner_email == owner_email)  # << filtro per utente
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




@app.get("/plots/{plot_id}/data")
def get_plot_data(plot_id: str, db: Session = Depends(get_db), current_user = Depends(get_current_user)):
    """
    Scarica il CSV del forecast (quello con ds,value,kind) e lo restituisce al frontend.
    """
    plot = db.get(ForecastPlot, plot_id)
    if not plot:
        raise HTTPException(404, "Plot non trovato")

    owner_email = str(current_user.email).strip().lower()
    if plot.owner_email != owner_email:  # 👈 NEW
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

from cleanlab.outlier import OutOfDistribution

@app.post("/datasets/{dataset_id}/clean-outliers")
def clean_outliers_on_dataset(
    dataset_id: str,
    body: CleanOutliersBody,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    email = str(current_user.email).strip().lower()
    # 1. l’utente ha scelto il dataset
    df, dataset = _load_dataset_as_df(db, dataset_id)

    n = len(df)
    y = df["value"].astype(float).to_numpy()
    outlier_mask_all = np.zeros(n, dtype=bool)

    # 2. scan a blocchi
    for start in range(0, n, body.chunk):
        end = min(start + body.chunk, n)
        seg = y[start:end].reshape(-1, 1)
        if (end - start) < 3:
            continue

        ood = OutOfDistribution()
        scores_seg = ood.fit_score(features=seg)

        mask_seg = scores_seg < body.threshold
        outlier_mask_all[start:end] = mask_seg

    # 3. metto NaN solo dove c’è outlier
    df.loc[outlier_mask_all, "value"] = np.nan

    # 4. salvo nuovo CSV NEL TUO BUCKET
    cleaned_key = f"{dataset.id}_cleaned.csv"
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    upload_to_bucket(cleaned_key, buf.getvalue().encode("utf-8"), bucket=SUPABASE_BUCKET)

    # 5. nuova riga in datasets
    cleaned_name = body.new_name or f"{dataset.name} (cleaned)"
    new_ds = _create_new_dataset_row(db, cleaned_name, cleaned_key, owner_email=email)  # ⬅️ PASSA L'OWNER

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
    # 1. l’utente sceglie il dataset
    df, dataset = _load_dataset_as_df(db, dataset_id)

    # 2. preparo i dati in 3D per PyPots
    values = df["value"].to_numpy(dtype=float)
    X = values.reshape(1, -1, 1)  # [1, n_steps, 1]
    data_dict = {"X": X}

    imputer = Lerp()
    result = imputer.predict(data_dict)
    imputed = result["imputation"].reshape(-1)

    # 3. metto i valori imputati nel df
    df["value"] = imputed

    # 4. salvo nuovo CSV nel TUO bucket
    imputed_key = f"{dataset.id}_imputed.csv"
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    upload_to_bucket(imputed_key, buf.getvalue().encode("utf-8"), bucket=SUPABASE_BUCKET)

    imputed_name = body.new_name if body and body.new_name else f"{dataset.name} (imputed)"
    new_ds = _create_new_dataset_row(
        db, imputed_name, imputed_key, owner_email=email  # ⬅️ PASSA L'OWNER
    )

    return {
        "original_dataset_id": str(dataset.id),
        "imputed_dataset_id": str(new_ds.id),
    }

@app.post("/auth/register")
def register(body: RegisterBody, db: Session = Depends(get_db)):
    # controlla se esiste già
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
    current_user = Depends(get_current_user),  # ← come nel tuo upload
):
    owner_email = str(current_user.email).strip().lower()
    delete_csv(db, dataset_id, owner_email)  # la funzione service mostrata prima
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
    # 1) Leggi metadati dal DB
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

    # 2) Path reale del file su Supabase
    file_path = row.path  # <-- IMPORTANTISSIMO

    # 3) Scarica dal bucket
    try:
        csv_bytes = sb.storage.from_(SUPABASE_BUCKET).download(file_path)  # <-- BYTES
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"CSV non trovato nello storage: {e}")

    if not csv_bytes:
        raise HTTPException(status_code=404, detail="CSV vuoto o non trovato")

        # 4) ritorna CSV
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
        "status": tr.status,     # PENDING | RUNNING | SUCCESS | FAILURE
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
        .join(Dataset, TrainingRun.dataset_id == Dataset.id)   # << join
        .filter(Dataset.owner_email == owner_email)            # << ora funziona
        .order_by(TrainingRun.created_at.desc())
        .all()
    )

    def metric_from(m: dict | None) -> str:
        m = m or {}
        # con ensemble disattivo stai salvando best_model
        # (se un domani aggiungi "model", prenderà quello)
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
    df = pd.read_csv(ds.path)  # se usi storage remoto, scarica in temp
    df["ds"] = pd.to_datetime(df["ds"])
    return df[["ds","value"]], ds

def _infer_freq(ds_col: pd.Series) -> str:
    try: return pd.infer_freq(ds_col.sort_values()) or "D"
    except Exception: return "D"


# ---------- ZERO-SHOT: salvataggio CSV (history+forecast) ----------
@app.post("/lag-llama/predict/save")
def lag_llama_predict_and_save(payload: ZeroShotPredictIn,
                               background: BackgroundTasks,
                               db: Session = Depends(get_db)):
    run_id = str(uuid.uuid4())
    db.add(TrainingRun(id=run_id, dataset_id=str(payload.dataset_id), status="PENDING",
                       metrics_json={}, error=None))
    db.commit()
    background.add_task(_run_lagllama_forecast, SessionLocal, run_id,
                        str(payload.dataset_id), int(payload.horizon), int(payload.context_len))
    return {"run_id": run_id}

def _run_lagllama_forecast(db_maker, run_id: str, dataset_id: str, horizon: int, context_len: int):
    db: Session = db_maker()
    try:
        run = db.get(TrainingRun, run_id); run.status="RUNNING"; db.commit()
        df, ds_row = _load_dataset_as_df(db, dataset_id)
        owner_email = str(ds_row.owner_email).strip().lower()

        s = pd.Series(df["value"].values, index=df["ds"])
        yhat = predict_series(s, horizon, context_len)
        future_index = pd.date_range(df["ds"].max() + pd.tseries.frequencies.to_offset(_infer_freq(df["ds"])),
                                     periods=horizon)
        fut = pd.DataFrame({"ds": future_index, "value": yhat[:horizon], "kind": "forecast"})
        hist = df.assign(kind="history")
        combined = pd.concat([hist, fut], ignore_index=True)

        csv_bytes = combined.to_csv(index=False).encode("utf-8")
        client = supa()
        plot_id = str(uuid.uuid4())
        safe_owner = owner_email.replace("@","_at_")
        object_key = f"{safe_owner}/{dataset_id}/{plot_id}.csv"
        client.storage.from_(SUPABASE_BUCKET).upload(
            path=object_key, file=csv_bytes,
            file_options={"content-type":"text/csv","upsert":"true"}
        )

        db.add(ForecastPlot(id=plot_id, training_run_id=run_id,
                            name=f"{ds_row.name} (Lag-Llama forecast)",
                            path=object_key, owner_email=owner_email))

        fm = db.query(Model).filter(Model.name=="Lag-Llama", Model.kind=="foundation").first()
        run.status="SUCCESS"
        run.metrics_json = (run.metrics_json or {}) | {"note":"Lag-Llama foundation",
                                                       "horizon":horizon, "context_len":context_len}
        if hasattr(run, "model_id_used") and fm:
            run.model_id_used = fm.id
        db.commit()
    except Exception as e:
        run = db.get(TrainingRun, run_id)
        if run: run.status="FAILURE"; run.error=str(e); db.commit()
        raise
    finally:
        db.close()

# ---------- FINETUNE: crea ZIP nello Storage + riga in models ----------
@app.post("/lag-llama/finetune")
def lag_llama_finetune(payload: FinetuneIn, db: Session = Depends(get_db), current_user = Depends(get_current_user),):

    owner_email = str(current_user.email).strip().lower()
    df, _ = _load_dataset_as_df(db, payload.dataset_id)
    if df is None or df.empty:
        raise HTTPException(400, "Dataset vuoto o non trovato")

    df = df.sort_values("ds")
    values = df["value"].to_numpy(dtype=float)
    start_ts = df["ds"].iloc[0]
    try:
        freq = pd.infer_freq(pd.to_datetime(df["ds"]).sort_values())
    except Exception:
        freq = None
    freq = freq or "D"

    # 1) FINE-TUNING → ckpt path locale
    ckpt_path = finetune_and_dump_ckpt(
        values=values,
        start=start_ts,
        freq=freq,
        prediction_length=payload.horizon,
        context_length=payload.context_len,
        lr=payload.lr,
        aug_prob=payload.aug_prob,
        max_epochs=payload.epochs,
        ckpt_base_dir=None,  # lascia che Lightning scelga/crei la dir
    )

    # 2) Upload ckpt su Supabase
    model_id = str(uuid.uuid4())
    object_key = f"models/{model_id}/weights.ckpt"
    with open(ckpt_path, "rb") as f:
        blob = f.read()
    supa().storage.from_(SUPABASE_BUCKET).upload(
        path=object_key,
        file=blob,
        file_options={"content-type": "application/octet-stream", "upsert": "true"},
    )

    # 3) Inserisci riga in DB
    m = Model(
        id=model_id,
        name="Lag-Llama FT",
        kind="fine_tuned",
        base_model="lag-llama",
        storage_path=object_key,
        params_json={
            "dataset_id": payload.dataset_id,
            "epochs": payload.epochs,
            "horizon": payload.horizon,
            "context_len": payload.context_len,
            "freq": freq,
            "lr": payload.lr,
            "aug_prob": payload.aug_prob,
        },
        metrics_json={},
        owner_email=owner_email,
        status="AVAILABLE",
    )
    db.add(m)
    db.commit()

    return {"model_id": m.id, "storage_path": m.storage_path}


# ---------- PREDICT FT: preview ----------
@app.post("/lag-llama-ft/{model_id}/predict/save")
def lag_llama_ft_predict_and_save(
    model_id: str,
    payload: PredictFTSaveIn,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
):
    # controlla che il modello FT esista
    m = db.query(Model).filter(Model.id == model_id).first()
    if not m:
        raise HTTPException(404, "Modello non trovato")
    if (m.kind or "").lower() != "fine_tuned" or (m.base_model or "").lower() != "lag-llama":
        raise HTTPException(400, "Il modello non è un Lag-Llama fine-tuned")

    run_id = str(uuid.uuid4())
    db.add(TrainingRun(
        id=run_id,
        dataset_id=str(payload.dataset_id),
        status="PENDING",
        metrics_json={},
        error=None,
        model_id_used=model_id,
    ))
    db.commit()

    background.add_task(
        _run_lagllama_ft_forecast_and_save,
        SessionLocal,
        run_id,
        model_id,
        str(payload.dataset_id),
        int(payload.horizon),
        int(payload.context_len),
    )
    return {"run_id": run_id}
# ---------- PREDICT FT: salvataggio CSV (history+forecast) ----------
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
                Model.owner_email.is_(None),  # <— GLOBALI
            )
        )  # << filtro per utente
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