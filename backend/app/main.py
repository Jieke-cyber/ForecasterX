# app/main.py
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Depends, BackgroundTasks, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Security, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt, ExpiredSignatureError, JOSEError
from matplotlib import pyplot as plt
from starlette.responses import RedirectResponse, StreamingResponse, PlainTextResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from sqlalchemy import text
import uuid
import os, httpx, re
import io

from .db import Base, engine, get_db
from .models import Dataset, TrainingRun, ForecastPlot
from .schemas import TrainRequest, JobStatus
from .services import save_csv, delete_csv
from .jobs import train_job
from .supa import SUPABASE_URL, SUPABASE_BUCKET, supa  # per costruire URL se bucket è pubblico

from cleanlab.outlier import OutOfDistribution

from .models import User
from .auth_utils import hash_password, verify_password, create_access_token

from .auth_utils import SECRET_KEY, ALGORITHM
app = FastAPI(title="TS WebApp Backend (MVP)")

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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/datasets/upload")
def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db), current_user = Depends(get_current_user),):
    owner_email = str(current_user.email).strip().lower()
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


@app.get("/jobs/{job_id}", response_model=JobStatus)
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