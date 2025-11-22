# app/services.py
import os
import tempfile
import uuid
import io
from typing import Iterable

import pandas as pd
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session

from FoundationModel.lagllama import load_predictor_from_ckpt, predict_series_with_predictor
from .utils import download_from_bucket
from .models import Dataset, ForecastPlot, TrainingRun, Model
from .supa import supa, SUPABASE_URL, SUPABASE_BUCKET  # <-- nuovo

BUCKET = os.getenv("SUPABASE_BUCKET", "datasets")

def save_csv(db: Session, file: UploadFile, owner_email: str) -> str:
    """
    Carica il CSV su Supabase Storage e registra una riga in 'datasets'
    con (id, name, path, owner_email). Restituisce dataset_id.
    """
    fname = (file.filename or "").lower()
    if not fname.endswith(".csv"):
        raise HTTPException(400, "Carica un file .csv")

    content: bytes = file.file.read()
    if not content:
        raise HTTPException(400, "File vuoto")

    # sanity check: leggibilità CSV
    try:
        pd.read_csv(io.BytesIO(content), nrows=3)
    except Exception:
        raise HTTPException(400, "CSV non leggibile (formato/encoding)")

    ds_id = str(uuid.uuid4())
    object_key = f"{ds_id}.csv"

    client = supa()
    try:
        # supabase-py: upload(bytes) → ok
        client.storage.from_(BUCKET).upload(path=object_key, file=content)
    except Exception as e:
        raise HTTPException(500, f"Upload su Supabase fallito: {e}")

    # registra nel DB includendo il proprietario
    db.add(Dataset(id=ds_id, name=file.filename, path=object_key, owner_email=owner_email))
    db.commit()
    return ds_id


def read_ts_for_training(db: Session, dataset_id: str) -> pd.DataFrame:
    """
    Scarica il CSV da Supabase Storage e lo normalizza a due colonne: ds, value.
    Accetta:
      - CSV con colonne 'ds,value'
      - CSV dove la prima colonna è data e la seconda è valore
    """
    ds = db.get(Dataset, dataset_id)
    if not ds or not ds.path:
        raise HTTPException(404, "Dataset non trovato")

    client = supa()
    try:
        # bucket privato: scarica i bytes
        data: bytes = client.storage.from_(BUCKET).download(path=ds.path)
        df = pd.read_csv(io.BytesIO(data))
        # Se il bucket è Public e preferisci via URL, usa:
        # url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{ds.path}"
        # df = pd.read_csv(url)
    except Exception as e:
        raise HTTPException(400, f"Impossibile leggere il CSV dal bucket: {e}")

    # normalizza a due colonne ds,value
    if {"ds", "value"}.issubset(df.columns):
        df = df[["ds", "value"]].copy()
    else:
        if df.shape[1] < 2:
            raise HTTPException(400, "CSV deve avere almeno due colonne (data, valore)")
        df = df.iloc[:, :2].copy()
        df.columns = ["ds", "value"]

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"]).sort_values("ds")
    if df.empty:
        raise HTTPException(400, "Nessun dato valido dopo la conversione delle date")
    return df

def _norm_email(v: str) -> str:
    return str(v).strip().lower()

def delete_csv(db: Session, dataset_id: str, owner_email: str) -> None:
    """
    Complementare di save_csv:
    - verifica che il dataset esista e sia dell'owner
    - elimina il file CSV da Storage (usando ds.path)
    - elimina la riga dal DB
    """
    owner = _norm_email(owner_email)

    ds = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.owner_email == owner)
        .first()
    )
    if not ds:
        raise HTTPException(404, "Dataset non trovato")

    if not BUCKET:
        raise HTTPException(500, "Config mancante: SUPABASE_BUCKET")

    # ⚠️ Nel tuo modello la key è ds.path (es. "630f8... .csv")
    object_key = (ds.path or "").strip()

    if not object_key:
        # niente file associato → elimina comunque la riga
        db.delete(ds)
        db.commit()
        return

    client = supa()
    storage = client.storage.from_(BUCKET)

    # (facoltativo ma utile) verifica che il file esista davvero, come fai nel GET
    base = object_key.rsplit("/", 1)[0] if "/" in object_key else ""
    try:
        entries = storage.list(path=base) or []
    except Exception as e:
        raise HTTPException(500, f"Errore nel list dello storage: {e}")

    filename = object_key.split("/")[-1]
    exists = any(it.get("name") == filename for it in entries)
    if not exists:
        # scegli tu: 404 per evitare incoerenze
        raise HTTPException(404, f"CSV non trovato in storage: {BUCKET}/{object_key}")

    # elimina il file (coerenza forte: se fallisce, non tocchiamo il DB)
    try:
        storage.remove([object_key])
    except Exception as e:
        raise HTTPException(409, f"Rimozione CSV fallita: {e}")

    # ora elimina la riga DB
    db.delete(ds)
    db.commit()

def _remove_storage_paths(paths: Iterable[str]) -> None:
    paths = [p for p in set(paths) if p]
    if not paths:
        return
    client = supa()
    client.storage.from_(BUCKET).remove(paths)

def delete_single_plot(db: Session, plot_id: str, owner_email: str) -> None:
    """
    Elimina UN plot dell'utente:
    - verifica ownership
    - se nessun altro record punta allo stesso path, rimuove anche il file dallo storage
    - cancella il record DB
    """
    owner = owner_email.strip().lower()

    row = (
        db.query(ForecastPlot)
        .filter(ForecastPlot.id == plot_id, Dataset.owner_email == owner)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Plot non trovato")

    # se il file è condiviso da altri record, non cancellarlo dallo storage
    same_path_count = (
        db.query(ForecastPlot)
        .filter(ForecastPlot.path == row.path, ForecastPlot.id != row.id)
        .count()
    )
    if not row:
        raise HTTPException(404, "Plot non trovato")

    if same_path_count == 0 and row.path:
        _remove_storage_paths([row.path])

    db.delete(row)
    db.commit()

def delete_training_run(db: Session, run_id: str, user_email: str) -> None:
    owner = user_email.strip().lower()

    run = (
        db.query(TrainingRun)
          .join(Dataset, TrainingRun.dataset_id == Dataset.id)        # <-- serve per ownership
          .filter(TrainingRun.id == run_id, Dataset.owner_email == owner)
          .first()
    )
    if not run:
        raise HTTPException(status_code=404, detail="Training run non trovato")

    if run.status == "RUNNING":
        raise HTTPException(status_code=409, detail="Il run è in esecuzione")

    # opzionale: pulizia record plot collegati (solo DB)
    db.query(ForecastPlot).filter(ForecastPlot.training_run_id == run_id) \
      .delete(synchronize_session=False)

    db.delete(run)
    db.commit()


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

        run = db.get(TrainingRun, run_id)
        if not run:
            return
        run.status = "RUNNING"
        db.commit()

        df, ds_row = _load_dataset_as_df(db, dataset_id)
        if df is None or df.empty:
            raise RuntimeError("Dataset vuoto o non trovato")

        df = df.sort_values("ds")
        owner_email = (str(ds_row.owner_email) if ds_row and ds_row.owner_email else "").strip().lower()
        values = df["value"].to_numpy(dtype=float)
        start_ts = df["ds"].iloc[0]

        try:
            freq = pd.infer_freq(pd.to_datetime(df["ds"]).sort_values())
        except Exception:
            freq = None
        freq = freq or "D"

        m = db.query(Model).filter(Model.id == model_id).first()
        if not m or not m.storage_path:
            raise RuntimeError("Modello FT senza storage_path")

        blob: bytes = supa().storage.from_(SUPABASE_BUCKET).download(m.storage_path)
        tmp_ckpt = tempfile.NamedTemporaryFile(delete=False, suffix=".ckpt")
        tmp_ckpt.write(blob); tmp_ckpt.flush(); tmp_ckpt.close()
        ckpt_local_path = tmp_ckpt.name

        predictor = load_predictor_from_ckpt(
            weights_ckpt_path=ckpt_local_path,
            horizon=horizon,
            context_len=context_len,
            freq=freq,
        )

        yhat = predict_series_with_predictor(
            predictor,
            series=values,
            horizon=horizon,
            freq=freq,
            start=start_ts,
        )

        try:
            offset = pd.tseries.frequencies.to_offset(freq)
        except Exception:
            offset = pd.tseries.frequencies.to_offset("D")

        last_ts = df["ds"].max()
        future_index = pd.date_range(last_ts + offset, periods=horizon, freq=freq)

        fut = pd.DataFrame({"ds": future_index, "value": yhat[:horizon], "kind": "forecast"})
        hist = df.assign(kind="history")
        combined = pd.concat([hist, fut], ignore_index=True)

        csv_bytes = combined.to_csv(index=False).encode("utf-8")
        client = supa()
        plot_id = str(uuid.uuid4())
        safe_owner = (owner_email or "public").replace("@", "_at_")
        object_key = f"{safe_owner}/{dataset_id}/{plot_id}.csv"

        client.storage.from_(SUPABASE_BUCKET).upload(
            path=object_key,
            file=csv_bytes,
            file_options={"content-type": "text/csv", "upsert": "true"},
        )

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
        raise
    finally:
        db.close()


def _create_new_dataset_row(db: Session, name: str, path: str, owner_email: str) -> Dataset:
    new_id = str(uuid.uuid4())
    new_ds = Dataset(
        id=new_id,
        name=name,
        path=path,
        owner_email=owner_email,
    )
    db.add(new_ds)
    db.commit()
    return new_ds
