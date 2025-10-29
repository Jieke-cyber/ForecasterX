# app/services.py
import os
import uuid
import io
import pandas as pd
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from .models import Dataset
from .supa import supa, SUPABASE_URL  # <-- nuovo

BUCKET = os.getenv("SUPABASE_BUCKET", "datasets")

def save_csv(db: Session, file: UploadFile) -> str:
    """
    Carica il CSV su Supabase Storage (bucket) e registra una riga in datasets
    con (id, name, path=<object_key>). Restituisce dataset_id.
    """
    fname = (file.filename or "").lower()
    if not fname.endswith(".csv"):
        raise HTTPException(400, "Carica un file .csv")

    content: bytes = file.file.read()
    if not content:
        raise HTTPException(400, "File vuoto")

    # sanity check: deve essere leggibile come CSV
    try:
        pd.read_csv(io.BytesIO(content), nrows=3)
    except Exception:
        raise HTTPException(400, "CSV non leggibile (formato/encoding)")

    ds_id = str(uuid.uuid4())
    object_key = f"{ds_id}.csv"

    # upload su Supabase Storage
    client = supa()
    try:
        client.storage.from_(BUCKET).upload(path=object_key, file=content)
    except Exception as e:
        raise HTTPException(500, f"Upload su Supabase fallito: {e}")

    # Salva in DB la CHIAVE dell’oggetto nel bucket (non l'URL)
    db.add(Dataset(id=ds_id, name=file.filename, path=object_key))
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
