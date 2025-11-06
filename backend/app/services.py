# app/services.py
import os
import uuid
import io
import pandas as pd
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from .models import Dataset
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
