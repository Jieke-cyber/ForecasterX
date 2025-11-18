# In backend/app/tasks.py

# --- Import di Celery ---
from .worker import celery_app  # La nostra app Celery da worker.py

# --- Import dal tuo vecchio app/jobs.py e logica LagLlama ---
import logging
import os
import io
import uuid
import pandas as pd
from sqlalchemy.orm import Session
from .models import TrainingRun, ForecastPlot, Dataset, Model
from .db import SessionLocal
from .supa import supa  # Assicurati che l'import 'supa' sia corretto
from autots import AutoTS

# --- Import specifici per i tuoi task ---
# Assicurati che questi percorsi siano corretti e importabili
from .services import read_ts_for_training, _load_dataset_as_df
from FoundationModel.lagllama import finetune_and_dump_ckpt
from .supa import SUPABASE_URL, SUPABASE_BUCKET, SUPABASE_BUCKET_PLOTS, supa

# --- Setup di base ---
logger = logging.getLogger(__name__)
BUCKET_PLOTS = os.getenv("SUPABASE_BUCKET", "plots")
SUPABASE_BUCKET_MODELS = os.getenv("SUPABASE_BUCKET", "models")


# --- Funzione helper 'naive_forecast' (copiata dal tuo jobs.py) ---
def naive_forecast(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    last = float(df["value"].tail(1).iloc[0])
    last_date = df["ds"].tail(1).iloc[0]
    freq_str = pd.infer_freq(df["ds"])
    if freq_str:
        offset = pd.tseries.frequencies.to_offset(freq_str)
    else:
        diffs = df["ds"].diff().dropna()
        offset = diffs.mode().iloc[0] if not diffs.empty else pd.Timedelta(days=1)
    dates = [last_date + offset * (i + 1) for i in range(horizon)]
    return pd.DataFrame({"ds": dates, "yhat": [last] * horizon})


# --- Funzione helper '_run_training' (copiata dal tuo jobs.py) ---
# Questa è la logica interna di AutoTS
def _run_training(db: Session, run_id: str, dataset_id: str, horizon: int):
    run = db.get(TrainingRun, run_id)
    if not run:
        logger.error(f"Task AutoTS avviato ma TrainingRun {run_id} non trovato.")
        return

    run.status = "RUNNING"
    db.commit()

    try:
        ds_row = db.get(Dataset, dataset_id)
        if not ds_row:
            raise ValueError("Dataset non trovato")
        owner_email = str(ds_row.owner_email).strip().lower()
        df = read_ts_for_training(db, dataset_id)

        try:
            model = AutoTS(forecast_length=horizon, frequency="infer", ensemble=None)
            model = model.fit(df, date_col="ds", value_col="value")
            prediction = model.predict()
            fcst = prediction.forecast
            yhat = fcst.iloc[:, 0].reset_index()
            yhat.columns = ["ds", "yhat"]
            metrics = {"note": "AutoTS", "best_model": getattr(model, "best_model_name", None)}
        except Exception:
            logger.exception("❌ AutoTS fallito, uso naive forecast:")
            yhat = naive_forecast(df, horizon)
            metrics = {"note": "naive"}

        hist = df[["ds", "value"]].copy().assign(kind="history")
        fut = yhat.rename(columns={"yhat": "value"}).assign(kind="forecast")
        combined = pd.concat([hist, fut], ignore_index=True)

        csv_buf = io.StringIO()
        combined.to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue().encode("utf-8")

        client = supa()
        plot_id = str(uuid.uuid4())
        safe_owner = owner_email.replace("@", "_at_")
        object_key = f"{safe_owner}/{dataset_id}/{plot_id}.csv"
        client.storage.from_(BUCKET_PLOTS).upload(
            path=object_key,
            file=csv_bytes,
            file_options={"content-type": "text/csv", "upsert": "true"},
        )

        db.add(ForecastPlot(
            id=plot_id,
            training_run_id=run_id,
            name=f"{ds_row.name} (forecasted)",
            path=object_key,
            owner_email=owner_email
        ))
        run.status = "SUCCESS"
        run.error = None  # Pulisce errori vecchi
        run.metrics_json = (run.metrics_json or {}) | metrics
        db.commit()

    except Exception as e:
        logger.error(f"❌ Fallimento catastrofico nel job AutoTS {run_id}: {e}")
        db.rollback()
        run.status = "FAILURE"
        run.error = str(e)
        db.commit()


# --- TASK CELERY 1 (AutoTS) ---
@celery_app.task(name="run_autots_training")
def run_autots_training_task(run_id: str, dataset_id: str, horizon: int):
    """Wrapper Task di Celery per AutoTS."""
    db = SessionLocal()
    try:
        _run_training(db, run_id, dataset_id, horizon)
    except Exception as e:
        logger.error(f"Fallimento del wrapper Celery AutoTS per {run_id}: {e}")
        try:
            run = db.get(TrainingRun, run_id)
            if run:
                run.status = "FAILURE"
                run.error = f"Errore wrapper: {e}"
                db.commit()
        except:
            pass
    finally:
        db.close()


# --- TASK CELERY 2 (Lag-Llama Finetune) ---
@celery_app.task(name="run_lagllama_finetuning")
def run_lagllama_finetuning_task(
        run_id: str,
        dataset_id: str,
        owner_email: str,
        base_model_id: str,
        payload_dict: dict
):
    """Wrapper Task di Celery per il Finetuning di Lag-Llama."""

    db = SessionLocal()
    run = db.get(TrainingRun, run_id)
    if not run:
        logger.error(f"Task Lag-Llama avviato ma TrainingRun {run_id} non trovato.")
        db.close()
        return

    try:
        # 1. Imposta lo stato su RUNNING
        run.status = "RUNNING"
        db.commit()

        # 2. Carica i dati
        df, _ = _load_dataset_as_df(db, dataset_id)
        if df is None or df.empty:
            raise ValueError("Dataset vuoto o non trovato")

        df = df.sort_values("ds")
        values = df["value"].to_numpy(dtype=float)
        start_ts = df["ds"].iloc[0]
        try:
            freq = pd.infer_freq(pd.to_datetime(df["ds"]).sort_values())
        except Exception:
            freq = None
        freq = freq or "D"

        # 3. ESEGUI IL FINETUNING (Lavoro Pesante 1)
        logger.info(f"Inizio finetuning per job {run_id}...")
        ckpt_path = finetune_and_dump_ckpt(
            values=values,
            start=start_ts,
            freq=freq,
            prediction_length=payload_dict['horizon'],
            context_length=payload_dict['context_len'],
            lr=payload_dict['lr'],
            aug_prob=payload_dict['aug_prob'],
            max_epochs=payload_dict['epochs'],
            ckpt_base_dir=None,
        )
        logger.info(f"Finetuning per job {run_id} completato. Ckpt: {ckpt_path}")

        # 4. UPLOAD SU SUPABASE (Lavoro Pesante 2)
        ffmodel_id = str(uuid.uuid4())
        object_key = f"models/{ffmodel_id}/weights.ckpt"
        with open(ckpt_path, "rb") as f:
            blob = f.read()

        supa().storage.from_(SUPABASE_BUCKET_MODELS).upload(
            path=object_key,
            file=blob,
            file_options={"content-type": "application/octet-stream", "upsert": "true"},
        )
        logger.info(f"Upload ckpt per job {run_id} completato.")

        # 5. Crea il record del nuovo Modello nel DB
        m = Model(
            id=ffmodel_id,
            name="Lag-Llama FT",
            kind="fine_tuned",
            base_model="lag-llama",
            storage_path=object_key,
            params_json={
                "dataset_id": dataset_id,
                "epochs": payload_dict['epochs'],
                "horizon": payload_dict['horizon'],
                "context_len": payload_dict['context_len'],
                "freq": freq,
                "lr": payload_dict['lr'],
                "aug_prob": payload_dict['aug_prob'],
            },
            metrics_json={},
            owner_email=owner_email,
            status="AVAILABLE",
        )
        db.add(m)

        # 6. Aggiorna il TrainingRun su SUCCESS
        run.status = "SUCCESS"
        run.error = None
        meta = (run.metrics_json or {})
        meta.update({
            "model": "Lag-Llama fine-tuned",
            "model_id_used": ffmodel_id,
            **payload_dict  # Salva tutti i parametri usati
        })
        run.metrics_json = meta

        db.commit()

    except Exception as e:
        # 7. GESTIONE FALLIMENTO
        logger.error(f"❌ Fallimento catastrofico nel job lag-llama {run_id}: {e}")
        db.rollback()
        run.status = "FAILURE"
        run.error = str(e)
        db.commit()
    finally:
        # 8. Chiudi sempre la sessione
        db.close()