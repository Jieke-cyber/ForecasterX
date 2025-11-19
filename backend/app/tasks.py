from .worker import celery_app

import logging
import os
import io
import uuid
import pandas as pd
from sqlalchemy.orm import Session
from .models import TrainingRun, ForecastPlot, Dataset, Model
from .db import SessionLocal
from autots import AutoTS

from .services import read_ts_for_training, _load_dataset_as_df
from FoundationModel.lagllama import finetune_and_dump_ckpt
from .supa import SUPABASE_URL, SUPABASE_BUCKET, SUPABASE_BUCKET_PLOTS, supa

logger = logging.getLogger(__name__)
BUCKET_PLOTS = os.getenv("SUPABASE_BUCKET", "plots")
SUPABASE_BUCKET_MODELS = os.getenv("SUPABASE_BUCKET", "models")


def _run_training(db: Session, run_id: str, dataset_ids: list[str], horizon: int):
    run = db.get(TrainingRun, run_id)
    if not run:
        logger.error(f"Task AutoTS avviato ma TrainingRun {run_id} non trovato.")
        return

    run.status = "RUNNING"
    db.commit()

    try:
        combined_dfs = []
        dataset_metadata = {}

        for ds_id in dataset_ids:
            ds_row = db.get(Dataset, ds_id)
            if not ds_row:
                raise ValueError(f"Dataset {ds_id} non trovato")

            df_single = read_ts_for_training(db, ds_id)

            if df_single.empty:
                logger.warning(f"Dataset {ds_id} vuoto, saltato.")
                continue

            df_single["series_id"] = ds_id
            combined_dfs.append(df_single)
            dataset_metadata[ds_id] = ds_row

        if not combined_dfs:
            raise ValueError("Nessun dataset valido è stato caricato.")

        df_long = pd.concat(combined_dfs, ignore_index=True)

        model = AutoTS(forecast_length=horizon, frequency="infer", ensemble=None)
        model = model.fit(
            df_long,
            date_col="ds",
            value_col="value",
            id_col="series_id"
        )
        prediction = model.predict()
        fcst = prediction.forecast

        forecast_plots = []
        owner_email = str(db.get(Dataset, dataset_ids[0]).owner_email).strip().lower()
        safe_owner = owner_email.replace("@", "_at_")

        for ds_id in fcst.columns:
            yhat = fcst[[ds_id]].reset_index()
            yhat.columns = ["ds", "yhat"]

            hist = df_long[df_long["series_id"] == ds_id][["ds", "value"]].copy().assign(kind="history")
            fut = yhat.rename(columns={"yhat": "value"}).assign(kind="forecast")
            combined = pd.concat([hist, fut], ignore_index=True)

            csv_buf = io.StringIO()
            combined.to_csv(csv_buf, index=False)
            csv_bytes = csv_buf.getvalue().encode("utf-8")

            client = supa()
            plot_id = str(uuid.uuid4())
            object_key = f"{safe_owner}/{ds_id}/{plot_id}.csv"

            client.storage.from_(BUCKET_PLOTS).upload(
                path=object_key,
                file=csv_bytes,
                file_options={"content-type": "text/csv", "upsert": "true"},
            )

            ds_name = dataset_metadata[ds_id].name
            forecast_plots.append(ForecastPlot(
                id=plot_id,
                training_run_id=run_id,
                name=f"{ds_name} (forecasted, combined run)",
                path=object_key,
                owner_email=owner_email
            ))

        db.add_all(forecast_plots)

        run.status = "SUCCESS"
        run.error = None
        metrics = {"note": f"AutoTS su {len(dataset_ids)} serie", "best_model": getattr(model, "best_model_name", None)}
        run.metrics_json = (run.metrics_json or {}) | metrics
        db.commit()

    except Exception as e:
        logger.error(f"❌ Fallimento catastrofico nel job AutoTS {run_id}: {e}")
        db.rollback()
        run.status = "FAILURE"
        run.error = str(e)
        db.commit()


@celery_app.task(name="run_autots_training")
def run_autots_training_task(run_id: str, dataset_ids: list[str], horizon: int):
    """Wrapper Task di Celery per AutoTS."""
    db = SessionLocal()
    try:
        _run_training(db, run_id, dataset_ids, horizon)
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


@celery_app.task(name="run_lagllama_finetuning")
def run_lagllama_finetuning_task(
        run_id: str,
        dataset_ids: list[str],
        owner_email: str,
        base_model_id: str,
        payload_dict: dict
):
    """Wrapper Task di Celery per il Finetuning di Lag-Llama (Multiserie)."""

    db = SessionLocal()
    run = db.get(TrainingRun, run_id)
    if not run:
        logger.error(f"Task Lag-Llama avviato ma TrainingRun {run_id} non trovato.")
        db.close()
        return

    try:
        run.status = "RUNNING"
        db.commit()

        df_list = []
        frequencies = []

        for ds_id in dataset_ids:
            df, _ = _load_dataset_as_df(db, ds_id)
            if df is None or df.empty:
                logger.warning(f"Dataset {ds_id} vuoto, saltato.")
                continue

            df = df.sort_values("ds")

            df['item_id'] = ds_id
            df_list.append(df)

            try:
                freq = pd.infer_freq(pd.to_datetime(df["ds"]).sort_values())
            except Exception:
                freq = None
            frequencies.append(freq or "D")

        if not df_list:
            raise ValueError("Nessun dataset valido è stato caricato per il fine-tuning.")

        df_combined = pd.concat(df_list, ignore_index=True)
        common_freq = frequencies[0]

        logger.info(f"Inizio finetuning su {len(dataset_ids)} serie per job {run_id}...")

        ckpt_path = finetune_and_dump_ckpt(
            df_long=df_combined,
            freq=common_freq,
            prediction_length=payload_dict['horizon'],
            context_length=payload_dict['context_len'],
            lr=payload_dict['lr'],
            aug_prob=payload_dict['aug_prob'],
            max_epochs=payload_dict['epochs'],
            ckpt_base_dir=None,
        )
        logger.info(f"Finetuning per job {run_id} completato. Ckpt: {ckpt_path}")

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

        m = Model(
            id=ffmodel_id,
            name="Lag-Llama FT",
            kind="fine_tuned",
            base_model="lag-llama",
            storage_path=object_key,
            params_json={
                "dataset_id": dataset_ids[0],
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

        run.status = "SUCCESS"
        run.error = None
        meta = (run.metrics_json or {})
        meta.update({
            "model": "Lag-Llama fine-tuned",
            "model_id_used": ffmodel_id,
            **payload_dict
        })
        run.metrics_json = meta

        db.commit()

    except Exception as e:
        logger.error(f"❌ Fallimento catastrofico nel job lag-llama {run_id}: {e}")
        db.rollback()
        run.status = "FAILURE"
        run.error = str(e)
        db.commit()
    finally:
        db.close()