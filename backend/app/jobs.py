import logging

from FoundationModel.lagllama import predict_series

logger = logging.getLogger(__name__)
import os
import io
import uuid
import pandas as pd
from sqlalchemy.orm import Session
from .models import TrainingRun, ForecastPlot, Dataset, Model

BUCKET_PLOTS = os.getenv("SUPABASE_BUCKET_PLOTS", "plots")

def _run_lagllama_forecast(db_maker, run_id: str, dataset_id: str, horizon: int, context_len: int):
    db: Session = db_maker()
    try:
        run = db.get(TrainingRun, run_id)
        run.status = "RUNNING"
        db.commit()

        ds_row = db.get(Dataset, dataset_id)
        if not ds_row:
            raise ValueError("Dataset non trovato")
        owner_email = str(ds_row.owner_email).strip().lower()

        try:
            from .services import read_ts_for_training
            df = read_ts_for_training(db, dataset_id)
        except Exception:
            df = pd.read_csv(ds_row.path)
            df["ds"] = pd.to_datetime(df["ds"])
            df = df[["ds","value"]]

        s = pd.Series(df["value"].values, index=df["ds"])
        yhat = predict_series(s, horizon=horizon, context_len=context_len)
        yhat = pd.Series(yhat, index=pd.date_range(
            start=df["ds"].max() + pd.tseries.frequencies.to_offset(_infer_freq(df["ds"])),
            periods=horizon
        )).reset_index()
        yhat.columns = ["ds", "yhat"]
        metrics = {"note": "Lag-Llama foundation", "horizon": horizon, "context_len": context_len}

        hist = df[["ds", "value"]].copy().assign(kind="history")
        fut  = yhat.rename(columns={"yhat": "value"}).assign(kind="forecast")
        combined = pd.concat([hist, fut], ignore_index=True)

        csv_buf = io.StringIO()
        combined.to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue().encode("utf-8")

        from .supa import supa, BUCKET_PLOTS
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
            name=f"{ds_row.name} (Lag-Llama forecast)",
            path=object_key,
            owner_email=owner_email
        ))

        fm = db.query(Model).filter(Model.name=="Lag-Llama", Model.kind=="foundation").first()
        run.status = "SUCCESS"
        run.metrics_json = (run.metrics_json or {}) | metrics
        if fm:
            setattr(run, "model_id_used", fm.id)
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


def _infer_freq(ds_col: pd.Series) -> str:
    """Inferisce la frequenza temporale per creare le date future."""
    try:
        return pd.infer_freq(ds_col.sort_values()) or "D"
    except Exception:
        return "D"

def _run_lagllama_ft_forecast(db_maker, run_id: str, model_id: str, dataset_id: str, horizon: int, context_len: int):
    db: Session = db_maker()
    try:
        run = db.get(TrainingRun, run_id); run.status = "RUNNING"; db.commit()
        ds_row = db.get(Dataset, dataset_id);
        if not ds_row: raise ValueError("Dataset non trovato")
        owner_email = str(ds_row.owner_email).strip().lower()

        try:
            from .services import read_ts_for_training
            df = read_ts_for_training(db, dataset_id)
        except Exception:
            df = pd.read_csv(ds_row.path); df["ds"]=pd.to_datetime(df["ds"]); df=df[["ds","value"]]

        s = pd.Series(df["value"].values, index=df["ds"])
        yhat = predict_series(s, horizon=horizon, context_len=context_len)

        yhat = pd.Series(yhat, index=pd.date_range(
            start=df["ds"].max() + pd.tseries.frequencies.to_offset(_infer_freq(df["ds"])),
            periods=horizon
        )).reset_index(); yhat.columns = ["ds","yhat"]
        hist = df[["ds","value"]].copy().assign(kind="history")
        fut  = yhat.rename(columns={"yhat":"value"}).assign(kind="forecast")
        combined = pd.concat([hist, fut], ignore_index=True)

        csv_buf = io.StringIO(); combined.to_csv(csv_buf, index=False)
        from .supa import supa, BUCKET_PLOTS
        client = supa()
        plot_id = str(uuid.uuid4())
        safe_owner = owner_email.replace("@","_at_")
        object_key = f"{safe_owner}/{dataset_id}/{plot_id}.csv"
        client.storage.from_(BUCKET_PLOTS).upload(
            path=object_key, file=csv_buf.getvalue().encode("utf-8"),
            file_options={"content-type":"text/csv","upsert":"true"}
        )
        db.add(ForecastPlot(id=plot_id, training_run_id=run_id, name=f"{ds_row.name} (FT forecast)",
                            path=object_key, owner_email=owner_email))

        m = db.get(Model, model_id)
        metrics = {"note": "Lag-Llama FT", "horizon": horizon, "context_len": context_len}
        run.status="SUCCESS"; run.metrics_json=(run.metrics_json or {}) | metrics
        if m: setattr(run, "model_id_used", m.id)
        db.commit()
    except Exception as e:
        run = db.get(TrainingRun, run_id)
        if run: run.status="FAILURE"; run.error=str(e); db.commit()
        raise
    finally:
        db.close()
