import logging

from FoundationModel.lagllama import predict_series

logger = logging.getLogger(__name__)
import os
import io
import uuid
import pandas as pd
from sqlalchemy.orm import Session
from .models import TrainingRun, ForecastPlot, Dataset, Model
from .services import read_ts_for_training
from .db import SessionLocal
from .supa import supa
from autots import AutoTS

BUCKET_PLOTS = os.getenv("SUPABASE_BUCKET_PLOTS", "plots")

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

def _run_training(db: Session, run_id: str, dataset_id: str, horizon: int):
    run = db.get(TrainingRun, run_id)
    run.status = "RUNNING"
    db.commit()

    try:
        ds_row = db.get(Dataset, dataset_id)
        if not ds_row:
            raise ValueError("Dataset non trovato")
        owner_email = str(ds_row.owner_email).strip().lower()
        df = read_ts_for_training(db, dataset_id)

        try:

            model = AutoTS(forecast_length=horizon, frequency="infer", ensemble= None)
            model = model.fit(df, date_col="ds", value_col="value")
            prediction = model.predict()
            fcst = prediction.forecast
            yhat = fcst.iloc[:, 0].reset_index()
            yhat.columns = ["ds", "yhat"]
            best_name = getattr(model, "best_model_name", None)
            best_params = getattr(model, "best_model_params", None)
            metrics = {"note": "AutoTS"
                       , "best_model": best_name
                       , "best_params": str(best_params)}
        except Exception:
            logger.exception("❌ AutoTS fallito:")
            yhat = naive_forecast(df, horizon)
            metrics = {"note": "naive"}

        hist = df[["ds", "value"]].copy().assign(kind="history")
        fut  = yhat.rename(columns={"yhat": "value"}).assign(kind="forecast")
        combined = pd.concat([hist, fut], ignore_index=True)

        csv_buf = io.StringIO()
        combined.to_csv(csv_buf, index=False)

        client = supa()
        plot_id = str(uuid.uuid4())
        safe_owner = owner_email.replace("@", "_at_")
        object_key = f"{safe_owner}/{dataset_id}/{plot_id}.csv"
        client.storage.from_(BUCKET_PLOTS).upload(
            path = object_key,
            file = csv_bytes,
            file_options = {
                "content-type": "text/csv",
                "upsert": "true",
            },
        )
        db.add(ForecastPlot(
            id=plot_id,
            training_run_id=run_id,
            name=f"{ds_row.name} (forecasted)",
            path=object_key,
            owner_email=owner_email
        ))
        run.status = "SUCCESS"
        run.metrics_json = (run.metrics_json or {}) | metrics
        db.commit()

    except Exception as e:
        run.status = "FAILURE"
        run.error = str(e)
        db.commit()

def train_job(run_id: str, dataset_id: str, horizon: int):
    db = SessionLocal()
    try:
        _run_training(db, run_id, dataset_id, horizon)
    finally:
        db.close()

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
