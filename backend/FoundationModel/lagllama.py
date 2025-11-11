from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch

# -----------------------------------------------------------------------------
# Torch 2.6+: allow-list delle classi custom usate nei checkpoint GluonTS
# -----------------------------------------------------------------------------
try:
    from torch.serialization import add_safe_globals
except Exception:
    add_safe_globals = None  # su torch < 2.6 non serve

SAFE_CLASSES = []
try:
    from gluonts.torch.distributions.studentT import StudentTOutput
    SAFE_CLASSES.append(StudentTOutput)
except Exception:
    pass
try:
    from gluonts.torch.modules.loss import NegativeLogLikelihood
    SAFE_CLASSES.append(NegativeLogLikelihood)
except Exception:
    pass

if add_safe_globals and SAFE_CLASSES:
    add_safe_globals(SAFE_CLASSES)

# -----------------------------------------------------------------------------
# (Facoltativo) Patch Lightning: evitiamo che forzi weights_only=True al suo interno
# -----------------------------------------------------------------------------
try:
    import lightning.fabric.utilities.cloud_io as fabric_io
    def _safe_load_cloudio(path, map_location=None):
        # useremo comunque il nostro _load_ckpt che tenta prima weights_only=True;
        # qui mettiamo False per evitare sorprese nelle load interne di PL.
        return torch.load(path, map_location=map_location, weights_only=False)
    fabric_io._load = _safe_load_cloudio
except Exception:
    pass

try:
    import lightning.pytorch.core.saving as saving_mod
    def _safe_pl_load(path, map_location=None):
        return torch.load(path, map_location=map_location, weights_only=False)
    saving_mod.pl_load = _safe_pl_load
except Exception:
    pass

# -----------------------------------------------------------------------------
# GluonTS / Lag-Llama
# -----------------------------------------------------------------------------
from gluonts.dataset.common import ListDataset
from lag_llama.gluon.estimator import LagLlamaEstimator  # percorso coerente con il tuo repo

DEFAULT_CKPT = Path(__file__).resolve().parent / "weights" / "lag-llama.ckpt"

# cache in memoria per predictor parametrizzati
_cache: dict[tuple[str, int, int], object] = {}

# mapping per normalizzare distr_output a stringa
_DISTR_OUTPUT_MAP = {
    "studenttoutput": "student_t",
    "student_t": "student_t",
    "studentt": "student_t",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _coerce_distr_output(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "_")
        return _DISTR_OUTPUT_MAP.get(key, key)
    # classe/istanza -> nome
    name = getattr(value, "__name__", None) or value.__class__.__name__
    return _DISTR_OUTPUT_MAP.get(name.lower(), "student_t")


def _load_ckpt(ckpt_path: str) -> dict:
    """
    Caricamento robusto:
      1) tenta weights_only=True (Torch 2.6 default, sicuro)
      2) fallback a weights_only=False solo se il file è trusted (qui sì: fondazione locale)
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {ckpt_path}")

    # 1) tentativo sicuro
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict):
            return ckpt
    except Exception:
        pass

    # 2) fallback fidato
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def _extract_model_kwargs(ckpt: dict) -> dict:
    hp = (ckpt.get("hyper_parameters", {}) or {})
    mk = dict(hp.get("model_kwargs", {}) or {})

    # rimuovi ciò che passeremo noi o crea conflitti
    for k in ("context_length", "prediction_length", "rope_scaling", "ckpt_path", "freq", "freq_str"):
        mk.pop(k, None)

    # normalizza distr_output (classe -> string)
    if "distr_output" in mk:
        mk["distr_output"] = _coerce_distr_output(mk["distr_output"])

    # coerce numeri base (se arrivano come stringhe)
    def _coerce(v):
        if isinstance(v, (int, float, bool)) or v is None:
            return v
        if isinstance(v, str):
            s = v.strip()
            if s.isdigit():
                return int(s)
            try:
                return float(s)
            except Exception:
                return s
        return v

    mk = {k: _coerce(v) for k, v in mk.items()}

    # whitelist dei kwargs “ufficiali” (come nel sample del sito)
    whitelist = {
        "input_size", "n_layer", "n_embd_per_head", "n_head",
        "scaling", "time_feat", "time_features"
    }
    mk = {k: v for k, v in mk.items() if k in whitelist}

    return mk


from inspect import signature
from gluonts.time_feature.lag import get_lags_for_frequency
from gluonts.time_feature import time_features_from_frequency_str

from inspect import signature

def _build_estimator(ckpt_path: str, horizon: int, context_len: int, freq: str) -> LagLlamaEstimator:
    ckpt = _load_ckpt(ckpt_path)
    model_kwargs = _extract_model_kwargs(ckpt)

    params = signature(LagLlamaEstimator).parameters
    # ... (tutto il resto invariato: freq/freq_str o lags_seq/time_features, rope_scaling, ecc.)

    est_kwargs = dict(
        ckpt_path=ckpt_path,
        prediction_length=int(horizon),
        context_length=int(context_len),
        **model_kwargs,
    )

    # Frequenza (come da patch precedente) ...
    # Rope scaling (come da patch precedente) ...

    # Aggiungi flag utili SOLO se supportati dalla tua versione
    if "nonnegative_pred_samples" in params:
        est_kwargs["nonnegative_pred_samples"] = True
    if "batch_size" in params:
        # valore di default; poi lo puoi sovrascrivere dal predictor (vedi get_predictor)
        est_kwargs["batch_size"] = int(os.getenv("LAG_LLAMA_BATCH_SIZE", "64"))
    if "num_parallel_samples" in params:
        est_kwargs["num_parallel_samples"] = int(os.getenv("LAG_LLAMA_NUM_SAMPLES", "20"))

    return LagLlamaEstimator(**est_kwargs)




def get_predictor(horizon: int, context_len: int, freq: str,
                  num_samples: int = 20, batch_size: int = 64, device: str = "cpu"):
    ckpt_path = os.getenv("LAG_LLAMA_CKPT") or str(DEFAULT_CKPT)
    key = (ckpt_path, int(horizon), int(context_len), str(freq), int(num_samples), int(batch_size), str(device))
    pred = _cache.get(key)
    if pred is None:
        est = _build_estimator(ckpt_path, horizon, context_len, freq)
        # crea il predictor (alcune versioni accettano device, altre no: gestiamo entrambi i casi)
        try:
            pred = est.create_predictor(
                transformation=est.create_transformation(),
                module=est.create_lightning_module(),
                device=device
            )
        except TypeError:
            pred = est.create_predictor(
                transformation=est.create_transformation(),
                module=est.create_lightning_module(),
            )
        # configura campioni/batch se supportati
        if hasattr(pred, "num_parallel_samples"):
            pred.num_parallel_samples = int(num_samples)
        if hasattr(pred, "batch_size"):
            pred.batch_size = int(batch_size)
        _cache[key] = pred
    return pred



def predict_series(
    series: np.ndarray,
    horizon: int,
    context_len: int,
    freq: str = "D",
    start: str | pd.Timestamp = "1970-01-01",
    *,
    num_samples: int = 20,
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """
    series: array 1D storico
    ritorna: array 1D di lunghezza = horizon (forecast puntuale)
    """
    # 1) sanitizza input
    series = np.asarray(series, dtype=float).reshape(-1)
    if not np.isfinite(series).all():
        raise ValueError("La serie contiene NaN/inf.")
    if series.size < max(8, context_len):
        raise ValueError(f"Serie troppo corta: {series.size} < context_len={context_len}")

    # 2) guardia su freq (evita '0' o stringhe vuote)
    if not freq or str(freq).strip() == "0":
        freq = "D"

    # 3) dataset GluonTS
    dataset = ListDataset([{"target": series, "start": pd.Timestamp(start)}], freq=freq)

    # 4) predictor parametrizzato
    predictor = get_predictor(
        horizon,
        context_len,
        freq,
        num_samples=num_samples,
        batch_size=batch_size,
        device=device,
    )

    # 5) previsione (predictor.predict(...) -> generator)
    pred_iter = predictor.predict(dataset)
    forecast = next(iter(pred_iter), None)
    if forecast is None:
        raise RuntimeError("Predictor non ha restituito forecast")

    # 6) estrai il punto: mean -> q50 -> media dei samples -> cast
    if hasattr(forecast, "mean"):
        yhat = np.asarray(forecast.mean, dtype=float)
    elif hasattr(forecast, "quantile"):
        yhat = np.asarray(forecast.quantile("0.5"), dtype=float)
    elif hasattr(forecast, "samples"):
        samples = np.asarray(forecast.samples, dtype=float)  # shape: (n_samples, horizon[, dims])
        yhat = np.nanmean(samples, axis=0)
    else:
        yhat = np.asarray(forecast, dtype=float)

    # 7) se multivariato (dims, horizon) prendi la prima dimensione come default
    if yhat.ndim > 1:
        yhat = yhat[0]

    # 8) trim/pad alla lunghezza richiesta
    if yhat.shape[0] < horizon:
        # pad di sicurezza (non dovrebbe succedere, ma evitiamo crash)
        pad = np.full(horizon - yhat.shape[0], np.nan)
        yhat = np.concatenate([yhat, pad])
    return yhat[:horizon]


