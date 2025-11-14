from __future__ import annotations

import glob

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
from lag_llama.lag_llama.gluon.estimator import LagLlamaEstimator  # percorso coerente con il tuo repo

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

def _normalize_time_feat_keys(model_kwargs: dict) -> dict:
    """Rende compatibili 'time_feat' e 'time_features' con la firma del tuo LagLlamaEstimator."""
    params = signature(LagLlamaEstimator).parameters
    mk = dict(model_kwargs or {})
    if "time_feat" in mk and "time_feat" not in params and "time_features" in params:
        mk["time_features"] = mk.pop("time_feat")
    if "time_features" in mk and "time_features" not in params and "time_feat" in params:
        mk["time_feat"] = mk.pop("time_features")
    return mk

def build_finetune_estimator_from_foundation(
    *,
    foundation_ckpt_path: str | None,
    prediction_length: int,
    context_length: int,
    freq: str = "D",
    lr: float = 1e-4,
    aug_prob: float = 0.2,
    max_epochs: int = 50,
    default_root_dir: str | None = None,   # dove Lightning salverà i ckpt
) -> LagLlamaEstimator:
    """
    Crea un Estimator per FINE-TUNING partendo dal checkpoint foundation (ckpt completo).
    Scrive i checkpoint Lightning in default_root_dir/version_*/checkpoints/*.ckpt
    """
    ckpt_path = foundation_ckpt_path or (os.getenv("LAG_LLAMA_CKPT") or str(DEFAULT_CKPT))
    ckpt = _load_ckpt(ckpt_path)
    base_kwargs = _extract_model_kwargs(ckpt)

    # opzionale: se hai già questa helper, usa la tua; altrimenti no-op
    try:
        base_kwargs = _normalize_time_feat_keys(base_kwargs)  # se presente nel tuo file
    except NameError:
        pass

    params = signature(LagLlamaEstimator).parameters
    est_kwargs = dict(
        ckpt_path=ckpt_path,
        prediction_length=int(prediction_length),
        context_length=int(context_length),
        lr=float(lr),
        aug_prob=float(aug_prob),
        **base_kwargs,
    )

    # freq/freq_str se supportati (altrimenti la freq sta nel ListDataset)
    if "freq" in params:
        est_kwargs["freq"] = str(freq)
    elif "freq_str" in params:
        est_kwargs["freq_str"] = str(freq)

    # rope_scaling se supportato
    base_ctx = ckpt.get("hyper_parameters", {}).get("model_kwargs", {}).get("context_length")
    if base_ctx is not None and "rope_scaling" in params:
        try:
            base_ctx = int(base_ctx)
            factor = max(1.0, (int(context_length) + int(prediction_length)) / float(base_ctx))
            est_kwargs["rope_scaling"] = {"type": "linear", "factor": factor}
        except Exception:
            pass

    # epoche + checkpointing
    tkwargs = {"max_epochs": int(max_epochs), "enable_checkpointing": True}
    if default_root_dir:
        tkwargs["default_root_dir"] = default_root_dir
    if "trainer_kwargs" in params:
        est_kwargs["trainer_kwargs"] = tkwargs
    elif "max_epochs" in params:
        est_kwargs["max_epochs"] = int(max_epochs)

    # batch/samples se previsti
    if "batch_size" in params:
        est_kwargs["batch_size"] = 64
    if "num_parallel_samples" in params:
        est_kwargs["num_parallel_samples"] = 20

    return LagLlamaEstimator(**est_kwargs)



def _latest_ckpt_path(root: str) -> str | None:
    """
    Cerca il checkpoint più recente sotto <root>/**/checkpoints/*.ckpt
    (va bene se fai un FT alla volta e hai un solo foundation).
    """
    pats = [
        os.path.join(root, "**", "checkpoints", "*.ckpt"),
        os.path.join(root, "**", "*.ckpt"),
    ]
    candidates = []
    for pat in pats:
        candidates.extend(glob.glob(pat, recursive=True))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def finetune_and_dump_ckpt(
    *,
    values: np.ndarray,
    start: str | pd.Timestamp,
    freq: str,
    prediction_length: int,
    context_length: int,
    foundation_ckpt_path: str | None = None,
    lr: float = 5e-4,
    aug_prob: float = 0.0,
    max_epochs: int = 50,
    ckpt_base_dir: str | None = None,
) -> str:
    """
    Esegue FINE-TUNING e ritorna il path del checkpoint Lightning (.ckpt) completo.
    """
    import tempfile, glob, os, pathlib
    import pandas as pd
    import numpy as np

    # --- sanitizzazione ---
    arr = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError("La serie contiene NaN/inf.")
    if arr.size < max(8, context_length):
        raise ValueError(f"Serie troppo corta: {arr.size} < context_length={context_length}")

    # --- freq robusta ---
    if not freq or str(freq).strip() == "0":
        freq = "D"

    # --- split 80/20 temporale (invece di arr[:-PL] vs arr) ---
    n = len(arr)
    split_idx = max(int(n * 0.8), 1)

    start_ts = pd.Timestamp(start)
    # calcola lo start della valid usando l'offset della freq
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
        valid_start = start_ts + (split_idx * offset)
    except Exception:
        # fallback: stesso start (non perfetto, ma funziona)
        valid_start = start_ts

    train_ds = ListDataset(
        [{"target": arr[:split_idx], "start": start_ts}],
        freq=freq,
    )
    valid_ds = ListDataset(
        [{"target": arr[split_idx:], "start": valid_start}],
        freq=freq,
    )

    # --- dove salvare i ckpt ---
    rootdir = ckpt_base_dir or tempfile.mkdtemp(prefix="llama_ft_ckpt_")

    # --- estimator da foundation ---
    est = build_finetune_estimator_from_foundation(
        foundation_ckpt_path=foundation_ckpt_path,
        prediction_length=int(prediction_length),
        context_length=int(context_length),
        freq=str(freq),
        lr=float(lr),
        aug_prob=float(aug_prob),
        max_epochs=int(max_epochs),
        default_root_dir=rootdir,
    )

    # --- training: Lightning salva i ckpt sotto rootdir/**/checkpoints ---
    _ = est.train(
        train_ds,
        validation_data=valid_ds,
        cache_data=True,
        shuffle_buffer_length=1000,
    )

    # --- prendi il ckpt più recente ---
    pats = [
        os.path.join(rootdir, "**", "checkpoints", "*.ckpt"),
        os.path.join(rootdir, "**", "*.ckpt"),
    ]
    candidates = []
    for pat in pats:
        candidates.extend(glob.glob(pat, recursive=True))
    if not candidates:
        raise RuntimeError(f"Nessun checkpoint salvato in {rootdir}")
    candidates.sort(key=lambda p: pathlib.Path(p).stat().st_mtime, reverse=True)
    return candidates[0]



# ---------------- PREDICTOR da CKPT FT (riuso in inferenza) -------------------
def load_predictor_from_ckpt(
    weights_ckpt_path: str,
    horizon: int,
    context_len: int,
    freq: str = "D",
    *,
    num_samples: int = 20,
    batch_size: int = 64,
    device: str = "cpu",
):
    """
    Ricrea l'estimator dal ckpt (completo) e costruisce il Predictor per l'inferenza.
    """
    est = _build_estimator(weights_ckpt_path, horizon, context_len, freq)
    try:
        pred = est.create_predictor(
            transformation=est.create_transformation(),
            module=est.create_lightning_module(),
            device=device,
        )
    except TypeError:
        pred = est.create_predictor(
            transformation=est.create_transformation(),
            module=est.create_lightning_module(),
        )
    if hasattr(pred, "num_parallel_samples"): pred.num_parallel_samples = int(num_samples)
    if hasattr(pred, "batch_size"): pred.batch_size = int(batch_size)
    return pred


# (Utility) Predici con un Predictor già pronto (foundation o FT)
from gluonts.model.predictor import Predictor as _GluonPredictor

def predict_series_with_predictor(
    predictor: _GluonPredictor,
    series: np.ndarray,
    horizon: int,
    *,
    freq: str = "D",
    start: str | pd.Timestamp = "1970-01-01",
) -> np.ndarray:
    arr = np.asarray(series, dtype=float).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError("La serie contiene NaN/inf.")
    if arr.size < max(8, horizon):
        raise ValueError(f"Serie troppo corta per horizon={horizon}")
    if not freq or str(freq).strip() == "0":
        freq = "D"
    ds = ListDataset([{"target": arr, "start": pd.Timestamp(start)}], freq=freq)
    it = predictor.predict(ds)
    fc = next(iter(it), None)
    if fc is None:
        raise RuntimeError("Predictor non ha restituito forecast")
    if hasattr(fc, "mean"): yhat = np.asarray(fc.mean, dtype=float)
    elif hasattr(fc, "quantile"): yhat = np.asarray(fc.quantile("0.5"), dtype=float)
    elif hasattr(fc, "samples"): yhat = np.asarray(fc.samples, dtype=float).mean(axis=0)
    else: yhat = np.asarray(fc, dtype=float)
    if yhat.ndim > 1: yhat = yhat[0]
    if yhat.shape[0] < horizon:
        yhat = np.concatenate([yhat, np.full(horizon - yhat.shape[0], np.nan)])
    return yhat[:horizon]
# ============================================================================#