# app/models_runtime/pypots_runtime.py

from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from pypots.forecasting import DLinear, TimesNet, FITS

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "Modelli"
PYPOTS_MODELS: dict[str, dict] = {}


def load_artifact(path):
    return torch.load(str(path), map_location="cpu")


def get_torch_module(model):
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "network"):
        return model.network
    import torch as _torch
    if isinstance(model, _torch.nn.Module):
        return model
    raise ValueError(f"Non trovo modulo torch interno per {type(model)}")


def build_model_from_artifact(artifact: dict):
    model_type = artifact["model_type"]
    init_kwargs = artifact["init_kwargs"]

    if model_type == "DLinear":
        model = DLinear(**init_kwargs)
    elif model_type == "TimesNet":
        model = TimesNet(**init_kwargs)
    elif model_type == "FITS":
        model = FITS(**init_kwargs)
    else:
        raise ValueError(f"Model type sconosciuto: {model_type}")

    core = get_torch_module(model)
    core.load_state_dict(artifact["state_dict"])
    if hasattr(core, "eval"):
        core.eval()
    return model


def build_scaler_from_artifact(artifact: dict) -> StandardScaler:
    sc = StandardScaler()
    sc.mean_ = np.array(artifact["scaler"]["mean"])
    sc.scale_ = np.array(artifact["scaler"]["scale"])
    return sc


def preload_pypots_models() -> None:
    if not ARTIFACT_DIR.exists():
        print(f"[PyPOTS] Nessuna cartella Modelli in {ARTIFACT_DIR}")
        return

    for path in ARTIFACT_DIR.glob("*.pt"):
        artifact = load_artifact(path)
        model = build_model_from_artifact(artifact)
        scaler = build_scaler_from_artifact(artifact)

        pattern = artifact.get("pattern", path.stem)
        model_type = artifact.get("model_type", "unknown")
        key = f"{pattern}_{model_type}"   # es: pattern1_DLinear

        PYPOTS_MODELS[key] = {
            "artifact": artifact,
            "model": model,
            "scaler": scaler,
            "L": artifact["L"],
            "H": artifact["H"],
            "path": str(path),
        }

        print(f"[PyPOTS] Caricato modello '{key}' da {path}")

def get_pypots_model(key: str) -> Dict[str, Any]:
    """
    Recupera un modello dalla cache (da usare negli endpoint).
    """
    if key not in PYPOTS_MODELS:
        raise KeyError(f"Modello PyPOTS non trovato in cache: {key}")
    return PYPOTS_MODELS[key]

def _as_forecast_array(obj, H: int) -> np.ndarray:
    """Adatta l'output del modello a un array [H]."""
    if isinstance(obj, dict):
        for k in ("forecast", "y", "pred", "output"):
            if k in obj:
                return _as_forecast_array(obj[k], H)
        for v in obj.values():
            try:
                return _as_forecast_array(v, H)
            except Exception:
                continue
        raise TypeError("Predict ha restituito un dict senza array utilizzabili.")

    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()

    arr = np.asarray(obj)
    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], arr.shape[1])
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr.reshape(arr.shape[1])
    return arr.astype("float32")


def predict_future(model, x_last: np.ndarray, H: int) -> np.ndarray:
    """
    x_last: [1, L, 1] -> ritorna ndarray [H] (float32) in scala NORMALIZZATA.
    """
    try:
        with torch.no_grad():
            out = model(torch.tensor(x_last, dtype=torch.float32))
        return _as_forecast_array(out, H)
    except Exception:
        x_pred = np.zeros((1, H, 1), dtype=x_last.dtype)
        out = model.predict({"X": x_last, "X_pred": x_pred})
        return _as_forecast_array(out, H)