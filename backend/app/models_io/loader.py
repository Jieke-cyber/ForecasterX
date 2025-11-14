# backend/app/models_io/loader.py
from pathlib import Path
import json

BAKED = Path("/app/app/models_runtime")
LOCAL = Path("/app/models_local")
CACHE = Path("/app/cache/models")

def _load_registry() -> dict:
    reg = BAKED / "registry.json"
    if not reg.exists():
        return {}  # nessun modello registrato
    try:
        with open(reg) as f:
            return json.load(f) or {}
    except Exception:
        return {}

def list_registered_models() -> list[dict]:
    reg = _load_registry()
    return [
        {
            "id": mid,
            "framework": meta.get("framework"),
            "path": meta.get("path"),
            "config": meta.get("config"),
            "description": meta.get("description", "")
        }
        for mid, meta in reg.items()
    ]

def get_model_metadata(model_id: str) -> dict:
    reg = _load_registry()
    if model_id not in reg:
        raise KeyError(f"Model ID non registrato: {model_id}")
    return reg[model_id]
