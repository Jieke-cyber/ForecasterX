from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List


class JobResult(BaseModel):
    metrics: Optional[Dict[str, Any]] = None
    plot: Optional[Dict[str, Any]] = None

class JobStatus(BaseModel):
    job_id: str
    status: str
    result: Optional[JobResult] = None

class RecentPlot(BaseModel):
    id: str
    plot_json: Dict[str, Any]
    created_at: str


class ZeroShotPredictIn(BaseModel):
    dataset_id: str
    horizon: int = 48
    context_len: int = 512

class FinetuneIn(BaseModel):
    dataset_id: str
    epochs: int = Field(ge=1, le=500, description="Numero di epoche di fine-tuning")
    horizon: int = Field(60, ge=1, le=10000)
    context_len: int = Field(256, ge=1, le=100000)
    lr: float = Field(1e-4, gt=0)
    aug_prob: float = Field(0.2, ge=0, le=1.0)


class PredictFTSaveIn(BaseModel):
    dataset_id: str
    horizon: int
    context_len: int


class TrainRequest(BaseModel):
    # Campo esistente per la lista di ID
    dataset_ids: List[str] = Field(..., min_length =1)

    # Campo di tracciamento esistente
    main_dataset_id: str | None = None

    # Campo esistente per l'orizzonte di previsione
    horizon: int = Field(ge=1, le=2000, default=24)

    # 👈 CAMPO AGGIUNTO
    model_name: str | None = None
