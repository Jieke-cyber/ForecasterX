from pydantic import BaseModel, Field
from typing import Optional, Any, Dict

class TrainRequest(BaseModel):
    dataset_id: str
    horizon: int = Field(ge=1, le=2000, default=24)

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
