import uuid

from sqlalchemy import Column, String, DateTime, JSON, ForeignKey, func
from sqlalchemy.orm import relationship
from .db import Base

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    path = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    owner_email = Column(String, nullable=False, index=True)

class TrainingRun(Base):
    __tablename__ = "training_runs"
    id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False)
    status = Column(String, nullable=False, default="PENDING")
    metrics_json = Column(JSON, nullable=True)
    error = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    model_id_used = Column(String, ForeignKey("models.id"), nullable=True)
    celery_task_id = Column(String(255), nullable=True)

    dataset = relationship("Dataset", backref="runs")

class ForecastPlot(Base):
    __tablename__ = "forecast_plots"
    id = Column(String, primary_key=True)
    training_run_id = Column(String, ForeignKey("training_runs.id"), nullable=True)
    name = Column(String, nullable=False)
    path = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    owner_email = Column(String, nullable=False, index=True)

    run = relationship("TrainingRun", backref="plots")



class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

def _uuid() -> str:
    return str(uuid.uuid4())

class Model(Base):
    __tablename__ = "models"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    kind = Column(String, nullable=False)
    base_model = Column(String, nullable=False)

    storage_path = Column(String, nullable=True)

    params_json = Column(JSON, nullable=True)
    metrics_json = Column(JSON, nullable=True)

    owner_email = Column(String, nullable=True, index=True)

    version = Column(String, nullable=True)
    source_uri = Column(String, nullable=True)
    status = Column(String, nullable=False, default="AVAILABLE")

    created_at = Column(DateTime, server_default=func.now())