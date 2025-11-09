# app/models.py
from sqlalchemy import Column, String, DateTime, JSON, ForeignKey, func
from sqlalchemy.orm import relationship
from .db import Base

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    path = Column(String, nullable=False)       # percorso del CSV su disco/bucket
    created_at = Column(DateTime, server_default=func.now())
    # (niente 'points' qui)
    owner_email = Column(String, nullable=False, index=True)  # <— nuovo campo

class TrainingRun(Base):
    __tablename__ = "training_runs"
    id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False)
    status = Column(String, nullable=False, default="PENDING")
    metrics_json = Column(JSON, nullable=True)
    error = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    dataset = relationship("Dataset", backref="runs")

class ForecastPlot(Base):
    __tablename__ = "forecast_plots"
    id = Column(String, primary_key=True)
    training_run_id = Column(String, ForeignKey("training_runs.id"), nullable=False)
    name = Column(String, nullable=False)        # es. forecast_<run>.csv
    path = Column(String, nullable=False)        # chiave o URL del CSV nello storage
    created_at = Column(DateTime, server_default=func.now())

    owner_email = Column(String, nullable=False, index=True)

    run = relationship("TrainingRun", backref="plots")



class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)  # come i tuoi dataset: stringa
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())