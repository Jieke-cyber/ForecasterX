import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(env_path)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL non impostata. Metti l'URI del Pooler Supabase in backend/.env"
    )

if ".pooler.supabase.com" not in DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL non è del Pooler. Usa l'URI '...pooler.supabase.com:6543/... ?sslmode=require'"
    )

if "sslmode=" not in DATABASE_URL:
    raise RuntimeError("DATABASE_URL senza sslmode. Aggiungi '?sslmode=require'")

CONNECT_ARGS = {
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 5,
    "application_name": "ts-webapp-backend",
}

engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,
    pool_pre_ping=True,
    connect_args=CONNECT_ARGS,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
