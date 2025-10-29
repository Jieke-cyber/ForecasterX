# app/db.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv

# carica .env dalla root di backend/
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(env_path)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL non impostata. Metti l'URI del Pooler Supabase in backend/.env")

if ".pooler.supabase.com" not in DATABASE_URL:
    raise RuntimeError("DATABASE_URL non è del Pooler. Usa l'URI '...pooler.supabase.com:6543/... ?sslmode=require'")

if "sslmode=" not in DATABASE_URL:
    raise RuntimeError("DATABASE_URL senza sslmode. Aggiungi '?sslmode=require'")

# deleghiamo il pooling a PgBouncer
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    poolclass=NullPool,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
