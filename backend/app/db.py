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
    raise RuntimeError(
        "DATABASE_URL non impostata. Metti l'URI del Pooler Supabase in backend/.env"
    )

# Esempio valido: postgresql+psycopg2://<user>:<pw>@<project>.pooler.supabase.com:6543/postgres?sslmode=require
if ".pooler.supabase.com" not in DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL non è del Pooler. Usa l'URI '...pooler.supabase.com:6543/... ?sslmode=require'"
    )

if "sslmode=" not in DATABASE_URL:
    raise RuntimeError("DATABASE_URL senza sslmode. Aggiungi '?sslmode=require'")

# Opzioni keepalive per ridurre chiusure inattese della connessione
CONNECT_ARGS = {
    "keepalives": 1,
    "keepalives_idle": 30,      # secondi di inattività prima del primo keepalive
    "keepalives_interval": 10,  # intervallo tra keepalive
    "keepalives_count": 5,      # tentativi prima di considerare la connessione morta
    # facoltativo ma utile:
    "application_name": "ts-webapp-backend",
}

# Con PgBouncer (pooler di Supabase) usiamo NullPool: ogni Session apre/chiude la sua connessione
engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,
    pool_pre_ping=True,   # ping prima di usare la connessione
    connect_args=CONNECT_ARGS,
)

# expire_on_commit=False evita ricarichi impliciti dopo commit (più sicuro con PgBouncer)
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
