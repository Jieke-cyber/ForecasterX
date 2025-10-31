# app/supa.py
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") # service role lato server
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")
SUPABASE_BUCKET_PLOTS = os.getenv("SUPABASE_BUCKET_PLOTS")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Config Supabase mancante (SUPABASE_URL / SUPABASE_SERVICE_KEY).")

def supa() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)
