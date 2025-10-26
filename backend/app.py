from fastapi import FastAPI
app = FastAPI()
@app.get("/health")
def ok():
    return {"ok": True}
