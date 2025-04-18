from __future__ import annotations
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
app=FastAPI(title="PathFinder-Ship Web API")
app.add_middleware(CORSMiddleware,allow_origins=[os.getenv("FRONTEND_ORIGIN","*")],allow_methods=["*"],allow_headers=["*"],allow_credentials=True)
@app.get("/api/health")
def health(): return {"ok": True}
