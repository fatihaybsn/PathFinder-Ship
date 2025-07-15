from __future__ import annotations
import os
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config import CFG
from services.nlu_classifier import NLUClassifier
from services.t5 import T5Service
class IntentRequest(BaseModel): text:str
class IntentResponse(BaseModel): intent:str; score:float; threshold:float; narration:Optional[str]=None
class ChatRequest(BaseModel): message:str
class ChatResponse(BaseModel): answer:str
app=FastAPI(title="PathFinder-Ship Web API")
app.add_middleware(CORSMiddleware,allow_origins=[CFG.get("FRONTEND_ORIGIN") or os.getenv("FRONTEND_ORIGIN","*")],allow_methods=["*"],allow_headers=["*"],allow_credentials=True)
NLU:Optional[NLUClassifier]=None; T5:Optional[T5Service]=None
@app.on_event("startup")
def startup_event():
    global NLU,T5; NLU=NLUClassifier(CFG); T5=T5Service(CFG)
@app.get("/api/health")
def health(): return {"ok":True}
@app.post("/api/intent",response_model=IntentResponse)
def intent_api(body:IntentRequest):
    label,score=NLU.predict(body.text); thr=float(CFG.get("CLS_ROUTE_THRESHOLD",0.60)); return {"intent":label,"score":float(score),"threshold":thr,"narration":None}
@app.post("/api/chat",response_model=ChatResponse)
def chat_api(body:ChatRequest): return {"answer":T5.chat(body.message)}
