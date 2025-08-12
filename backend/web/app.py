from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import List, Optional
import os, cv2, numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from config import CFG
from services.nlu_classifier import NLUClassifier
from services.t5 import T5Service
from services.yolo import YOLOService
from utils.vision import draw_dets
class IntentRequest(BaseModel): text:str
class IntentResponse(BaseModel): intent:str; score:float; threshold:float; narration:Optional[str]=None
class ChatRequest(BaseModel): message:str
class ChatResponse(BaseModel): answer:str
class DetectResponse(BaseModel): labels:List[str]; summary:str; boxes:Optional[List[List[float]]]=None; image_url:Optional[str]=None; narration:Optional[str]=None
app=FastAPI(title="PathFinder-Ship Web API"); Path('data').mkdir(exist_ok=True); app.mount('/static',StaticFiles(directory='data'),name='static')
app.add_middleware(CORSMiddleware,allow_origins=[CFG.get('FRONTEND_ORIGIN') or os.getenv('FRONTEND_ORIGIN','*')],allow_methods=['*'],allow_headers=['*'],allow_credentials=True)
NLU:Optional[NLUClassifier]=None; T5:Optional[T5Service]=None; YOLO:Optional[YOLOService]=None
@app.on_event('startup')
def startup_event():
    global NLU,T5,YOLO; NLU=NLUClassifier(CFG); T5=T5Service(CFG); YOLO=YOLOService(CFG)
@app.get('/api/health')
def health(): return {'ok':True}
@app.post('/api/intent',response_model=IntentResponse)
def intent_api(body:IntentRequest):
    label,score=NLU.predict(body.text); thr=float(CFG.get('CLS_ROUTE_THRESHOLD',0.60)); return {'intent':label,'score':float(score),'threshold':thr,'narration':None}
@app.post('/api/chat',response_model=ChatResponse)
def chat_api(body:ChatRequest): return {'answer':T5.chat(body.message)}
@app.post('/api/photo')
async def take_photo_api(background_tasks:BackgroundTasks,file:UploadFile=File(...)):
    out=Path(CFG.get('PHOTO_DIR','data/web_out/photo')); out.mkdir(parents=True,exist_ok=True); target=out/'photo_latest.jpg'; target.write_bytes(await file.read()); return {'ok':True,'stored':str(target)}
@app.post('/api/detect',response_model=DetectResponse)
async def detect_api(background_tasks:BackgroundTasks,file:UploadFile=File(...),draw:int=Form(1)):
    data=await file.read(); img=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    boxes,labels,scores,cls_ids=YOLO.detect_from_bgr(img); summary=', '.join([f'{c} {n}' for c,n in Counter(labels).items()]) if labels else 'no objects'
    return {'labels':labels,'summary':summary,'boxes':boxes,'image_url':None,'narration':None}
