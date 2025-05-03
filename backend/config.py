from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception: pass
def _s(k:str,d:Optional[str]=None):
    v=os.getenv(k,d); return None if v is None else str(v).strip()
def _i(k:str,d:int):
    try: return int(str(os.getenv(k,d)).strip())
    except Exception: return d
def _f(k:str,d:float):
    try: return float(str(os.getenv(k,d)).strip())
    except Exception: return d
def _b(k:str,d:bool): return str(os.getenv(k,str(d))).lower() in ("1","true","yes","y","on")
def _p(v): return None if not v else str(Path(v).as_posix())
def build_config()->Dict[str,Any]:
    return {"APP_NAME":_s("APP_NAME","PathFinder-Ship"),"DEFAULT_USER_NAME":_s("DEFAULT_USER_NAME","Passenger"),"BOT_NAME":_s("BOT_NAME","Passenger-Bot"),"DEBUG":_b("DEBUG",False),"API_HOST":_s("API_HOST","0.0.0.0"),"API_PORT":_i("API_PORT",8000),"FRONTEND_ORIGIN":_s("FRONTEND_ORIGIN","http://localhost:5173"),"PHOTO_DIR":_p(_s("PHOTO_DIR","data/web_out/photo")),"DETECT_DIR":_p(_s("DETECT_DIR","data/web_out/detect")),"MAX_FILES_PER_DIR":_i("MAX_FILES_PER_DIR",10)}
CFG=build_config()
