from __future__ import annotations
import os, uvicorn
if __name__ == "__main__":
    uvicorn.run("web.app:app", host=os.getenv("API_HOST","0.0.0.0"), port=int(os.getenv("API_PORT","8000")), reload=True)
