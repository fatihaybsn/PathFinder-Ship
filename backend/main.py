import uvicorn
from config import CFG

if __name__ == "__main__":
    host = str(CFG.get("API_HOST", "0.0.0.0"))
    port = int(CFG.get("API_PORT", 8000))
    uvicorn.run("web.app:app", host=host, port=port, reload=True)
