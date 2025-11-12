# main.py (yeni)
import os
import uvicorn

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("web.app:app", host=host, port=port, reload=True)
