# PathFinder-Ship

Minimal local setup notes for the current backend/frontend prototype.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create local config without committing secrets:

```powershell
Copy-Item .env.example backend\.env
```

Fill `backend/.env` with local SMTP values only if email sending is needed. Keep `ENABLE_EMAIL=false` unless those values are ready.

## Run

Backend:

```powershell
Set-Location backend
python .\main.py
```

Frontend:

```powershell
Set-Location frontend
python -m http.server 5173
```

Open `http://localhost:5173`.

## Readiness

With the backend running:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/readiness
```

The readiness check reports whether config is loaded and whether expected T5, NLU, YOLO, RAG corpus, Chroma, and SQLite paths exist. It does not load model files and does not return secret values.

## Prompt 9 Manual Smoke Checks

With the backend and frontend running:

- Chat message -> `/api/run` returns `final_answer` and the UI shows it.
- RAG-style question -> `/api/run` returns `final_answer` and the UI shows it.
- Open camera command -> `/api/run` returns `client_action=open_camera` and the browser camera preview opens.
- Close camera command -> `/api/run` returns `client_action=close_camera` and the browser camera preview closes.
- Capture/detect command -> frontend captures or uses the uploaded image, then sends it to `/api/detect`.
- `/api/run` failure -> the UI shows a safe error or uses the legacy endpoint fallback without dumping internal JSON.

## Path Rules

Relative model, corpus, index, photo, and detect paths are resolved from `backend/`.

Default locations:

- T5: `backend/assets/models/t5`
- NLU: `backend/assets/models/nlu`
- YOLO: `backend/assets/models/yolo_nas`
- RAG corpus: `backend/data/rag/corpus`
- RAG index: `backend/assets/rag/chroma_db`

This step only prepares the project for safer local execution and configuration. Diagent integration is not implemented here.
