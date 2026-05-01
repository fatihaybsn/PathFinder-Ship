from typing import Any, Dict, List
import os, re
from pathlib import Path

import glob
from config import CFG

try:
    import docx
except Exception:
    docx = None

try:
    import fitz
except Exception:
    fitz = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf", ".html", ".htm"}

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def read_docx(file_path):
    if docx is None:
        raise RuntimeError("python-docx is not available")
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def read_pdf(file_path):
    if fitz is None:
        raise RuntimeError("PyMuPDF is not available")
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def read_html(file_path):
    raw = read_txt(file_path)
    if BeautifulSoup is not None:
        soup = BeautifulSoup(raw, "html.parser")
        return soup.get_text(" ", strip=True)
    return re.sub(r"<[^>]+>", " ", raw)

def load_document_from_file(file_path, metadata: Dict[str, Any] | None = None):
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        content = read_txt(path)
    elif suffix == ".docx":
        content = read_docx(path)
    elif suffix == ".pdf":
        content = read_pdf(path)
    elif suffix in {".html", ".htm"}:
        content = read_html(path)
    else:
        raise ValueError("unsupported_file_type")

    metadata = dict(metadata or {})
    return {
        "file_name": metadata.get("file_name") or path.name,
        "content": content,
        "metadata": metadata,
        **metadata,
    }

def load_documents_from_folder(folder_path=None):
    folder = str(folder_path or CFG.get("RAG_CORPUS_DIR", "data/rag/corpus"))
    documents = []
    for file_path in glob.glob(os.path.join(folder, "*")):
        if Path(file_path).suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        documents.append(load_document_from_file(file_path, metadata={"file_name": os.path.basename(file_path)}))
    return documents
