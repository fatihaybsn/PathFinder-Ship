from typing import List, Dict
import os, docx, fitz
from pathlib import Path

import glob
from config import CFG

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def read_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def load_documents_from_folder(folder_path=None):
    folder = str(folder_path or CFG.get("RAG_CORPUS_DIR", "data/rag/corpus"))
    documents = []
    for file_path in glob.glob(os.path.join(folder, "*")):
        suffix = Path(file_path).suffix.lower()
        if suffix == ".txt":
            content = read_txt(file_path)
        elif suffix == ".docx":
            content = read_docx(file_path)
        elif suffix == ".pdf":
            content = read_pdf(file_path)
        else:
            continue
        documents.append({"file_name": os.path.basename(file_path), "content": content})
    return documents
