from typing import List, Dict
import os, docx, fitz

import glob

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

def load_documents_from_folder(folder_path="data/rag/corpus"):
    documents = []
    for file_path in glob.glob(os.path.join(folder_path, "*")):
        if file_path.endswith(".txt"):
            content = read_txt(file_path)
        elif file_path.endswith(".docx"):
            content = read_docx(file_path)
        elif file_path.endswith(".pdf"):
            content = read_pdf(file_path)
        else:
            continue
        documents.append({"file_name": os.path.basename(file_path), "content": content})
    return documents