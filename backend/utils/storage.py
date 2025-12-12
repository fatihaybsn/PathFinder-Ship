# backend/utils/storage.py
from __future__ import annotations
import os
from pathlib import Path

def ensure_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True) if Path(path).suffix else Path(path).mkdir(parents=True, exist_ok=True)

def _read_idx(idx_file: Path) -> int:
    try:
        return int(idx_file.read_text().strip())
    except Exception:
        return 0

def _write_idx(idx_file: Path, val: int) -> None:
    idx_file.write_text(str(val))

def save_with_ring_buffer(folder: str | Path, filename_prefix: str, ext: str, max_count: int) -> Path:
    """
    folder/ -> .ring.idx içinde bir sayaç tutar.
    Kaydetme için slot = (idx % max_count) + 1 kullanır.
    Geriye hedef dosya yolunu döndürür (üstüne yazılacak).
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    idx_file = folder / ".ring.idx"

    idx = _read_idx(idx_file)
    slot = (idx % max_count) + 1
    target = folder / f"{filename_prefix}_{slot:02d}.{ext.lstrip('.')}"
    _write_idx(idx_file, idx + 1)
    return target
