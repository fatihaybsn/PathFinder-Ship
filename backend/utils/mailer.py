# backend/utils/mailer.py
from __future__ import annotations
import os, smtplib, mimetypes, ssl
from email.message import EmailMessage
from pathlib import Path

def _truth(x: str) -> bool:
    return str(x).strip().lower() in ("1", "true", "yes", "y", "on")

def send_image_via_email(image_path: str | Path, subject: str, body: str = "") -> None:
    # ENV'leri fonksiyon içinde oku (import sırasına bağımlı kalma)
    SMTP_HOST = os.getenv("EMAIL_SMTP_HOST", "")
    SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "465"))
    SMTP_USER = os.getenv("EMAIL_USER", "")
    SMTP_PASSWORD = (os.getenv("EMAIL_PASSWORD", "") or "").strip().replace(" ", "")
    EMAIL_FROM = os.getenv("EMAIL_FROM") or SMTP_USER
    EMAIL_TO = os.getenv("EMAIL_TO_PHONE", "")
    USE_TLS = _truth(os.getenv("EMAIL_USE_TLS", "0"))  # 587 ise 1 yap

    # --- HIZLI TANI (DEBUG) BLOĞUNU BURAYA YAPIŞTIR ---
    print("[mailer] host:", SMTP_HOST, "port:", SMTP_PORT, "tls:", USE_TLS)
    print("[mailer] user:", SMTP_USER, "from:", EMAIL_FROM, "to:", EMAIL_TO)
    print("[mailer] pwd_len:", len(SMTP_PASSWORD), "file:", str(image_path))
    # (DİKKAT: Parolayı ASLA yazdırma; sadece uzunluğunu gösteriyoruz.)

    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD or not EMAIL_TO:
        print("[mailer] Email config missing; skipping send.")
        return

    p = Path(image_path)
    if not p.exists():
        print(f"[mailer] Image not found: {p}")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.set_content(body or "Gönderilen görsel ektedir.")

    ctype, _ = mimetypes.guess_type(str(p))
    maintype, subtype = (ctype.split("/", 1) if ctype else ("application", "octet-stream"))
    msg.add_attachment(p.read_bytes(), maintype=maintype, subtype=subtype, filename=p.name)

    # Tanı için istersen aç (parola gösterme!)
    # print("[mailer] host:", SMTP_HOST, "port:", SMTP_PORT, "tls:", USE_TLS)
    # print("[mailer] user:", SMTP_USER, "from:", EMAIL_FROM, "to:", EMAIL_TO)
    # print("[mailer] pwd_len:", len(SMTP_PASSWORD))

    if USE_TLS:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
            s.starttls(context=ssl.create_default_context())
            s.login(SMTP_USER, SMTP_PASSWORD)
            s.send_message(msg)
    else:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=20) as s:
            s.login(SMTP_USER, SMTP_PASSWORD)
            s.send_message(msg)
