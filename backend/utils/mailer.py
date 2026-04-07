from __future__ import annotations

import logging
import mimetypes
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

from config import CFG

logger = logging.getLogger(__name__)


def send_image_via_email(image_path: str | Path, subject: str, body: str = "") -> None:
    if not bool(CFG.get("ENABLE_EMAIL", False)):
        logger.info("Email delivery is disabled; skipping send.")
        return

    smtp_host = str(CFG.get("EMAIL_SMTP_HOST", "") or "")
    smtp_port = int(CFG.get("EMAIL_SMTP_PORT", 465))
    smtp_user = str(CFG.get("EMAIL_USER", "") or "")
    smtp_password = str(CFG.get("EMAIL_PASSWORD", "") or "").strip().replace(" ", "")
    email_from = str(CFG.get("EMAIL_FROM", "") or smtp_user)
    email_to = str(CFG.get("EMAIL_TO_PHONE", "") or "")
    use_tls = bool(CFG.get("EMAIL_USE_TLS", False))

    missing = [
        name
        for name, value in {
            "EMAIL_SMTP_HOST": smtp_host,
            "EMAIL_USER": smtp_user,
            "EMAIL_PASSWORD": smtp_password,
            "EMAIL_TO_PHONE": email_to,
        }.items()
        if not value
    ]
    if missing:
        logger.warning("Email config missing; skipping send. missing=%s", ",".join(missing))
        return

    p = Path(image_path)
    if not p.exists():
        logger.warning("Email attachment not found; skipping send. file=%s", p.name)
        return

    logger.debug("Sending email attachment. file=%s tls=%s", p.name, use_tls)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = email_to
    msg.set_content(body or "Gonderilen gorsel ektedir.")

    ctype, _ = mimetypes.guess_type(str(p))
    maintype, subtype = (ctype.split("/", 1) if ctype else ("application", "octet-stream"))
    msg.add_attachment(p.read_bytes(), maintype=maintype, subtype=subtype, filename=p.name)

    if use_tls:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as s:
            s.starttls(context=ssl.create_default_context())
            s.login(smtp_user, smtp_password)
            s.send_message(msg)
    else:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20) as s:
            s.login(smtp_user, smtp_password)
            s.send_message(msg)
