"""Outbound email — password resets and invites.

Stdlib smtplib only, matching auth.py's no-new-dependency rule. Mail is
optional: with no SMTP host configured (the pilot default) every send is a
no-op returning False, and the caller logs the link instead so an operator
can pass it on by hand. Nothing breaks; delivery just isn't automatic yet.

Configure in .env to turn it on:
    SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASSWORD,
    SMTP_FROM (defaults to SMTP_USER), SMTP_STARTTLS (default true)
"""
from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage

from . import config

log = logging.getLogger("hanuman.notify")


def configured() -> bool:
    return bool(config.SMTP_HOST and config.SMTP_FROM)


def _send(to: str, subject: str, body: str) -> bool:
    """Send one plain-text message. Returns True only if it was accepted.

    Never raises: a mail outage must not turn a password reset into a 500,
    and the caller falls back to logging the link.
    """
    if not configured():
        return False
    msg = EmailMessage()
    msg["From"] = config.SMTP_FROM
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=10) as s:
            if config.SMTP_STARTTLS:
                s.starttls(context=ssl.create_default_context())
            if config.SMTP_USER:
                s.login(config.SMTP_USER, config.SMTP_PASSWORD)
            s.send_message(msg)
        return True
    except Exception:                                  # noqa: BLE001
        log.exception("could not send mail to %s", to)
        return False


def send_password_reset(to: str, link: str) -> bool:
    return _send(
        to,
        "Reset your hanuman.ai password",
        "Someone asked to reset the password for this hanuman.ai account.\n\n"
        f"Choose a new one here:\n{link}\n\n"
        "The link works once and expires in 7 days.\n"
        "If this wasn't you, you can ignore this email — nothing has changed.\n",
    )


def send_invite(to: str, link: str, company_name: str = "") -> bool:
    at = f" at {company_name}" if company_name else ""
    return _send(
        to,
        "You've been added to hanuman.ai",
        f"You've been given access to the hanuman.ai dashboard{at}.\n\n"
        f"Choose your password here:\n{link}\n\n"
        "The link works once and expires in 7 days.\n",
    )
