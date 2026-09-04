"""
Multi-channel alerting: email + SMS notifications to frontline staff when
impersonation risk crosses a configured threshold (config.yaml's
alert_on / sms_alert_on). Kept deliberately fail-open -- every public
function here swallows its own exceptions and just logs them, since a
broken SMTP/SMS config or a network blip must never take down live call
scoring. The two channels are independent: each has its own enable flag,
threshold list, and rate-limit bucket, so a HIGH risk event can fire an
email, an SMS, both, or neither, and one channel failing never blocks
the other.

Email has two send paths: plain smtplib+TLS (stdlib only, always
available) or SendGrid's API (used automatically when SENDGRID_API_KEY
is set) as a higher-reliability alternative for production use, where a
flaky SMTP connection is a worse failure mode than an API call.

SMS uses Fast2SMS's Quick SMS route (route=q) -- free-text messages, no
DLT template pre-registration needed, which is why it's the right choice
for demo use versus a template-gated transactional route.
"""
import logging
import os
import smtplib
import ssl
import threading
import time
from email.mime.text import MIMEText

import requests

logger = logging.getLogger(__name__)

RATE_LIMIT_SECONDS = 60

_last_email_time = {}  # call_id (or "_default") -> unix timestamp of last sent email
_last_email_lock = threading.Lock()
_last_sms_time = {}    # separate bucket -- email and SMS rate-limit independently
_last_sms_lock = threading.Lock()

FAST2SMS_URL = "https://www.fast2sms.com/dev/bulkV2"


def _get_alerting_config():
    # Deferred import: avoids a circular import with risk_engine.py,
    # which imports send_email_alert_async/send_sms_alert_async from
    # this module at its own top level.
    from risk_engine import load_config
    return load_config().get("alerting", {})


def _rate_limited(call_id, tracking, lock):
    """True if an alert was already sent (on this channel) for this
    call_id within the rate-limit window -- so a HIGH risk level held
    across many consecutive chunks of the SAME call fires one alert at
    first detection, not one per chunk. call_id is optional (falls back
    to a single global bucket) since the one-shot /analyze_with_context
    path has no natural per-call identifier the way a live phone call
    does. `tracking`/`lock` select which channel's independent bucket to
    check (see _last_email_time vs _last_sms_time)."""
    key = call_id or "_default"
    now = time.time()
    with lock:
        last = tracking.get(key)
        if last is not None and (now - last) < RATE_LIMIT_SECONDS:
            return True
        tracking[key] = now
        return False


def _build_alert_body(risk_level, voice_risk, final_risk, context_flags=None, profile=None,
                       speaker_id=None, consistency_risk=None, recommended_action=None):
    lines = [
        f"AI Voice Detector -- {risk_level} impersonation risk alert",
        "",
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Risk level: {risk_level}",
        f"Voice risk score: {voice_risk}",
        f"Final combined risk score: {final_risk}",
    ]
    if profile:
        lines.append(f"Active profile: {profile}")
    if recommended_action:
        lines.append(f"Recommended action: {recommended_action}")

    contributing = {k: v for k, v in (context_flags or {}).items() if v} if isinstance(context_flags, dict) else {}
    if contributing:
        lines.append("")
        lines.append("Contributing context flags:")
        for k, v in contributing.items():
            lines.append(f"  - {k}: {v}")

    if speaker_id:
        lines.append("")
        lines.append(f"Speaker ID checked: {speaker_id}")
        if consistency_risk is not None:
            lines.append(f"Speaker consistency risk: {consistency_risk} "
                         f"(0 = confirmed same person, 70 = likely different person)")

    return "\n".join(lines)


def _send_via_sendgrid(sender_email, recipients, subject, body):
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail

    message = Mail(from_email=sender_email, to_emails=recipients, subject=subject, plain_text_content=body)
    sg = SendGridAPIClient(os.environ["SENDGRID_API_KEY"])
    response = sg.send(message)
    if response.status_code >= 300:
        raise RuntimeError(f"SendGrid returned status {response.status_code}: {response.body}")


def _send_via_smtp(sender_email, sender_password, smtp_host, smtp_port, recipients, subject, body):
    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = ", ".join(recipients)

    tls_context = ssl.create_default_context()
    with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
        server.starttls(context=tls_context)
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipients, msg.as_string())


def _dispatch(subject, body, recipients):
    """Picks SendGrid if configured, else falls back to SMTP. Raises on
    failure -- callers decide whether that's fatal (send_test_alert) or
    just logged (send_email_alert)."""
    sender_email = os.environ.get("ALERT_SENDER_EMAIL")
    if not sender_email:
        raise RuntimeError("ALERT_SENDER_EMAIL environment variable is not set")

    if os.environ.get("SENDGRID_API_KEY"):
        _send_via_sendgrid(sender_email, recipients, subject, body)
        return "sendgrid"

    sender_password = os.environ.get("ALERT_SENDER_PASSWORD")
    if not sender_password:
        raise RuntimeError("ALERT_SENDER_PASSWORD environment variable is not set (and SENDGRID_API_KEY isn't either)")

    config = _get_alerting_config()
    smtp_host = config.get("smtp_host", "smtp.gmail.com")
    smtp_port = config.get("smtp_port", 587)
    _send_via_smtp(sender_email, sender_password, smtp_host, smtp_port, recipients, subject, body)
    return "smtp"


def send_email_alert(risk_level, voice_risk, final_risk, context_flags=None, speaker_id=None,
                      recommended_action=None, profile=None, consistency_risk=None, call_id=None):
    """Fire-and-forget: never raises. No-ops silently if email_enabled is
    false, this risk_level isn't in alert_on, or an alert was already
    sent for this call_id within the rate-limit window. Call via
    send_email_alert_async() from anywhere in the scoring pipeline so a
    slow/broken SMTP connection can never add latency there."""
    try:
        config = _get_alerting_config()
        if not config.get("email_enabled", False):
            return
        if risk_level not in config.get("alert_on", ["HIGH"]):
            return

        recipients = config.get("alert_recipients", [])
        if not recipients:
            logger.warning("alerting: email_enabled but no alert_recipients configured in config.yaml")
            return

        if _rate_limited(call_id, _last_email_time, _last_email_lock):
            logger.info("alerting: email rate-limited (call_id=%s), skipping duplicate %s alert", call_id, risk_level)
            return

        body = _build_alert_body(risk_level, voice_risk, final_risk, context_flags, profile,
                                  speaker_id, consistency_risk, recommended_action)
        subject = f"[AI Voice Detector] {risk_level} impersonation risk detected"
        via = _dispatch(subject, body, recipients)
        logger.info("alerting: sent %s alert to %d recipient(s) via %s", risk_level, len(recipients), via)
    except Exception:
        logger.exception("alerting: failed to send email alert (scoring pipeline unaffected)")


def send_email_alert_async(*args, **kwargs):
    """Thread wrapper so callers (risk_engine.compute_final_risk) never
    block the scoring pipeline on network/SMTP latency."""
    threading.Thread(target=send_email_alert, args=args, kwargs=kwargs, daemon=True).start()


def send_test_alert():
    """Bypasses email_enabled/alert_on/rate-limiting (all meant to
    filter real risk events) so this can verify SMTP/SendGrid config
    works on its own, independent of live scoring. Returns a result dict
    rather than raising, so the /test-alert route can report a clean
    error message instead of a 500."""
    config = _get_alerting_config()
    recipients = config.get("alert_recipients", [])
    if not recipients:
        return {"ok": False, "error": "no alert_recipients configured in config.yaml"}

    body = (
        "This is a test alert from the AI Voice Detector.\n\n"
        f"Sent at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        "If you received this, email alerting is configured correctly."
    )
    try:
        via = _dispatch("[AI Voice Detector] Test alert", body, recipients)
        return {"ok": True, "recipients": recipients, "via": via}
    except Exception as e:
        logger.exception("alerting: test alert failed")
        return {"ok": False, "error": str(e)}


def _build_sms_body(risk_level, final_risk, recommended_action=None):
    """Kept under 160 chars (single SMS segment) -- Fast2SMS's Quick SMS
    route sends free text as-is, no template, but a longer message either
    gets truncated by carriers or billed as multiple segments."""
    action = recommended_action or "Verify caller before proceeding."
    msg = f"VoiceGuard ALERT: {risk_level} risk call detected. Score: {final_risk:.0f}. Action: {action}"
    if len(msg) > 160:
        msg = msg[:157] + "..."
    return msg


def _send_via_fast2sms(api_key, numbers, message):
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    payload = {"route": "q", "message": message, "numbers": ",".join(numbers)}
    r = requests.post(FAST2SMS_URL, headers=headers, json=payload, timeout=15)
    data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    if r.status_code >= 300 or not data.get("return", False):
        raise RuntimeError(f"Fast2SMS returned status={r.status_code} body={r.text}")
    return data


def send_sms_alert(risk_level, final_risk, recommended_action=None, call_id=None):
    """Fire-and-forget: never raises. No-ops silently if sms_enabled is
    false, this risk_level isn't in sms_alert_on, or an SMS was already
    sent for this call_id within the rate-limit window. Independent of
    the email channel -- each has its own enable flag and rate-limit
    bucket, so a HIGH event can fire both, either, or neither depending
    on config. Call via send_sms_alert_async() so a slow/broken API call
    never adds latency to the scoring pipeline."""
    try:
        config = _get_alerting_config()
        if not config.get("sms_enabled", False):
            return
        if risk_level not in config.get("sms_alert_on", ["HIGH"]):
            return

        recipients = config.get("sms_recipients", [])
        if not recipients:
            logger.warning("alerting: sms_enabled but no sms_recipients configured in config.yaml")
            return

        if _rate_limited(call_id, _last_sms_time, _last_sms_lock):
            logger.info("alerting: SMS rate-limited (call_id=%s), skipping duplicate %s alert", call_id, risk_level)
            return

        api_key = os.environ.get("FAST2SMS_API_KEY")
        if not api_key:
            logger.error("alerting: FAST2SMS_API_KEY environment variable is not set")
            return

        message = _build_sms_body(risk_level, final_risk, recommended_action)
        _send_via_fast2sms(api_key, recipients, message)
        logger.info("alerting: sent %s SMS alert to %d recipient(s) via fast2sms", risk_level, len(recipients))
    except Exception:
        logger.exception("alerting: failed to send SMS alert (scoring pipeline unaffected)")


def send_sms_alert_async(*args, **kwargs):
    """Thread wrapper so callers (risk_engine.compute_final_risk) never
    block the scoring pipeline on network/API latency."""
    threading.Thread(target=send_sms_alert, args=args, kwargs=kwargs, daemon=True).start()


def send_test_sms():
    """Bypasses sms_enabled/sms_alert_on/rate-limiting (all meant to
    filter real risk events) so this can verify Fast2SMS config works on
    its own, independent of live scoring. Returns a result dict rather
    than raising, so the /test-sms route can report a clean error
    message instead of a 500."""
    config = _get_alerting_config()
    recipients = config.get("sms_recipients", [])
    if not recipients:
        return {"ok": False, "error": "no sms_recipients configured in config.yaml"}

    api_key = os.environ.get("FAST2SMS_API_KEY")
    if not api_key:
        return {"ok": False, "error": "FAST2SMS_API_KEY environment variable is not set"}

    message = "VoiceGuard: This is a test SMS. If you received this, SMS alerting is configured correctly."
    try:
        _send_via_fast2sms(api_key, recipients, message)
        return {"ok": True, "recipients": recipients, "via": "fast2sms"}
    except Exception as e:
        logger.exception("alerting: test SMS failed")
        return {"ok": False, "error": str(e)}
