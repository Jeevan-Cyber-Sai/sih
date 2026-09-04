"""
Twilio credentials, read from environment variables only -- never
hardcoded, matching this project's existing privacy/config conventions
(see PRIVACY.md, config.yaml).
"""
import os

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")

_REQUIRED = {
    "TWILIO_ACCOUNT_SID": TWILIO_ACCOUNT_SID,
    "TWILIO_AUTH_TOKEN": TWILIO_AUTH_TOKEN,
    "TWILIO_PHONE_NUMBER": TWILIO_PHONE_NUMBER,
}


def validate_config():
    """Raises RuntimeError listing every missing variable, so a
    misconfiguration fails loudly at startup instead of surfacing as a
    confusing error the first time a real call comes in."""
    missing = [name for name, val in _REQUIRED.items() if not val]
    if missing:
        raise RuntimeError(
            "Missing required Twilio environment variable(s): " + ", ".join(missing) +
            ". Set them before starting the server -- see README.md."
        )
    return True
