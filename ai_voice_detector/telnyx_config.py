"""
Telnyx credentials, read from environment variables only -- never
hardcoded, matching this project's existing privacy/config conventions
(see PRIVACY.md, config.yaml, twilio_config.py).
"""
import os

TELNYX_API_KEY = os.environ.get("TELNYX_API_KEY")

# Needed only for the dial+bridge flow (connecting the inbound caller to a
# second real party) -- not secrets exactly, but account-specific config
# that has no business being hardcoded into a script pushed to a public
# GitHub repo.
TELNYX_CONNECTION_ID = os.environ.get("TELNYX_CONNECTION_ID")
TELNYX_FROM_NUMBER = os.environ.get("TELNYX_FROM_NUMBER")
TELNYX_BRIDGE_TO_NUMBER = os.environ.get("TELNYX_BRIDGE_TO_NUMBER")

_REQUIRED = {"TELNYX_API_KEY": TELNYX_API_KEY}
_REQUIRED_FOR_BRIDGE = {
    "TELNYX_CONNECTION_ID": TELNYX_CONNECTION_ID,
    "TELNYX_FROM_NUMBER": TELNYX_FROM_NUMBER,
    "TELNYX_BRIDGE_TO_NUMBER": TELNYX_BRIDGE_TO_NUMBER,
}


def validate_bridge_config():
    missing = [name for name, val in _REQUIRED_FOR_BRIDGE.items() if not val]
    if missing:
        raise RuntimeError(
            "Missing required Telnyx bridge environment variable(s): " + ", ".join(missing) +
            ". Set them before starting the server -- see README.md."
        )
    return True


def validate_config():
    """Raises RuntimeError listing every missing variable, so a
    misconfiguration fails loudly at startup instead of surfacing as a
    confusing error the first time a real call comes in."""
    missing = [name for name, val in _REQUIRED.items() if not val]
    if missing:
        raise RuntimeError(
            "Missing required Telnyx environment variable(s): " + ", ".join(missing) +
            ". Set them before starting the server -- see README.md."
        )
    return True
