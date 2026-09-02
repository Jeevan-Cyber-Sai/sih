"""
Privacy/compliance layer: enforces that raw audio never persists beyond a
single request's lifetime, and that any logging captures only risk-score
metadata -- never audio, never transcripts, never file contents.

See PRIVACY.md for the full policy this module implements.
"""
import json
import os
import time

from risk_engine import load_config

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, "logs")
LOG_PATH = os.path.join(LOG_DIR, "risk_events.jsonl")

# Hard allowlist: only these keys are ever written to the log, regardless
# of what a caller passes in. This is the actual enforcement mechanism --
# not a promise, a filter.
_ALLOWED_LOG_KEYS = {
    "timestamp", "voice_risk", "context_risk", "final_risk",
    "risk_level", "profile", "context_flags", "chunk_index",
}


def get_retention_mode():
    return load_config().get("privacy", {}).get("retention_mode", "none")


def zero_buffer(audio_array):
    """Best-effort in-place zeroing of an audio buffer once inference is
    done with it, so the raw waveform doesn't linger in memory/GC churn
    longer than necessary. This is defense-in-depth, not a substitute for
    never persisting the buffer in the first place (we don't)."""
    try:
        audio_array[:] = 0
    except Exception:
        pass  # not every caller passes a mutable array; never fail scoring over this


def log_risk_event(event: dict):
    """Structured, feature-only logging. Filters to an explicit allowlist
    of keys -- timestamp, risk scores, risk level, profile, context FLAGS
    -- so no accidental future field (a file path, a transcript, raw
    audio) can leak into the log just by being added to the event dict
    upstream. No-ops entirely when retention_mode is "none" (the default).
    """
    if get_retention_mode() == "none":
        return None

    safe_event = {k: v for k, v in event.items() if k in _ALLOWED_LOG_KEYS}
    safe_event.setdefault("timestamp", time.time())

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(safe_event) + "\n")

    return safe_event
