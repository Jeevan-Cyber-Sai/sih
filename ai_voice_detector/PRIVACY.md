# Privacy & Data Handling

This document describes what the AI Voice Detector retains, what it
never retains, and how that supports data-protection compliance (e.g.
GDPR-style purpose limitation and data minimization principles).

## Inference runs locally

All voice-cloning detection happens on-device. No audio is sent to any
third-party API or external service for scoring. The model files
(`models/*.joblib`) are loaded once at process startup and run entirely
within this process — nothing about the audio content leaves the machine
running this software.

## What is retained

**By default: nothing derived from a specific call is retained.**

- Uploaded audio (`/analyze`, `/analyze_stream`) is written to a temporary
  file only for the duration of the request, and deleted immediately
  after scoring completes — see the `finally` blocks in `app.py`.
- In-memory audio buffers (the decoded waveform, and each streaming
  chunk) are explicitly zeroed out (`privacy.zero_buffer()`) as soon as
  scoring is done with them, rather than left to linger until garbage
  collection.
- Live capture (`live_capture.py`) never writes captured audio to disk at
  all — chunks exist only as in-memory arrays during scoring, and are
  dropped (not queued) if the previous chunk is still being processed.

If `privacy.retention_mode` in `config.yaml` is set to `"features_only"`,
one thing changes: a structured risk event is appended to
`logs/risk_events.jsonl` per analysis. That event is filtered through an
explicit allowlist (`privacy._ALLOWED_LOG_KEYS`) containing only:

- `timestamp`
- `voice_risk`, `context_risk`, `final_risk`
- `risk_level`
- `profile`
- `context_flags` (booleans/categories only — e.g. `new_beneficiary:
  true`, never the raw transaction amount or caller number)

## What is never retained, in any mode

- Raw audio bytes or waveforms
- Transcripts or any derived text content
- Uploaded filenames or file paths
- Caller phone numbers, account numbers, or transaction amounts (only
  derived boolean flags, e.g. "was this a large transaction", not the
  figure itself)
- Anything that could be used to reconstruct or replay the original audio

The logging allowlist in `privacy.py` is the actual enforcement
mechanism, not just a policy statement: even if an upstream caller adds
an unexpected field to an event dict (a stray file path, a name), it is
silently dropped before the log line is written, because only keys in
`_ALLOWED_LOG_KEYS` are ever included.

## Retention modes

Set in `config.yaml` under `privacy.retention_mode`:

| mode | behavior |
|---|---|
| `none` (default) | No persistent logging at all. `log_risk_event()` is a no-op. |
| `features_only` | Persists the feature-level risk event described above to `logs/risk_events.jsonl`. Still never audio. |

## Why this matters for compliance

Because scoring never persists raw audio and logging (when enabled) is
restricted to numeric risk scores and boolean context flags, this system
avoids processing or storing personal biometric voiceprint data beyond
the moment of inference. That significantly narrows the compliance
surface compared to a system that stores call recordings or transcripts
— there is no audio data at rest to secure, encrypt, retain-limit, or
respond to a deletion request for, because none is kept in the first
place.
