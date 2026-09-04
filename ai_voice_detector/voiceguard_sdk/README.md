# VoiceGuard SDK

A lightweight Python client for the VoiceGuard voice-clone/impersonation
detection engine. Talks to a running VoiceGuard server over gRPC (default,
port 50051) or REST (port 5000), and hides that choice completely behind
plain dataclasses and a small set of exceptions — integrating systems
never touch raw gRPC messages, REST JSON, or transport-level errors.

Only three dependencies: `grpcio`, `requests`, `numpy`. The heavy ML
stack (librosa, torch, transformers) stays server-side, where it belongs
— this SDK is meant to be installed in a bank's application tier, not
next to a GPU.

## Install

```bash
pip install -e .   # from the project root, editable install for development
```

## Quick start

```python
from voiceguard_sdk import VoiceGuardClient

client = VoiceGuardClient(host="localhost")  # gRPC by default

result = client.analyze(
    "incoming_call.wav",
    profile="high_value_transaction",
    context={"caller_known": False, "transaction_amount": 500000, "new_beneficiary": True},
    speaker_id="CFO_Rajesh",
)

if result.risk_level == "HIGH":
    print(result.recommended_action)
```

## API

- `VoiceGuardClient(host="localhost", port=50051, use_grpc=True, rest_port=5000, timeout=15)`
- `client.analyze(audio_path, profile="routine", context=None, speaker_id=None) -> RiskResponse`
- `client.enroll_speaker(speaker_id, audio_path) -> SpeakerProfile`
- `client.verify_speaker(speaker_id, audio_path) -> ConsistencyResponse`
- `client.get_profiles() -> list[str]`
- `client.test_connection() -> bool`

Every method raises `voiceguard_sdk.VoiceGuardException` (or a subclass:
`ConnectionError`, `AudioFormatError`, `SpeakerNotFoundError`) on failure
— never a raw `grpc.RpcError` or `requests` exception.

## Audio format note

The gRPC path (`use_grpc=True`, the default) reads audio via Python's
stdlib `wave` module only — **uncompressed WAV**, not MP3/FLAC — since
this package deliberately doesn't depend on librosa. The REST path
(`use_grpc=False`) has no such limit: it uploads the raw file bytes and
lets the server (which already has librosa) decode whatever format it's
given.

## CLI

Installing this package also installs a `voiceguard` command:

```bash
voiceguard analyze audio.wav --profile high_value_transaction
voiceguard enroll --speaker-id "CFO_Rajesh" --audio cfo_sample.wav
voiceguard verify --speaker-id "CFO_Rajesh" --audio live_call.wav
voiceguard test-connection
```

Add `--rest` to any command to use REST instead of gRPC, or `--host`/`--port`
to point at a different server.
