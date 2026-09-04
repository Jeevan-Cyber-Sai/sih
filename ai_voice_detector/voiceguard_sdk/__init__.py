"""
VoiceGuard SDK: a clean Python client for the VoiceGuard voice-clone/
impersonation detection engine. Integrating systems (banking cores, IVRs,
call centers) work with plain dataclasses and a small set of exceptions
-- never with raw gRPC protobuf messages, REST JSON, or transport-level
error types.
"""
from .client import VoiceGuardClient
from .exceptions import AudioFormatError, ConnectionError, SpeakerNotFoundError, VoiceGuardException
from .models import AudioAnalysisRequest, ConsistencyResponse, RiskResponse, SpeakerProfile

__version__ = "0.1.0"

__all__ = [
    "VoiceGuardClient",
    "VoiceGuardException",
    "ConnectionError",
    "AudioFormatError",
    "SpeakerNotFoundError",
    "AudioAnalysisRequest",
    "RiskResponse",
    "ConsistencyResponse",
    "SpeakerProfile",
]
