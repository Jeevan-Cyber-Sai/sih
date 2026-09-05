"""
Clean, dependency-free data classes for VoiceGuard SDK requests and
responses -- callers work with these, never with raw gRPC protobuf
messages or REST JSON dicts, so the transport (gRPC vs REST) is fully
hidden behind VoiceGuardClient.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioAnalysisRequest:
    audio_path: str
    profile: str = "routine"
    speaker_id: Optional[str] = None
    context: Optional[dict] = None


@dataclass
class RiskResponse:
    voice_risk: float
    final_risk: float
    risk_level: str
    recommended_action: str
    conflicted: bool = False
    speaker_similarity: Optional[float] = None
    ssl_score: Optional[float] = None
    mfcc_score: Optional[float] = None
    phase_score: Optional[float] = None
    conflict_detail: Optional[str] = None


@dataclass
class ConsistencyResponse:
    similarity: float
    match: bool
    consistency_risk: float


@dataclass
class SpeakerProfile:
    speaker_id: str
    enrolled_at: Optional[str] = None
    embedding_dim: Optional[int] = None
