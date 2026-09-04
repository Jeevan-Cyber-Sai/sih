"""
VoiceGuardClient: the main SDK entry point. Talks to a running VoiceGuard
server over gRPC (default, port 50051) or REST (port 5000, use_grpc=False),
and translates every failure into a VoiceGuardException subclass -- callers
never need to know or handle raw gRPC/HTTP error types.

Deliberately lightweight: this package's only dependencies are grpcio,
requests, and numpy (see setup.py) -- the heavy ML stack (librosa, torch,
transformers) stays server-side where it belongs. That means the gRPC
path reads audio via Python's stdlib `wave` module rather than librosa,
which limits it to uncompressed WAV (not MP3/FLAC) -- an intentional
tradeoff for a distributable client, not an oversight. The REST path has
no such limit, since it just uploads the raw file bytes and lets the
server (which already has librosa) decode whatever format it's given.
"""
import wave

import numpy as np

from .exceptions import AudioFormatError, ConnectionError, SpeakerNotFoundError, VoiceGuardException
from .models import ConsistencyResponse, RiskResponse, SpeakerProfile

DEFAULT_SAMPLE_RATE = 16000
# Intrinsic to the pretrained resemblyzer encoder used server-side
# (speaker_consistency.py) -- not something that varies per profile.
RESEMBLYZER_EMBEDDING_DIM = 256


def _read_wav_as_float32(audio_path):
    """Reads a WAV file into a mono float32 numpy array plus its native
    sample rate, using only the stdlib `wave` module -- see the module
    docstring for why this avoids librosa."""
    try:
        with wave.open(audio_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
    except Exception as e:
        raise AudioFormatError(
            f"could not read '{audio_path}' as a WAV file: {e}. The VoiceGuard SDK's "
            "gRPC path reads uncompressed WAV directly with no extra audio-decoding "
            "dependencies -- convert other formats to WAV first, or use use_grpc=False "
            "(REST), which accepts any format the server can decode."
        ) from e

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise AudioFormatError(f"unsupported WAV sample width in '{audio_path}': {sample_width} bytes")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1).astype(np.float32)

    return audio, frame_rate


class VoiceGuardClient:
    def __init__(self, host="localhost", port=50051, use_grpc=True, rest_port=5000, timeout=15):
        self.host = host
        self.port = port
        self.rest_port = rest_port
        self.use_grpc = use_grpc
        self.timeout = timeout
        self._grpc = self._pb2 = self._pb2_grpc = None
        self._grpc_stub = None

        if self.use_grpc:
            self._init_grpc()

    def _init_grpc(self):
        try:
            import grpc
        except ImportError as e:
            raise VoiceGuardException("grpcio is required for use_grpc=True (pip install grpcio).") from e

        try:
            import voiceguard_pb2
            import voiceguard_pb2_grpc
        except ImportError:
            # A console-script entry point (e.g. the installed
            # `voiceguard` CLI) doesn't add the current working
            # directory to sys.path the way `python script.py` does, so
            # the generated stubs next to app.py/grpc_server.py at the
            # project root aren't found purely by cwd. Locate them
            # relative to this SDK package's own file instead -- robust
            # regardless of how the caller invoked Python.
            import os
            import sys
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            try:
                import voiceguard_pb2
                import voiceguard_pb2_grpc
            except ImportError as e:
                raise VoiceGuardException(
                    f"could not find the generated voiceguard_pb2/voiceguard_pb2_grpc modules "
                    f"(looked in {project_root!r}). Regenerate them from proto/voiceguard.proto, "
                    "or pass use_grpc=False to use REST instead."
                ) from e
        self._grpc = grpc
        self._pb2 = voiceguard_pb2
        self._pb2_grpc = voiceguard_pb2_grpc
        self._grpc_stub = voiceguard_pb2_grpc.VoiceGuardStub(grpc.insecure_channel(f"{self.host}:{self.port}"))

    def _rest_url(self, path):
        return f"http://{self.host}:{self.rest_port}{path}"

    def _raise_clean(self, e):
        """Translates a raw gRPC/HTTP/network exception into the
        appropriate VoiceGuardException subclass."""
        if self.use_grpc and isinstance(e, self._grpc.RpcError):
            code = e.code()
            if code == self._grpc.StatusCode.NOT_FOUND:
                raise SpeakerNotFoundError(e.details() or "speaker not found") from e
            if code in (self._grpc.StatusCode.UNAVAILABLE, self._grpc.StatusCode.DEADLINE_EXCEEDED):
                raise ConnectionError(f"could not reach VoiceGuard gRPC server at {self.host}:{self.port}") from e
            raise VoiceGuardException(e.details() or str(e)) from e

        import requests
        if isinstance(e, requests.exceptions.ConnectionError):
            raise ConnectionError(f"could not reach VoiceGuard REST server at {self.host}:{self.rest_port}") from e
        if isinstance(e, requests.exceptions.Timeout):
            raise ConnectionError(f"VoiceGuard REST server at {self.host}:{self.rest_port} timed out") from e
        raise VoiceGuardException(str(e)) from e

    def test_connection(self):
        """Returns True/False rather than raising -- this is meant to be
        a cheap, safe health check, not something callers need to wrap
        in a try/except."""
        try:
            if self.use_grpc:
                # A cheap RPC that needs no real audio or enrolled
                # speaker: VerifyConsistency on a bogus speaker_id, which
                # a reachable server answers with NOT_FOUND (a real
                # response), versus UNAVAILABLE if nothing's listening.
                silence = np.zeros(DEFAULT_SAMPLE_RATE, dtype=np.float32)
                request = self._pb2.VerifyRequest(
                    speaker_id="__voiceguard_connection_test__",
                    audio_data=silence.tobytes(), sample_rate=DEFAULT_SAMPLE_RATE,
                )
                try:
                    self._grpc_stub.VerifyConsistency(request, timeout=self.timeout)
                except self._grpc.RpcError as e:
                    return e.code() == self._grpc.StatusCode.NOT_FOUND
                return True

            import requests
            r = requests.get(self._rest_url("/profiles"), timeout=self.timeout)
            return r.status_code == 200
        except Exception:
            return False

    def get_profiles(self):
        """Available named risk profiles (config.yaml). No dedicated RPC
        exists for this in voiceguard.proto -- it's a small, read-only
        listing that already exists via REST, so this always uses REST
        regardless of use_grpc, rather than expanding the gRPC contract
        for a non-scoring convenience call."""
        try:
            import requests
            r = requests.get(self._rest_url("/profiles"), timeout=self.timeout)
            r.raise_for_status()
            return list(r.json()["profiles"].keys())
        except VoiceGuardException:
            raise
        except Exception as e:
            self._raise_clean(e)

    def analyze(self, audio_path, profile="routine", context=None, speaker_id=None):
        context = context or {}
        try:
            if self.use_grpc:
                audio, sr = _read_wav_as_float32(audio_path)
                request = self._pb2.ContextualRequest(
                    audio_data=audio.tobytes(), sample_rate=sr, profile=profile,
                    caller_known=bool(context.get("caller_known", False)),
                    channel=context.get("channel") or "",
                    transaction_amount=float(context.get("transaction_amount", 0.0)),
                    new_beneficiary=bool(context.get("new_beneficiary", False)),
                    outside_business_hours=bool(context.get("outside_business_hours", False)),
                    previously_flagged=bool(context.get("previously_flagged", False)),
                    speaker_id=speaker_id or "",
                )
                resp = self._grpc_stub.AnalyzeWithContext(request, timeout=self.timeout)
                return RiskResponse(
                    voice_risk=resp.voice_risk, final_risk=resp.final_risk,
                    risk_level=resp.risk_level, recommended_action=resp.recommended_action,
                    conflicted=resp.conflicted, speaker_similarity=resp.speaker_similarity or None,
                )

            import requests
            data = {
                "profile": profile,
                "caller_known": context.get("caller_known", False),
                "origin": context.get("origin") or "",
                "channel": context.get("channel") or "",
                "transaction_amount": context.get("transaction_amount", 0),
                "new_beneficiary": context.get("new_beneficiary", False),
                "outside_business_hours": context.get("outside_business_hours", False),
                "previously_flagged": context.get("previously_flagged", False),
            }
            if speaker_id:
                data["speaker_id"] = speaker_id
            with open(audio_path, "rb") as f:
                r = requests.post(self._rest_url("/analyze_with_context"), files={"audio": f}, data=data, timeout=self.timeout)
            if r.status_code != 200:
                raise VoiceGuardException(f"server returned {r.status_code}: {r.text}")
            body = r.json()
            return RiskResponse(
                voice_risk=body["voice_risk"], final_risk=body["final_risk"],
                risk_level=body["risk_level"], recommended_action=body["recommended_action"],
                conflicted=body.get("conflicted", False),
                speaker_similarity=body.get("speaker_similarity"),
            )
        except VoiceGuardException:
            raise
        except FileNotFoundError as e:
            raise AudioFormatError(f"audio file not found: {audio_path}") from e
        except Exception as e:
            self._raise_clean(e)

    def enroll_speaker(self, speaker_id, audio_path):
        try:
            if self.use_grpc:
                audio, sr = _read_wav_as_float32(audio_path)
                request = self._pb2.EnrollRequest(speaker_id=speaker_id, audio_data=audio.tobytes(), sample_rate=sr)
                resp = self._grpc_stub.EnrollSpeaker(request, timeout=self.timeout)
                if not resp.success:
                    raise VoiceGuardException(resp.message or "enrollment failed")
            else:
                import requests
                with open(audio_path, "rb") as f:
                    r = requests.post(
                        self._rest_url("/enroll-speaker"), files={"audio": f},
                        data={"speaker_id": speaker_id}, timeout=self.timeout,
                    )
                if r.status_code != 200:
                    raise VoiceGuardException(f"server returned {r.status_code}: {r.text}")

            # Neither transport's enroll response currently carries back
            # enrollment metadata (see voiceguard.proto's EnrollResponse),
            # so this reports what the SDK itself knows at call time --
            # the embedding dimension is intrinsic to the server's
            # pretrained encoder, not something that varies per speaker.
            import datetime
            return SpeakerProfile(
                speaker_id=speaker_id,
                enrolled_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                embedding_dim=RESEMBLYZER_EMBEDDING_DIM,
            )
        except VoiceGuardException:
            raise
        except FileNotFoundError as e:
            raise AudioFormatError(f"audio file not found: {audio_path}") from e
        except Exception as e:
            self._raise_clean(e)

    def verify_speaker(self, speaker_id, audio_path):
        try:
            if self.use_grpc:
                audio, sr = _read_wav_as_float32(audio_path)
                request = self._pb2.VerifyRequest(speaker_id=speaker_id, audio_data=audio.tobytes(), sample_rate=sr)
                resp = self._grpc_stub.VerifyConsistency(request, timeout=self.timeout)
                return ConsistencyResponse(
                    similarity=resp.similarity, match=resp.match, consistency_risk=resp.consistency_risk,
                )

            import requests
            with open(audio_path, "rb") as f:
                r = requests.post(
                    self._rest_url("/verify-consistency"), files={"audio": f},
                    data={"speaker_id": speaker_id}, timeout=self.timeout,
                )
            if r.status_code == 404:
                raise SpeakerNotFoundError(f"speaker_id '{speaker_id}' is not enrolled")
            if r.status_code != 200:
                raise VoiceGuardException(f"server returned {r.status_code}: {r.text}")
            body = r.json()
            return ConsistencyResponse(
                similarity=body["similarity"], match=body["match"], consistency_risk=body["consistency_risk"],
            )
        except VoiceGuardException:
            raise
        except FileNotFoundError as e:
            raise AudioFormatError(f"audio file not found: {audio_path}") from e
        except Exception as e:
            self._raise_clean(e)
