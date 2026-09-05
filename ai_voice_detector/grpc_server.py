"""
gRPC API for the AI voice detector, running alongside the existing Flask
REST API (app.py) on a separate port -- required for enterprise banking
integration per the PS (both REST and gRPC).

Every RPC here is a thin wrapper: it converts the incoming raw audio
bytes into whatever input shape the underlying function already expects,
then calls the EXACT SAME functions from predict.py/speaker_consistency.py
that app.py's REST routes call -- zero duplicate scoring/business logic
between the two APIs, so they can never silently drift apart.

Wire format: audio_data on every RPC is raw 32-bit float PCM samples
(mono), not a WAV/container file -- see proto/voiceguard.proto's header
comment and grpc_client_test.py for the encode-side convention.
"""
import logging
import os
import tempfile
from concurrent import futures

import grpc
import numpy as np
import soundfile as sf

import voiceguard_pb2
import voiceguard_pb2_grpc
from predict import analyze_file_with_context
from speaker_consistency import enroll_speaker, verify_speaker

logger = logging.getLogger(__name__)

GRPC_PORT = 50051


def _bytes_to_array(audio_data):
    return np.frombuffer(audio_data, dtype=np.float32)


def _bytes_to_temp_wav(audio_data, sample_rate):
    """Writes raw float32 PCM bytes to a temp .wav file so we can hand
    it to analyze_file_with_context() completely unmodified -- the same
    function app.py's /analyze_with_context route calls. Caller must
    remove the returned path when done."""
    audio = _bytes_to_array(audio_data)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, audio, sample_rate)
    return path


class VoiceGuardServicer(voiceguard_pb2_grpc.VoiceGuardServicer):
    def AnalyzeAudio(self, request, context):
        """Voice-only risk assessment, no call/transaction context --
        reuses analyze_file_with_context() with context=None so the
        response shape (voice_risk, final_risk, conflicted) matches
        AnalyzeWithContext exactly, just without any context blended in."""
        path = _bytes_to_temp_wav(request.audio_data, request.sample_rate)
        try:
            result = analyze_file_with_context(path, context=None, profile=request.profile or None)
        except Exception as e:
            logger.exception("gRPC AnalyzeAudio failed")
            context.abort(grpc.StatusCode.INTERNAL, str(e))
            return voiceguard_pb2.RiskResponse()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

        return voiceguard_pb2.RiskResponse(
            voice_risk=result["voice_risk"],
            final_risk=result["final_risk"],
            risk_level=result["risk_level"],
            recommended_action=result["recommended_action"],
            conflicted=result.get("conflicted", False),
            speaker_similarity=result.get("speaker_similarity", 0.0),
            ssl_score=result.get("ssl_score", 0.0),
            mfcc_score=result.get("mfcc_score", 0.0),
            phase_score=result.get("phase_score", 0.0),
            conflict_detail=result.get("conflict_detail") or "",
        )

    def AnalyzeWithContext(self, request, context):
        ctx = {
            "caller_known": request.caller_known,
            "channel": request.channel or None,
            "transaction_amount": request.transaction_amount,
            "new_beneficiary": request.new_beneficiary,
            "outside_business_hours": request.outside_business_hours,
            "previously_flagged": request.previously_flagged,
        }
        path = _bytes_to_temp_wav(request.audio_data, request.sample_rate)
        try:
            result = analyze_file_with_context(
                path, context=ctx, profile=request.profile or None,
                speaker_id=request.speaker_id or None,
            )
        except Exception as e:
            logger.exception("gRPC AnalyzeWithContext failed")
            context.abort(grpc.StatusCode.INTERNAL, str(e))
            return voiceguard_pb2.RiskResponse()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

        return voiceguard_pb2.RiskResponse(
            voice_risk=result["voice_risk"],
            final_risk=result["final_risk"],
            risk_level=result["risk_level"],
            recommended_action=result["recommended_action"],
            conflicted=result.get("conflicted", False),
            speaker_similarity=result.get("speaker_similarity", 0.0),
            ssl_score=result.get("ssl_score", 0.0),
            mfcc_score=result.get("mfcc_score", 0.0),
            phase_score=result.get("phase_score", 0.0),
            conflict_detail=result.get("conflict_detail") or "",
        )

    def EnrollSpeaker(self, request, context):
        try:
            audio = _bytes_to_array(request.audio_data)
            enroll_speaker(request.speaker_id, audio, sr=request.sample_rate)
        except Exception as e:
            logger.exception("gRPC EnrollSpeaker failed")
            context.abort(grpc.StatusCode.INTERNAL, str(e))
            return voiceguard_pb2.EnrollResponse()

        return voiceguard_pb2.EnrollResponse(
            success=True, speaker_id=request.speaker_id,
            message=f"enrolled speaker '{request.speaker_id}'",
        )

    def VerifyConsistency(self, request, context):
        try:
            audio = _bytes_to_array(request.audio_data)
            result = verify_speaker(request.speaker_id, audio, sr=request.sample_rate)
        except Exception as e:
            logger.exception("gRPC VerifyConsistency failed")
            context.abort(grpc.StatusCode.INTERNAL, str(e))
            return voiceguard_pb2.ConsistencyResponse()

        if result is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"speaker_id '{request.speaker_id}' is not enrolled")
            return voiceguard_pb2.ConsistencyResponse()

        return voiceguard_pb2.ConsistencyResponse(
            similarity=result["similarity"],
            match=result["match"],
            consistency_risk=result["consistency_risk"],
        )


def serve(port=GRPC_PORT, block=True):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    voiceguard_pb2_grpc.add_VoiceGuardServicer_to_server(VoiceGuardServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("gRPC server listening on port %d", port)
    print(f"gRPC server listening on port {port}")
    if block:
        server.wait_for_termination()
    return server


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
