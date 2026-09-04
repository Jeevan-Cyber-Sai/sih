"""
Standalone gRPC client that exercises every RPC on VoiceGuard
(grpc_server.py) against a locally running server -- run this after
starting app.py (which starts the gRPC server on port 50051 alongside
the Flask REST API on port 5000).

Usage: python grpc_client_test.py
"""
import librosa
import numpy as np

import grpc
import voiceguard_pb2
import voiceguard_pb2_grpc

HOST = "localhost:50051"
SAMPLE_RATE = 16000

REAL_FILE = "test/pc_bonafide.wav"
FAKE_FILE = "data_generators/elevenlabs/elevenlabs_audio1.wav"


def load_as_pcm_bytes(path, sr=SAMPLE_RATE):
    """Matches the wire convention documented in proto/voiceguard.proto:
    raw float32 PCM bytes, mono, at `sr`."""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32).tobytes()


def print_risk_response(label, resp):
    print(f"--- {label} ---")
    print(f"  voice_risk:          {resp.voice_risk:.2f}")
    print(f"  final_risk:          {resp.final_risk:.2f}")
    print(f"  risk_level:          {resp.risk_level}")
    print(f"  recommended_action:  {resp.recommended_action}")
    print(f"  conflicted:          {resp.conflicted}")
    print(f"  speaker_similarity:  {resp.speaker_similarity:.4f}")
    print()


def main():
    channel = grpc.insecure_channel(HOST)
    stub = voiceguard_pb2_grpc.VoiceGuardStub(channel)

    print(f"=== connected to {HOST} ===\n")

    # --- AnalyzeAudio: one real, one fake ---
    real_bytes = load_as_pcm_bytes(REAL_FILE)
    resp = stub.AnalyzeAudio(voiceguard_pb2.AudioRequest(
        audio_data=real_bytes, sample_rate=SAMPLE_RATE, profile="routine",
    ))
    print_risk_response(f"AnalyzeAudio: real file ({REAL_FILE})", resp)

    fake_bytes = load_as_pcm_bytes(FAKE_FILE)
    resp = stub.AnalyzeAudio(voiceguard_pb2.AudioRequest(
        audio_data=fake_bytes, sample_rate=SAMPLE_RATE, profile="routine",
    ))
    print_risk_response(f"AnalyzeAudio: fake file ({FAKE_FILE})", resp)

    # --- AnalyzeWithContext: sample high-value-transaction context ---
    resp = stub.AnalyzeWithContext(voiceguard_pb2.ContextualRequest(
        audio_data=fake_bytes, sample_rate=SAMPLE_RATE, profile="high_value_transaction",
        caller_known=False, channel="voip", transaction_amount=25000.0,
        new_beneficiary=True, outside_business_hours=True, previously_flagged=False,
        speaker_id="",
    ))
    print_risk_response("AnalyzeWithContext: fake file, aggravating context, high_value_transaction profile", resp)

    # --- EnrollSpeaker then VerifyConsistency ---
    speaker_id = "grpc_test_speaker"
    enroll_resp = stub.EnrollSpeaker(voiceguard_pb2.EnrollRequest(
        speaker_id=speaker_id, audio_data=real_bytes, sample_rate=SAMPLE_RATE,
    ))
    print("--- EnrollSpeaker ---")
    print(f"  success:    {enroll_resp.success}")
    print(f"  speaker_id: {enroll_resp.speaker_id}")
    print(f"  message:    {enroll_resp.message}")
    print()

    verify_resp = stub.VerifyConsistency(voiceguard_pb2.VerifyRequest(
        speaker_id=speaker_id, audio_data=real_bytes, sample_rate=SAMPLE_RATE,
    ))
    print("--- VerifyConsistency (same clip used to enroll) ---")
    print(f"  similarity:        {verify_resp.similarity:.4f}")
    print(f"  match:             {verify_resp.match}")
    print(f"  consistency_risk:  {verify_resp.consistency_risk:.2f}")
    print()


if __name__ == "__main__":
    main()
