"""
Command-line interface for the VoiceGuard SDK:

    voiceguard analyze audio.wav --profile high_value_transaction
    voiceguard enroll --speaker-id "CFO_Rajesh" --audio cfo_sample.wav
    voiceguard verify --speaker-id "CFO_Rajesh" --audio live_call.wav
    voiceguard test-connection

All four accept --host/--port/--rest to point at a different server or
use REST instead of gRPC (default).
"""
import argparse
import sys

from .client import VoiceGuardClient
from .exceptions import VoiceGuardException


def _add_connection_args(parser):
    parser.add_argument("--host", default="localhost", help="VoiceGuard server host (default: localhost)")
    parser.add_argument("--port", type=int, default=50051, help="gRPC port (default: 50051)")
    parser.add_argument("--rest", action="store_true", help="use REST (port 5000) instead of gRPC")


def _client_from_args(args):
    return VoiceGuardClient(host=args.host, port=args.port, use_grpc=not args.rest)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="voiceguard", description="VoiceGuard SDK command-line tool.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="Analyze an audio file for voice-cloning/impersonation risk.")
    p_analyze.add_argument("audio", help="path to the audio file (WAV for gRPC; any format the server supports for --rest)")
    p_analyze.add_argument("--profile", default="routine", help="risk profile (default: routine)")
    p_analyze.add_argument("--speaker-id", default=None, help="check against an enrolled speaker profile")
    _add_connection_args(p_analyze)

    p_enroll = sub.add_parser("enroll", help="Enroll a speaker's voice profile.")
    p_enroll.add_argument("--speaker-id", required=True)
    p_enroll.add_argument("--audio", required=True)
    _add_connection_args(p_enroll)

    p_verify = sub.add_parser("verify", help="Check a clip against an enrolled speaker profile.")
    p_verify.add_argument("--speaker-id", required=True)
    p_verify.add_argument("--audio", required=True)
    _add_connection_args(p_verify)

    p_test = sub.add_parser("test-connection", help="Check whether the VoiceGuard server is reachable.")
    _add_connection_args(p_test)

    args = parser.parse_args(argv)
    client = _client_from_args(args)

    try:
        if args.command == "analyze":
            result = client.analyze(args.audio, profile=args.profile, speaker_id=args.speaker_id)
            print(f"risk_level:         {result.risk_level}")
            print(f"voice_risk:         {result.voice_risk:.2f}")
            print(f"final_risk:         {result.final_risk:.2f}")
            print(f"recommended_action: {result.recommended_action}")
            print(f"conflicted:         {result.conflicted}")
            if result.speaker_similarity is not None:
                print(f"speaker_similarity: {result.speaker_similarity:.4f}")

        elif args.command == "enroll":
            profile = client.enroll_speaker(args.speaker_id, args.audio)
            print(f"enrolled speaker_id: {profile.speaker_id}")
            print(f"enrolled_at:         {profile.enrolled_at}")
            print(f"embedding_dim:       {profile.embedding_dim}")

        elif args.command == "verify":
            result = client.verify_speaker(args.speaker_id, args.audio)
            print(f"similarity:       {result.similarity:.4f}")
            print(f"match:            {result.match}")
            print(f"consistency_risk: {result.consistency_risk:.2f}")

        elif args.command == "test-connection":
            ok = client.test_connection()
            print("OK -- server reachable" if ok else "FAILED -- server unreachable")
            sys.exit(0 if ok else 1)

    except VoiceGuardException as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
