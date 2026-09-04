"""
Example: how a banking system would integrate the VoiceGuard SDK to
screen an incoming call before authorizing a transaction.

Requires a running VoiceGuard server (python app.py) -- gRPC on 50051,
REST on 5000.
"""
from voiceguard_sdk import VoiceGuardClient

REAL_FILE = "test/pc_bonafide.wav"
FAKE_FILE = "data_generators/elevenlabs/elevenlabs_audio1.wav"


def block_transaction():
    print("  >>> ACTION: transaction blocked.")


def notify_supervisor():
    print("  >>> ACTION: supervisor notified.")


def screen_call(client, audio_path, label):
    print(f"=== screening: {label} ({audio_path}) ===")
    result = client.analyze(
        audio_path,
        profile="high_value_transaction",
        context={
            "caller_known": False,
            "transaction_amount": 500000,
            "new_beneficiary": True,
            "channel": "voip",
        },
        speaker_id="CFO_Rajesh",
    )

    print(f"  risk_level:         {result.risk_level}")
    print(f"  voice_risk:         {result.voice_risk:.2f}")
    print(f"  final_risk:         {result.final_risk:.2f}")
    print(f"  recommended_action: {result.recommended_action}")
    print(f"  conflicted:         {result.conflicted}")

    if result.risk_level == "HIGH":
        block_transaction()
        notify_supervisor()
    else:
        print("  >>> ACTION: proceed normally.")
    print()


def main():
    client = VoiceGuardClient(host="localhost")  # gRPC by default

    print(f"test_connection(): {client.test_connection()}")
    print(f"get_profiles():    {client.get_profiles()}\n")

    screen_call(client, REAL_FILE, "genuine call")
    screen_call(client, FAKE_FILE, "AI-cloned voice attempting a wire transfer")


if __name__ == "__main__":
    main()
