"""
Simulates a Twilio Media Stream against a locally running /media-stream
websocket: encodes a local audio file into the same base64 mulaw-8kHz
frames Twilio sends, and streams them paced at Twilio's real cadence
(20ms/frame) -- so the whole live-call pipeline (decode, resample,
buffer, rolling-score, backpressure) can be verified end-to-end without
needing an actual phone call.

Usage: python simulate_twilio_stream.py path/to/audio.wav [--host localhost:5000]
"""
import argparse
import asyncio
import audioop
import base64
import json
import threading
import time
import uuid

import librosa
import numpy as np
import requests
import websockets

FRAME_MS = 20
FRAME_SAMPLES = int(8000 * FRAME_MS / 1000)  # 160 samples @ 8kHz, matches Twilio's real framing


def encode_mulaw_frames(path):
    """Loads the file at 8kHz mono (telephone quality, same as a real
    call) and mulaw-encodes it into Twilio-sized 20ms frames."""
    audio, sr = librosa.load(path, sr=8000, mono=True)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    pcm16 = (audio * 32767).astype(np.int16).tobytes()
    mulaw = audioop.lin2ulaw(pcm16, 2)

    frame_bytes = FRAME_SAMPLES  # 1 byte per mulaw sample
    frames = [mulaw[i:i + frame_bytes] for i in range(0, len(mulaw), frame_bytes)]
    return [f for f in frames if f]


def poll_status(host, stop_event, call_sid):
    """Runs in a background thread, printing the rolling score every
    time /live-call-status reports a new chunk for THIS call -- the same
    info the dashboard's live gauge would show. Filters on call_sid so a
    leftover final state from a previous run against the same
    (persistent) server never gets printed as if it were this run's."""
    last_chunk_count = -1
    while not stop_event.is_set():
        try:
            r = requests.get(f"http://{host}/live-call-status", timeout=2)
            data = r.json()
            if data.get("call_sid") != call_sid:
                time.sleep(0.3)
                continue
            if data.get("chunk_count", 0) != last_chunk_count and data.get("rolling_score") is not None:
                last_chunk_count = data["chunk_count"]
                print(f"  chunk {data['chunk_count']:3d}: rolling_score={data['rolling_score']:6.2f}  "
                      f"risk={data['risk_level']:<6}  action={data['recommended_action']}")
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.3)


async def stream_file(path, host):
    frames = encode_mulaw_frames(path)
    duration_s = len(frames) * FRAME_MS / 1000
    print(f"\n=== simulating live call: {path} ===")
    print(f"{len(frames)} frames, {duration_s:.1f}s of audio, paced at {FRAME_MS}ms/frame (real Twilio cadence)")

    call_sid = f"CAsimulated{uuid.uuid4().hex[:24]}"
    uri = f"ws://{host}/media-stream"

    stop_event = threading.Event()
    poll_thread = threading.Thread(target=poll_status, args=(host, stop_event, call_sid), daemon=True)
    poll_thread.start()

    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        await ws.send(json.dumps({
            "event": "start",
            "start": {
                "callSid": call_sid,
                "streamSid": "MZsimulated",
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
            },
        }))

        for i, frame in enumerate(frames):
            payload = base64.b64encode(frame).decode("ascii")
            await ws.send(json.dumps({
                "event": "media",
                "media": {"track": "inbound", "chunk": str(i), "payload": payload},
            }))
            await asyncio.sleep(FRAME_MS / 1000)

        await ws.send(json.dumps({"event": "stop", "stop": {"callSid": call_sid}}))

    time.sleep(1.5)  # let the final chunk finish scoring and get polled once more
    stop_event.set()
    poll_thread.join(timeout=2)

    try:
        final = requests.get(f"http://{host}/live-call-status", timeout=2).json()
        print(f"final: rolling_score={final['rolling_score']}  risk={final['risk_level']}  "
              f"chunks_scored={final['chunk_count']}")
    except requests.exceptions.RequestException as e:
        print(f"could not fetch final status: {e}")


def main():
    parser = argparse.ArgumentParser(description="Simulate a Twilio Media Stream against a local /media-stream endpoint.")
    parser.add_argument("audio_path")
    parser.add_argument("--host", default="localhost:5000", help="host:port the Flask app is running on")
    args = parser.parse_args()

    asyncio.run(stream_file(args.audio_path, args.host))


if __name__ == "__main__":
    main()
