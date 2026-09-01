"""
Live audio capture for the AI voice detector.

Captures from a virtual audio input device (e.g. VB-Cable fed by a Zoom /
Google Meet call's output via "Listen to this device" loopback routing)
and scores it live using the same feature pipeline as file-based analysis.
"""
import argparse
import queue
import time

import numpy as np
import sounddevice as sd

from predict import load_model, risk_level, score_single_clip


def list_audio_devices():
    devices = sd.query_devices()
    print(f"{'idx':>4}  {'name':<50} {'in_ch':>6}  {'default_sr':>10}")
    print("-" * 76)
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            print(f"{i:>4}  {d['name']:<50} {d['max_input_channels']:>6}  "
                  f"{d['default_samplerate']:>10.0f}")
    return devices


def _open_input_stream(device_index, sr, chunk_samples, callback):
    try:
        return sd.InputStream(
            device=device_index, channels=1, samplerate=sr,
            blocksize=chunk_samples, dtype="float32", callback=callback,
        )
    except sd.PortAudioError:
        # some devices (e.g. VB-Cable) only expose a stereo input
        return sd.InputStream(
            device=device_index, channels=2, samplerate=sr,
            blocksize=chunk_samples, dtype="float32", callback=callback,
        )


def capture_and_analyze(device_index, chunk_seconds=2.0, sr=16000, max_duration=None,
                         alpha=0.6, stop_event=None):
    """Generator: yields a risk result dict for each chunk captured live.
    Runs until max_duration elapses, stop_event is set, or the caller
    stops iterating (e.g. via generator.close(), which cleanly tears down
    the input stream)."""
    clf, scaler = load_model()

    chunk_samples = int(sr * chunk_seconds)
    audio_q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[stream status] {status}")
        mono = indata.mean(axis=1) if indata.ndim > 1 and indata.shape[1] > 1 else indata[:, 0]
        audio_q.put(mono.copy())

    rolling_score = None
    start_time = time.time()

    with _open_input_stream(device_index, sr, chunk_samples, callback):
        while True:
            if max_duration is not None and (time.time() - start_time) >= max_duration:
                break
            if stop_event is not None and stop_event.is_set():
                break
            try:
                audio = audio_q.get(timeout=1.0)
            except queue.Empty:
                continue

            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak

            instant_score = score_single_clip(audio, clf, scaler)
            rolling_score = instant_score if rolling_score is None else (
                alpha * instant_score + (1 - alpha) * rolling_score
            )
            rolling_score = round(rolling_score, 2)
            level, action = risk_level(rolling_score)

            yield {
                "timestamp": time.strftime("%H:%M:%S"),
                "instant_score": instant_score,
                "rolling_score": rolling_score,
                "risk_level": level,
                "recommended_action": action,
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live audio capture for the AI voice detector.")
    parser.add_argument("--device", type=int, default=None,
                         help="input device index; omit to just list devices")
    parser.add_argument("--chunk-seconds", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=None,
                         help="stop after N seconds (default: run until Ctrl+C)")
    args = parser.parse_args()

    if args.device is None:
        print("Available audio input devices:")
        list_audio_devices()
        print("\nRun again with --device <idx> to start live capture.")
    else:
        print(f"Listening on device {args.device} ... Ctrl+C to stop.")
        try:
            for result in capture_and_analyze(
                args.device, chunk_seconds=args.chunk_seconds, max_duration=args.max_duration
            ):
                print(f"[{result['timestamp']}] instant={result['instant_score']:6.2f}  "
                      f"rolling={result['rolling_score']:6.2f}  risk={result['risk_level']}")
        except KeyboardInterrupt:
            print("\nStopped by user.")
