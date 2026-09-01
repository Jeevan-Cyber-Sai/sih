"""
Live audio capture for the AI voice detector.

Captures from a virtual audio input device (e.g. VB-Cable fed by a Zoom /
Google Meet call's output via "Listen to this device" loopback routing)
and scores it live using the same feature pipeline as file-based analysis.
"""
import argparse
import queue
import time

import librosa
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


def _resolve_stream_params(device_index, target_sr):
    """Find a (channels, samplerate) combo the device/driver actually
    accepts. Many WDM/WASAPI-exclusive devices reject an arbitrary
    samplerate (like our model's 16kHz) and only work at their native
    rate, so we probe and fall back to that, resampling in software."""
    info = sd.query_devices(device_index)
    max_ch = info["max_input_channels"]
    if max_ch <= 0:
        raise ValueError(f"device {device_index} has no input channels")

    channel_options = list(dict.fromkeys([1, max_ch]))
    sr_options = list(dict.fromkeys([target_sr, int(round(info["default_samplerate"]))]))

    for channels in channel_options:
        for sr in sr_options:
            try:
                sd.check_input_settings(device=device_index, channels=channels, samplerate=sr)
                return channels, sr
            except Exception:
                continue

    raise RuntimeError(
        f"device {device_index} ('{info['name']}') doesn't support any of the "
        f"tried channel/samplerate combinations: channels={channel_options}, "
        f"samplerates={sr_options}"
    )


def capture_and_analyze(device_index, chunk_seconds=2.0, sr=16000, max_duration=None,
                         alpha=0.6, stop_event=None):
    """Generator: yields a risk result dict for each chunk captured live.
    Runs until max_duration elapses, stop_event is set, or the caller
    stops iterating (e.g. via generator.close(), which cleanly tears down
    the input stream)."""
    clf, scaler = load_model()

    channels, capture_sr = _resolve_stream_params(device_index, sr)
    if capture_sr != sr:
        print(f"device {device_index} doesn't support {sr}Hz directly; "
              f"capturing at its native {capture_sr}Hz (channels={channels}) "
              f"and resampling to {sr}Hz per chunk.")

    chunk_samples = int(capture_sr * chunk_seconds)
    audio_q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[stream status] {status}")
        mono = indata.mean(axis=1) if indata.ndim > 1 and indata.shape[1] > 1 else indata[:, 0]
        audio_q.put(mono.copy())

    rolling_score = None
    start_time = time.time()

    with sd.InputStream(device=device_index, channels=channels, samplerate=capture_sr,
                         blocksize=chunk_samples, dtype="float32", callback=callback):
        while True:
            if max_duration is not None and (time.time() - start_time) >= max_duration:
                break
            if stop_event is not None and stop_event.is_set():
                break
            try:
                audio = audio_q.get(timeout=1.0)
            except queue.Empty:
                continue

            if capture_sr != sr:
                audio = librosa.resample(audio, orig_sr=capture_sr, target_sr=sr)

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


def _resolve_device_arg(device_arg):
    """Windows/PortAudio device indices can shift between process launches
    (e.g. a Bluetooth headset connecting/disconnecting), so also accept a
    case-insensitive substring of the device name (e.g. 'CABLE')."""
    try:
        return int(device_arg)
    except ValueError:
        pass

    devices = sd.query_devices()
    matches = [i for i, d in enumerate(devices)
               if d["max_input_channels"] > 0 and device_arg.lower() in d["name"].lower()]
    if not matches:
        raise ValueError(f"no input device name contains '{device_arg}'")
    if len(matches) > 1:
        names = ", ".join(f"{i}:{devices[i]['name']}" for i in matches)
        raise ValueError(f"multiple devices match '{device_arg}': {names} -- use an exact index instead")
    return matches[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live audio capture for the AI voice detector.")
    parser.add_argument("--device", type=str, default=None,
                         help="input device index, or a substring of its name (e.g. 'CABLE'); "
                              "omit to just list devices")
    parser.add_argument("--chunk-seconds", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=None,
                         help="stop after N seconds (default: run until Ctrl+C)")
    args = parser.parse_args()

    if args.device is None:
        print("Available audio input devices:")
        list_audio_devices()
        print("\nRun again with --device <idx or name substring> to start live capture.")
    else:
        device_index = _resolve_device_arg(args.device)
        print(f"Listening on device {device_index} ... Ctrl+C to stop.")
        try:
            for result in capture_and_analyze(
                device_index, chunk_seconds=args.chunk_seconds, max_duration=args.max_duration
            ):
                print(f"[{result['timestamp']}] instant={result['instant_score']:6.2f}  "
                      f"rolling={result['rolling_score']:6.2f}  risk={result['risk_level']}")
        except KeyboardInterrupt:
            print("\nStopped by user.")
