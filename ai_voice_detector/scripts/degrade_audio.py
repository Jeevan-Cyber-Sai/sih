"""
Randomized channel-degradation augmentation: simulates the lossy
compression / background noise / gain variation that real-world audio
(phone recordings, messaging apps) naturally has.

Why this matters: if only the REAL class gets diversified with real-world
(degraded) samples, a classifier can shortcut on "clean studio = fake,
degraded = real" instead of learning genuine synthetic-speech artifacts --
which is exactly what happened when data_realworld/real/ was added without
a matching data_realworld/fake/ (see the ElevenLabs regression test).
This module is applied to BOTH new real-world-style fake samples and a
degraded copy of existing ASVspoof fake samples, so the fake class spans
clean and degraded conditions symmetrically with the real class.
"""
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from ensure_ffmpeg import ensure_ffmpeg_on_path

ensure_ffmpeg_on_path()

CLEAN_PROBABILITY = 0.3  # fraction of samples left undegraded, so the model sees both


def _codec_roundtrip(audio, sr, codec, bitrate_k):
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "in.wav")
        enc_ext = "mp3" if codec == "mp3" else "opus"
        enc_path = os.path.join(tmp, f"enc.{enc_ext}")
        out_wav = os.path.join(tmp, "out.wav")

        sf.write(wav_path, audio, sr)

        codec_args = ["-codec:a", "libmp3lame"] if codec == "mp3" else ["-codec:a", "libopus"]
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, *codec_args, "-b:a", f"{bitrate_k}k", enc_path],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", enc_path, "-ar", str(sr), "-ac", "1", out_wav],
            check=True, capture_output=True,
        )
        y, _ = sf.read(out_wav)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y.astype(np.float32)


def _add_noise(audio, rng, snr_db_range=(15, 30)):
    snr_db = rng.uniform(*snr_db_range)
    sig_power = np.mean(audio ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return (audio + noise).astype(np.float32)


def _vary_amplitude(audio, rng, gain_range=(0.6, 1.4)):
    gain = rng.uniform(*gain_range)
    out = audio * gain
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


def apply_random_degradation(audio, sr, rng):
    """rng: a numpy.random.Generator (np.random.default_rng(seed)).
    Returns (degraded_audio, description_str)."""
    audio = np.asarray(audio, dtype=np.float32)

    if rng.random() < CLEAN_PROBABILITY:
        return audio, "clean"

    applied = []
    out = audio

    if rng.random() < 0.7:
        codec = rng.choice(["mp3", "opus"])
        bitrate = int(rng.choice([16, 24, 32, 48, 64, 96]))
        try:
            out = _codec_roundtrip(out, sr, codec, bitrate)
            applied.append(f"{codec}@{bitrate}k")
        except subprocess.CalledProcessError:
            pass  # fall through to the other degradations

    if rng.random() < 0.5:
        out = _add_noise(out, rng)
        applied.append("noise")

    if rng.random() < 0.5:
        out = _vary_amplitude(out, rng)
        applied.append("gain")

    if not applied:
        out = _add_noise(out, rng)
        applied.append("noise")

    return out, "+".join(applied)
