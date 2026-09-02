"""
Audio preprocessing for the AI voice detector.

load_and_preprocess() prepares a single fixed-length clip for the
classifier. chunk_audio() is kept separate because it serves a different
purpose: slicing longer audio into overlapping windows for simulated
real-time / streaming analysis later on.
"""
import numpy as np
import librosa

from ensure_ffmpeg import ensure_ffmpeg_on_path

ensure_ffmpeg_on_path()


def load_and_preprocess(file_path, sr=16000, duration=3.0):
    y, sr = librosa.load(file_path, sr=sr, mono=True)

    y, _ = librosa.effects.trim(y, top_db=20)

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    return y


def chunk_audio(audio, sr=16000, chunk_seconds=2.0, overlap=0.5):
    chunk_len = int(sr * chunk_seconds)
    hop_len = int(chunk_len * (1 - overlap))
    if hop_len <= 0:
        raise ValueError("overlap must be < 1.0")

    n = len(audio)
    if n <= chunk_len:
        chunk = np.zeros(chunk_len, dtype=audio.dtype)
        chunk[:n] = audio
        return [chunk]

    chunks = []
    start = 0
    while start < n:
        end = start + chunk_len
        if end <= n:
            chunks.append(audio[start:end])
        else:
            chunk = np.zeros(chunk_len, dtype=audio.dtype)
            remaining = audio[start:n]
            chunk[: len(remaining)] = remaining
            chunks.append(chunk)
            break
        start += hop_len

    return chunks


if __name__ == "__main__":
    import os
    import random

    import soundfile as sf

    ROOT = os.path.dirname(os.path.abspath(__file__))
    REAL_DIR = os.path.join(ROOT, "data", "real")
    FAKE_DIR = os.path.join(ROOT, "data", "fake")

    def wavs(d):
        return [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".wav")]

    real_files = wavs(REAL_DIR)
    fake_files = wavs(FAKE_DIR)

    random.seed(0)
    sample_real = random.sample(real_files, 3)
    sample_fake = random.sample(fake_files, 3)

    print("=" * 60)
    print("load_and_preprocess() test")
    print("=" * 60)
    for label, files in [("real", sample_real), ("fake", sample_fake)]:
        for path in files:
            try:
                out = load_and_preprocess(path)
                print(f"[{label}] {os.path.basename(path)} -> shape {out.shape}, "
                      f"min={out.min():.3f}, max={out.max():.3f}")
            except Exception as e:
                print(f"[{label}] {os.path.basename(path)} -> ERROR: {e}")

    print()
    print("=" * 60)
    print("chunk_audio() test")
    print("=" * 60)

    all_files = [(p, "real") for p in real_files] + [(p, "fake") for p in fake_files]
    best_path, best_label, best_diff = None, None, float("inf")
    for path, label in all_files:
        try:
            info = sf.info(path)
            diff = abs(info.duration - 8.0)
            if diff < best_diff:
                best_diff = diff
                best_path = path
                best_label = label
        except Exception:
            continue

    print(f"closest-to-8s file: [{best_label}] {os.path.basename(best_path)} "
          f"(duration={sf.info(best_path).duration:.2f}s)")

    y, sr = librosa.load(best_path, sr=16000, mono=True)
    print(f"loaded full length: {len(y)} samples ({len(y) / sr:.2f}s)")

    chunks = chunk_audio(y, sr=16000, chunk_seconds=2.0, overlap=0.5)
    print(f"chunk_audio produced {len(chunks)} chunks, each shape {chunks[0].shape}")
