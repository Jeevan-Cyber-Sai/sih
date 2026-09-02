"""
Builds data_realworld/fake/: the symmetric counterpart to
data_realworld/real/, needed to close the "compressed/noisy = real"
shortcut the model learned when only the real class was diversified.

Two sources, both run through the same randomized channel degradation as
data_realworld/real/'s natural conditions:
  1. Modern TTS speech (gTTS, several English accents) synthesized from
     LibriSpeech transcripts -- genuinely different synthesis technology
     from ASVspoof2019's 2019-era spoofing systems.
  2. A degraded copy of a random subset of the existing ASVspoof
     data/fake/ samples, so the fake class spans clean AND degraded
     conditions symmetrically with the real class. The clean_fake holdout
     files (see holdout.py) are excluded from this sampling pool.
"""
import io
import os
import random
import sys
import tarfile
import time

import librosa
import numpy as np
import soundfile as sf
from gtts import gTTS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from degrade_audio import apply_random_degradation  # noqa: E402
from ensure_ffmpeg import ensure_ffmpeg_on_path  # noqa: E402
from holdout import get_exclude_set  # noqa: E402

ensure_ffmpeg_on_path()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data_realworld", "fake")
FAKE_DIR = os.path.join(ROOT, "data", "fake")
TARBALL_PATH = os.path.join(ROOT, "scripts", "_dev-clean.tar.gz")
TMP_DIR = os.path.join(ROOT, "scripts", "_gtts_tmp")

N_TTS_TARGET = 200
N_DEGRADED_COPIES = 200
SEED = 77
ACCENTS = ["com", "co.uk", "com.au", "co.in", "ca"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_librispeech_sentences(tar_path, n_needed, seed):
    sentences = []
    with tarfile.open(tar_path, "r:gz") as tar:
        trans_members = [m for m in tar.getmembers() if m.name.endswith(".trans.txt")]
        rng = random.Random(seed)
        rng.shuffle(trans_members)
        for m in trans_members:
            f = tar.extractfile(m)
            text = f.read().decode("utf-8", errors="ignore")
            for line in text.strip().split("\n"):
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    sentences.append(parts[1].strip())
            if len(sentences) >= n_needed * 4:
                break

    rng.shuffle(sentences)
    seen, filtered = set(), []
    for s in sentences:
        if 25 <= len(s) <= 180 and s not in seen:
            seen.add(s)
            filtered.append(s)
    return filtered[:n_needed]


def synth_gtts(text, tld, out_path, max_retries=10):
    for _ in range(max_retries):
        try:
            gTTS(text=text, lang="en", tld=tld).save(out_path)
            return True
        except Exception:
            time.sleep(1)
    return False


def build_modern_tts_fake(n_target, seed):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    sentences = load_librispeech_sentences(TARBALL_PATH, n_target, seed)
    log(f"loaded {len(sentences)} candidate sentences from LibriSpeech transcripts")

    py_rng = random.Random(seed)
    rng = np.random.default_rng(seed)

    n_ok = 0
    for i, text in enumerate(sentences):
        out_path = os.path.join(OUT_DIR, f"tts_gtts_{i:04d}.wav")
        if os.path.exists(out_path):
            n_ok += 1
            continue

        tld = py_rng.choice(ACCENTS)
        mp3_path = os.path.join(TMP_DIR, f"tts_{i:04d}.mp3")
        if not synth_gtts(text, tld, mp3_path):
            log(f"  gTTS failed for sample {i}, skipping")
            continue

        try:
            audio, sr = librosa.load(mp3_path, sr=16000, mono=True)
            degraded, desc = apply_random_degradation(audio, sr, rng)
            sf.write(out_path, degraded, sr)
            n_ok += 1
        except Exception as e:
            log(f"  SKIPPED tts sample {i}: {e}")

        if (i + 1) % 25 == 0:
            log(f"  synthesized {i + 1}/{len(sentences)}")

    log(f"modern TTS fake: {n_ok}/{len(sentences)} converted")
    return n_ok


def build_degraded_asvspoof_copies(n_target, seed, exclude_paths):
    files = [f for f in os.listdir(FAKE_DIR) if f.lower().endswith(".wav")]
    files = [f for f in files if os.path.abspath(os.path.join(FAKE_DIR, f)) not in exclude_paths]

    py_rng = random.Random(seed)
    selected = py_rng.sample(files, min(n_target, len(files)))
    rng = np.random.default_rng(seed)

    n_ok = 0
    for i, fname in enumerate(selected):
        out_path = os.path.join(OUT_DIR, f"degraded_{fname}")
        if os.path.exists(out_path):
            n_ok += 1
            continue
        try:
            audio, sr = sf.read(os.path.join(FAKE_DIR, fname))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            degraded, desc = apply_random_degradation(audio.astype(np.float32), sr, rng)
            sf.write(out_path, degraded, sr)
            n_ok += 1
        except Exception as e:
            log(f"  SKIPPED {fname}: {e}")

        if (i + 1) % 50 == 0:
            log(f"  degraded {i + 1}/{len(selected)}")

    log(f"degraded ASVspoof copies: {n_ok}/{len(selected)} converted")
    return n_ok


def main():
    exclude_paths = get_exclude_set()
    log(f"holdout exclude set: {len(exclude_paths)} files")

    log("building modern TTS fake samples (gTTS)...")
    n_tts = build_modern_tts_fake(N_TTS_TARGET, SEED)

    log("building degraded copies of ASVspoof fake samples...")
    n_degraded = build_degraded_asvspoof_copies(N_DEGRADED_COPIES, SEED, exclude_paths)

    total = len([f for f in os.listdir(OUT_DIR) if f.lower().endswith(".wav")])
    log(f"DONE. data_realworld/fake/ now has {total} wav files "
        f"(tts={n_tts}, degraded_asvspoof={n_degraded})")


if __name__ == "__main__":
    main()
