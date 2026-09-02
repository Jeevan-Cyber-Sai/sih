"""
Builds data_realworld/real/: supplementary genuine-speech samples spanning
recording conditions the ASVspoof2019 LA "real" class doesn't cover
(phone mics, everyday compression, varied home-recording setups), to fix
the domain-shift misclassification found via diagnose_mismatch.py.

Sources:
  1. The user's own WhatsApp voice notes (mp3/mp4/mpeg), decoded via the
     ffmpeg fallback set up in ensure_ffmpeg.py.
  2. A random subset of LibriSpeech dev-clean (OpenSLR, ungated, plain
     HTTP with Range support) -- many different LibriVox volunteers'
     home-recording setups, as a stand-in for Common Voice (which needs
     either a multi-GB signed download or gated HF auth, both impractical
     given this session's flaky/gated network conditions).

Resumable: the tar.gz download resumes via Range on failure, and already-
converted output wavs are skipped on re-run.
"""
import io
import os
import random
import sys
import tarfile
import time

import librosa
import requests
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ensure_ffmpeg import ensure_ffmpeg_on_path  # noqa: E402

ensure_ffmpeg_on_path()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data_realworld", "real")

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
TARBALL_PATH = os.path.join(ROOT, "scripts", "_dev-clean.tar.gz")
LIBRISPEECH_TARGET_N = 220
SEED = 42

WHATSAPP_DIR = os.path.join(ROOT, "test")
WHATSAPP_EXTS = (".mp3", ".mpeg", ".mp4")
WHATSAPP_EXCLUDE = ("elevenlabs",)  # AI-generated, not genuine


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def convert_whatsapp_files():
    if not os.path.isdir(WHATSAPP_DIR):
        log(f"no {WHATSAPP_DIR} folder found, skipping WhatsApp sources")
        return 0

    files = [f for f in os.listdir(WHATSAPP_DIR) if f.lower().endswith(WHATSAPP_EXTS)]
    files = [f for f in files if not any(x in f.lower() for x in WHATSAPP_EXCLUDE)]

    n_ok = 0
    for fname in files:
        src = os.path.join(WHATSAPP_DIR, fname)
        out_name = "whatsapp_" + os.path.splitext(fname)[0].replace(" ", "_").replace(":", "") + ".wav"
        out_path = os.path.join(OUT_DIR, out_name)
        if os.path.exists(out_path):
            n_ok += 1
            continue
        try:
            y, sr = librosa.load(src, sr=16000, mono=True)
            sf.write(out_path, y, sr)
            n_ok += 1
            log(f"converted {fname} -> {out_name}")
        except Exception as e:
            log(f"SKIPPED {fname}: {e}")

    return n_ok


def download_resumable(url, dest, chunk_size=1024 * 1024, max_retries=40):
    session = requests.Session()

    total = None
    for _ in range(max_retries):
        try:
            r = session.head(url, timeout=30, allow_redirects=True)
            total = int(r.headers.get("Content-Length"))
            break
        except Exception:
            time.sleep(1)

    existing = os.path.getsize(dest) if os.path.exists(dest) else 0
    if total is not None and existing >= total:
        log(f"tarball already fully downloaded ({existing} bytes)")
        return

    pos = existing
    retries_left = max_retries
    mode = "ab" if existing else "wb"
    with open(dest, mode) as f:
        while total is None or pos < total:
            try:
                headers = {"Range": f"bytes={pos}-"}
                with session.get(url, headers=headers, stream=True, timeout=60) as resp:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pos += len(chunk)
                    f.flush()
                if total is None or pos >= total:
                    break
            except Exception as e:
                retries_left -= 1
                if retries_left <= 0:
                    raise RuntimeError(f"download failed after {max_retries} retries at byte {pos}: {e}")
                log(f"chunk failed at byte {pos} ({e}); resuming...")
                time.sleep(1)
            if total:
                log(f"download progress: {pos}/{total} bytes ({100 * pos / total:.1f}%)")

    log(f"download complete: {pos} bytes")


def extract_librispeech_subset(tar_path, out_dir, n, seed):
    log("scanning tarball for .flac entries...")
    with tarfile.open(tar_path, "r:gz") as tar:
        flac_members = [m for m in tar.getmembers() if m.name.lower().endswith(".flac")]
        log(f"found {len(flac_members)} flac utterances in archive")

        rng = random.Random(seed)
        selected = rng.sample(flac_members, min(n, len(flac_members)))

        n_ok = 0
        for i, member in enumerate(selected, 1):
            out_name = "librispeech_" + os.path.basename(member.name).replace(".flac", ".wav")
            out_path = os.path.join(out_dir, out_name)
            if os.path.exists(out_path):
                n_ok += 1
                continue
            try:
                fobj = tar.extractfile(member)
                raw = fobj.read()
                audio, sr = sf.read(io.BytesIO(raw))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != 16000:
                    audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=16000)
                    sr = 16000
                sf.write(out_path, audio, sr)
                n_ok += 1
            except Exception as e:
                log(f"  skipped {member.name}: {e}")

            if i % 50 == 0:
                log(f"  extracted {i}/{len(selected)}")

    log(f"LibriSpeech extraction done: {n_ok}/{len(selected)} converted")
    return n_ok


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    log("converting WhatsApp voice notes...")
    n_whatsapp = convert_whatsapp_files()
    log(f"WhatsApp: {n_whatsapp} files converted")

    log(f"downloading LibriSpeech dev-clean ({LIBRISPEECH_URL})...")
    download_resumable(LIBRISPEECH_URL, TARBALL_PATH)

    n_libri = extract_librispeech_subset(TARBALL_PATH, OUT_DIR, LIBRISPEECH_TARGET_N, SEED)

    total = len([f for f in os.listdir(OUT_DIR) if f.lower().endswith(".wav")])
    log(f"DONE. data_realworld/real/ now has {total} wav files "
        f"(whatsapp={n_whatsapp}, librispeech={n_libri})")


if __name__ == "__main__":
    main()
