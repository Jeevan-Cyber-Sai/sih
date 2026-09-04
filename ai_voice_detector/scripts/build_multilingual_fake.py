"""
Synthesizes fake (AI-generated) Hindi/Tamil speech to pair with the
genuine IndicTTS samples pulled by extract_indictts_real.py.

Uses Microsoft Edge's neural TTS (via the edge-tts package) rather than
Coqui/XTTS: XTTS v2 does not support Tamil at all (Hindi-only among
Indian languages), and edge-tts is already a proven generator in this
repo (see build_generator_dataset.py's build_edge_tts) with real hi-IN
and ta-IN neural voices, so this reuses that exact same
isolated-subprocess-per-call pattern instead of introducing a second,
partially-unsupported TTS stack.

Synthesizes from the SAME transcripts.json produced alongside the real
extraction, so real/fake pairs share sentence content and only differ
on the genuine-vs-synthetic axis (mirrors how ASVspoof/real-world pairs
are built elsewhere in this project).

No degradation is applied here -- raw data_<lang>/{real,fake}/ stays
undegraded, matching data_indian/{real,fake}/'s pattern; the balanced
channel-degradation augmentation step comes later as its own script,
same as degrade_indian_dataset.py did for data_indian/.

Usage:
    python build_multilingual_fake.py hindi
    python build_multilingual_fake.py tamil
"""
import argparse
import json
import os
import random
import subprocess
import sys

import librosa
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EDGE_VOICES = {
    "hindi": ["hi-IN-MadhurNeural", "hi-IN-SwaraNeural"],
    "tamil": ["ta-IN-ValluvarNeural", "ta-IN-PallaviNeural"],
}


def _synth_one_edge(text, voice, tmp_path, timeout=15):
    """Isolated subprocess per call so one stalled network call can't
    hang the whole batch -- same rationale as build_generator_dataset.py."""
    code = (
        "import asyncio, edge_tts\n"
        "async def main():\n"
        f"    c = edge_tts.Communicate({text!r}, {voice!r})\n"
        f"    await c.save({tmp_path!r})\n"
        "asyncio.run(main())\n"
    )
    try:
        subprocess.run([sys.executable, "-c", code], timeout=timeout,
                        capture_output=True, check=True)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("language", choices=sorted(EDGE_VOICES))
    args = parser.parse_args()

    transcripts_path = os.path.join(ROOT, f"data_{args.language}", "transcripts.json")
    if not os.path.exists(transcripts_path):
        raise FileNotFoundError(
            f"{transcripts_path} not found -- run extract_indictts_real.py {args.language} first."
        )
    with open(transcripts_path, "r", encoding="utf-8") as f:
        transcripts = json.load(f)

    dest_dir = os.path.join(ROOT, f"data_{args.language}", "fake")
    os.makedirs(dest_dir, exist_ok=True)

    voices = EDGE_VOICES[args.language]
    rng = random.Random(42)

    n_ok = n_skip = 0
    for i, (real_name, text) in enumerate(sorted(transcripts.items())):
        if not text.strip():
            n_skip += 1
            continue
        out_path = os.path.join(dest_dir, f"edgetts_{args.language}_{i:04d}.wav")
        if os.path.exists(out_path):
            n_ok += 1
            continue

        voice = voices[i % len(voices)]  # alternate for gender balance
        tmp_path = out_path + ".raw.mp3"
        if not _synth_one_edge(text, voice, tmp_path):
            print(f"  SKIPPED {i} ({voice}): synthesis timed out or failed")
            n_skip += 1
            continue
        try:
            audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
            sf.write(out_path, audio.astype(np.float32), sr)
            n_ok += 1
        except Exception as e:
            print(f"  SKIPPED {i}: {e}")
            n_skip += 1
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if (i + 1) % 20 == 0:
            print(f"  {args.language}: {i + 1}/{len(transcripts)}  (ok={n_ok} skip={n_skip})", flush=True)

    print(f"\nwrote {n_ok} fake samples -> {dest_dir} ({n_skip} skipped)")


if __name__ == "__main__":
    main()
