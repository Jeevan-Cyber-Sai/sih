"""
Processes the real ElevenLabs & Respeecher dataset (600 genuine commercial
voice-conversion/TTS samples, 335 ElevenLabs + 265 Respeecher, from
metadata.xlsx) into data_generators/elevenlabs/ and data_generators/respeecher/:
resamples 22050Hz -> 16000Hz, applies the same balanced channel degradation
as every other generator (so this doesn't reintroduce the compression
shortcut), and writes them out as 16kHz mono wav.
"""
import os
import random
import sys

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from degrade_audio import apply_random_degradation  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(
    ROOT, "data_generators", "_elevenlabs_respeecher_extract",
    "Fake Audio Dataset (ElevenLabs & Respeecher)",
)
AUDIO_DIR = os.path.join(SRC_DIR, "extracted", "Fake_ElevenLabs_Respeecher")
METADATA_PATH = os.path.join(SRC_DIR, "metadata.xlsx")
OUT_ROOT = os.path.join(ROOT, "data_generators")

SEED = 313


def main():
    df = pd.read_excel(METADATA_PATH)
    print(f"metadata rows: {len(df)}")

    rng = np.random.default_rng(SEED)

    counts = {"ElevenLabs": 0, "Respeecher": 0}
    skipped = 0

    for _, row in df.iterrows():
        audio_name = str(row["Audio"]).strip()
        tool = str(row["Tool"]).strip()
        tool_key = tool.lower()

        src_path = os.path.join(AUDIO_DIR, audio_name + ".wav")
        if not os.path.exists(src_path):
            print(f"  MISSING source file: {src_path}")
            skipped += 1
            continue

        out_dir = os.path.join(OUT_ROOT, tool_key)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{tool_key}_{audio_name}.wav")
        if os.path.exists(out_path):
            counts[tool] = counts.get(tool, 0) + 1
            continue

        try:
            audio, sr = librosa.load(src_path, sr=16000, mono=True)
            audio = audio.astype(np.float32)
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak
            degraded, _ = apply_random_degradation(audio, 16000, rng)
            sf.write(out_path, degraded, 16000)
            counts[tool] = counts.get(tool, 0) + 1
        except Exception as e:
            print(f"  SKIPPED {audio_name}: {e}")
            skipped += 1

    print(f"\nDONE. converted: {counts}  skipped: {skipped}")


if __name__ == "__main__":
    main()
