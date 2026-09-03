"""
Sanity-checks data_indian/real/ and data_indian/fake/ (the IndieFake
Dataset, organized by scripts/organize_indian_dataset.py): counts files,
tries to load each with librosa, reports/removes anything unreadable, and
prints duration / sample-rate / class-balance stats.

Same pattern as verify_dataset.py -- kept as a separate script rather than
a parameterized one so each dataset's report stays a simple, obvious
command to run.

Usage:
    python verify_indian_dataset.py              # removes corrupted files
    python verify_indian_dataset.py --no-remove   # only flags them, leaves files in place
"""
import argparse
import os
from collections import Counter

import librosa
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REAL_DIR = os.path.join(ROOT, "data_indian", "real")
FAKE_DIR = os.path.join(ROOT, "data_indian", "fake")

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
IMBALANCE_THRESHOLD = 2.0


def list_audio_files(d):
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in AUDIO_EXTS)


def scan(label, d, remove_corrupt):
    stats, corrupted = [], []
    for fname in list_audio_files(d):
        path = os.path.join(d, fname)
        try:
            y, sr = librosa.load(path, sr=None, mono=True)
            if len(y) == 0:
                raise ValueError("zero-length audio")
            stats.append({"label": label, "file": fname, "duration": len(y) / sr, "sr": sr})
        except Exception as e:
            corrupted.append((path, str(e)))

    if remove_corrupt:
        for path, _ in corrupted:
            try:
                os.remove(path)
            except OSError:
                pass

    return stats, corrupted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-remove", action="store_true",
        help="flag corrupted files in the report but don't delete them",
    )
    args = parser.parse_args()
    remove_corrupt = not args.no_remove

    real_stats, real_bad = scan("real", REAL_DIR, remove_corrupt)
    fake_stats, fake_bad = scan("fake", FAKE_DIR, remove_corrupt)

    print("=" * 60)
    print("INDIEFAKE DATASET (INDIAN-ENGLISH) VERIFICATION REPORT")
    print("=" * 60)
    print(f"data_indian/real/: {len(real_stats)} valid files ({len(real_bad)} corrupted)")
    print(f"data_indian/fake/: {len(fake_stats)} valid files ({len(fake_bad)} corrupted)")

    bad = real_bad + fake_bad
    if bad:
        action = "removed" if remove_corrupt else "flagged, NOT removed"
        print(f"\nCorrupted/unreadable files ({action}):")
        for path, err in bad:
            print(f"  {path}: {err}")

    all_stats = real_stats + fake_stats
    if not all_stats:
        print("\nNo valid audio files found.")
        return

    durations = np.array([s["duration"] for s in all_stats])
    sr_counts = Counter(s["sr"] for s in all_stats)

    print("\n--- Duration stats (seconds, all valid files) ---")
    print(f"  average: {durations.mean():.2f}")
    print(f"  min:     {durations.min():.2f}")
    print(f"  max:     {durations.max():.2f}")

    print("\n--- Duration stats by class ---")
    for label, stats in (("real", real_stats), ("fake", fake_stats)):
        if not stats:
            continue
        d = np.array([s["duration"] for s in stats])
        print(f"  {label}: avg={d.mean():.2f}s  min={d.min():.2f}s  max={d.max():.2f}s")

    print("\n--- Sample rate distribution ---")
    for sr, count in sorted(sr_counts.items()):
        print(f"  {sr} Hz: {count} files")

    n_real, n_fake = len(real_stats), len(fake_stats)
    print("\n--- Class balance ---")
    print(f"  real: {n_real}   fake: {n_fake}")
    if n_real and n_fake:
        ratio = max(n_real, n_fake) / min(n_real, n_fake)
        if ratio > IMBALANCE_THRESHOLD:
            larger = "real" if n_real > n_fake else "fake"
            print(f"  WARNING: class imbalance -- {larger} has {ratio:.2f}x as many samples as the other class")
        else:
            print(f"  balanced (ratio {ratio:.2f}x)")
    else:
        print("  WARNING: at least one class has zero valid files")

    speakers = set()
    for stats in (real_stats, fake_stats):
        for s in stats:
            speakers.add(s["file"].split("_")[0])
    print("\n--- Speaker coverage ---")
    print(f"  distinct speaker/source tags: {len(speakers)}")


if __name__ == "__main__":
    main()
