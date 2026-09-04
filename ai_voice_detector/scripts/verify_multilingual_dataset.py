"""
Sanity-checks data_hindi/{real,fake}/ or data_tamil/{real,fake}/: counts
files, tries to load each with librosa, reports/removes anything
unreadable, and prints duration / sample-rate / class-balance stats.

Parameterized (unlike verify_dataset.py / verify_indian_dataset.py)
since this needs to run identically for two languages rather than
being a one-off report for a single fixed dataset.

Usage:
    python verify_multilingual_dataset.py hindi
    python verify_multilingual_dataset.py tamil
    python verify_multilingual_dataset.py hindi --no-remove
"""
import argparse
import os
from collections import Counter

import librosa
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    parser.add_argument("language", choices=["hindi", "tamil"])
    parser.add_argument("--no-remove", action="store_true",
                         help="flag corrupted files in the report but don't delete them")
    args = parser.parse_args()
    remove_corrupt = not args.no_remove

    real_dir = os.path.join(ROOT, f"data_{args.language}", "real")
    fake_dir = os.path.join(ROOT, f"data_{args.language}", "fake")

    real_stats, real_bad = scan("real", real_dir, remove_corrupt)
    fake_stats, fake_bad = scan("fake", fake_dir, remove_corrupt)

    print("=" * 60)
    print(f"{args.language.upper()} DATASET VERIFICATION REPORT")
    print("=" * 60)
    print(f"data_{args.language}/real/: {len(real_stats)} valid files ({len(real_bad)} corrupted)")
    print(f"data_{args.language}/fake/: {len(fake_stats)} valid files ({len(fake_bad)} corrupted)")

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


if __name__ == "__main__":
    main()
