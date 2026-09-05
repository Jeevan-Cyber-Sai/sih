"""
Applies the same channel-degradation augmentation used to build
data_realworld/{real,fake} and data_indian_augmented/{real,fake} (see
degrade_audio.apply_random_degradation) to data_hindi/ and data_tamil/,
symmetrically across both classes. Critical: both real and fake get the
same treatment, or a classifier can shortcut on "clean = one class,
degraded = the other" instead of learning genuine synthetic-speech
artifacts (see degrade_audio.py's module docstring for the ElevenLabs
regression that motivated this rule originally, and the sample-rate leak
just fixed in extract_indictts_real.py for the same class of bug).

Parameterized by language (unlike degrade_indian_dataset.py, which is a
one-off for a single fixed dataset) since this needs to run identically
for Hindi and Tamil.

Each source file gets exactly one pass through apply_random_degradation()
-- which itself leaves ~30% of files (CLEAN_PROBABILITY) untouched --
written to data_<lang>_augmented/{real,fake}/ under the same filename.
The original data_<lang>/{real,fake}/ files are left untouched, so both
the raw and augmented corpora stay available.

Usage:
    python degrade_multilingual_dataset.py hindi
    python degrade_multilingual_dataset.py tamil
"""
import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from degrade_audio import apply_random_degradation  # noqa: E402

SEED = 202


def _seed_for(seed, label, fname):
    h = hashlib.md5(f"{seed}:{label}:{fname}".encode()).hexdigest()
    return int(h[:8], 16)


def _process_one(args):
    src_dir, dest_dir, seed, label, fname = args
    src_path = os.path.join(src_dir, fname)
    dest_path = os.path.join(dest_dir, fname)
    if os.path.exists(dest_path):
        return label, fname, "skip", None
    try:
        audio, sr = sf.read(src_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        rng = np.random.default_rng(_seed_for(seed, label, fname))
        degraded, desc = apply_random_degradation(audio, sr, rng)
        sf.write(dest_path, degraded, sr)
        return label, fname, "ok", desc
    except Exception as e:
        return label, fname, "error", str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("language", choices=["hindi", "tamil"])
    args = parser.parse_args()

    src_dirs = {
        "real": os.path.join(ROOT, f"data_{args.language}", "real"),
        "fake": os.path.join(ROOT, f"data_{args.language}", "fake"),
    }
    dest_dirs = {
        "real": os.path.join(ROOT, f"data_{args.language}_augmented", "real"),
        "fake": os.path.join(ROOT, f"data_{args.language}_augmented", "fake"),
    }
    for d in dest_dirs.values():
        os.makedirs(d, exist_ok=True)

    tasks = []
    for label, src_dir in src_dirs.items():
        files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".wav"))
        tasks.extend((src_dir, dest_dirs[label], SEED, label, fname) for fname in files)

    print(f"total files to process: {len(tasks)} (real={sum(1 for t in tasks if t[3]=='real')}, "
          f"fake={sum(1 for t in tasks if t[3]=='fake')})")
    t0 = time.time()
    n_ok = n_skip = n_err = 0
    desc_counts = {}

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = [ex.submit(_process_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            label, fname, status, info = fut.result()
            if status == "ok":
                n_ok += 1
                desc_counts[info] = desc_counts.get(info, 0) + 1
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                print(f"ERROR {label}/{fname}: {info}")
            if i % 500 == 0:
                print(f"  processed {i}/{len(tasks)} ({time.time()-t0:.0f}s elapsed)", flush=True)

    elapsed = time.time() - t0
    print(f"\nDONE in {elapsed:.1f}s: ok={n_ok} skipped(existing)={n_skip} errors={n_err}")

    n_real = len([f for f in os.listdir(dest_dirs["real"]) if f.lower().endswith(".wav")])
    n_fake = len([f for f in os.listdir(dest_dirs["fake"]) if f.lower().endswith(".wav")])
    print(f"data_{args.language}_augmented/real/: {n_real} files")
    print(f"data_{args.language}_augmented/fake/: {n_fake} files")

    n_clean = desc_counts.get("clean", 0)
    n_degraded = n_ok - n_clean
    print(f"\nclean (left untouched): {n_clean} ({100*n_clean/max(n_ok,1):.1f}%)")
    print(f"degraded: {n_degraded} ({100*n_degraded/max(n_ok,1):.1f}%)")
    print("\ndegradation type breakdown:")
    for desc, count in sorted(desc_counts.items(), key=lambda x: -x[1]):
        print(f"  {desc}: {count}")


if __name__ == "__main__":
    main()
