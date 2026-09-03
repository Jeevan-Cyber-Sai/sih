"""
Applies the same channel-degradation augmentation used to build
data_realworld/{real,fake} (see degrade_audio.apply_random_degradation)
to the Indian dataset (data_indian/{real,fake}), symmetrically across
both classes. Critical: both real and fake get the same treatment, or a
classifier can shortcut on "clean = one class, degraded = the other"
instead of learning genuine synthetic-speech artifacts (see
degrade_audio.py's module docstring for the ElevenLabs regression that
motivated this rule originally).

Each source file gets exactly one pass through
apply_random_degradation() -- which itself leaves ~30% of files
(CLEAN_PROBABILITY) untouched -- written to
data_indian_augmented/{real,fake}/ under the same filename. The original
data_indian/{real,fake}/ files are left untouched, so both the raw and
augmented Indian corpora stay available (mirrors data/ vs data_realworld/
being separate, additive directories).
"""
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
SRC_DIRS = {
    "real": os.path.join(ROOT, "data_indian", "real"),
    "fake": os.path.join(ROOT, "data_indian", "fake"),
}
DEST_DIRS = {
    "real": os.path.join(ROOT, "data_indian_augmented", "real"),
    "fake": os.path.join(ROOT, "data_indian_augmented", "fake"),
}
SEED = 202


def _seed_for(label, fname):
    h = hashlib.md5(f"{SEED}:{label}:{fname}".encode()).hexdigest()
    return int(h[:8], 16)


def _process_one(args):
    label, fname = args
    src_path = os.path.join(SRC_DIRS[label], fname)
    dest_path = os.path.join(DEST_DIRS[label], fname)
    if os.path.exists(dest_path):
        return label, fname, "skip", None
    try:
        audio, sr = sf.read(src_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        rng = np.random.default_rng(_seed_for(label, fname))
        degraded, desc = apply_random_degradation(audio, sr, rng)
        sf.write(dest_path, degraded, sr)
        return label, fname, "ok", desc
    except Exception as e:
        return label, fname, "error", str(e)


def main():
    for d in DEST_DIRS.values():
        os.makedirs(d, exist_ok=True)

    tasks = []
    for label, src_dir in SRC_DIRS.items():
        files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".wav"))
        tasks.extend((label, fname) for fname in files)

    print(f"total files to process: {len(tasks)} (real={sum(1 for t in tasks if t[0]=='real')}, "
          f"fake={sum(1 for t in tasks if t[0]=='fake')})")
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
            if i % 1000 == 0:
                print(f"  processed {i}/{len(tasks)} ({time.time()-t0:.0f}s elapsed)", flush=True)

    elapsed = time.time() - t0
    print(f"\nDONE in {elapsed:.1f}s: ok={n_ok} skipped(existing)={n_skip} errors={n_err}")

    n_real = len([f for f in os.listdir(DEST_DIRS["real"]) if f.lower().endswith(".wav")])
    n_fake = len([f for f in os.listdir(DEST_DIRS["fake"]) if f.lower().endswith(".wav")])
    print(f"data_indian_augmented/real/: {n_real} files")
    print(f"data_indian_augmented/fake/: {n_fake} files")

    n_clean = desc_counts.get("clean", 0)
    n_degraded = n_ok - n_clean
    print(f"\nclean (left untouched): {n_clean} ({100*n_clean/max(n_ok,1):.1f}%)")
    print(f"degraded: {n_degraded} ({100*n_degraded/max(n_ok,1):.1f}%)")
    print("\ndegradation type breakdown:")
    for desc, count in sorted(desc_counts.items(), key=lambda x: -x[1]):
        print(f"  {desc}: {count}")


if __name__ == "__main__":
    main()
