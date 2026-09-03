"""
Step 4: extracts SSL embeddings for the Indian dataset (data_indian/ +
its channel-degraded counterpart data_indian_augmented/, see
degrade_indian_dataset.py) using the production XLS-R-300M "truncated"
extractor (layer-6 truncated, encoder layers 7-24 never execute --
numerically EXACT match to the full model per features_ssl.py's
docstring, RTF ~0.116 vs ~0.358 for the full model).

Deliberately NOT the quantized variant: predict.py documents that
"truncated" (fp32) is the actual production default (SSL_VARIANT =
"truncated") -- quantized was measured strictly worse on this machine
(same speed, higher peak RAM) and is only kept around for other
hardware. Using "truncated" also keeps these new embeddings numerically
identical to the ASVspoof/real-world embeddings already cached by
train_ssl.py (which reads layer 6 off the FULL model -- numerically
equal to "truncated" by construction), so the combined training set in
step 5 draws from one consistent feature space instead of mixing in a
quantization-perturbed subset.

Reuses train_ssl.py's per-file embedding cache (same cache key scheme:
model_name + absolute path -> sha1 -> cache/ssl_embeddings/<hash>.npy),
so files already embedded by an earlier run are free, and any other
script that later touches these same files (with the same model_name)
gets instant cache hits too.

Held-out files (see build_holdout_manifest_indian.py's indian_real /
indian_fake quadrants) are excluded from BOTH data_indian/ and their
degraded copy in data_indian_augmented/ (matched by basename) -- a plain
path-based exclude set would miss the degraded copy sitting under a
different directory, silently leaking a "held-out" test file into
training via its degraded twin.

Output: features_indian.npy in the project root -- a pickled dict
{"X": np.ndarray, "y": np.ndarray, "paths": list[str]} covering every
non-excluded Indian file (clean + degraded), for step 5 to load directly
instead of re-walking the directories.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from features_ssl import extract_ssl_features_truncated_direct  # noqa: E402
from holdout import get_exclude_set, load_manifest  # noqa: E402
from preprocess import load_and_preprocess  # noqa: E402
from train_ssl import XLSR_MODEL, _embed_cache_path  # noqa: E402

INDIAN_REAL_DIR = os.path.join(ROOT, "data_indian", "real")
INDIAN_FAKE_DIR = os.path.join(ROOT, "data_indian", "fake")
INDIAN_AUG_REAL_DIR = os.path.join(ROOT, "data_indian_augmented", "real")
INDIAN_AUG_FAKE_DIR = os.path.join(ROOT, "data_indian_augmented", "fake")
CACHE_DIR = os.path.join(ROOT, "cache", "ssl_embeddings")
OUT_PATH = os.path.join(ROOT, "features_indian.npy")

# This machine has only ~8GB RAM and is already running near its commit
# limit (observed <1GB physically available, committed memory well past
# physical RAM via page file even at idle). Tried 2 workers first -- each
# loading its own ~900MB truncated-model copy triggered heavy swapping
# and measured SLOWER (~0.7 files/s) than a single process's benchmark
# (~2 files/s), i.e. parallelism was net negative here. 1 avoids the
# contention entirely.
MAX_WORKERS = 1

SOURCES = [
    (INDIAN_REAL_DIR, 0),
    (INDIAN_AUG_REAL_DIR, 0),
    (INDIAN_FAKE_DIR, 1),
    (INDIAN_AUG_FAKE_DIR, 1),
]


def build_augmented_exclude_basenames():
    """Basenames of held-out indian_real/indian_fake files -- their
    degraded twin in data_indian_augmented/ must be excluded too."""
    names = set()
    for e in load_manifest():
        if e["quadrant"] in ("indian_real", "indian_fake"):
            names.add(os.path.basename(e["path"]))
    return names


def _process_one(args):
    """Runs in a worker process. Each worker loads (and caches, within
    itself) its own copy of the truncated model on first call; torch is
    pinned to 1 thread per worker so N worker processes on N cores don't
    oversubscribe via BLAS/MKL internal threading."""
    import torch
    torch.set_num_threads(1)

    path, label = args
    cache_path = _embed_cache_path(path, XLSR_MODEL)
    if os.path.exists(cache_path):
        return path, label, np.load(cache_path), True, None
    try:
        audio = load_and_preprocess(path)
        vec = extract_ssl_features_truncated_direct(audio, model_name=XLSR_MODEL, quantize=False)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, vec)
        return path, label, vec, False, None
    except Exception as e:
        return path, label, None, False, str(e)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    exclude_paths = get_exclude_set()
    exclude_basenames = build_augmented_exclude_basenames()
    print(f"holdout exclude: {len(exclude_paths)} exact paths, "
          f"{len(exclude_basenames)} indian basenames (also excluded from data_indian_augmented/)")

    tasks = []
    for d, label in SOURCES:
        if not os.path.isdir(d):
            print(f"WARNING: {d} does not exist, skipping")
            continue
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(".wav"))
        for fname in files:
            path = os.path.join(d, fname)
            abspath = os.path.abspath(path)
            if abspath in exclude_paths or fname in exclude_basenames:
                continue
            tasks.append((path, label))

    n_excluded = sum(len(os.listdir(d)) for d, _ in SOURCES if os.path.isdir(d)) - len(tasks)
    print(f"total files to embed: {len(tasks)} (excluded: {n_excluded})")

    X, y, paths = [], [], []
    n_skipped = n_cache_hits = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_process_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            path, label, vec, was_cached, err = fut.result()
            if err is not None:
                n_skipped += 1
                print(f"SKIPPED {path}: {err}")
                continue
            if was_cached:
                n_cache_hits += 1
            X.append(vec)
            y.append(label)
            paths.append(path)

            if i % 1000 == 0:
                elapsed = time.time() - t0
                print(f"processed {i}/{len(tasks)} "
                      f"(cache hits: {n_cache_hits}, {elapsed:.0f}s elapsed)", flush=True)

    X = np.array(X)
    y = np.array(y)
    elapsed = time.time() - t0
    print(f"\ndone in {elapsed:.1f}s: processed={len(paths)} excluded(holdout)={n_excluded} "
          f"skipped(errors)={n_skipped} cache_hits={n_cache_hits}")
    print(f"X.shape={X.shape}  y.shape={y.shape}  (real={int((y==0).sum())}, fake={int((y==1).sum())})")

    np.save(OUT_PATH, {"X": X, "y": y, "paths": paths}, allow_pickle=True)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
