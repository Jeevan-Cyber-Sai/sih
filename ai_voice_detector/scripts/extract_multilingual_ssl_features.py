"""
Step 5 (SSL half): extracts SSL embeddings for data_hindi/ + its
channel-degraded counterpart data_hindi_augmented/ (and same for tamil),
using the production XLS-R-300M "truncated" extractor -- same model,
same "truncated" (fp32, not quantized) variant, same per-file disk cache
scheme as extract_indian_ssl_features.py, so this stays numerically
consistent with the ASVspoof/real-world/IndieFake embeddings already
cached by train_ssl.py.

Deliberately single-worker (MAX_WORKERS=1): extract_indian_ssl_features.py
found 2 workers net-negative on this machine's 8GB RAM (each worker's own
~900MB model copy triggers swapping). At Hindi+Tamil's much smaller scale
here (1600 raw + 1600 augmented per language, vs IndieFake's ~19.5k raw
files that motivated moving that extraction to Colab GPU), single-worker
CPU throughput (~2 files/s measured on this machine) keeps total runtime
in the tens of minutes rather than hours, so this stays local.

No holdout-manifest exclusion (unlike the Indian version) -- there is no
Hindi/Tamil held-out-generator manifest yet; Part 4's per-language 20%
holdout is a stratified split at train/eval time instead (see
retrain_multilingual.py / the per-language eval script), not a
file-exclusion step here.

Output: features_<language>.npy in the project root -- a pickled dict
{"X": np.ndarray, "y": np.ndarray, "paths": list[str]}.

Usage:
    python extract_multilingual_ssl_features.py hindi
    python extract_multilingual_ssl_features.py tamil
"""
import argparse
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from features_ssl import extract_ssl_features_truncated_direct  # noqa: E402
from preprocess import load_and_preprocess  # noqa: E402
from train_ssl import XLSR_MODEL, _embed_cache_path  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "cache", "ssl_embeddings")
MAX_WORKERS = 1


def _process_one(args):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("language", choices=["hindi", "tamil"])
    args = parser.parse_args()

    sources = [
        (os.path.join(ROOT, f"data_{args.language}", "real"), 0),
        (os.path.join(ROOT, f"data_{args.language}_augmented", "real"), 0),
        (os.path.join(ROOT, f"data_{args.language}", "fake"), 1),
        (os.path.join(ROOT, f"data_{args.language}_augmented", "fake"), 1),
    ]
    out_path = os.path.join(ROOT, f"features_{args.language}.npy")
    os.makedirs(CACHE_DIR, exist_ok=True)

    tasks = []
    for d, label in sources:
        if not os.path.isdir(d):
            print(f"WARNING: {d} does not exist, skipping")
            continue
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(".wav"))
        tasks.extend((os.path.join(d, fname), label) for fname in files)

    print(f"total files to embed: {len(tasks)}")

    X, y, paths = [], [], []
    n_skipped = n_cache_hits = 0
    t0 = time.time()

    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=mp_context) as ex:
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

            if i % 200 == 0:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - i) / rate if rate > 0 else float("inf")
                print(f"processed {i}/{len(tasks)} (cache hits: {n_cache_hits}, "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", flush=True)

    X = np.array(X)
    y = np.array(y)
    elapsed = time.time() - t0
    print(f"\ndone in {elapsed:.1f}s: processed={len(paths)} skipped(errors)={n_skipped} "
          f"cache_hits={n_cache_hits}")
    print(f"X.shape={X.shape}  y.shape={y.shape}  (real={int((y==0).sum())}, fake={int((y==1).sum())})")

    np.save(out_path, {"X": X, "y": y, "paths": paths}, allow_pickle=True)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
