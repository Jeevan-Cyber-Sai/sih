"""One-off: migrate the old bulk ssl_features_X.npy/y.npy cache (from the
pre-holdout, pre-per-file-cache train_ssl.py run) into the new per-file
embedding cache, so the augmented run doesn't have to re-extract 1408
already-computed embeddings."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_ssl import REAL_DIR, FAKE_DIR, EMBED_CACHE_DIR, _embed_cache_path  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "cache")
X_CACHE = os.path.join(CACHE_DIR, "ssl_features_X.npy")
Y_CACHE = os.path.join(CACHE_DIR, "ssl_features_y.npy")
MODEL_NAME_CACHE = os.path.join(CACHE_DIR, "ssl_model_name.txt")


def main():
    X = np.load(X_CACHE)
    y = np.load(Y_CACHE)
    model_name = open(MODEL_NAME_CACHE).read().strip()
    print(f"loaded old cache: X.shape={X.shape} model={model_name}")

    real_files = sorted(f for f in os.listdir(REAL_DIR) if f.lower().endswith(".wav"))
    fake_files = sorted(f for f in os.listdir(FAKE_DIR) if f.lower().endswith(".wav"))
    paths = [os.path.join(REAL_DIR, f) for f in real_files] + [os.path.join(FAKE_DIR, f) for f in fake_files]

    assert len(paths) == len(X), f"mismatch: {len(paths)} files vs {len(X)} cached rows"

    os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
    n_written = 0
    for path, vec in zip(paths, X):
        cache_path = _embed_cache_path(path, model_name)
        if not os.path.exists(cache_path):
            np.save(cache_path, vec)
            n_written += 1

    print(f"migrated {n_written} embeddings into per-file cache ({EMBED_CACHE_DIR})")


if __name__ == "__main__":
    main()
