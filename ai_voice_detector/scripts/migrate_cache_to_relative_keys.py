"""
One-off: re-keys cache/ssl_embeddings/*.npy from the old absolute-path
cache key to the new path-relative-to-project-root key (see the
_embed_cache_path docstring in train_ssl.py for why: an absolute path is
machine-specific and silently breaks cache reuse the moment this project
runs somewhere else, e.g. Colab).

Walks every directory this project has ever pulled SSL embeddings from,
computes each file's OLD cache path (old scheme) and NEW cache path (new
scheme), and copies the old cache file to the new name if the old one
exists and the new one doesn't yet. Old files are left in place (cheap,
and harmless if orphaned) rather than deleted, so this is safe to re-run
or abort halfway through.
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_ssl import EMBED_CACHE_DIR, GENERATOR_FAKE_DIRS  # noqa: E402
from train_ssl import BASE_MODEL, FAKE_DIR, REAL_DIR, REALWORLD_FAKE_DIR, REALWORLD_REAL_DIR, ROOT, XLSR_MODEL  # noqa: E402

MODEL_NAMES = [XLSR_MODEL, BASE_MODEL]

SOURCE_DIRS = [
    REAL_DIR, FAKE_DIR, REALWORLD_REAL_DIR, REALWORLD_FAKE_DIR,
    *GENERATOR_FAKE_DIRS,
    os.path.join(ROOT, "data_generators", "sapi"),
    os.path.join(ROOT, "data_generators", "piper"),
    os.path.join(ROOT, "data_generators", "edgetts"),
    os.path.join(ROOT, "data_indian", "real"),
    os.path.join(ROOT, "data_indian", "fake"),
    os.path.join(ROOT, "data_indian_augmented", "real"),
    os.path.join(ROOT, "data_indian_augmented", "fake"),
]


def old_cache_path(path, model_name):
    key = f"{model_name}::{os.path.abspath(path)}"
    h = hashlib.sha1(key.encode()).hexdigest()
    return os.path.join(EMBED_CACHE_DIR, h + ".npy")


def new_cache_path(path, model_name):
    rel = os.path.relpath(os.path.abspath(path), ROOT).replace(os.sep, "/")
    key = f"{model_name}::{rel}"
    h = hashlib.sha1(key.encode()).hexdigest()
    return os.path.join(EMBED_CACHE_DIR, h + ".npy")


def main():
    n_migrated = n_already = n_missing = 0

    for d in SOURCE_DIRS:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.lower().endswith(".wav"):
                continue
            path = os.path.join(d, fname)
            for model_name in MODEL_NAMES:
                old_path = old_cache_path(path, model_name)
                if not os.path.exists(old_path):
                    continue
                new_path = new_cache_path(path, model_name)
                if os.path.exists(new_path):
                    n_already += 1
                    continue
                with open(old_path, "rb") as src, open(new_path, "wb") as dst:
                    dst.write(src.read())
                n_migrated += 1

    print(f"migrated: {n_migrated}")
    print(f"already at new key: {n_already}")


if __name__ == "__main__":
    main()
