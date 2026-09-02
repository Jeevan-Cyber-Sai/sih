"""
Training pipeline for the SSL-embedding-based voice detector (wav2vec2 /
XLS-R features instead of hand-crafted MFCCs).

Supports two modes:
  - default: data/real/ + data/fake/ only (matches the original MFCC
    baseline's dataset, for apples-to-apples CV comparison)
  - --augmented: adds data_realworld/real/ and data_realworld/fake/ (the
    balanced real-world diversification -- see degrade_audio.py) and
    saves as a separate _v2 model so the original SSL run stays available
    for comparison

Embedding extraction is slow on CPU, so each file's embedding is cached
individually in cache/ssl_embeddings/ (keyed by model name + file path),
so adding new files to the dataset only costs extraction time for the new
files, not a full recompute.

Files in the held-out evaluation manifest (holdout.py) are always
excluded from training.
"""
import argparse
import hashlib
import os
import random
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from features_ssl import SSL_LAYER, extract_ssl_features, load_ssl_model
from holdout import get_exclude_set
from preprocess import load_and_preprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
REAL_DIR = os.path.join(ROOT, "data", "real")
FAKE_DIR = os.path.join(ROOT, "data", "fake")
REALWORLD_REAL_DIR = os.path.join(ROOT, "data_realworld", "real")
REALWORLD_FAKE_DIR = os.path.join(ROOT, "data_realworld", "fake")
GENERATORS_DIR = os.path.join(ROOT, "data_generators")
# Real commercial voice-conversion/TTS sources (ElevenLabs, Respeecher) plus
# our own kNN-VC voice conversion samples -- closes the LOGO generalization
# gap. sapi/piper/edgetts deliberately stay OUT of training (pure held-out
# generalization checks).
GENERATOR_FAKE_DIRS = [
    os.path.join(GENERATORS_DIR, "elevenlabs"),
    os.path.join(GENERATORS_DIR, "respeecher"),
    os.path.join(GENERATORS_DIR, "knnvc"),
]
MODELS_DIR = os.path.join(ROOT, "models")
CACHE_DIR = os.path.join(ROOT, "cache")
EMBED_CACHE_DIR = os.path.join(CACHE_DIR, "ssl_embeddings")

XLSR_MODEL = "facebook/wav2vec2-xls-r-300m"
BASE_MODEL = "facebook/wav2vec2-base"
# if projected full-dataset extraction time with XLS-R-300M exceeds this,
# fall back to the lighter wav2vec2-base
MAX_PROJECTED_MINUTES = 90


def _embed_cache_path(path, model_name):
    key = f"{model_name}::{os.path.abspath(path)}"
    h = hashlib.sha1(key.encode()).hexdigest()
    return os.path.join(EMBED_CACHE_DIR, h + ".npy")


def get_or_compute_embedding(path, model_name):
    cache_path = _embed_cache_path(path, model_name)
    if os.path.exists(cache_path):
        return np.load(cache_path)

    audio = load_and_preprocess(path)
    vec = extract_ssl_features(audio, model_name=model_name)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, vec)
    return vec


def choose_model(n_files, sample_paths):
    print(f"benchmarking {XLSR_MODEL} on {len(sample_paths)} sample files...")
    t0 = time.time()
    load_ssl_model(XLSR_MODEL)
    print(f"  model load time: {time.time() - t0:.1f}s")

    per_file_times = []
    for path in sample_paths:
        audio = load_and_preprocess(path)
        t0 = time.time()
        extract_ssl_features(audio, model_name=XLSR_MODEL)
        per_file_times.append(time.time() - t0)

    avg_time = float(np.mean(per_file_times))
    projected_minutes = avg_time * n_files / 60
    print(f"  avg extraction time: {avg_time:.2f}s/file -> projected {projected_minutes:.1f} min "
          f"for {n_files} files (CPU-only, no CUDA)")

    if projected_minutes <= MAX_PROJECTED_MINUTES:
        print(f"  -> using {XLSR_MODEL}")
        return XLSR_MODEL

    print(f"  -> too slow for this machine; falling back to {BASE_MODEL} "
          f"(note: base is English-only, XLS-R is multilingual)")
    return BASE_MODEL


def build_dataset_ssl(sources, model_name):
    """sources: list of (dir, label) pairs. Uses the per-file embedding
    cache, so previously-processed files are essentially free."""
    exclude = get_exclude_set()
    X, y = [], []
    n_processed = 0
    n_skipped = 0
    n_excluded = 0
    n_cache_hits = 0

    for d, label in sources:
        if not os.path.isdir(d):
            print(f"WARNING: {d} does not exist, skipping")
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.lower().endswith(".wav"):
                continue
            path = os.path.join(d, fname)
            if os.path.abspath(path) in exclude:
                n_excluded += 1
                continue
            try:
                was_cached = os.path.exists(_embed_cache_path(path, model_name))
                feats = get_or_compute_embedding(path, model_name)
                if was_cached:
                    n_cache_hits += 1
                X.append(feats)
                y.append(label)
            except Exception as e:
                n_skipped += 1
                print(f"SKIPPED {path}: {e}")
                continue

            n_processed += 1
            if n_processed % 100 == 0:
                print(f"processed {n_processed} files... (cache hits so far: {n_cache_hits})", flush=True)

    print(f"build_dataset_ssl done: processed={n_processed} skipped={n_skipped} "
          f"excluded(holdout)={n_excluded} cache_hits={n_cache_hits}")
    return np.array(X), np.array(y)


def evaluate_classifier(name, clf_factory, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = clf_factory()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        metrics["accuracy"].append(accuracy_score(y_test, preds))
        metrics["precision"].append(precision_score(y_test, preds))
        metrics["recall"].append(recall_score(y_test, preds))
        metrics["f1"].append(f1_score(y_test, preds))

    summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in metrics.items()}
    print(f"\n=== {name}: 5-fold stratified CV (mean +/- std) ===")
    for k, (m, s) in summary.items():
        print(f"{k}: {m:.4f} +/- {s:.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmented", action="store_true",
                         help="train v2 on data/ + data_realworld/ (balanced real+fake) instead of "
                              "data/ alone; saves as voice_classifier_ssl_v2.joblib / scaler_ssl_v2.joblib")
    parser.add_argument("--model-name", default=None,
                         help="skip the auto speed benchmark and force a specific HF model name")
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.augmented:
        sources = [(REAL_DIR, 0), (REALWORLD_REAL_DIR, 0), (FAKE_DIR, 1), (REALWORLD_FAKE_DIR, 1)]
        sources += [(d, 1) for d in GENERATOR_FAKE_DIRS]
        model_out, scaler_out = "voice_classifier_ssl_v2.joblib", "scaler_ssl_v2.joblib"
    else:
        sources = [(REAL_DIR, 0), (FAKE_DIR, 1)]
        model_out, scaler_out = "voice_classifier_ssl.joblib", "scaler_ssl.joblib"

    t_start = time.time()

    all_files = []
    for d, _ in sources:
        if os.path.isdir(d):
            all_files += [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".wav")]
    n_files = len(all_files)

    if args.model_name:
        model_name = args.model_name
        print(f"using forced model: {model_name}")
    else:
        random.seed(0)
        sample_paths = random.sample(all_files, min(6, n_files))
        model_name = choose_model(n_files, sample_paths)

    print(f"\nextracting SSL embeddings ({model_name}, layer {SSL_LAYER}) for up to {n_files} files "
          f"(per-file cache in {EMBED_CACHE_DIR})...")
    t0 = time.time()
    X, y = build_dataset_ssl(sources, model_name)
    extraction_time = time.time() - t0
    print(f"extraction done in {extraction_time:.1f}s")

    print(f"\nX.shape={X.shape}  y.shape={y.shape}  embedding_dim={X.shape[1]}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    candidates = {
        "RandomForest": lambda: RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "MLP (128,64)": lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
        "LogisticRegression": lambda: LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
    }
    summaries = {name: evaluate_classifier(name, factory, X_scaled, y)
                 for name, factory in candidates.items()}

    best_name = max(summaries, key=lambda k: summaries[k]["f1"][0])
    print(f"\nbest head by mean F1: {best_name} (F1={summaries[best_name]['f1'][0]:.4f})")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    holdout_clf = candidates[best_name]()
    holdout_clf.fit(X_train, y_train)
    preds = holdout_clf.predict(X_test)

    print(f"\n=== Held-out 20% test set ({best_name}) ===")
    print("Confusion matrix (rows=true, cols=pred, order=[real, fake]):")
    print(confusion_matrix(y_test, preds))
    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=["real", "fake"]))

    final_model = candidates[best_name]()
    final_model.fit(X_scaled, y)

    model_path = os.path.join(MODELS_DIR, model_out)
    scaler_path = os.path.join(MODELS_DIR, scaler_out)
    joblib.dump(final_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nsaved model  -> {model_path}  (head: {best_name})")
    print(f"saved scaler -> {scaler_path}")

    print(f"\nSSL model used: {model_name}")
    print(f"embedding layer: {SSL_LAYER}")
    print(f"embedding vector length: {X.shape[1]}")
    print(f"extraction time: {extraction_time:.1f}s")
    print(f"total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
