"""
Phase-spectrum classifier: the fourth independent detection layer (see
phase_analysis.py for the feature extraction and rationale). A
GradientBoostingClassifier trained purely on phase-spectrum features --
none of the magnitude-spectrum-derived MFCC/SSL/prosody signals the
other three layers use, so this one can't share their blind spots.

Trained across every available language/domain source (ASVspoof,
real-world, Indian, Hindi, Tamil) rather than just the original ASVspoof
set, since phase artifacts from vocoders are a synthesis-technique
property, not an accent/language one -- more diverse fake sources should
generalize the phase classifier better than ASVspoof alone did.
"""
import os

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from holdout import get_exclude_set
from phase_analysis import extract_phase_features
from preprocess import load_and_preprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
PHASE_MODEL_PATH = os.path.join(MODELS_DIR, "phase_classifier.joblib")
PHASE_SCALER_PATH = os.path.join(MODELS_DIR, "phase_scaler.joblib")

FEATURES_CACHE_PATH = os.path.join(ROOT, "phase_features_cache.npy")
LABELS_CACHE_PATH = os.path.join(ROOT, "phase_labels_cache.npy")

# (directory, label) pairs -- label 0=real, 1=fake. Only directories that
# actually exist on disk are used; others are skipped with a warning
# rather than failing the whole run (e.g. if Hindi/Tamil data isn't
# present on a given machine).
SOURCES = [
    (os.path.join(ROOT, "data", "real"), 0),
    (os.path.join(ROOT, "data", "fake"), 1),
    (os.path.join(ROOT, "data_realworld", "real"), 0),
    (os.path.join(ROOT, "data_realworld", "fake"), 1),
    (os.path.join(ROOT, "data_indian", "real"), 0),
    (os.path.join(ROOT, "data_indian", "fake"), 1),
    (os.path.join(ROOT, "data_hindi", "real"), 0),
    (os.path.join(ROOT, "data_hindi", "fake"), 1),
    (os.path.join(ROOT, "data_tamil", "real"), 0),
    (os.path.join(ROOT, "data_tamil", "fake"), 1),
]

GB_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42)

_phase_model_cache = {}


def load_phase_model():
    if "clf" in _phase_model_cache:
        return _phase_model_cache["clf"], _phase_model_cache["scaler"]

    if not os.path.exists(PHASE_MODEL_PATH) or not os.path.exists(PHASE_SCALER_PATH):
        raise FileNotFoundError(
            f"Phase model/scaler not found in {MODELS_DIR}. Run "
            "phase_classifier.py (train_phase_classifier()) first."
        )
    clf = joblib.load(PHASE_MODEL_PATH)
    scaler = joblib.load(PHASE_SCALER_PATH)
    _phase_model_cache["clf"] = clf
    _phase_model_cache["scaler"] = scaler
    return clf, scaler


def score_phase(audio):
    """Returns a 0-100 phase-anomaly score (probability the phase
    spectrum looks synthetic) -- same scale/semantics as
    predict.score_single_clip(), so it drops straight into the same
    weighted-blend/conflict-detection logic as ssl_score and mfcc_score."""
    clf, scaler = load_phase_model()
    feats = extract_phase_features(audio).reshape(1, -1)
    feats_scaled = scaler.transform(feats)
    proba_fake = clf.predict_proba(feats_scaled)[0][1]
    return round(float(proba_fake) * 100, 2)


def _build_dataset(sources, use_cache=True):
    if use_cache and os.path.exists(FEATURES_CACHE_PATH) and os.path.exists(LABELS_CACHE_PATH):
        print(f"loading cached features from {FEATURES_CACHE_PATH} / {LABELS_CACHE_PATH}")
        return np.load(FEATURES_CACHE_PATH), np.load(LABELS_CACHE_PATH)

    exclude = get_exclude_set()
    X, y = [], []
    n_processed = n_skipped = n_excluded = 0

    for d, label in sources:
        if not os.path.isdir(d):
            print(f"WARNING: {d} does not exist, skipping")
            continue
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(".wav"))
        print(f"{d}: {len(files)} files (label={label})", flush=True)
        for fname in files:
            path = os.path.join(d, fname)
            if os.path.abspath(path) in exclude:
                n_excluded += 1
                continue
            try:
                audio = load_and_preprocess(path)
                X.append(extract_phase_features(audio))
                y.append(label)
            except Exception as e:
                n_skipped += 1
                print(f"SKIPPED {path}: {e}")
                continue
            n_processed += 1
            if n_processed % 500 == 0:
                print(f"  processed {n_processed} files...", flush=True)

    print(f"build_dataset done: processed={n_processed} skipped={n_skipped} excluded(holdout)={n_excluded}")
    X, y = np.array(X), np.array(y)

    np.save(FEATURES_CACHE_PATH, X)
    np.save(LABELS_CACHE_PATH, y)
    print(f"cached features -> {FEATURES_CACHE_PATH}")
    print(f"cached labels   -> {LABELS_CACHE_PATH}")

    return X, y


def train_phase_classifier(use_cache=True):
    os.makedirs(MODELS_DIR, exist_ok=True)

    X, y = _build_dataset(SOURCES, use_cache=use_cache)
    print(f"X.shape={X.shape}  y.shape={y.shape}  (real={int((y==0).sum())}, fake={int((y==1).sum())})")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    def make_clf():
        return GradientBoostingClassifier(**GB_PARAMS)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
        clf = make_clf()
        clf.fit(X_scaled[train_idx], y[train_idx])
        preds = clf.predict(X_scaled[test_idx])

        acc = accuracy_score(y[test_idx], preds)
        prec = precision_score(y[test_idx], preds)
        rec = recall_score(y[test_idx], preds)
        f1 = f1_score(y[test_idx], preds)
        for k, v in [("accuracy", acc), ("precision", prec), ("recall", rec), ("f1", f1)]:
            metrics[k].append(v)
        print(f"fold {fold}: accuracy={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")

    print("\n=== phase classifier: 5-fold stratified CV (mean +/- std) ===")
    for k, v in metrics.items():
        print(f"{k}: {np.mean(v):.4f} +/- {np.std(v):.4f}")

    mean_f1 = float(np.mean(metrics["f1"]))
    if mean_f1 < 0.80:
        print(f"\n*** F1 ({mean_f1:.4f}) is below the 0.80 gate -- per instructions, the intended "
              "0.15 phase_weight is likely too high; consider dropping to 0.08 before wiring this "
              "into predict.py/config.yaml. Stopping here for a decision rather than proceeding. ***")
    else:
        print(f"\nF1 ({mean_f1:.4f}) clears the 0.80 gate -- 0.15 phase_weight can proceed as planned.")

    final_clf = make_clf()
    final_clf.fit(X_scaled, y)

    joblib.dump(final_clf, PHASE_MODEL_PATH)
    joblib.dump(scaler, PHASE_SCALER_PATH)
    print(f"\nsaved model  -> {PHASE_MODEL_PATH}")
    print(f"saved scaler -> {PHASE_SCALER_PATH}")

    return final_clf, scaler


if __name__ == "__main__":
    train_phase_classifier()
