"""
Language identification layer (Part 5 of the multilingual PS requirement):
detects whether a clip is English, Hindi, or Tamil so the dashboard can
report e.g. "detected Hindi speech" alongside the risk score -- direct,
visible evidence of language-aware processing.

Deliberately NOT speechbrain/VoxLingua107 (the option named in the
original spec): that pulls in a new heavy dependency stack (speechbrain +
its own pretrained-model download) on a machine already tight on RAM
(see extract_indian_ssl_features.py's notes on this box's ~8GB ceiling).
Instead this reuses the SSL (XLS-R) embeddings already being computed for
every clip during real/fake scoring -- language is a coarse, easily
linearly-separable property of those embeddings (unlike real-vs-fake),
so a simple LogisticRegression head on top of embeddings we're computing
anyway costs one extra cheap sklearn .predict() call, no new model, no
new download, no extra RAM.

Trained on the exact same cached SSL embeddings used for
train_ssl.py --multilingual: English (ASVspoof + real-world + generator
sources), Hindi, and Tamil. IndieFake (Indian-accented ENGLISH) is folded
into the "english" class here -- it's a different accent, not a different
language, and this layer answers "which language," not "which accent."
"""
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
LANG_MODEL_PATH = os.path.join(MODELS_DIR, "language_id_classifier.joblib")
LANG_SCALER_PATH = os.path.join(MODELS_DIR, "language_id_scaler.joblib")

LANGUAGES = ["english", "hindi", "tamil"]  # index = label used during training

_lang_model_cache = {}


def load_language_model():
    if "clf" in _lang_model_cache:
        return _lang_model_cache["clf"], _lang_model_cache["scaler"]

    if not os.path.exists(LANG_MODEL_PATH) or not os.path.exists(LANG_SCALER_PATH):
        raise FileNotFoundError(
            f"Language ID model/scaler not found in {MODELS_DIR}. Run "
            "language_id.py (train_language_id()) first."
        )
    clf = joblib.load(LANG_MODEL_PATH)
    scaler = joblib.load(LANG_SCALER_PATH)
    _lang_model_cache["clf"] = clf
    _lang_model_cache["scaler"] = scaler
    return clf, scaler


def detect_language(ssl_feats):
    """ssl_feats: the SAME 2048-dim SSL embedding already computed for
    real/fake scoring (extract_ssl_features_truncated_direct output),
    reshaped to (1, -1). Returns (language_label, confidence_0_to_100)."""
    clf, scaler = load_language_model()
    feats_scaled = scaler.transform(ssl_feats)
    proba = clf.predict_proba(feats_scaled)[0]
    idx = int(np.argmax(proba))
    return str(clf.classes_[idx]), round(float(proba[idx]) * 100, 2)


def _load_language_features():
    import sys
    sys.path.insert(0, ROOT)
    from train_ssl import (  # noqa: E402
        FAKE_DIR,
        GENERATOR_FAKE_DIRS,
        REAL_DIR,
        REALWORLD_FAKE_DIR,
        REALWORLD_REAL_DIR,
        XLSR_MODEL,
        build_dataset_ssl,
    )

    english_sources = [(REAL_DIR, 0), (REALWORLD_REAL_DIR, 0), (FAKE_DIR, 1), (REALWORLD_FAKE_DIR, 1)]
    english_sources += [(d, 1) for d in GENERATOR_FAKE_DIRS]
    X_en, _ = build_dataset_ssl(english_sources, XLSR_MODEL)

    indian_path = os.path.join(ROOT, "features_indian.npy")
    indian = np.load(indian_path, allow_pickle=True).item()
    X_indian = indian["X"]

    hindi = np.load(os.path.join(ROOT, "features_hindi.npy"), allow_pickle=True).item()
    tamil = np.load(os.path.join(ROOT, "features_tamil.npy"), allow_pickle=True).item()

    X_english = np.concatenate([X_en, X_indian], axis=0)  # IndieFake = accented English, not a new language
    X = np.concatenate([X_english, hindi["X"], tamil["X"]], axis=0)
    y = np.array(
        ["english"] * len(X_english) + ["hindi"] * len(hindi["X"]) + ["tamil"] * len(tamil["X"])
    )
    return X, y


def train_language_id():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("loading SSL embeddings for language ID training (cache-backed, should be fast)...")
    X, y = _load_language_features()
    for lang in LANGUAGES:
        print(f"  {lang}: {int((y == lang).sum())} samples")
    print(f"X.shape={X.shape}  y.shape={y.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    def make_clf():
        return LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
        clf = make_clf()
        clf.fit(X_scaled[train_idx], y[train_idx])
        preds = clf.predict(X_scaled[test_idx])
        acc = accuracy_score(y[test_idx], preds)
        accs.append(acc)
        print(f"fold {fold}: accuracy={acc:.4f}")

    print(f"\n=== language ID: 5-fold stratified CV accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f} ===")

    final_clf = make_clf()
    final_clf.fit(X_scaled, y)

    joblib.dump(final_clf, LANG_MODEL_PATH)
    joblib.dump(scaler, LANG_SCALER_PATH)
    print(f"\nsaved model  -> {LANG_MODEL_PATH}")
    print(f"saved scaler -> {LANG_SCALER_PATH}")

    return final_clf, scaler


if __name__ == "__main__":
    train_language_id()
