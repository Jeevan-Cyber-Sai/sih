"""
Training pipeline for the AI voice detector: builds a feature dataset from
data/real/ and data/fake/, cross-validates a RandomForest classifier, then
fits a final model on everything and saves it for inference.
"""
import argparse
import os
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from features import extract_features
from holdout import get_exclude_set
from preprocess import load_and_preprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
REAL_DIR = os.path.join(ROOT, "data", "real")
FAKE_DIR = os.path.join(ROOT, "data", "fake")
REALWORLD_REAL_DIR = os.path.join(ROOT, "data_realworld", "real")
REALWORLD_FAKE_DIR = os.path.join(ROOT, "data_realworld", "fake")
MODELS_DIR = os.path.join(ROOT, "models")

LABELS = {"real": 0, "fake": 1}


def _build_dataset_from_sources(sources):
    """sources: list of (dir, label) pairs. Files in the held-out
    evaluation manifest (holdout.py) are always excluded."""
    exclude = get_exclude_set()
    X, y = [], []
    n_processed = 0
    n_skipped = 0
    n_excluded = 0

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
                audio = load_and_preprocess(path)
                feats = extract_features(audio)
                X.append(feats)
                y.append(label)
            except Exception as e:
                n_skipped += 1
                print(f"SKIPPED {path}: {e}")
                continue

            n_processed += 1
            if n_processed % 100 == 0:
                print(f"processed {n_processed} files...", flush=True)

    print(f"build_dataset done: processed={n_processed} skipped={n_skipped} excluded(holdout)={n_excluded}")
    return np.array(X), np.array(y)


def build_dataset():
    """Original dataset: ASVspoof data/real/ + data/fake/ only."""
    return _build_dataset_from_sources([(REAL_DIR, 0), (FAKE_DIR, 1)])


def build_dataset_augmented():
    """v3 dataset: adds data_realworld/real/ (real-world genuine speech)
    to the real class AND data_realworld/fake/ (modern TTS + degraded
    ASVspoof copies) to the fake class -- balanced diversification on
    both sides, so the model can't shortcut on "degraded = real"."""
    return _build_dataset_from_sources([
        (REAL_DIR, 0),
        (REALWORLD_REAL_DIR, 0),
        (FAKE_DIR, 1),
        (REALWORLD_FAKE_DIR, 1),
    ])


def train_and_evaluate(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        metrics["accuracy"].append(accuracy_score(y_test, preds))
        metrics["precision"].append(precision_score(y_test, preds))
        metrics["recall"].append(recall_score(y_test, preds))
        metrics["f1"].append(f1_score(y_test, preds))
        print(f"fold {fold}: acc={metrics['accuracy'][-1]:.4f} "
              f"prec={metrics['precision'][-1]:.4f} "
              f"rec={metrics['recall'][-1]:.4f} f1={metrics['f1'][-1]:.4f}")

    print("\n=== 5-fold stratified CV results (mean +/- std) ===")
    for name, vals in metrics.items():
        vals = np.array(vals)
        print(f"{name}: {vals.mean():.4f} +/- {vals.std():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    holdout_clf = RandomForestClassifier(n_estimators=200, random_state=42)
    holdout_clf.fit(X_train, y_train)
    preds = holdout_clf.predict(X_test)

    print("\n=== Held-out 20% test set ===")
    print("Confusion matrix (rows=true, cols=pred, order=[real, fake]):")
    print(confusion_matrix(y_test, preds))
    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=["real", "fake"]))

    final_model = RandomForestClassifier(n_estimators=200, random_state=42)
    final_model.fit(X_scaled, y)

    return final_model, scaler


def main():
    parser = argparse.ArgumentParser(description="Train the AI voice detector.")
    parser.add_argument("--augmented", action="store_true",
                         help="train v3 on the fully balanced dataset (data/ + data_realworld/real/ "
                              "+ data_realworld/fake/) instead of data/ alone; "
                              "saves as voice_classifier_v3.joblib / scaler_v3.joblib")
    args = parser.parse_args()

    t0 = time.time()

    if args.augmented:
        X, y = build_dataset_augmented()
        model_name, scaler_name = "voice_classifier_v3.joblib", "scaler_v3.joblib"
    else:
        X, y = build_dataset()
        model_name, scaler_name = "voice_classifier.joblib", "scaler.joblib"

    print(f"\nX.shape={X.shape}  y.shape={y.shape}")

    model, scaler = train_and_evaluate(X, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, model_name)
    scaler_path = os.path.join(MODELS_DIR, scaler_name)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nsaved model  -> {model_path}")
    print(f"saved scaler -> {scaler_path}")

    elapsed = time.time() - t0
    print(f"\ntotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
