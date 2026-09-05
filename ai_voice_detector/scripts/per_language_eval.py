"""
Part 4 of the multilingual PS requirement: does the SSL model actually
work per-language, or is the 99% overall number from train_ssl.py
--multilingual propped up by one easy subset?

IMPORTANT: this can't reuse models/voice_classifier_ssl_multilingual.joblib
for eval -- that file is refit on ALL data (train_ssl.py's standard
"report CV metrics, then refit final model on everything for production"
pattern), so every Hindi/Tamil/Indian-Eng file was already in its
training set. Testing it on a "held-out" slice would just be evaluating
training data. This script trains its own fresh model on an 80% split
(stratified per language AND class) so the 20% per-language test sets are
genuinely unseen.

Two things, both using the winning SSL head (MLP 128,64):

1. Per-language breakdown (--breakdown): one combined model trained on
   the union of each language's 80% train split; report real/fake/overall
   accuracy separately per language on its own 20% test split.

2. Leave-one-language-out (--logo): train on the OTHER three languages'
   FULL data (not just 80% -- the held-out language contributes nothing
   at all), test on the held-out language's FULL data. Answers "does this
   generalize to an Indian language with zero training exposure, or does
   it need explicit per-language data."

Usage:
    python per_language_eval.py --breakdown
    python per_language_eval.py --logo
    python per_language_eval.py --breakdown --logo
"""
import argparse
import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

LANGUAGES = ["english", "indian_eng", "hindi", "tamil"]


def load_language_features():
    """Returns {language: (X, y)} using the exact same cached SSL
    embeddings train_ssl.py --multilingual already computed -- nothing
    here re-extracts anything, it's all cache hits."""
    feats = {}

    english_sources = [(REAL_DIR, 0), (REALWORLD_REAL_DIR, 0), (FAKE_DIR, 1), (REALWORLD_FAKE_DIR, 1)]
    english_sources += [(d, 1) for d in GENERATOR_FAKE_DIRS]
    X_en, y_en = build_dataset_ssl(english_sources, XLSR_MODEL)
    feats["english"] = (X_en, y_en)

    for lang, fname in [("indian_eng", "features_indian.npy"), ("hindi", "features_hindi.npy"),
                         ("tamil", "features_tamil.npy")]:
        path = os.path.join(ROOT, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        d = np.load(path, allow_pickle=True).item()
        feats[lang] = (d["X"], d["y"])

    return feats


def per_class_accuracy(y_true, y_pred):
    real_mask = y_true == 0
    fake_mask = y_true == 1
    real_acc = accuracy_score(y_true[real_mask], y_pred[real_mask]) if real_mask.any() else float("nan")
    fake_acc = accuracy_score(y_true[fake_mask], y_pred[fake_mask]) if fake_mask.any() else float("nan")
    overall_acc = accuracy_score(y_true, y_pred)
    return real_acc, fake_acc, overall_acc


def make_head():
    return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True)


def run_breakdown(feats):
    print("\n" + "=" * 70)
    print("PART 4 STEP 7: PER-LANGUAGE BREAKDOWN (held-out 20% per language)")
    print("=" * 70)

    train_X, train_y, test_sets = [], [], {}
    for lang in LANGUAGES:
        X, y = feats[lang]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        train_X.append(X_train)
        train_y.append(y_train)
        test_sets[lang] = (X_test, y_test)
        print(f"{lang}: train={len(y_train)} test={len(y_test)} "
              f"(test real={int((y_test==0).sum())} fake={int((y_test==1).sum())})")

    X_train_all = np.concatenate(train_X, axis=0)
    y_train_all = np.concatenate(train_y, axis=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_all)

    print(f"\nfitting MLP(128,64) on combined train set: X.shape={X_train_scaled.shape}", flush=True)
    t0 = time.time()
    clf = make_head()
    clf.fit(X_train_scaled, y_train_all)
    print(f"fit done in {time.time()-t0:.1f}s")

    rows = []
    all_test_X, all_test_y = [], []
    display_names = {"english": "English", "indian_eng": "Indian-Eng", "hindi": "Hindi", "tamil": "Tamil"}
    for lang in LANGUAGES:
        X_test, y_test = test_sets[lang]
        X_test_scaled = scaler.transform(X_test)
        preds = clf.predict(X_test_scaled)
        real_acc, fake_acc, overall_acc = per_class_accuracy(y_test, preds)
        rows.append((display_names[lang], real_acc, fake_acc, overall_acc))
        all_test_X.append(X_test)
        all_test_y.append(y_test)

    X_overall = np.concatenate(all_test_X, axis=0)
    y_overall = np.concatenate(all_test_y, axis=0)
    preds_overall = clf.predict(scaler.transform(X_overall))
    real_acc, fake_acc, overall_acc = per_class_accuracy(y_overall, preds_overall)
    rows.append(("OVERALL", real_acc, fake_acc, overall_acc))

    print(f"\n{'Language':<14}{'Real acc':>10}{'Fake acc':>10}{'Overall':>10}")
    for name, real_acc, fake_acc, overall_acc in rows:
        print(f"{name:<14}{100*real_acc:>9.2f}%{100*fake_acc:>9.2f}%{100*overall_acc:>9.2f}%")


def run_logo(feats):
    print("\n" + "=" * 70)
    print("PART 4 STEP 8: LEAVE-ONE-LANGUAGE-OUT")
    print("=" * 70)

    for held_out in ["hindi", "tamil"]:
        train_langs = [l for l in LANGUAGES if l != held_out]
        print(f"\n--- held out: {held_out}  (train on {', '.join(train_langs)}) ---")

        X_train = np.concatenate([feats[l][0] for l in train_langs], axis=0)
        y_train = np.concatenate([feats[l][1] for l in train_langs], axis=0)
        X_test, y_test = feats[held_out]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"train: X.shape={X_train_scaled.shape}  test ({held_out}): X.shape={X_test_scaled.shape}",
              flush=True)
        t0 = time.time()
        clf = make_head()
        clf.fit(X_train_scaled, y_train)
        print(f"fit done in {time.time()-t0:.1f}s")

        preds = clf.predict(X_test_scaled)
        real_acc, fake_acc, overall_acc = per_class_accuracy(y_test, preds)
        print(f"\n{held_out} detection (zero training exposure):")
        print(f"  real acc:    {100*real_acc:.2f}%")
        print(f"  fake acc:    {100*fake_acc:.2f}%")
        print(f"  overall acc: {100*overall_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--logo", action="store_true")
    args = parser.parse_args()
    if not args.breakdown and not args.logo:
        args.breakdown = args.logo = True

    print("loading per-language SSL features (cache-backed, should be fast)...")
    feats = load_language_features()
    for lang in LANGUAGES:
        X, y = feats[lang]
        print(f"  {lang}: X.shape={X.shape}  (real={int((y==0).sum())}, fake={int((y==1).sum())})")

    if args.breakdown:
        run_breakdown(feats)
    if args.logo:
        run_logo(feats)


if __name__ == "__main__":
    main()
