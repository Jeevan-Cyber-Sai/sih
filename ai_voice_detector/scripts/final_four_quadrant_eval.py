"""
Final four-quadrant evaluation of the CURRENT production models (post
ElevenLabs/Respeecher/kNN-VC retrain) on the expanded held-out set, with
Wilson score confidence intervals per quadrant. Reports three rows: SSL
v2 alone, MFCC v3 alone, and the Dual-combined score actually used in
production (predict.score_dual: 0.7*SSL + 0.3*MFCC).
"""
import math
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import extract_features  # noqa: E402
from features_ssl import extract_ssl_features_truncated_direct  # noqa: E402
from holdout import get_quadrants  # noqa: E402
from preprocess import load_and_preprocess  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

QUADRANT_TRUE_LABEL = {
    "clean_real": 0,
    "clean_fake": 1,
    "realworld_real": 0,
    "realworld_fake": 1,
}
QUADRANT_ORDER = ["clean_real", "clean_fake", "realworld_real", "realworld_fake"]
DUAL_WEIGHTS = {"ssl": 0.7, "mfcc": 0.3}  # must match predict.DUAL_SCORE_WEIGHTS


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def score_ssl(path, clf, scaler):
    audio = load_and_preprocess(path)
    feats = extract_ssl_features_truncated_direct(audio)
    return float(clf.predict_proba(scaler.transform(feats.reshape(1, -1)))[0][1]) * 100


def score_mfcc(path, clf, scaler):
    audio = load_and_preprocess(path)
    feats = extract_features(audio)
    return float(clf.predict_proba(scaler.transform(feats.reshape(1, -1)))[0][1]) * 100


def evaluate(label, score_fn, quadrants):
    print(f"\nevaluating {label}...")
    results = {}
    for q in QUADRANT_ORDER:
        files = quadrants.get(q, [])
        true_label = QUADRANT_TRUE_LABEL[q]
        correct = n_ok = 0
        for path in files:
            try:
                s = score_fn(path)
                pred = 1 if s >= 50 else 0
                n_ok += 1
                correct += int(pred == true_label)
            except Exception as e:
                print(f"  SKIPPED {path}: {e}")
        results[q] = (correct, n_ok)
    return results


def print_table(label, results):
    print(f"\n--- {label} ---")
    print(f"{'quadrant':<20}{'n':>6}{'correct':>10}{'accuracy':>12}{'95% CI':>20}")
    total_correct = total_n = 0
    for q in QUADRANT_ORDER:
        c, n = results[q]
        total_correct += c
        total_n += n
        acc = c / n if n else 0
        lo, hi = wilson_ci(c, n)
        print(f"{q:<20}{n:>6}{c:>10}{100 * acc:>11.1f}%{f'[{100*lo:.1f}%, {100*hi:.1f}%]':>20}")
    overall_acc = total_correct / total_n if total_n else 0
    lo, hi = wilson_ci(total_correct, total_n)
    print(f"{'OVERALL':<20}{total_n:>6}{total_correct:>10}{100 * overall_acc:>11.1f}%"
          f"{f'[{100*lo:.1f}%, {100*hi:.1f}%]':>20}")
    return total_correct, total_n


def main():
    quadrants = get_quadrants()
    for q in QUADRANT_ORDER:
        print(f"{q}: {len(quadrants.get(q, []))} held-out files")

    ssl_clf = joblib.load(os.path.join(MODELS_DIR, "voice_classifier_ssl_v2.joblib"))
    ssl_scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_ssl_v2.joblib"))
    mfcc_clf = joblib.load(os.path.join(MODELS_DIR, "voice_classifier_v3.joblib"))
    mfcc_scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_v3.joblib"))

    # cache both raw scores per file so the dual row doesn't redo work
    _cache = {}

    def cached_scores(path):
        if path not in _cache:
            _cache[path] = (score_ssl(path, ssl_clf, ssl_scaler), score_mfcc(path, mfcc_clf, mfcc_scaler))
        return _cache[path]

    ssl_results = evaluate("SSL v2 (production, current)", lambda p: cached_scores(p)[0], quadrants)
    mfcc_results = evaluate("MFCC v3 (current)", lambda p: cached_scores(p)[1], quadrants)
    dual_results = evaluate(
        "Dual combined (0.7*SSL + 0.3*MFCC, actual production formula)",
        lambda p: cached_scores(p)[0] * DUAL_WEIGHTS["ssl"] + cached_scores(p)[1] * DUAL_WEIGHTS["mfcc"],
        quadrants,
    )

    print()
    print("=" * 100)
    print("FINAL FOUR-QUADRANT ACCURACY -- CURRENT MODELS (post ElevenLabs/Respeecher/kNN-VC retrain)")
    print("=" * 100)
    print_table("SSL v2 alone", ssl_results)
    print_table("MFCC v3 alone", mfcc_results)
    print_table("Dual combined (production)", dual_results)


if __name__ == "__main__":
    main()
