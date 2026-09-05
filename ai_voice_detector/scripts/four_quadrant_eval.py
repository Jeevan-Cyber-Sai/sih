"""
Evaluates every trained model on the four held-out quadrants (clean_real,
clean_fake, realworld_real, realworld_fake) and reports per-quadrant
accuracy -- overall accuracy can hide a shortcut (e.g. "degraded = real"),
per-quadrant accuracy can't.

None of these files were used in training any model (see holdout.py).
"""
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import extract_features  # noqa: E402
from features_ssl import extract_ssl_features, extract_ssl_features_truncated  # noqa: E402
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

XLSR_MODEL = "facebook/wav2vec2-xls-r-300m"
BASE_MODEL = "facebook/wav2vec2-base"

# (label, model_file, scaler_file, kind)
# kind routes to the feature extractor: "mfcc" | "ssl_xlsr" | "ssl_base" |
# "ssl_truncated" | "ssl_quantized". (c)/(d) intentionally reuse the SAME
# classifier/scaler as the full XLS-R SSL v2 model -- the whole point is
# testing whether a faster/quantized *feature extractor* still works with
# the existing decision boundary, not training a new one.
MODELS = [
    ("v1 MFCC (ASVspoof-only)", "voice_classifier.joblib", "scaler.joblib", "mfcc"),
    ("v2 MFCC (real-only aug, broken)", "voice_classifier_v2.joblib", "scaler_v2.joblib", "mfcc"),
    ("v3 MFCC (balanced aug)", "voice_classifier_v3.joblib", "scaler_v3.joblib", "mfcc"),
    ("SSL v1 (ASVspoof-only, broken)", "voice_classifier_ssl.joblib", "scaler_ssl.joblib", "ssl_xlsr"),
    ("SSL v2 (balanced aug, wav2vec2-base)", "voice_classifier_ssl_v2_base.joblib", "scaler_ssl_v2_base.joblib", "ssl_base"),
    ("SSL v2 (balanced aug, XLS-R-300M)", "voice_classifier_ssl_v2.joblib", "scaler_ssl_v2.joblib", "ssl_xlsr"),
    ("(c) SSL v2, XLS-R truncated", "voice_classifier_ssl_v2.joblib", "scaler_ssl_v2.joblib", "ssl_truncated"),
    ("(d) SSL v2, XLS-R truncated+int8", "voice_classifier_ssl_v2.joblib", "scaler_ssl_v2.joblib", "ssl_quantized"),
    # +IndieFake+Hindi+Tamil (train_ssl.py --multilingual) -- same
    # "truncated" fp32 extractor as (c), numerically identical to the
    # full XLS-R model's layer-6 output, so this is an apples-to-apples
    # comparison against SSL v2 on the SAME four English/real-world
    # quadrants (Hindi/Tamil generalization is evaluated separately in
    # scripts/per_language_eval.py -- this script only answers "did adding
    # more languages regress the original production quadrants").
    ("SSL multilingual (+IndieFake+Hindi+Tamil)", "voice_classifier_ssl_multilingual.joblib",
     "scaler_ssl_multilingual.joblib", "ssl_truncated"),
]


def score(path, clf, scaler, kind):
    audio = load_and_preprocess(path)
    if kind == "mfcc":
        feats = extract_features(audio)
    elif kind == "ssl_xlsr":
        feats = extract_ssl_features(audio, model_name=XLSR_MODEL)
    elif kind == "ssl_base":
        feats = extract_ssl_features(audio, model_name=BASE_MODEL)
    elif kind == "ssl_truncated":
        feats = extract_ssl_features_truncated(audio, model_name=XLSR_MODEL, quantize=False)
    elif kind == "ssl_quantized":
        feats = extract_ssl_features_truncated(audio, model_name=XLSR_MODEL, quantize=True)
    else:
        raise ValueError(f"unknown kind {kind}")
    feats_scaled = scaler.transform(feats.reshape(1, -1))
    proba_fake = clf.predict_proba(feats_scaled)[0][1]
    return float(proba_fake) * 100


def main():
    quadrants = get_quadrants()
    for q in QUADRANT_ORDER:
        print(f"{q}: {len(quadrants.get(q, []))} held-out files")
    print()

    results = {}  # model_label -> quadrant -> (correct, total)

    for label, model_file, scaler_file, kind in MODELS:
        model_path = os.path.join(MODELS_DIR, model_file)
        scaler_path = os.path.join(MODELS_DIR, scaler_file)
        if not os.path.exists(model_path):
            print(f"skipping {label}: {model_path} not found")
            continue

        clf = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print(f"evaluating {label}...")

        results[label] = {}
        for q in QUADRANT_ORDER:
            files = quadrants.get(q, [])
            true_label = QUADRANT_TRUE_LABEL[q]
            correct = 0
            for path in files:
                try:
                    s = score(path, clf, scaler, kind)
                    pred = 1 if s >= 50 else 0
                    if pred == true_label:
                        correct += 1
                except Exception as e:
                    print(f"  SKIPPED {path}: {e}")
            results[label][q] = (correct, len(files))

    print()
    print("=" * 100)
    print("FOUR-QUADRANT ACCURACY")
    print("=" * 100)
    header = f"{'model':<34}" + "".join(f"{q:>18}" for q in QUADRANT_ORDER) + f"{'overall':>12}"
    print(header)
    for label, _, _, _ in MODELS:
        if label not in results:
            continue
        row = f"{label:<34}"
        total_correct = total_n = 0
        for q in QUADRANT_ORDER:
            c, n = results[label][q]
            total_correct += c
            total_n += n
            pct = f"{c}/{n} ({100*c/n:.0f}%)" if n else "n/a"
            row += f"{pct:>18}"
        overall = f"{100*total_correct/total_n:.1f}%" if total_n else "n/a"
        row += f"{overall:>12}"
        print(row)


if __name__ == "__main__":
    main()
