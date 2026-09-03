"""
Step 6: evaluates voice_classifier_ssl_indian.joblib (trained on ASVspoof
+ real-world + Indian, see train_ssl.py --indian) on all six held-out
buckets -- clean_real, clean_fake, realworld_real, realworld_fake,
indian_real, indian_fake -- reporting each one separately. Overall
accuracy can hide a shortcut (e.g. "Indian accent = real" or
"clean = fake"); per-bucket accuracy can't.

Also evaluates SSL v2 (the pre-Indian production model) on the same six
buckets for comparison, so the indian_real/indian_fake numbers show
whether adding the Indian data actually helped, not just what the new
model's absolute accuracy is.

Uses the "truncated" (non-quantized) extractor throughout -- numerically
identical to the full model's layer-6 output, and the same one used to
build features_indian.npy, so eval-time features match train-time
features exactly.
"""
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    "indian_real": 0,
    "indian_fake": 1,
}
QUADRANT_ORDER = ["clean_real", "clean_fake", "realworld_real", "realworld_fake", "indian_real", "indian_fake"]

XLSR_MODEL = "facebook/wav2vec2-xls-r-300m"

MODELS = [
    ("SSL v2 (pre-Indian, ASVspoof+realworld)", "voice_classifier_ssl_v2.joblib", "scaler_ssl_v2.joblib"),
    ("SSL indian (ASVspoof+realworld+IndieFake)", "voice_classifier_ssl_indian.joblib", "scaler_ssl_indian.joblib"),
]


def score(path, clf, scaler):
    audio = load_and_preprocess(path)
    feats = extract_ssl_features_truncated_direct(audio, model_name=XLSR_MODEL, quantize=False)
    feats_scaled = scaler.transform(feats.reshape(1, -1))
    proba_fake = clf.predict_proba(feats_scaled)[0][1]
    return float(proba_fake) * 100


def main():
    quadrants = get_quadrants()
    for q in QUADRANT_ORDER:
        print(f"{q}: {len(quadrants.get(q, []))} held-out files")
    print()

    results = {}

    for label, model_file, scaler_file in MODELS:
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
                    s = score(path, clf, scaler)
                    pred = 1 if s >= 50 else 0
                    if pred == true_label:
                        correct += 1
                except Exception as e:
                    print(f"  SKIPPED {path}: {e}")
            results[label][q] = (correct, len(files))

    print()
    print("=" * 115)
    print("SIX-BUCKET ACCURACY (clean / real-world / Indian x real / fake) -- reported separately, never blended")
    print("=" * 115)
    header = f"{'model':<44}" + "".join(f"{q:>14}" for q in QUADRANT_ORDER)
    print(header)
    for label, _, _ in MODELS:
        if label not in results:
            continue
        row = f"{label:<44}"
        for q in QUADRANT_ORDER:
            c, n = results[label][q]
            pct = f"{c}/{n}({100*c/n:.0f}%)" if n else "n/a"
            row += f"{pct:>14}"
        print(row)


if __name__ == "__main__":
    main()
