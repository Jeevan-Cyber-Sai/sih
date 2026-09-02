"""
Re-runs the four-quadrant held-out evaluation after simulating G.711
mu-law telephony quality (16kHz -> 8kHz -> mu-law 8-bit -> 16kHz) on every
file, using the production model (config (c): XLS-R truncated). Reports
telephony accuracy next to the original clean accuracy per quadrant, so
we know the real cost before committing to a telephony integration.
"""
import os
import sys

import joblib
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features_ssl import extract_ssl_features_truncated_direct  # noqa: E402
from holdout import get_quadrants  # noqa: E402
from telephony_simulate import simulate_telephony  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

QUADRANT_TRUE_LABEL = {
    "clean_real": 0,
    "clean_fake": 1,
    "realworld_real": 0,
    "realworld_fake": 1,
}
QUADRANT_ORDER = ["clean_real", "clean_fake", "realworld_real", "realworld_fake"]

MODEL_FILE = "voice_classifier_ssl_v2.joblib"
SCALER_FILE = "scaler_ssl_v2.joblib"


def load_and_prep(path, telephony):
    y, sr = librosa.load(path, sr=16000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    import numpy as np
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    target_len = int(sr * 3.0)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    if telephony:
        y = simulate_telephony(y, sr=sr)
    return y


def score(path, clf, scaler, telephony):
    audio = load_and_prep(path, telephony)
    feats = extract_ssl_features_truncated_direct(audio, quantize=False)
    feats_scaled = scaler.transform(feats.reshape(1, -1))
    proba_fake = clf.predict_proba(feats_scaled)[0][1]
    return float(proba_fake) * 100


def evaluate(clf, scaler, quadrants, telephony):
    results = {}
    for q in QUADRANT_ORDER:
        files = quadrants.get(q, [])
        true_label = QUADRANT_TRUE_LABEL[q]
        correct = 0
        for path in files:
            try:
                s = score(path, clf, scaler, telephony)
                pred = 1 if s >= 50 else 0
                if pred == true_label:
                    correct += 1
            except Exception as e:
                print(f"  SKIPPED {path}: {e}")
        results[q] = (correct, len(files))
    return results


def print_row(label, results):
    row = f"{label:<28}"
    total_correct = total_n = 0
    for q in QUADRANT_ORDER:
        c, n = results[q]
        total_correct += c
        total_n += n
        pct = f"{c}/{n} ({100 * c / n:.0f}%)" if n else "n/a"
        row += f"{pct:>18}"
    overall = f"{100 * total_correct / total_n:.1f}%" if total_n else "n/a"
    row += f"{overall:>12}"
    print(row)


def main():
    quadrants = get_quadrants()
    for q in QUADRANT_ORDER:
        print(f"{q}: {len(quadrants.get(q, []))} held-out files")

    clf = joblib.load(os.path.join(MODELS_DIR, MODEL_FILE))
    scaler = joblib.load(os.path.join(MODELS_DIR, SCALER_FILE))

    print("\nevaluating clean (16kHz)...")
    clean_results = evaluate(clf, scaler, quadrants, telephony=False)

    print("evaluating telephony-simulated (8kHz mu-law roundtrip)...")
    telephony_results = evaluate(clf, scaler, quadrants, telephony=True)

    print()
    print("=" * 100)
    print("FOUR-QUADRANT ACCURACY: clean vs telephony-simulated")
    print("=" * 100)
    header = f"{'condition':<28}" + "".join(f"{q:>18}" for q in QUADRANT_ORDER) + f"{'overall':>12}"
    print(header)
    print_row("clean (16kHz)", clean_results)
    print_row("telephony (8kHz mu-law)", telephony_results)

    print()
    print("Per-quadrant accuracy delta (telephony - clean):")
    for q in QUADRANT_ORDER:
        c_clean, n = clean_results[q]
        c_tel, _ = telephony_results[q]
        delta = 100 * (c_tel - c_clean) / n if n else 0
        print(f"  {q:<20} clean={100*c_clean/n:.0f}%  telephony={100*c_tel/n:.0f}%  delta={delta:+.0f}pp")


if __name__ == "__main__":
    main()
