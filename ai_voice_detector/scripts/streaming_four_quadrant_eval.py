"""
Runs the four-quadrant held-out evaluation through the CHUNKED STREAMING
path (mirrors predict.stream_chunks()'s logic exactly: full-length load,
trim, normalize, chunk_audio, per-chunk score, exponential rolling
average) instead of single whole-clip scoring, to check whether 2-second
chunks degrade accuracy versus predict.analyze_file()'s full-clip path --
since the live demo (analyze_stream / the Flask SSE route) uses this path.

predict.py itself is intentionally left unmodified: this reimplements its
stream_chunks() logic, parametrized by an arbitrary (clf, scaler, feature
kind) so it can run against multiple candidate models.
"""
import os
import sys

import joblib
import librosa
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import extract_features  # noqa: E402
from features_ssl import extract_ssl_features  # noqa: E402
from holdout import get_quadrants  # noqa: E402
from preprocess import chunk_audio, load_and_preprocess  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

QUADRANT_TRUE_LABEL = {
    "clean_real": 0,
    "clean_fake": 1,
    "realworld_real": 0,
    "realworld_fake": 1,
}
QUADRANT_ORDER = ["clean_real", "clean_fake", "realworld_real", "realworld_fake"]

SSL_DIM_TO_MODEL = {
    2048: "facebook/wav2vec2-xls-r-300m",
    1536: "facebook/wav2vec2-base",
}

MODELS = [
    ("v3 MFCC (balanced aug)", "voice_classifier_v3.joblib", "scaler_v3.joblib", "mfcc"),
    ("SSL v2 (balanced aug, XLS-R-300M)", "voice_classifier_ssl_v2.joblib", "scaler_ssl_v2.joblib", "ssl"),
]


def score_chunk(audio, clf, scaler, kind):
    if kind == "mfcc":
        feats = extract_features(audio)
    else:
        expected_dim = scaler.mean_.shape[0]
        model_name = SSL_DIM_TO_MODEL.get(expected_dim)
        feats = extract_ssl_features(audio, model_name=model_name)
    feats_scaled = scaler.transform(feats.reshape(1, -1))
    proba_fake = clf.predict_proba(feats_scaled)[0][1]
    return round(float(proba_fake) * 100, 2)


def stream_score_file(path, clf, scaler, kind, chunk_seconds=2.0, overlap=0.5, alpha=0.6):
    """Mirrors predict.stream_chunks()'s logic exactly, returns the final
    rolling score (what the live demo would end up displaying)."""
    y, sr = librosa.load(path, sr=16000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    chunks = chunk_audio(y, sr=16000, chunk_seconds=chunk_seconds, overlap=overlap)

    rolling_score = None
    for chunk in chunks:
        instant = score_chunk(chunk, clf, scaler, kind)
        rolling_score = instant if rolling_score is None else (alpha * instant + (1 - alpha) * rolling_score)
        rolling_score = round(rolling_score, 2)

    return rolling_score, len(chunks)


def whole_clip_score(path, clf, scaler, kind):
    audio = load_and_preprocess(path)
    return score_chunk(audio, clf, scaler, kind)


def main():
    quadrants = get_quadrants()

    for label, model_file, scaler_file, kind in MODELS:
        model_path = os.path.join(MODELS_DIR, model_file)
        scaler_path = os.path.join(MODELS_DIR, scaler_file)
        if not os.path.exists(model_path):
            print(f"skipping {label}: {model_path} not found")
            continue
        clf = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        print(f"\n{'=' * 100}")
        print(f"{label}  --  streaming vs whole-clip")
        print(f"{'=' * 100}")

        per_quadrant = {}
        for q in QUADRANT_ORDER:
            files = quadrants.get(q, [])
            true_label = QUADRANT_TRUE_LABEL[q]
            n_stream_correct = 0
            n_match = 0
            rows = []
            for path in files:
                try:
                    stream_score, n_chunks = stream_score_file(path, clf, scaler, kind)
                    clip_score = whole_clip_score(path, clf, scaler, kind)
                    stream_pred = 1 if stream_score >= 50 else 0
                    clip_pred = 1 if clip_score >= 50 else 0
                    correct = stream_pred == true_label
                    matches = stream_pred == clip_pred
                    if correct:
                        n_stream_correct += 1
                    if matches:
                        n_match += 1
                    rows.append((os.path.basename(path), clip_score, stream_score, n_chunks, correct, matches))
                except Exception as e:
                    print(f"  SKIPPED {path}: {e}")

            per_quadrant[q] = (n_stream_correct, len(files), n_match)
            print(f"\n-- {q} --")
            print(f"{'file':<42}{'whole-clip':>12}{'streaming':>12}{'chunks':>8}{'correct':>9}{'matches':>9}")
            for fname, cs, ss, nc, correct, matches in rows:
                print(f"{fname:<42}{cs:>12.2f}{ss:>12.2f}{nc:>8}{str(correct):>9}{str(matches):>9}")

        print(f"\n{'quadrant':<20}{'streaming accuracy':>22}{'matches whole-clip verdict':>28}")
        total_correct = total_n = total_match = 0
        for q in QUADRANT_ORDER:
            c, n, m = per_quadrant[q]
            total_correct += c
            total_n += n
            total_match += m
            print(f"{q:<20}{f'{c}/{n} ({100 * c / n:.0f}%)':>22}{f'{m}/{n} ({100 * m / n:.0f}%)':>28}")
        print(f"{'OVERALL':<20}{f'{total_correct}/{total_n} ({100 * total_correct / total_n:.1f}%)':>22}"
              f"{f'{total_match}/{total_n} ({100 * total_match / total_n:.1f}%)':>28}")


if __name__ == "__main__":
    main()
