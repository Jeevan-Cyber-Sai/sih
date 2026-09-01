"""
Inference and risk-scoring engine for the AI voice detector: loads the
trained model, scores single clips or a simulated real-time audio stream,
and maps scores to actionable risk levels.
"""
import os
import time

import joblib
import librosa
import numpy as np

from features import extract_features
from preprocess import chunk_audio, load_and_preprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "voice_classifier.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Model/scaler not found in {MODELS_DIR}. Run train.py first to train and save them."
        )
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return clf, scaler


def risk_level(score):
    if score < 30:
        return "LOW", "No action needed. Proceed normally."
    elif score < 70:
        return "MEDIUM", "Recommend secondary verification (callback or OTP)."
    else:
        return "HIGH", "High impersonation risk. Block transaction / escalate to supervisor."


def score_single_clip(audio, clf, scaler):
    feats = extract_features(audio).reshape(1, -1)
    feats_scaled = scaler.transform(feats)
    proba_fake = clf.predict_proba(feats_scaled)[0][1]
    return round(float(proba_fake) * 100, 2)


def analyze_file(file_path):
    clf, scaler = load_model()
    audio = load_and_preprocess(file_path)
    score = score_single_clip(audio, clf, scaler)
    level, action = risk_level(score)
    return {
        "file": file_path,
        "risk_score": score,
        "risk_level": level,
        "recommended_action": action,
    }


def stream_chunks(file_path, chunk_seconds=2.0, overlap=0.5, alpha=0.6):
    """Yield a per-chunk risk result dict as each chunk is scored. Shared by
    analyze_stream() (console demo) and the Flask SSE endpoint (app.py)."""
    clf, scaler = load_model()

    y, sr = librosa.load(file_path, sr=16000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    chunks = chunk_audio(y, sr=16000, chunk_seconds=chunk_seconds, overlap=overlap)

    rolling_score = None
    for i, chunk in enumerate(chunks):
        instant_score = score_single_clip(chunk, clf, scaler)
        rolling_score = instant_score if rolling_score is None else (
            alpha * instant_score + (1 - alpha) * rolling_score
        )
        rolling_score = round(rolling_score, 2)
        level, action = risk_level(rolling_score)

        yield {
            "chunk_index": i,
            "total_chunks": len(chunks),
            "instant_score": instant_score,
            "rolling_score": rolling_score,
            "risk_level": level,
            "recommended_action": action,
        }


def analyze_stream(file_path, chunk_seconds=2.0, overlap=0.5, delay=0.3, alpha=0.6):
    results = []
    for result in stream_chunks(file_path, chunk_seconds, overlap, alpha):
        results.append(result)
        print(f"chunk {result['chunk_index']}: instant={result['instant_score']:6.2f}  "
              f"rolling={result['rolling_score']:6.2f}  risk={result['risk_level']}")
        time.sleep(delay)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze an audio file for voice cloning risk.")
    parser.add_argument("file_path", help="path to a .wav file")
    parser.add_argument("--stream", action="store_true", help="simulate real-time chunked streaming analysis")
    args = parser.parse_args()

    if args.stream:
        analyze_stream(args.file_path)
    else:
        result = analyze_file(args.file_path)
        print(result)
