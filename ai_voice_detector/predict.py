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
from features_ssl import extract_ssl_features, extract_ssl_features_truncated, load_ssl_model, load_ssl_model_truncated
from preprocess import chunk_audio, load_and_preprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")

# Production backend switch: "ssl" (wav2vec2/XLS-R embeddings, current
# best per the four-quadrant eval) or "mfcc" (original hand-crafted
# features). Flip this one constant to roll back.
ACTIVE_BACKEND = "ssl"

# Which SSL feature extractor to use, when ACTIVE_BACKEND == "ssl":
#   "full"      -- untruncated XLS-R-300M. Most accurate on paper, but
#                  RTF 1.616 on this machine -- CANNOT sustain real-time
#                  streaming (a 2s chunk takes 3.2s to process).
#   "quantized" -- XLS-R-300M truncated to layer 6 + dynamic int8
#                  quantization. Same 99.0% four-quadrant accuracy as
#                  "full" (verified: truncation is numerically exact,
#                  quantization didn't flip a single verdict on the eval
#                  set), RTF 0.527 -- actually real-time-capable. This is
#                  the production default.
SSL_VARIANT = "quantized"

MFCC_MODEL_PATH = os.path.join(MODELS_DIR, "voice_classifier.joblib")
MFCC_SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
SSL_MODEL_PATH = os.path.join(MODELS_DIR, "voice_classifier_ssl_v2.joblib")
SSL_SCALER_PATH = os.path.join(MODELS_DIR, "scaler_ssl_v2.joblib")

if ACTIVE_BACKEND == "ssl":
    MODEL_PATH, SCALER_PATH = SSL_MODEL_PATH, SSL_SCALER_PATH
else:
    MODEL_PATH, SCALER_PATH = MFCC_MODEL_PATH, MFCC_SCALER_PATH

_model_cache = {}  # clf/scaler cached globally -- loaded once, never per-request


def load_model():
    if "clf" in _model_cache:
        return _model_cache["clf"], _model_cache["scaler"]

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Model/scaler not found in {MODELS_DIR}. Run train.py "
            f"{'--augmented' if ACTIVE_BACKEND == 'mfcc' else ''} "
            f"(or train_ssl.py --augmented) first to train and save them."
        )
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    _model_cache["clf"] = clf
    _model_cache["scaler"] = scaler
    return clf, scaler


def risk_level(score):
    if score < 30:
        return "LOW", "No action needed. Proceed normally."
    elif score < 70:
        return "MEDIUM", "Recommend secondary verification (callback or OTP)."
    else:
        return "HIGH", "High impersonation risk. Block transaction / escalate to supervisor."


def score_single_clip(audio, clf, scaler):
    if ACTIVE_BACKEND == "ssl":
        if SSL_VARIANT == "quantized":
            feats = extract_ssl_features_truncated(audio, quantize=True).reshape(1, -1)
        else:
            feats = extract_ssl_features(audio).reshape(1, -1)
    else:
        feats = extract_features(audio).reshape(1, -1)
    feats_scaled = scaler.transform(feats)
    proba_fake = clf.predict_proba(feats_scaled)[0][1]
    return round(float(proba_fake) * 100, 2)


def _warm_up():
    """Loads the classifier/scaler and (for the SSL backend) the
    underlying transformer model once at import time, so the first real
    request isn't the one paying the load cost."""
    try:
        load_model()
        if ACTIVE_BACKEND == "ssl":
            if SSL_VARIANT == "quantized":
                load_ssl_model_truncated(quantize=True)
            else:
                load_ssl_model()
    except FileNotFoundError:
        pass  # models not trained yet -- let load_model() raise properly on first real use


_warm_up()


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
