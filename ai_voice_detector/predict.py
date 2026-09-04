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
from features_ssl import (
    extract_ssl_features,
    extract_ssl_features_truncated_direct,
    load_ssl_model,
    load_ssl_model_truncated_direct,
)
from preprocess import chunk_audio, load_and_preprocess
from privacy import log_risk_event, zero_buffer
from risk_engine import compute_final_risk, risk_level_for_profile
from speaker_consistency import verify_speaker

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")

# Production backend switch: "ssl" (wav2vec2/XLS-R embeddings, current
# best per the four-quadrant eval) or "mfcc" (original hand-crafted
# features). Flip this one constant to roll back.
ACTIVE_BACKEND = "ssl"

# Which SSL feature extractor to use, when ACTIVE_BACKEND == "ssl". Uses
# the "_direct" loaders (build the architecture at num_hidden_layers=N
# from the start) rather than the deepcopy-then-slice ones -- deepcopy
# transiently holds the full 24-layer model AND its copy at once, which
# inflates peak RAM without reflecting real standalone usage.
#   "full"       -- untruncated XLS-R-300M. RTF 0.358, peak RAM ~1768MB
#                   (clean, isolated measurements) -- real-time-capable on
#                   its own, just slower/heavier than the alternative below.
#   "truncated"  -- XLS-R-300M truncated to layer 6 (the deeper layers are
#                   never executed since we only read hidden_states[6]).
#                   Numerically EXACT match to "full" (verified, max abs
#                   diff 0.0), same 99.0% four-quadrant accuracy, RTF
#                   0.116 (~3x speedup) AND peak RAM ~894MB (~2x lighter)
#                   -- strictly better than "full" on every axis measured.
#                   This is the production default.
#   "quantized"  -- truncated + dynamic int8 quantization on top. Speed is
#                   statistically indistinguishable from "truncated" alone
#                   (231.4ms vs 231.0ms, within each other's std), AND peak
#                   RAM is actually *higher* (~1352MB vs ~894MB) -- the
#                   conversion step transiently holds both the fp32 and
#                   int8 weights at once. Strictly worse than "truncated"
#                   on this machine; kept available in case it helps more
#                   elsewhere (e.g. a CPU with stronger int8 SIMD support).
SSL_VARIANT = "truncated"

MFCC_MODEL_PATH = os.path.join(MODELS_DIR, "voice_classifier.joblib")
MFCC_SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
# voice_classifier_ssl_indian.joblib supersedes v2: same ASVspoof +
# real-world training data, PLUS the IndieFake Indian-accent dataset --
# scored 99-100% across all six clean/real-world/Indian eval buckets and
# 92.9% detecting unseen Indian-accent clones in leave-one-generator-out
# testing (see the Colab training session), with no regression on the
# original categories. v2's files are left in place for comparison/
# rollback, not deleted.
SSL_MODEL_PATH = os.path.join(MODELS_DIR, "voice_classifier_ssl_indian.joblib")
SSL_SCALER_PATH = os.path.join(MODELS_DIR, "scaler_ssl_indian.joblib")

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


# MFCC v3 loaded as a second, independent opinion alongside whichever
# model ACTIVE_BACKEND points at -- see score_dual().
MFCC_V3_MODEL_PATH = os.path.join(MODELS_DIR, "voice_classifier_v3.joblib")
MFCC_V3_SCALER_PATH = os.path.join(MODELS_DIR, "scaler_v3.joblib")

_mfcc_v3_cache = {}


def load_mfcc_v3_model():
    if "clf" in _mfcc_v3_cache:
        return _mfcc_v3_cache["clf"], _mfcc_v3_cache["scaler"]

    if not os.path.exists(MFCC_V3_MODEL_PATH) or not os.path.exists(MFCC_V3_SCALER_PATH):
        raise FileNotFoundError(
            f"MFCC v3 model/scaler not found in {MODELS_DIR}. Run train.py --augmented first."
        )
    clf = joblib.load(MFCC_V3_MODEL_PATH)
    scaler = joblib.load(MFCC_V3_SCALER_PATH)
    _mfcc_v3_cache["clf"] = clf
    _mfcc_v3_cache["scaler"] = scaler
    return clf, scaler


def risk_level(score, profile=None):
    """LOW/MEDIUM/HIGH classification using the named profile's thresholds
    (config.yaml) -- defaults to the "routine" profile (30/70), so every
    existing caller that doesn't pass `profile` keeps its old behavior."""
    return risk_level_for_profile(score, profile)


def score_single_clip(audio, clf, scaler):
    if ACTIVE_BACKEND == "ssl":
        if SSL_VARIANT == "quantized":
            feats = extract_ssl_features_truncated_direct(audio, quantize=True).reshape(1, -1)
        elif SSL_VARIANT == "truncated":
            feats = extract_ssl_features_truncated_direct(audio, quantize=False).reshape(1, -1)
        else:
            feats = extract_ssl_features(audio).reshape(1, -1)
    else:
        feats = extract_features(audio).reshape(1, -1)
    feats_scaled = scaler.transform(feats)
    proba_fake = clf.predict_proba(feats_scaled)[0][1]
    return round(float(proba_fake) * 100, 2)


# Dual-scoring: SSL is the primary/production model, MFCC v3 runs as an
# independent second opinion using entirely different features (hand-
# crafted spectral/pitch stats vs. a pretrained transformer's learned
# representation). Two architectures rarely share the same blind spots,
# so large disagreement between them is itself a useful signal.
DUAL_SCORE_WEIGHTS = {"ssl": 0.7, "mfcc": 0.3}
CONFLICT_THRESHOLD = 40  # point gap between the two scores that counts as "conflicted"


def score_dual(audio):
    """Scores `audio` with both the SSL model and the MFCC v3 model
    independently, returning both raw scores plus a weighted combination.
    Flags large disagreement (e.g. one says LOW, the other HIGH) as
    "conflicted" -- surfaced separately rather than silently averaged
    away, since that disagreement is diagnostically useful on its own."""
    ssl_clf, ssl_scaler = load_model()
    mfcc_clf, mfcc_scaler = load_mfcc_v3_model()

    ssl_score = score_single_clip(audio, ssl_clf, ssl_scaler)

    mfcc_feats = extract_features(audio).reshape(1, -1)
    mfcc_feats_scaled = mfcc_scaler.transform(mfcc_feats)
    mfcc_score = round(float(mfcc_clf.predict_proba(mfcc_feats_scaled)[0][1]) * 100, 2)

    final_voice_risk = round(
        ssl_score * DUAL_SCORE_WEIGHTS["ssl"] + mfcc_score * DUAL_SCORE_WEIGHTS["mfcc"], 2
    )
    conflicted = abs(ssl_score - mfcc_score) > CONFLICT_THRESHOLD

    return {
        "ssl_score": ssl_score,
        "mfcc_score": mfcc_score,
        "final_voice_risk": final_voice_risk,
        "conflicted": conflicted,
    }


def _warm_up():
    """Loads the classifier/scaler and (for the SSL backend) the
    underlying transformer model once at import time, so the first real
    request isn't the one paying the load cost. Also warms up the MFCC v3
    "second opinion" model used by score_dual()."""
    try:
        load_model()
        if ACTIVE_BACKEND == "ssl":
            if SSL_VARIANT == "quantized":
                load_ssl_model_truncated_direct(quantize=True)
            elif SSL_VARIANT == "truncated":
                load_ssl_model_truncated_direct(quantize=False)
            else:
                load_ssl_model()
    except FileNotFoundError:
        pass  # models not trained yet -- let load_model() raise properly on first real use

    try:
        load_mfcc_v3_model()
    except FileNotFoundError:
        pass  # same -- let score_dual() raise properly on first real use


_warm_up()


def analyze_file(file_path, profile=None):
    clf, scaler = load_model()
    audio = load_and_preprocess(file_path)
    score = score_single_clip(audio, clf, scaler)
    zero_buffer(audio)  # privacy: don't let the raw waveform linger in memory
    level, action = risk_level(score, profile)
    return {
        "file": file_path,
        "risk_score": score,
        "risk_level": level,
        "recommended_action": action,
    }


def analyze_file_with_context(file_path, context=None, profile=None, speaker_id=None):
    """Voice risk + contextual risk + (if speaker_id given) cross-session
    speaker consistency, combined per the active profile's weights (see
    risk_engine.compute_final_risk for the security-critical asymmetry:
    context/consistency can escalate, never suppress).

    Voice risk itself comes from score_dual() -- SSL (primary) + MFCC v3
    (independent second opinion) -- rather than a single model.
    speaker_id is optional: with no enrolled profile for that ID (or no
    speaker_id at all), consistency simply isn't part of the blend rather
    than being treated as a confirmed match or a confirmed risk."""
    audio = load_and_preprocess(file_path)
    dual = score_dual(audio)

    consistency = None
    if speaker_id:
        consistency = verify_speaker(speaker_id, audio, sr=16000)

    zero_buffer(audio)

    consistency_risk = consistency["consistency_risk"] if consistency else None
    result = compute_final_risk(dual["final_voice_risk"], context, profile, consistency_risk,
                                 speaker_id=speaker_id)
    result["ssl_score"] = dual["ssl_score"]
    result["mfcc_score"] = dual["mfcc_score"]
    result["conflicted"] = dual["conflicted"]
    if consistency:
        result["speaker_similarity"] = consistency["similarity"]
        result["speaker_match"] = consistency["match"]

    log_risk_event({
        "voice_risk": result["voice_risk"],
        "context_risk": result["context_risk"],
        "final_risk": result["final_risk"],
        "risk_level": result["risk_level"],
        "profile": result["profile"],
        "context_flags": _extract_context_flags(context),
    })
    return result


def _extract_context_flags(context):
    """Boolean/categorical flags only, for logging -- never the raw
    context dict verbatim in case a caller ever adds a free-text field."""
    context = context or {}
    return {
        "caller_known": context.get("caller_known"),
        "origin": context.get("origin"),
        "channel": context.get("channel"),
        "large_transaction": bool(context.get("transaction_amount")),
        "new_beneficiary": bool(context.get("new_beneficiary")),
        "outside_business_hours": bool(context.get("outside_business_hours")),
        "previously_flagged": bool(context.get("previously_flagged")),
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
        zero_buffer(chunk)  # privacy: don't let the raw chunk linger in memory
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

    zero_buffer(y)  # privacy: full-file buffer, done with it after chunking


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
