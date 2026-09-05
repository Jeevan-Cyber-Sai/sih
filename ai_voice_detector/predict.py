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
from language_id import detect_language, load_language_model
from phase_classifier import load_phase_model, score_phase
from preprocess import chunk_audio, load_and_preprocess
from privacy import log_risk_event, zero_buffer
from risk_engine import compute_final_risk, get_profile, risk_level_for_profile
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
# voice_classifier_ssl_multilingual.joblib supersedes ssl_indian (which
# itself superseded v2): same ASVspoof + real-world + IndieFake training
# data, PLUS Hindi and Tamil (IndicTTS real + edge-tts synthetic fakes,
# see scripts/extract_indictts_real.py / build_multilingual_fake.py) --
# matched SSL v2 exactly on all four original clean/real-world quadrants
# (scripts/four_quadrant_eval.py: 100%/100%/100%/99%, zero regression)
# while adding 99.5%+ per-language accuracy on Hindi/Tamil when the
# language is in training (scripts/per_language_eval.py --breakdown).
# Older files are left in place for comparison/rollback, not deleted.
SSL_MODEL_PATH = os.path.join(MODELS_DIR, "voice_classifier_ssl_multilingual.joblib")
SSL_SCALER_PATH = os.path.join(MODELS_DIR, "scaler_ssl_multilingual.joblib")

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
# model ACTIVE_BACKEND points at -- see score_all_layers().
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


# Four-layer scoring: SSL is the primary/production model, MFCC v3 and
# the phase-spectrum classifier run as independent second/third opinions
# using entirely different features (hand-crafted spectral/pitch stats;
# STFT phase-only statistics) from SSL's pretrained-transformer
# embeddings and from each other. Three architectures examining three
# different aspects of the signal rarely share the same blind spots, so
# disagreement between any pair of them is itself a useful signal.
#
# Sourced from config.yaml (ssl_weight/mfcc_weight/phase_weight) rather
# than hardcoded, so there's one place to tune them -- read once from the
# default profile at import time. score_all_layers() runs BEFORE any
# per-call profile is applied (it produces the single voice_risk number
# that compute_final_risk() later blends against context/consistency for
# a specific profile), so if these three weights are ever made to
# genuinely differ per profile rather than just being duplicated
# identically in each one, this would need to become profile-aware too.
_, _default_profile = get_profile()
LAYER_WEIGHTS = {
    "ssl": _default_profile["ssl_weight"],
    "mfcc": _default_profile["mfcc_weight"],
    "phase": _default_profile["phase_weight"],
}
CONFLICT_THRESHOLD = 40  # point gap between any two of the three scores that counts as "conflicted"


def score_all_layers(audio):
    """Scores `audio` with the SSL, MFCC v3, and phase-spectrum models
    independently, returning all three raw scores plus a weighted
    combination. Flags "conflicted" if ANY two of the three scores
    disagree by more than CONFLICT_THRESHOLD points -- surfaced
    separately rather than silently averaged away, since disagreement
    between independent detectors is diagnostically useful on its own."""
    ssl_clf, ssl_scaler = load_model()
    mfcc_clf, mfcc_scaler = load_mfcc_v3_model()

    ssl_score = score_single_clip(audio, ssl_clf, ssl_scaler)

    mfcc_feats = extract_features(audio).reshape(1, -1)
    mfcc_feats_scaled = mfcc_scaler.transform(mfcc_feats)
    mfcc_score = round(float(mfcc_clf.predict_proba(mfcc_feats_scaled)[0][1]) * 100, 2)

    phase_score = score_phase(audio)

    # Language ID (Part 5 of the multilingual PS requirement): its own
    # cheap "truncated" SSL forward pass -- numerically identical to
    # score_single_clip's internal one when ACTIVE_BACKEND=="ssl", but
    # kept independent so language detection stays correct even if the
    # active backend/variant is ever flipped away from SSL.
    lang_feats = extract_ssl_features_truncated_direct(audio, quantize=False).reshape(1, -1)
    detected_language, language_confidence = detect_language(lang_feats)

    final_voice_risk = round(
        ssl_score * LAYER_WEIGHTS["ssl"]
        + mfcc_score * LAYER_WEIGHTS["mfcc"]
        + phase_score * LAYER_WEIGHTS["phase"],
        2,
    )

    # Named per-pair gaps rather than just a boolean -- "SSL vs Phase"
    # disagreeing is a different diagnostic story than "SSL vs MFCC", and
    # collapsing them into one flag would throw that away. Sorted by gap
    # size so conflict_detail names the WORST disagreement first when
    # more than one pair crosses the threshold.
    pair_gaps = [
        ("SSL vs MFCC", abs(ssl_score - mfcc_score)),
        ("SSL vs Phase", abs(ssl_score - phase_score)),
        ("MFCC vs Phase", abs(mfcc_score - phase_score)),
    ]
    conflicting_pairs = sorted(
        (name for name, gap in pair_gaps if gap > CONFLICT_THRESHOLD),
        key=lambda name: -dict(pair_gaps)[name],
    )
    conflicted = len(conflicting_pairs) > 0
    conflict_detail = ", ".join(conflicting_pairs) if conflicted else None

    return {
        "ssl_score": ssl_score,
        "mfcc_score": mfcc_score,
        "phase_score": phase_score,
        "final_voice_risk": final_voice_risk,
        "conflicted": conflicted,
        "conflict_detail": conflict_detail,
        "detected_language": detected_language,
        "language_confidence": language_confidence,
    }


def _warm_up():
    """Loads the classifier/scaler and (for the SSL backend) the
    underlying transformer model once at import time, so the first real
    request isn't the one paying the load cost. Also warms up the MFCC v3
    and phase-spectrum "second/third opinion" models used by
    score_all_layers()."""
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
        pass  # same -- let score_all_layers() raise properly on first real use

    try:
        load_phase_model()
    except FileNotFoundError:
        pass  # same -- let score_all_layers() raise properly on first real use

    try:
        load_language_model()
    except FileNotFoundError:
        pass  # same -- let score_all_layers() raise properly on first real use


_warm_up()


def analyze_file(file_path, profile=None):
    """Voice-only risk assessment (no call/transaction context) via
    score_all_layers() -- SSL + MFCC v3 + phase-spectrum combined."""
    audio = load_and_preprocess(file_path)
    layers = score_all_layers(audio)
    zero_buffer(audio)  # privacy: don't let the raw waveform linger in memory
    level, action = risk_level(layers["final_voice_risk"], profile)
    return {
        "file": file_path,
        "risk_score": layers["final_voice_risk"],
        "risk_level": level,
        "recommended_action": action,
        "ssl_score": layers["ssl_score"],
        "mfcc_score": layers["mfcc_score"],
        "phase_score": layers["phase_score"],
        "conflicted": layers["conflicted"],
        "conflict_detail": layers["conflict_detail"],
        "detected_language": layers["detected_language"],
        "language_confidence": layers["language_confidence"],
    }


def analyze_file_with_context(file_path, context=None, profile=None, speaker_id=None):
    """Voice risk + contextual risk + (if speaker_id given) cross-session
    speaker consistency, combined per the active profile's weights (see
    risk_engine.compute_final_risk for the security-critical asymmetry:
    context/consistency can escalate, never suppress).

    Voice risk itself comes from score_all_layers() -- SSL (primary) +
    MFCC v3 + phase-spectrum (two independent second/third opinions) --
    rather than a single model.
    speaker_id is optional: with no enrolled profile for that ID (or no
    speaker_id at all), consistency simply isn't part of the blend rather
    than being treated as a confirmed match or a confirmed risk."""
    audio = load_and_preprocess(file_path)
    layers = score_all_layers(audio)

    consistency = None
    if speaker_id:
        consistency = verify_speaker(speaker_id, audio, sr=16000)

    zero_buffer(audio)

    consistency_risk = consistency["consistency_risk"] if consistency else None
    result = compute_final_risk(layers["final_voice_risk"], context, profile, consistency_risk,
                                 speaker_id=speaker_id, ssl_score=layers["ssl_score"])
    result["ssl_score"] = layers["ssl_score"]
    result["mfcc_score"] = layers["mfcc_score"]
    result["phase_score"] = layers["phase_score"]
    result["conflicted"] = layers["conflicted"]
    result["conflict_detail"] = layers["conflict_detail"]
    result["detected_language"] = layers["detected_language"]
    result["language_confidence"] = layers["language_confidence"]
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
        "detected_language": result["detected_language"],
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
    """Yield a per-chunk risk result dict as each chunk is scored, using
    all four layers (SSL + MFCC v3 + phase-spectrum) per chunk -- not
    just SSL. Shared by analyze_stream() (console demo) and the Flask
    SSE endpoint (app.py). NOTE: the live Twilio/Telnyx call path
    (twilio_handler.CallSession) calls score_single_clip() directly
    rather than going through this function, and is unchanged by this --
    it stays SSL-only for now, since folding in three model calls per
    live 2-second chunk risks real-time latency and wasn't asked for here."""
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    chunks = chunk_audio(y, sr=16000, chunk_seconds=chunk_seconds, overlap=overlap)

    rolling_score = None
    for i, chunk in enumerate(chunks):
        layers = score_all_layers(chunk)
        instant_score = layers["final_voice_risk"]
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
            "ssl_score": layers["ssl_score"],
            "mfcc_score": layers["mfcc_score"],
            "phase_score": layers["phase_score"],
            "conflicted": layers["conflicted"],
            "conflict_detail": layers["conflict_detail"],
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
