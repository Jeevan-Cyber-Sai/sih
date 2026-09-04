"""
Cross-session speaker consistency: answers "is this the same specific
person who called before", independent of spoof detection ("is this
audio human or AI-generated" -- see predict.py/score_dual). The two
checks are complementary: a voice can pass spoof detection (sound
human) while still being the WRONG human, and a voice can match an
enrolled profile while still being a synthetic clone of that person --
neither check alone tells the whole story, which is why risk_engine.py
combines both rather than treating either as sufficient on its own.

Uses resemblyzer's pretrained speaker-embedding model (256-dim
d-vectors) -- no training from scratch, matching this project's existing
"reuse a pretrained model" pattern for the SSL backend (features_ssl.py).

Profiles are plain JSON (speaker_profiles.json) keyed by speaker_id
(e.g. phone number or name), each holding a running-average embedding
across every enrollment call -- more calls builds a more representative
profile instead of just capturing whatever the most recent call sounded
like.
"""
import json
import os

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

ROOT = os.path.dirname(os.path.abspath(__file__))
PROFILES_PATH = os.path.join(ROOT, "speaker_profiles.json")

# Thresholds are on cosine similarity between d-vectors (both unit-norm,
# so dot product == cosine similarity).
HIGH_SIMILARITY_THRESHOLD = 0.75  # same person -> consistency_risk 0
LOW_SIMILARITY_THRESHOLD = 0.5    # below this -> likely a different person

_encoder = None


def _get_encoder():
    """Lazily loaded and cached globally -- loaded once, never per call,
    matching predict.py's model-caching convention."""
    global _encoder
    if _encoder is None:
        _encoder = VoiceEncoder()
    return _encoder


def _extract_embedding(audio_array, sr=16000):
    wav = preprocess_wav(audio_array, source_sr=sr)
    return _get_encoder().embed_utterance(wav)


def _load_profiles():
    if not os.path.exists(PROFILES_PATH):
        return {}
    with open(PROFILES_PATH) as f:
        return json.load(f)


def _save_profiles(profiles):
    with open(PROFILES_PATH, "w") as f:
        json.dump(profiles, f, indent=2)


def enroll_speaker(speaker_id, audio_array, sr=16000):
    """Extracts a d-vector from `audio_array` and stores/updates
    `speaker_id`'s profile. An existing profile is updated via a running
    average weighted by prior enrollment count, not overwritten, then
    re-normalized (averaging two unit vectors doesn't stay unit-length).
    Returns the embedding actually stored (post-averaging if applicable)."""
    embedding = _extract_embedding(audio_array, sr)

    profiles = _load_profiles()
    if speaker_id in profiles:
        existing = np.array(profiles[speaker_id]["embedding"], dtype=np.float32)
        n = profiles[speaker_id]["n_enrollments"]
        embedding = (existing * n + embedding) / (n + 1)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        profiles[speaker_id] = {"embedding": embedding.tolist(), "n_enrollments": n + 1}
    else:
        profiles[speaker_id] = {"embedding": embedding.tolist(), "n_enrollments": 1}

    _save_profiles(profiles)
    return embedding


def _consistency_risk_for_similarity(similarity):
    if similarity > HIGH_SIMILARITY_THRESHOLD:
        return 0
    elif similarity >= LOW_SIMILARITY_THRESHOLD:
        return 30
    else:
        return 70


def verify_speaker(speaker_id, audio_array, sr=16000):
    """Returns {"similarity", "match", "consistency_risk"}, or None if
    `speaker_id` has never been enrolled -- an unknown caller is a data
    gap, not evidence of risk, so callers must treat None differently
    from a low-similarity result (see risk_engine.py)."""
    profiles = _load_profiles()
    if speaker_id not in profiles:
        return None

    stored = np.array(profiles[speaker_id]["embedding"], dtype=np.float32)
    live = _extract_embedding(audio_array, sr)

    similarity = float(
        np.dot(stored, live) / (np.linalg.norm(stored) * np.linalg.norm(live))
    )
    return {
        "similarity": round(similarity, 4),
        "match": similarity > HIGH_SIMILARITY_THRESHOLD,
        "consistency_risk": _consistency_risk_for_similarity(similarity),
    }


def get_enrolled_speakers():
    return list(_load_profiles().keys())
