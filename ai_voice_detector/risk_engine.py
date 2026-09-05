"""
Risk-profile and contextual-risk enrichment layer, sitting on top of the
raw voice-only risk score from predict.py.

A single voice_risk number isn't enough to decide what to DO: authorizing
a large wire transfer and a routine account-balance call shouldn't be
held to the same bar. This module combines voice_risk (from audio) with
context_risk (from call/transaction metadata) into a final_risk, using
named, swappable "profiles" (config.yaml) so the same audio can produce
different recommended actions depending on what's actually at stake.
"""
import os

import yaml

from alerting import send_email_alert_async, send_sms_alert_async

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.yaml")

_config_cache = None


def load_config(force_reload=False):
    global _config_cache
    if _config_cache is None or force_reload:
        with open(CONFIG_PATH) as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache


def list_profiles():
    return list(load_config()["profiles"].keys())


def get_profile(profile_name=None):
    config = load_config()
    profile_name = profile_name or config.get("default_profile", "routine")
    profiles = config["profiles"]
    if profile_name not in profiles:
        raise ValueError(f"unknown profile '{profile_name}'; available: {list(profiles)}")
    return profile_name, profiles[profile_name]


def risk_level_for_profile(score, profile_name=None):
    """Same LOW/MEDIUM/HIGH classification predict.risk_level() used to do
    with hardcoded 30/70, but reading thresholds from the active profile."""
    profile_name, profile = get_profile(profile_name)
    low, high = profile["low_threshold"], profile["high_threshold"]
    if score < low:
        return "LOW", "No action needed. Proceed normally."
    elif score < high:
        return "MEDIUM", "Recommend secondary verification (callback or OTP)."
    else:
        return "HIGH", "High impersonation risk. Block transaction / escalate to supervisor."


def compute_context_risk(context=None):
    """Additive context risk score, capped at 100. Every field in
    `context` is optional; missing fields simply don't contribute.

    context: {caller_known, caller_number, origin, channel,
              transaction_amount, new_beneficiary,
              outside_business_hours, previously_flagged}
    """
    context = context or {}
    weights = load_config()["context_scoring"]

    score = 0
    if context.get("caller_known") is False:
        score += weights["unknown_caller"]
    if context.get("channel") == "voip":
        score += weights["voip_origin"]
    if context.get("origin") == "international":
        score += weights["international_origin"]

    amount = context.get("transaction_amount")
    if amount is not None and amount >= weights["large_transaction_threshold"]:
        score += weights["large_transaction"]

    if context.get("new_beneficiary"):
        score += weights["new_beneficiary"]
    if context.get("outside_business_hours"):
        score += weights["outside_business_hours"]
    if context.get("previously_flagged"):
        score += weights["previously_flagged"]

    return min(score, weights.get("cap", 100))


def compute_final_risk(voice_risk, context=None, profile_name=None, consistency_risk=None,
                        speaker_id=None, call_id=None, ssl_score=None):
    """Combines voice_risk (0-100, from the audio model), context_risk
    (0-100, from call/transaction metadata), and consistency_risk (0-100,
    from speaker_consistency.verify_speaker() -- "is this the same
    specific person who called before", independent of "is this a human
    or AI voice") into a single final_risk, weighted per the active
    profile.

    consistency_risk is None when no speaker_id was enrolled/provided for
    this call. An unknown caller is a missing data point, not a signal --
    treating None as 0 would silently read as "confirmed same person"
    (the BEST possible consistency score) and artificially deflate
    final_risk for every caller who simply hasn't been enrolled yet. So
    when it's None, that weight is excluded and voice/context are
    renormalized over just themselves, preserving their relative
    emphasis instead of leaving a chunk of the total weight unused.

    SECURITY-CRITICAL ASYMMETRY, deliberate and non-negotiable: context
    and consistency can only push risk UP, never down. If voice_risk
    ALONE, or ssl_score ALONE, already clears the active profile's HIGH
    threshold, final_risk is floored at that threshold no matter how
    favourable context or consistency look -- a known caller number, a
    familiar beneficiary, business hours, or even a voice that matches
    the enrolled profile, none of it can pull a voice-level HIGH back
    down. This matters because a known/trusted number, and a voice
    matching a stored profile, are *exactly* what a real attacker's
    clone is built to produce -- a synthetic clone that successfully
    matches the victim's own stored voiceprint is the worst-case attack,
    not a reassuring result. Context and consistency are corroborating
    evidence for escalating risk, never grounds for dismissing a
    voice-level HIGH finding. Without this floor, the weighted blend
    (voice_risk * voice_weight, with voice_weight < 1) could
    mathematically dilute a HIGH voice score below the HIGH threshold
    purely because the other signals were low -- this floor exists
    specifically to prevent that.

    The ssl_score-alone check is a SEPARATE, additional trigger for the
    same floor, not a replacement for the voice_risk one: SSL is the
    primary/most-trusted layer (predict.py's LAYER_WEIGHTS), so its own
    HIGH verdict must survive even if MFCC v3 or phase-spectrum disagree
    strongly enough to pull the blended voice_risk itself back under the
    threshold -- an independent second/third opinion disagreeing with
    SSL is grounds to flag "conflicted" (see predict.score_all_layers),
    never grounds to override SSL's own confident HIGH call. ssl_score is
    optional (None from any caller that doesn't have it split out from
    voice_risk) -- when absent, only the voice_risk-based floor applies,
    unchanged from before this layer existed.
    """
    profile_name, profile = get_profile(profile_name)
    voice_weight = profile["voice_weight"]
    context_weight = profile["context_weight"]
    consistency_weight = profile.get("consistency_weight", 0.0)
    high_threshold = profile["high_threshold"]

    context_risk = compute_context_risk(context)

    if consistency_risk is None:
        total_weight = voice_weight + context_weight
        weighted = (voice_risk * voice_weight + context_risk * context_weight) / total_weight
    else:
        weighted = (
            voice_risk * voice_weight
            + context_risk * context_weight
            + consistency_risk * consistency_weight
        )

    floor_triggered = voice_risk >= high_threshold or (ssl_score is not None and ssl_score >= high_threshold)
    if floor_triggered:
        final_risk = max(weighted, high_threshold)  # <-- the security floor
    else:
        final_risk = weighted

    final_risk = min(final_risk, 100.0)
    level, action = risk_level_for_profile(final_risk, profile_name)

    result = {
        "profile": profile_name,
        "voice_risk": round(float(voice_risk), 2),
        "context_risk": round(float(context_risk), 2),
        "final_risk": round(float(final_risk), 2),
        "risk_level": level,
        "recommended_action": action,
    }
    if consistency_risk is not None:
        result["consistency_risk"] = round(float(consistency_risk), 2)

    # Async so a slow/broken SMTP or SMS API call can never add latency
    # to scoring; alerting.py itself gates each channel independently on
    # its own enabled/alert_on/rate-limit config, so both calls are cheap
    # and safe to make unconditionally -- one channel failing or being
    # disabled never affects the other.
    send_email_alert_async(
        level, result["voice_risk"], result["final_risk"], context_flags=context,
        speaker_id=speaker_id, recommended_action=action, profile=profile_name,
        consistency_risk=result.get("consistency_risk"), call_id=call_id,
    )
    send_sms_alert_async(level, result["final_risk"], recommended_action=action, call_id=call_id)

    return result
