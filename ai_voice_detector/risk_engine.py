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


def compute_final_risk(voice_risk, context=None, profile_name=None):
    """Combines voice_risk (0-100, from the audio model) with context_risk
    (0-100, from call/transaction metadata) into a single final_risk,
    weighted per the active profile.

    SECURITY-CRITICAL ASYMMETRY, deliberate and non-negotiable: context can
    only push risk UP, never down. If voice_risk alone already clears the
    active profile's HIGH threshold, final_risk is floored at that
    threshold no matter how favourable the context looks -- a known caller
    number, a familiar beneficiary, business hours, none of it can pull a
    voice-level HIGH back down. This matters because a known/trusted
    number is *exactly* what a real attacker spoofs; treating "looks
    familiar" as exculpatory would hand attackers the easiest possible
    bypass. Context is corroborating evidence for escalating risk, never
    grounds for dismissing a voice-level HIGH finding. Without this floor,
    the weighted blend (voice_risk * voice_weight, with voice_weight < 1)
    could mathematically dilute a HIGH voice score below the HIGH
    threshold purely because context_risk was low -- this floor exists
    specifically to prevent that.
    """
    profile_name, profile = get_profile(profile_name)
    voice_weight = profile["voice_weight"]
    context_weight = profile["context_weight"]
    high_threshold = profile["high_threshold"]

    context_risk = compute_context_risk(context)
    weighted = voice_risk * voice_weight + context_risk * context_weight

    if voice_risk >= high_threshold:
        final_risk = max(weighted, high_threshold)  # <-- the security floor
    else:
        final_risk = weighted

    final_risk = min(final_risk, 100.0)
    level, action = risk_level_for_profile(final_risk, profile_name)

    return {
        "profile": profile_name,
        "voice_risk": round(float(voice_risk), 2),
        "context_risk": round(float(context_risk), 2),
        "final_risk": round(float(final_risk), 2),
        "risk_level": level,
        "recommended_action": action,
    }
