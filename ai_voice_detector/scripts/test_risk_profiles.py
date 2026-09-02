"""
End-to-end test of the risk-profile + contextual-enrichment layer:
  1. Same audio file scored under "routine" vs "high_value_transaction"
     with different context metadata -- shows different final_risk and
     different recommended actions.
  2. The asymmetry test: a HIGH voice-risk file with maximally favourable
     context must still report HIGH, in every profile.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import analyze_file_with_context  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_FILE = os.path.join(ROOT, "data", "real", sorted(
    f for f in os.listdir(os.path.join(ROOT, "data", "real")) if f.endswith(".wav"))[5])
FAKE_FILE = os.path.join(ROOT, "data", "fake", sorted(
    f for f in os.listdir(os.path.join(ROOT, "data", "fake")) if f.endswith(".wav"))[5])


def print_result(label, r):
    print(f"  [{label}] profile={r['profile']:<22} voice={r['voice_risk']:6.2f}  "
          f"context={r['context_risk']:6.2f}  final={r['final_risk']:6.2f}  "
          f"level={r['risk_level']:<6}  action={r['recommended_action']}")


def main():
    print("=" * 100)
    print("TEST 1: same audio, different profile + context -> different outcome")
    print("=" * 100)

    print("\n-- genuine caller, routine call --")
    ctx_routine = {
        "caller_known": True, "origin": "domestic", "channel": "landline",
        "transaction_amount": 200, "new_beneficiary": False,
        "outside_business_hours": False, "previously_flagged": False,
    }
    r1 = analyze_file_with_context(REAL_FILE, ctx_routine, "routine")
    print_result("REAL file, routine profile, low-risk context", r1)

    print("\n-- same genuine caller, but now authorizing a large transfer to a new beneficiary --")
    ctx_hv = {
        "caller_known": True, "origin": "domestic", "channel": "voip",
        "transaction_amount": 25000, "new_beneficiary": True,
        "outside_business_hours": True, "previously_flagged": False,
    }
    r2 = analyze_file_with_context(REAL_FILE, ctx_hv, "high_value_transaction")
    print_result("SAME REAL file, high_value_transaction profile, risky context", r2)

    print(f"\n  Same audio (voice_risk identical: {r1['voice_risk']} vs {r2['voice_risk']}), "
          f"different outcome: {r1['risk_level']} -> {r2['risk_level']}")

    print("\n-- fake/cloned caller, routine call --")
    r3 = analyze_file_with_context(FAKE_FILE, ctx_routine, "routine")
    print_result("FAKE file, routine profile, low-risk context", r3)

    print("\n-- same fake/cloned caller, high value transaction profile --")
    r4 = analyze_file_with_context(FAKE_FILE, ctx_hv, "high_value_transaction")
    print_result("SAME FAKE file, high_value_transaction profile, risky context", r4)

    print()
    print("=" * 100)
    print("TEST 2: ASYMMETRY -- HIGH voice risk + maximally favourable context must stay HIGH")
    print("=" * 100)

    maximally_favorable_context = {
        "caller_known": True,
        "caller_number": "+1-555-0100-KNOWN",
        "origin": "domestic",
        "channel": "landline",
        "transaction_amount": 0,
        "new_beneficiary": False,
        "outside_business_hours": False,
        "previously_flagged": False,
    }

    all_high = True
    for profile in ["routine", "high_value_transaction", "privileged_access"]:
        r = analyze_file_with_context(FAKE_FILE, maximally_favorable_context, profile)
        print_result(f"FAKE file + best-case context", r)
        if r["risk_level"] != "HIGH":
            all_high = False
            print(f"    !!! ASYMMETRY VIOLATED for profile {profile}: expected HIGH, got {r['risk_level']}")

    print()
    if all_high:
        print("ASYMMETRY HOLDS: favourable context did NOT suppress a voice-level HIGH finding, "
              "in any profile.")
    else:
        print("ASYMMETRY FAILED -- see violations above.")


if __name__ == "__main__":
    main()
