"""
Part 3, item 8: five key test cases through the full four-layer pipeline.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import analyze_file_with_context  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
random.seed(7)


def pick(d, n=1, contains=None):
    files = [f for f in os.listdir(d) if f.lower().endswith(".wav")]
    if contains:
        files = [f for f in files if contains in f.lower()]
    return [os.path.join(d, f) for f in random.sample(files, n)]


CASES = [
    ("clean genuine (ASVspoof bonafide)", pick(os.path.join(ROOT, "data", "real"))[0]),
    ("clean fake (ASVspoof spoof)", pick(os.path.join(ROOT, "data", "fake"))[0]),
    ("real-world genuine (WhatsApp voice note)",
     pick(os.path.join(ROOT, "data_realworld", "real"), contains="whatsapp")[0]),
    ("real-world fake (ElevenLabs)", pick(os.path.join(ROOT, "data_generators", "elevenlabs"))[0]),
    ("kNN-VC clone (hardest case)", pick(os.path.join(ROOT, "data_generators", "knnvc"))[0]),
]

print(f"{'case':<45}{'ssl':>8}{'mfcc':>8}{'phase':>8}{'voice_risk':>12}{'risk_level':>12}{'conflicted':>12}  conflict_detail")
print("-" * 140)
for label, path in CASES:
    r = analyze_file_with_context(path, context=None, profile="routine")
    print(f"{label:<45}{r['ssl_score']:>8.2f}{r['mfcc_score']:>8.2f}{r['phase_score']:>8.2f}"
          f"{r['voice_risk']:>12.2f}{r['risk_level']:>12}{str(r['conflicted']):>12}  {r['conflict_detail'] or '-'}")
