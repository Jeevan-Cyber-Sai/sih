"""
Scores every genuinely held-out ElevenLabs and Respeecher sample (never
seen in training) with the current production models, to give a real,
aggregate answer to "does it now work for ElevenLabs" instead of judging
from one file.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import analyze_file  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(ROOT, "_eval_paths.json")) as f:
    eval_paths = json.load(f)

for gen_name, paths in eval_paths.items():
    print(f"\n=== {gen_name} ({len(paths)} held-out samples) ===")
    n_correct = 0
    scores = []
    for rel_path in paths:
        full_path = os.path.join(ROOT, rel_path)
        r = analyze_file(full_path)
        scores.append(r["risk_score"])
        correct = r["risk_level"] != "LOW"  # these are all genuinely fake
        n_correct += correct
        flag = "" if correct else "  <-- MISSED"
        print(f"  {os.path.basename(rel_path):<30} score={r['risk_score']:6.2f}  level={r['risk_level']:<6}{flag}")

    print(f"\n  {gen_name}: {n_correct}/{len(paths)} correctly flagged "
          f"({100*n_correct/len(paths):.1f}%)")
    print(f"  score range: min={min(scores):.2f} max={max(scores):.2f} "
          f"avg={sum(scores)/len(scores):.2f}")
