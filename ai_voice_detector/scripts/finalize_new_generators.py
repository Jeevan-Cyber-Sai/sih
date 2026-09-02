"""
One-time bookkeeping after processing elevenlabs/respeecher (and covering
knnvc, generated earlier):
  1. Tags all elevenlabs/respeecher/knnvc files in generator_manifest.json
     (for LOGO evaluation).
  2. Reserves a genuine held-out eval slice per generator into
     holdout_manifest.json's realworld_fake quadrant, so the four-quadrant
     eval keeps measuring true generalization, not memorized training data.
  3. Leaves sapi/piper/edgetts exactly as they already are (100% held out,
     never used for production training) -- untouched.

The remaining (non-held-out) elevenlabs/respeecher/knnvc files become
available as new production training sources once train.py / train_ssl.py
are updated to read data_generators/{elevenlabs,respeecher,knnvc}.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from holdout import load_manifest, save_manifest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_MANIFEST_PATH = os.path.join(ROOT, "data_realworld", "generator_manifest.json")

HOLDOUT_COUNTS = {"elevenlabs": 30, "respeecher": 30, "knnvc": 12}
SEED = 727


def update_generator_manifest():
    manifest = []
    if os.path.exists(GEN_MANIFEST_PATH):
        with open(GEN_MANIFEST_PATH) as f:
            manifest = json.load(f)
    existing = {e["path"] for e in manifest}

    new = []
    for gen in ["elevenlabs", "respeecher", "knnvc"]:
        d = os.path.join(ROOT, "data_generators", gen)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".wav"):
                path = f"data_generators/{gen}/{f}"
                if path not in existing:
                    new.append({"path": path, "generator": gen})

    manifest += new
    with open(GEN_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    from collections import Counter
    print(f"generator_manifest.json: +{len(new)} new entries, "
          f"{len(manifest)} total: {dict(Counter(e['generator'] for e in manifest))}")
    return manifest


def update_holdout_manifest():
    entries = load_manifest()
    already = {e["path"] for e in entries}

    rng = random.Random(SEED)
    new_holdout = []
    for gen, n_holdout in HOLDOUT_COUNTS.items():
        d = os.path.join(ROOT, "data_generators", gen)
        if not os.path.isdir(d):
            print(f"  skip {gen}: no directory")
            continue
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(".wav"))
        selected = rng.sample(files, min(n_holdout, len(files)))
        for f in selected:
            path = f"data_generators/{gen}/{f}"
            if path not in already:
                new_holdout.append({"path": path, "quadrant": "realworld_fake"})

    entries += new_holdout
    save_manifest(entries)

    from collections import Counter
    print(f"holdout_manifest.json: +{len(new_holdout)} new realworld_fake entries, "
          f"{len(entries)} total: {dict(Counter(e['quadrant'] for e in entries))}")


if __name__ == "__main__":
    update_generator_manifest()
    update_holdout_manifest()
