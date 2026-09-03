"""
Registers the IndieFake Dataset's synthetic samples as their own
"indiefake" generator in data_realworld/generator_manifest.json, so
leave_one_generator_out_eval.py can hold the whole generator out as one
fold -- the direct test of whether the detector catches Indian-accent
deepfakes it has never trained on, rather than samples from a generator
it has partially seen.

Includes BOTH the clean IndieFake originals (data_indian/fake/) and their
channel-degraded copies (data_indian_augmented/fake/, see
degrade_indian_dataset.py) under the same "indiefake" generator tag --
they're the same underlying synthesis source, just post-processed, so
excluding only the clean copies while leaving degraded copies of the SAME
recordings in the training set would leak the generator's fingerprint
into "unseen" testing.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(ROOT, "data_realworld", "generator_manifest.json")
INDIAN_FAKE_DIR = os.path.join(ROOT, "data_indian", "fake")
INDIAN_AUGMENTED_FAKE_DIR = os.path.join(ROOT, "data_indian_augmented", "fake")

GENERATOR_NAME = "indiefake"


def main():
    with open(MANIFEST_PATH) as f:
        entries = json.load(f)
    already = {e["path"] for e in entries}

    new_entries = []
    for d, rel_prefix in [
        (INDIAN_FAKE_DIR, "data_indian/fake"),
        (INDIAN_AUGMENTED_FAKE_DIR, "data_indian_augmented/fake"),
    ]:
        if not os.path.isdir(d):
            print(f"WARNING: {d} does not exist, skipping")
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.lower().endswith(".wav"):
                continue
            rel_path = f"{rel_prefix}/{fname}"
            if rel_path in already:
                continue
            new_entries.append({"path": rel_path, "generator": GENERATOR_NAME})

    entries += new_entries
    with open(MANIFEST_PATH, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"indiefake += {len(new_entries)}")
    print(f"total generator manifest size: {len(entries)}")


if __name__ == "__main__":
    main()
