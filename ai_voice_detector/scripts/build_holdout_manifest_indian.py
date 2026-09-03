"""
Adds indian_real / indian_fake quadrants to the shared holdout manifest
(holdout.py), reserved from data_indian/{real,fake}/ (the clean IndieFake
originals -- never the degraded copies in data_indian_augmented/, so this
quadrant's provenance stays as clean/unambiguous as clean_real/clean_fake).

Sized to match the existing clean_real/clean_fake quadrants (30 each) so
five-quadrant accuracy numbers are comparable across quadrants.

These paths are excluded from training via holdout.get_exclude_set() the
same way every other quadrant is -- but that only matches the exact path.
Their degraded counterparts in data_indian_augmented/ share the same
basename in a different directory, so scripts/extract_indian_ssl_features.py
and train_ssl.py's --indian mode ALSO exclude any augmented file whose
basename matches a held-out indian file, or a degraded copy of a held-out
test file would leak into training.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from holdout import load_manifest, save_manifest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDIAN_REAL_DIR = os.path.join(ROOT, "data_indian", "real")
INDIAN_FAKE_DIR = os.path.join(ROOT, "data_indian", "fake")

SEED = 202  # distinct from phase1/phase2's 123, still fixed for reproducibility
N_INDIAN_REAL = 30
N_INDIAN_FAKE = 30


def main():
    entries = load_manifest()
    already = {e["path"] for e in entries}

    real_files = sorted(f for f in os.listdir(INDIAN_REAL_DIR) if f.lower().endswith(".wav"))
    fake_files = sorted(f for f in os.listdir(INDIAN_FAKE_DIR) if f.lower().endswith(".wav"))

    rng = random.Random(SEED)
    real_selected = rng.sample(real_files, min(N_INDIAN_REAL, len(real_files)))
    rng = random.Random(SEED + 1)
    fake_selected = rng.sample(fake_files, min(N_INDIAN_FAKE, len(fake_files)))

    new_entries = (
        [{"path": f"data_indian/real/{f}", "quadrant": "indian_real"} for f in real_selected]
        + [{"path": f"data_indian/fake/{f}", "quadrant": "indian_fake"} for f in fake_selected]
    )
    new_entries = [e for e in new_entries if e["path"] not in already]
    entries += new_entries
    save_manifest(entries)

    print(f"indian_real += {sum(1 for e in new_entries if e['quadrant']=='indian_real')}")
    print(f"indian_fake += {sum(1 for e in new_entries if e['quadrant']=='indian_fake')}")
    print(f"total manifest size: {len(entries)}")


if __name__ == "__main__":
    main()
