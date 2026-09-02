"""
Reserves held-out files for 3 of the 4 evaluation quadrants (clean_real,
clean_fake, realworld_real) BEFORE data_realworld/fake/ is built, so the
degraded-ASVspoof-copy sourcing step can exclude the clean_fake holdout
files from its sampling pool (no near-duplicate leakage between train and
eval). The realworld_fake quadrant is added in phase 2, after the modern
TTS fake samples exist.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from holdout import load_manifest, save_manifest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(ROOT, "data", "real")
FAKE_DIR = os.path.join(ROOT, "data", "fake")
REALWORLD_REAL_DIR = os.path.join(ROOT, "data_realworld", "real")

SEED = 123
N_CLEAN_REAL = 30
N_CLEAN_FAKE = 30
N_REALWORLD_REAL = 20


def wavs(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(".wav"))


def main():
    rng = random.Random(SEED)

    clean_real = rng.sample(wavs(REAL_DIR), N_CLEAN_REAL)
    clean_fake = rng.sample(wavs(FAKE_DIR), N_CLEAN_FAKE)
    realworld_real = rng.sample(wavs(REALWORLD_REAL_DIR), N_REALWORLD_REAL)

    entries = []
    entries += [{"path": f"data/real/{f}", "quadrant": "clean_real"} for f in clean_real]
    entries += [{"path": f"data/fake/{f}", "quadrant": "clean_fake"} for f in clean_fake]
    entries += [{"path": f"data_realworld/real/{f}", "quadrant": "realworld_real"} for f in realworld_real]

    save_manifest(entries)
    print(f"phase 1 manifest written: clean_real={len(clean_real)} "
          f"clean_fake={len(clean_fake)} realworld_real={len(realworld_real)}")


if __name__ == "__main__":
    main()
