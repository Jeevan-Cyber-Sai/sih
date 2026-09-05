"""
Step 5 (MFCC/prosody half): extracts the expanded MFCC + prosody feature
vector (features.extract_features -- MFCC stats, pitch contour dynamics,
and the rest of the behavioral layer described in that module's
docstring) for data_hindi/ + data_hindi_augmented/ (and same for tamil),
matching the MFCC branch's production feature set (train.py / train_ssl.py
--v3) rather than the SSL branch.

Unlike the SSL extractor, this is cheap pure-CPU librosa work (no model
load, no GB-scale RAM pressure), so it runs single-pass with no disk
cache -- fast enough that caching would add more complexity than it saves.

Output: features_<language>_mfcc.npy in the project root -- a pickled
dict {"X": np.ndarray, "y": np.ndarray, "paths": list[str]}.

Usage:
    python extract_multilingual_mfcc_features.py hindi
    python extract_multilingual_mfcc_features.py tamil
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from features import extract_features  # noqa: E402
from preprocess import load_and_preprocess  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("language", choices=["hindi", "tamil"])
    args = parser.parse_args()

    sources = [
        (os.path.join(ROOT, f"data_{args.language}", "real"), 0),
        (os.path.join(ROOT, f"data_{args.language}_augmented", "real"), 0),
        (os.path.join(ROOT, f"data_{args.language}", "fake"), 1),
        (os.path.join(ROOT, f"data_{args.language}_augmented", "fake"), 1),
    ]
    out_path = os.path.join(ROOT, f"features_{args.language}_mfcc.npy")

    X, y, paths = [], [], []
    n_skipped = 0
    t0 = time.time()
    n_total = 0

    for d, label in sources:
        if not os.path.isdir(d):
            print(f"WARNING: {d} does not exist, skipping")
            continue
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(".wav"))
        for fname in files:
            path = os.path.join(d, fname)
            try:
                audio = load_and_preprocess(path)
                feats = extract_features(audio)
                X.append(feats)
                y.append(label)
                paths.append(path)
            except Exception as e:
                n_skipped += 1
                print(f"SKIPPED {path}: {e}")
            n_total += 1
            if n_total % 500 == 0:
                print(f"processed {n_total} ({time.time()-t0:.0f}s elapsed)", flush=True)

    X = np.array(X)
    y = np.array(y)
    elapsed = time.time() - t0
    print(f"\ndone in {elapsed:.1f}s: processed={len(paths)} skipped(errors)={n_skipped}")
    print(f"X.shape={X.shape}  y.shape={y.shape}  (real={int((y==0).sum())}, fake={int((y==1).sum())})")

    np.save(out_path, {"X": X, "y": y, "paths": paths}, allow_pickle=True)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
