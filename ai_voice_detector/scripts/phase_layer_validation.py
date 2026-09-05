"""
Part 3 validation for the phase-spectrum detection layer: compares
phase-alone, the previous three-signal voice score (SSL+MFCC), and the
new four-layer voice score (SSL+MFCC+Phase) across the four standard
held-out quadrants -- does phase add value, or is it redundant with SSL?
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from holdout import get_quadrants  # noqa: E402
from predict import score_all_layers  # noqa: E402
from preprocess import load_and_preprocess  # noqa: E402

QUADRANT_TRUE_LABEL = {
    "clean_real": 0, "clean_fake": 1,
    "realworld_real": 0, "realworld_fake": 1,
}
QUADRANT_ORDER = ["clean_real", "clean_fake", "realworld_real", "realworld_fake"]

# Previous production combination (predict.py before this task): SSL 0.7, MFCC 0.3.
OLD_WEIGHTS = {"ssl": 0.7, "mfcc": 0.3}


def main():
    quadrants = get_quadrants()
    total_files = sum(len(quadrants.get(q, [])) for q in QUADRANT_ORDER)
    for q in QUADRANT_ORDER:
        print(f"{q}: {len(quadrants.get(q, []))} held-out files")
    print(f"total: {total_files} files -- one score_all_layers() call each (single SSL pass, "
          f"old-3-layer and phase-alone scores derived from the same call, not recomputed)")
    print()

    results = {"phase_alone": {}, "old_3layer_ssl_mfcc": {}, "new_4layer_ssl_mfcc_phase": {}}
    t0 = time.time()
    n_done = 0

    for q in QUADRANT_ORDER:
        files = quadrants.get(q, [])
        true_label = QUADRANT_TRUE_LABEL[q]
        correct = {k: 0 for k in results}
        n = 0

        for path in files:
            try:
                audio = load_and_preprocess(path)
                layers = score_all_layers(audio)
            except Exception as e:
                print(f"  SKIPPED {path}: {e}")
                continue
            n += 1
            n_done += 1

            phase = layers["phase_score"]
            old3 = round(layers["ssl_score"] * OLD_WEIGHTS["ssl"] + layers["mfcc_score"] * OLD_WEIGHTS["mfcc"], 2)
            new4 = layers["final_voice_risk"]

            if (1 if phase >= 50 else 0) == true_label:
                correct["phase_alone"] += 1
            if (1 if old3 >= 50 else 0) == true_label:
                correct["old_3layer_ssl_mfcc"] += 1
            if (1 if new4 >= 50 else 0) == true_label:
                correct["new_4layer_ssl_mfcc_phase"] += 1

            if n_done % 20 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (total_files - n_done) / rate if rate > 0 else 0
                print(f"  progress {n_done}/{total_files} ({elapsed:.0f}s elapsed, "
                      f"ETA {eta:.0f}s)", flush=True)

        for k in results:
            results[k][q] = (correct[k], n)
        print(f"  [{q}] done: {n} files scored")

    print("=" * 100)
    print("PER-QUADRANT ACCURACY: phase alone vs. previous 3-layer (SSL+MFCC) vs. new 4-layer (SSL+MFCC+Phase)")
    print("=" * 100)
    header = f"{'system':<32}" + "".join(f"{q:>17}" for q in QUADRANT_ORDER)
    print(header)
    for name, label in [
        ("phase_alone", "phase_alone"),
        ("old_3layer_ssl_mfcc", "old_3layer_ssl_mfcc"),
        ("new_4layer_ssl_mfcc_phase", "new_4layer_ssl_mfcc_phase"),
    ]:
        row = f"{label:<32}"
        for q in QUADRANT_ORDER:
            c, n = results[name][q]
            pct = f"{c}/{n}({100*c/n:.0f}%)" if n else "n/a"
            row += f"{pct:>17}"
        print(row)


if __name__ == "__main__":
    main()
