"""
Leave-one-generator-out (LOGO) evaluation: the direct answer to "did the
detector learn genuine synthesis artifacts, or memorize the specific
fingerprints of ASVspoof/ElevenLabs/gTTS?"

For each fake-class generator G, trains a fresh classifier head on ALL
data EXCLUDING G's samples, then tests purely on G (plus a fixed matched
pool of genuine samples reused across every fold). If accuracy holds up
on a generator the model never saw during training, that's real evidence
of generalization; if it collapses, the model was overfitting to specific
generators' artifacts.

Reuses train_ssl.py's per-file embedding cache -- most files here were
already embedded during earlier SSL training runs.
"""
import json
import os
import random
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from holdout import get_exclude_set  # noqa: E402
from train_ssl import XLSR_MODEL, get_or_compute_embedding  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(ROOT, "data_realworld", "generator_manifest.json")
REAL_DIR = os.path.join(ROOT, "data", "real")
REALWORLD_REAL_DIR = os.path.join(ROOT, "data_realworld", "real")

MIN_SAMPLES_FOR_LOGO = 20
N_REAL_TEST = 50
SEED = 999


def load_generator_manifest():
    with open(MANIFEST_PATH) as f:
        entries = json.load(f)
    by_generator = {}
    for e in entries:
        by_generator.setdefault(e["generator"], []).append(os.path.join(ROOT, e["path"]))
    return by_generator


def load_real_pool(exclude):
    files = []
    for d in [REAL_DIR, REALWORLD_REAL_DIR]:
        for f in os.listdir(d):
            if f.lower().endswith(".wav"):
                p = os.path.join(d, f)
                if os.path.abspath(p) not in exclude:
                    files.append(p)
    return files


def embed(path):
    return get_or_compute_embedding(path, XLSR_MODEL)


def main():
    by_generator = load_generator_manifest()
    exclude = get_exclude_set()
    real_pool = load_real_pool(exclude)

    print(f"real pool (excluding four-quadrant holdout): {len(real_pool)} files")
    for g, files in sorted(by_generator.items(), key=lambda x: -len(x[1])):
        print(f"  generator '{g}': {len(files)} fake samples")

    rng = random.Random(SEED)
    real_test = rng.sample(real_pool, min(N_REAL_TEST, len(real_pool)))
    real_test_set = set(real_test)
    real_train_pool = [f for f in real_pool if f not in real_test_set]

    print(f"\nmatched real test set (fixed, reused across all folds): {len(real_test)} files")
    print(f"real training pool: {len(real_train_pool)} files\n")

    print("pre-embedding real train/test pools...")
    real_train_embeds = {f: embed(f) for f in real_train_pool}
    real_test_embeds = {f: embed(f) for f in real_test}
    print("done.\n")

    results = []
    for g, g_files in sorted(by_generator.items()):
        if len(g_files) < MIN_SAMPLES_FOR_LOGO:
            print(f"skipping '{g}': only {len(g_files)} samples (< {MIN_SAMPLES_FOR_LOGO})\n")
            continue

        print(f"=== held out: {g} ({len(g_files)} test samples) ===")
        train_fake_files = [f for gen, files in by_generator.items() if gen != g for f in files]

        X_train, y_train = [], []
        for f in real_train_pool:
            X_train.append(real_train_embeds[f])
            y_train.append(0)
        for f in train_fake_files:
            try:
                X_train.append(embed(f))
                y_train.append(1)
            except Exception as e:
                print(f"  skip fake train {f}: {e}")

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        clf = LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")
        clf.fit(X_train_scaled, y_train)

        n_correct_fake = n_fake = 0
        for f in g_files:
            try:
                feat = embed(f).reshape(1, -1)
                pred = clf.predict(scaler.transform(feat))[0]
                n_fake += 1
                n_correct_fake += int(pred == 1)
            except Exception as e:
                print(f"  skip test fake {f}: {e}")

        n_correct_real = n_real = 0
        for f in real_test:
            feat = real_test_embeds[f].reshape(1, -1)
            pred = clf.predict(scaler.transform(feat))[0]
            n_real += 1
            n_correct_real += int(pred == 0)

        fake_acc = n_correct_fake / n_fake if n_fake else 0
        real_acc = n_correct_real / n_real if n_real else 0
        overall = (n_correct_fake + n_correct_real) / (n_fake + n_real) if (n_fake + n_real) else 0

        print(f"  detection rate on unseen '{g}': {n_correct_fake}/{n_fake} ({100 * fake_acc:.1f}%)")
        print(f"  matched real accuracy: {n_correct_real}/{n_real} ({100 * real_acc:.1f}%)")
        print(f"  overall: {100 * overall:.1f}%\n")

        results.append({
            "generator": g, "n_test_fake": n_fake, "detection_rate": fake_acc,
            "real_accuracy": real_acc, "overall": overall,
        })

    print("=" * 95)
    print("LEAVE-ONE-GENERATOR-OUT SUMMARY")
    print("=" * 95)
    print(f"{'held-out generator':<20}{'n test samples':>16}{'detection rate':>18}"
          f"{'matched real acc':>20}{'overall':>12}")
    for r in results:
        print(f"{r['generator']:<20}{r['n_test_fake']:>16}{100 * r['detection_rate']:>17.1f}%"
              f"{100 * r['real_accuracy']:>19.1f}%{100 * r['overall']:>11.1f}%")


if __name__ == "__main__":
    main()
