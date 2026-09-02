"""
Diagnose why real-world mp3 speech samples get misclassified as fake.

Compares their raw feature values against the ASVspoof data/real/ training
distribution, and checks whether losslessly re-encoding to WAV changes the
verdict -- if it does, that isolates an mp3 codec artifact rather than a
genuine content difference.
"""
import argparse
import os
import random
import tempfile

import librosa
import numpy as np
import soundfile as sf

from features import extract_features
from predict import load_model, risk_level, score_single_clip
from preprocess import load_and_preprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
ASVSPOOF_REAL_DIR = os.path.join(ROOT, "data", "real")

# Indices into the 60-dim vector extract_features() produces, per its
# concatenation order (see features.py):
#   [0:20]=mfcc_mean [20:40]=mfcc_std [40]=centroid_mean [41]=centroid_std
#   [42]=rolloff_mean [43]=rolloff_std [44]=bw_mean [45]=bw_std
#   [46:53]=contrast_mean [53]=pitch_mean [54]=pitch_std [55]=voiced_ratio
#   [56]=zcr_mean [57]=zcr_std [58]=rms_mean [59]=rms_std
KEY_FEATURES = {
    "mfcc0_mean": 0,
    "centroid_mean": 40,
    "zcr_mean": 56,
    "rms_mean": 58,
}

# mp3-family containers: plain .mp3, WhatsApp's ".mp3.mpeg" exports, and
# its .mp4-container voice notes (audio-only, decodable the same way).
AUDIO_EXTS = (".mp3", ".mpeg", ".mp4")


def _list_audio_files(folder, exclude=()):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(AUDIO_EXTS))
    if exclude:
        excluded = [f for f in files if any(x.lower() in f.lower() for x in exclude)]
        for f in excluded:
            print(f"  excluded (matches {exclude}): {f}")
        files = [f for f in files if f not in excluded]
    return files


def analyze_folder(mp3_dir, clf, scaler, exclude=()):
    files = _list_audio_files(mp3_dir, exclude)
    if not files:
        raise ValueError(f"no audio files ({AUDIO_EXTS}) found in {mp3_dir}")

    results = []
    for fname in files:
        path = os.path.join(mp3_dir, fname)
        try:
            audio = load_and_preprocess(path)
            feats = extract_features(audio)
            score = score_single_clip(audio, clf, scaler)
            level, _ = risk_level(score)
            row = {"file": fname, "risk_score": score, "risk_level": level}
            for name, idx in KEY_FEATURES.items():
                row[name] = float(feats[idx])
            results.append(row)
        except Exception as e:
            print(f"  SKIPPED {fname}: {e}")
    return results


def compute_baseline_averages(real_dir, sample_size=200, seed=0):
    files = [f for f in os.listdir(real_dir) if f.lower().endswith(".wav")]
    rng = random.Random(seed)
    if sample_size and sample_size < len(files):
        files = rng.sample(files, sample_size)

    accum = {name: [] for name in KEY_FEATURES}
    for fname in files:
        path = os.path.join(real_dir, fname)
        try:
            audio = load_and_preprocess(path)
            feats = extract_features(audio)
            for name, idx in KEY_FEATURES.items():
                accum[name].append(float(feats[idx]))
        except Exception as e:
            print(f"  [baseline] skipped {fname}: {e}")

    return {name: float(np.mean(v)) for name, v in accum.items()}, len(files)


def wav_control_test(mp3_dir, clf, scaler, n=5, exclude=()):
    files = _list_audio_files(mp3_dir, exclude)[:n]
    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for fname in files:
            mp3_path = os.path.join(mp3_dir, fname)
            try:
                y, sr = librosa.load(mp3_path, sr=16000, mono=True)
                wav_path = os.path.join(tmpdir, os.path.splitext(fname)[0] + ".wav")
                sf.write(wav_path, y, sr, subtype="PCM_16")

                mp3_audio = load_and_preprocess(mp3_path)
                mp3_score = score_single_clip(mp3_audio, clf, scaler)
                mp3_level, _ = risk_level(mp3_score)

                wav_audio = load_and_preprocess(wav_path)
                wav_score = score_single_clip(wav_audio, clf, scaler)
                wav_level, _ = risk_level(wav_score)

                rows.append({
                    "file": fname,
                    "mp3_score": mp3_score, "mp3_level": mp3_level,
                    "wav_score": wav_score, "wav_level": wav_level,
                })
            except Exception as e:
                print(f"  SKIPPED control for {fname}: {e}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Diagnose real-mp3 misclassification.")
    parser.add_argument("mp3_dir", help="folder of known-genuine mp3-family speech samples")
    parser.add_argument("--baseline-sample-size", type=int, default=200,
                         help="number of data/real/ files to average for the baseline "
                              "(default 200; use 0 for all)")
    parser.add_argument("--exclude", nargs="*", default=[],
                         help="filename substrings to exclude (e.g. a known-synthetic file "
                              "that shouldn't be treated as genuine)")
    args = parser.parse_args()

    clf, scaler = load_model()

    print("=" * 70)
    print(f"Analyzing test mp3s in: {args.mp3_dir}")
    print("=" * 70)
    results = analyze_folder(args.mp3_dir, clf, scaler, exclude=args.exclude)
    for r in results:
        print(f"{r['file']:<35} score={r['risk_score']:6.2f} ({r['risk_level']:<6}) "
              f"mfcc0={r['mfcc0_mean']:8.2f}  centroid={r['centroid_mean']:8.1f}  "
              f"zcr={r['zcr_mean']:.4f}  rms={r['rms_mean']:.4f}")

    n_flagged = sum(1 for r in results if r["risk_level"] != "LOW")
    print(f"\n{n_flagged}/{len(results)} test files scored MEDIUM or HIGH "
          f"(not classified as clearly real)")

    mp3_avgs = {name: float(np.mean([r[name] for r in results])) for name in KEY_FEATURES}

    print()
    print("=" * 70)
    print("Computing ASVspoof data/real/ baseline averages...")
    print("=" * 70)
    baseline_size = args.baseline_sample_size or None
    baseline_avgs, n_baseline = compute_baseline_averages(ASVSPOOF_REAL_DIR, sample_size=baseline_size)
    print(f"(averaged over {n_baseline} ASVspoof real samples)")

    print()
    print("=" * 70)
    print("Side-by-side feature comparison")
    print("=" * 70)
    print(f"{'feature':<16} {'test-mp3 avg':>14} {'ASVspoof-real avg':>20} {'abs diff':>10} {'ratio':>8}")
    for name in KEY_FEATURES:
        a, b = mp3_avgs[name], baseline_avgs[name]
        diff = a - b
        ratio = (a / b) if b != 0 else float("nan")
        print(f"{name:<16} {a:>14.4f} {b:>20.4f} {diff:>10.4f} {ratio:>8.2f}")

    print()
    print("=" * 70)
    n_control = min(5, len(results))
    print(f"WAV control test (first {n_control} files, losslessly re-encoded)")
    print("=" * 70)
    control_rows = wav_control_test(args.mp3_dir, clf, scaler, n=5, exclude=args.exclude)
    n_flipped = 0
    for c in control_rows:
        flipped = c["mp3_level"] != "LOW" and c["wav_level"] == "LOW"
        n_flipped += flipped
        flag = " <-- FLIPPED (mp3 wrong, wav correct)" if flipped else ""
        print(f"{c['file']:<35} mp3={c['mp3_score']:6.2f}({c['mp3_level']:<6})  "
              f"wav={c['wav_score']:6.2f}({c['wav_level']:<6}){flag}")

    print()
    if control_rows:
        print(f"{n_flipped}/{len(control_rows)} files flipped to LOW after lossless "
              f"WAV re-encoding.")
        if n_flipped == len(control_rows):
            print("-> Verdict changes with container/codec alone: this points to an "
                  "mp3/codec artifact, not a genuine content difference.")
        elif n_flipped == 0:
            print("-> Verdict unchanged after re-encoding: the mismatch is NOT a "
                  "codec artifact -- look at genuine content/recording differences "
                  "(mic, room, sample rate of source, speaker) instead.")
        else:
            print("-> Mixed result: codec artifacts may explain some but not all "
                  "of the misclassifications.")


if __name__ == "__main__":
    main()
