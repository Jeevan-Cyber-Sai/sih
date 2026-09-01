"""
Feature extraction for the AI voice detector: turns a preprocessed audio
array into a single fixed-length feature vector for a classical ML
classifier (MFCCs + spectral shape + pitch + energy/zero-crossing stats).
"""
import numpy as np
import librosa


def extract_features(audio, sr=16000, n_mfcc=20):
    audio = np.asarray(audio, dtype=np.float32)
    feats = []

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    feats.append(mfcc.mean(axis=1))
    feats.append(mfcc.std(axis=1))

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    feats.append([centroid.mean(), centroid.std()])

    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    feats.append([rolloff.mean(), rolloff.std()])

    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    feats.append([bandwidth.mean(), bandwidth.std()])

    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    feats.append(contrast.mean(axis=1))

    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, sr=sr,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
    )
    voiced_f0 = f0[~np.isnan(f0)]
    if len(voiced_f0) > 0:
        pitch_mean, pitch_std = voiced_f0.mean(), voiced_f0.std()
    else:
        pitch_mean, pitch_std = 0.0, 0.0
    voiced_ratio = np.mean(voiced_flag) if len(voiced_flag) > 0 else 0.0
    feats.append([pitch_mean, pitch_std, voiced_ratio])

    zcr = librosa.feature.zero_crossing_rate(y=audio)
    feats.append([zcr.mean(), zcr.std()])

    rms = librosa.feature.rms(y=audio)
    feats.append([rms.mean(), rms.std()])

    return np.concatenate([np.atleast_1d(f).astype(np.float32) for f in feats])


def get_feature_vector_length(sr=16000, n_mfcc=20):
    dummy = (np.random.randn(int(sr * 3.0)) * 0.01).astype(np.float32)
    return len(extract_features(dummy, sr=sr, n_mfcc=n_mfcc))


if __name__ == "__main__":
    import os
    import random

    from preprocess import load_and_preprocess

    ROOT = os.path.dirname(os.path.abspath(__file__))
    REAL_DIR = os.path.join(ROOT, "data", "real")
    FAKE_DIR = os.path.join(ROOT, "data", "fake")

    def wavs(d):
        return [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".wav")]

    random.seed(0)
    sample_real = random.sample(wavs(REAL_DIR), 5)
    sample_fake = random.sample(wavs(FAKE_DIR), 5)

    expected_len = get_feature_vector_length()
    print(f"expected feature vector length: {expected_len}")
    print()

    def process(files, label):
        vectors = []
        for path in files:
            audio = load_and_preprocess(path)
            vec = extract_features(audio)
            ok = "OK" if len(vec) == expected_len else "LENGTH MISMATCH"
            print(f"[{label}] {os.path.basename(path)} -> len={len(vec)} ({ok})")
            vectors.append(vec)
        return np.stack(vectors)

    print("=" * 60)
    print("real samples")
    print("=" * 60)
    real_vecs = process(sample_real, "real")

    print()
    print("=" * 60)
    print("fake samples")
    print("=" * 60)
    fake_vecs = process(sample_fake, "fake")

    all_ok = all(len(v) == expected_len for v in list(real_vecs) + list(fake_vecs))
    print()
    print(f"all vectors consistent length: {all_ok}")

    n_mfcc = 20
    mfcc0_idx = 0
    pitch_mean_idx = 2 * n_mfcc + 2 + 2 + 2 + 7

    print()
    print("=" * 60)
    print("real vs fake sanity check (group averages)")
    print("=" * 60)
    print(f"mean MFCC[0]:   real={real_vecs[:, mfcc0_idx].mean():.3f}   "
          f"fake={fake_vecs[:, mfcc0_idx].mean():.3f}")
    print(f"mean pitch(Hz): real={real_vecs[:, pitch_mean_idx].mean():.3f}   "
          f"fake={fake_vecs[:, pitch_mean_idx].mean():.3f}")
