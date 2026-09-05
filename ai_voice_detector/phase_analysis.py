"""
Phase-spectrum analysis for the AI voice detector: a fourth, independent
detection layer alongside SSL embeddings, MFCC/prosody, and speaker
consistency.

Rationale: vocoders that synthesize speech (converting a predicted
spectrogram back to a waveform, or generating one directly) tend to
either discard phase entirely and reconstruct it algorithmically
(Griffin-Lim-style), or learn it end-to-end in a way that's smoother and
more structured than a real vocal tract's genuinely chaotic phase
behavior. Either way, the RELATIONSHIP between phase across frequency
bins and across time tends to look measurably different from natural
speech, even when the magnitude spectrum (which the other three layers
lean on far more) sounds convincing. This module targets that gap
directly, via the STFT phase spectrum -- something none of the other
three layers examine explicitly.
"""
import numpy as np
import librosa

N_FFT = 2048
HOP_LENGTH = 512
# 6 bands per the spec: 0-1k, 1-2k, 2-4k, 4-6k, 6-7k, 7-8k Hz (8kHz being
# Nyquist for the project's standard 16kHz sample rate).
PHASE_BAND_EDGES_HZ = [0, 1000, 2000, 4000, 6000, 7000, 8000]


def _wrapped_diff(phase, axis):
    """Frame-to-frame or bin-to-bin phase difference, wrapped back into
    (-pi, pi] -- phase is circular, so a naive np.diff() produces a
    spurious near-2*pi jump every time the true difference crosses the
    +/-pi boundary. Wrapping via the unit-phasor trick avoids that."""
    diff = np.diff(phase, axis=axis)
    return np.angle(np.exp(1j * diff))


def _frame_phase_entropy(phase, n_bins=36):
    """Shannon entropy (bits) of the phase-value distribution within one
    frame (across frequency bins), using a fixed-width histogram over
    the full (-pi, pi] range."""
    hist, _ = np.histogram(phase, bins=n_bins, range=(-np.pi, np.pi))
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist[hist > 0] / total
    return float(-np.sum(p * np.log2(p)))


def _band_indices(freqs, lo, hi):
    idx = np.where((freqs >= lo) & (freqs < hi))[0]
    return idx


def _plv(angles):
    """Phase-locking value: the length of the mean unit phasor over a
    set of angles. 1.0 = perfectly consistent phase, 0.0 = uniformly
    random -- the standard circular-statistics measure of how
    concentrated (vs. scattered) a set of phase values is."""
    if len(angles) == 0:
        return 0.0
    return float(np.abs(np.mean(np.exp(1j * angles))))


def extract_phase_features(audio, sr=16000):
    """Returns a fixed-length 1D feature vector (17 values) derived
    purely from the STFT phase spectrum. Never raises -- clips too short
    to produce at least 2 STFT frames (needed for every frame-to-frame
    or bin-to-bin difference below) return an all-zero vector of the
    correct length instead."""
    audio = np.asarray(audio, dtype=np.float32)
    # 2 (entropy, variance) + 2 (inter-frame diff) + 2 (group delay) + 1
    # (phase-magnitude corr) + 2 (instantaneous freq) + 6 (band coherence)
    # + 2 (cross-band PLV) = 17. Kept as an explicit constant (rather than
    # computed from PHASE_BAND_EDGES_HZ) so the zero-fallback below can't
    # silently drift out of sync with the real feature list.
    N_PHASE_FEATURES = 17

    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    if phase.shape[1] < 2:
        return np.zeros(N_PHASE_FEATURES, dtype=np.float32)

    feats = []

    # --- phase consistency ---
    frame_entropies = [_frame_phase_entropy(phase[:, t]) for t in range(phase.shape[1])]
    feats.append(float(np.mean(frame_entropies)))

    feats.append(float(np.mean(np.var(phase, axis=0))))

    time_diff = _wrapped_diff(phase, axis=1)  # (freq_bins, n_frames-1)
    feats.append(float(np.mean(np.abs(time_diff))))
    feats.append(float(np.std(time_diff)))

    if phase.shape[0] >= 2:
        freq_diff = _wrapped_diff(phase, axis=0)  # (freq_bins-1, n_frames)
        group_delay = -freq_diff
        feats.append(float(np.mean(group_delay)))
        feats.append(float(np.std(group_delay)))
    else:
        feats += [0.0, 0.0]

    # --- phase-magnitude relationship ---
    correlations = []
    for t in range(phase.shape[1]):
        ph, mag = phase[:, t], magnitude[:, t]
        if np.std(ph) > 1e-8 and np.std(mag) > 1e-8:
            corr = np.corrcoef(ph, mag)[0, 1]
            if np.isfinite(corr):
                correlations.append(corr)
    feats.append(float(np.mean(correlations)) if correlations else 0.0)

    # instantaneous frequency deviation (Hz) from each bin's center
    # frequency -- the same underlying time_diff as above, scaled from
    # rad/frame into Hz, so this captures actual frequency-modulation
    # magnitude rather than duplicating the raw phase-difference feature.
    inst_freq_hz = time_diff * sr / (2 * np.pi * HOP_LENGTH)
    feats.append(float(np.mean(inst_freq_hz)))
    feats.append(float(np.std(inst_freq_hz)))

    # --- spectral phase coherence ---
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    band_mean_phase = []  # one circular-mean-phase time series per band
    for lo, hi in zip(PHASE_BAND_EDGES_HZ[:-1], PHASE_BAND_EDGES_HZ[1:]):
        idx = _band_indices(freqs, lo, hi)
        if len(idx) == 0:
            feats.append(0.0)
            band_mean_phase.append(np.zeros(phase.shape[1], dtype=np.float32))
            continue
        band_phase = phase[idx, :]
        # Phase-locking value of this band's frame-to-frame phase
        # change -- "consistency within the band" per the spec.
        band_diff = _wrapped_diff(band_phase, axis=1).flatten()
        feats.append(_plv(band_diff))
        # circular mean phase per frame, for the cross-band step below
        band_mean_phase.append(np.angle(np.mean(np.exp(1j * band_phase), axis=0)))

    # Cross-band phase relationship: PLV of the phase DIFFERENCE between
    # each pair of adjacent bands, across frames -- a circular-statistics
    # analogue of correlation (ordinary Pearson correlation isn't
    # meaningful on wrapped angles), summarized as mean +/- std over the
    # 5 adjacent-band pairs rather than reported individually.
    cross_band_plvs = []
    for i in range(len(band_mean_phase) - 1):
        cross_band_plvs.append(_plv(band_mean_phase[i] - band_mean_phase[i + 1]))
    feats.append(float(np.mean(cross_band_plvs)))
    feats.append(float(np.std(cross_band_plvs)))

    return np.array(feats, dtype=np.float32)


def get_phase_feature_length(sr=16000):
    dummy = (np.random.randn(int(sr * 3.0)) * 0.01).astype(np.float32)
    return len(extract_phase_features(dummy, sr=sr))


if __name__ == "__main__":
    import os
    import random

    from preprocess import load_and_preprocess

    ROOT = os.path.dirname(os.path.abspath(__file__))
    REAL_DIR = os.path.join(ROOT, "data", "real")
    FAKE_DIR = os.path.join(ROOT, "data", "fake")

    def wavs(d):
        return [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".wav")]

    random.seed(42)
    real_files = random.sample(wavs(REAL_DIR), 3)
    fake_files = random.sample(wavs(FAKE_DIR), 3)

    expected_len = get_phase_feature_length()
    print(f"expected phase feature vector length: {expected_len}")
    print()

    NAMES = [
        "phase_entropy", "phase_variance",
        "inter_frame_diff_mean", "inter_frame_diff_std",
        "group_delay_mean", "group_delay_std",
        "phase_magnitude_corr",
        "inst_freq_mean_hz", "inst_freq_std_hz",
        "band_coherence_0_1k", "band_coherence_1_2k", "band_coherence_2_4k",
        "band_coherence_4_6k", "band_coherence_6_7k", "band_coherence_7_8k",
        "cross_band_plv_mean", "cross_band_plv_std",
    ]
    assert len(NAMES) == expected_len, (len(NAMES), expected_len)

    def process(files, label):
        vectors = []
        for path in files:
            audio = load_and_preprocess(path)
            vec = extract_phase_features(audio)
            ok = "OK" if len(vec) == expected_len else "LENGTH MISMATCH"
            print(f"[{label}] {os.path.basename(path)} -> len={len(vec)} ({ok})")
            vectors.append(vec)
        return np.stack(vectors)

    print("=" * 60)
    real_vecs = process(real_files, "real")
    print()
    fake_vecs = process(fake_files, "fake")

    all_ok = all(len(v) == expected_len for v in list(real_vecs) + list(fake_vecs))
    print()
    print(f"all vectors consistent length: {all_ok}")
    print()

    print(f"{'feature':<24}{'real avg':>14}{'fake avg':>14}{'abs gap':>14}")
    print("-" * 66)
    gaps = []
    for i, name in enumerate(NAMES):
        r, f = real_vecs[:, i].mean(), fake_vecs[:, i].mean()
        gap = abs(r - f)
        gaps.append((name, gap))
        print(f"{name:<24}{r:>14.4f}{f:>14.4f}{gap:>14.4f}")

    print()
    print("largest real-vs-fake gaps (by |difference|, unnormalized -- a rough signal, not a final ranking):")
    for name, gap in sorted(gaps, key=lambda x: -x[1])[:5]:
        print(f"  {name}: {gap:.4f}")
