"""
Feature extraction for the AI voice detector: turns a preprocessed audio
array into a single fixed-length feature vector for a classical ML
classifier (MFCCs + spectral shape + pitch + energy/zero-crossing stats),
plus an extensive prosody/behavioral layer -- pitch contour dynamics,
pause/rhythm structure, jitter/shimmer/HNR microvariations, and speaking-
style dynamics -- appended after the original features so nothing in the
existing vector's layout changes; only new values are added at the end.

Rationale for the prosody layer: TTS/voice-conversion systems tend to be
UNNATURALLY stable where real human speech is variable -- flatter pitch
contours, more regular pause timing, lower jitter/shimmer, and narrower
energy dynamics. Static MFCC means/stds (the original feature set) can't
see any of that; these features specifically target how the voice moves
over time, not just its average spectral shape.
"""
import numpy as np
import librosa

SILENCE_RMS_THRESHOLD = 0.01
PITCH_RESET_THRESHOLD_HZ = 20.0


def _pitch_contour_features(voiced_f0):
    """How pitch moves over time -- slope, range, variability, frame-to-
    frame dynamics, abrupt resets, and distribution shape. All zeroed out
    (rather than raising) when there's under 2 voiced frames to compute
    a contour from at all, e.g. a near-silent or unvoiced clip."""
    if len(voiced_f0) < 2:
        return [0.0] * 9

    x = np.arange(len(voiced_f0), dtype=np.float32)
    f0_slope = float(np.polyfit(x, voiced_f0, 1)[0])
    f0_range = float(voiced_f0.max() - voiced_f0.min())
    f0_mean = float(voiced_f0.mean())
    f0_cv = float(voiced_f0.std() / f0_mean) if f0_mean > 0 else 0.0

    diffs = np.diff(voiced_f0)
    # mean of ABSOLUTE frame-to-frame change (average movement magnitude
    # -- a signed mean would hover near zero for any roughly-stable
    # contour and wash out the naturalness signal this is meant to
    # capture); std of the SIGNED differences still captures spread.
    f0_diff_mean = float(np.abs(diffs).mean())
    f0_diff_std = float(diffs.std())
    n_pitch_resets = float(np.sum(np.abs(diffs) > PITCH_RESET_THRESHOLD_HZ))

    q25, q50, q75 = np.percentile(voiced_f0, [25, 50, 75])

    return [f0_slope, f0_range, f0_cv, f0_diff_mean, f0_diff_std,
            n_pitch_resets, float(q25), float(q50), float(q75)]


def _pause_rhythm_features(rms_frames, voiced_flag, sr, hop_length):
    """Silence-segment structure and voicing rhythm -- natural speech has
    irregular pause lengths and occasional long pauses; TTS tends toward
    regular, shorter silences."""
    frame_duration = hop_length / sr
    is_silence = rms_frames < SILENCE_RMS_THRESHOLD

    silence_durations = []
    run = 0
    for v in is_silence:
        if v:
            run += 1
        else:
            if run > 0:
                silence_durations.append(run * frame_duration)
            run = 0
    if run > 0:
        silence_durations.append(run * frame_duration)

    n_silence_segments = float(len(silence_durations))
    mean_silence = float(np.mean(silence_durations)) if silence_durations else 0.0
    std_silence = float(np.std(silence_durations)) if silence_durations else 0.0
    longest_silence = float(np.max(silence_durations)) if silence_durations else 0.0

    voiced_bool = np.asarray(voiced_flag, dtype=bool) if len(voiced_flag) > 0 else np.array([], dtype=bool)
    speech_rate = float(voiced_bool.mean()) if len(voiced_bool) > 0 else 0.0

    voiced_segment_starts = []
    in_seg = False
    for i, v in enumerate(voiced_bool):
        if v and not in_seg:
            voiced_segment_starts.append(i)
            in_seg = True
        elif not v:
            in_seg = False
    n_voiced_segments = float(len(voiced_segment_starts))

    if len(voiced_segment_starts) >= 2:
        intervals = np.diff(voiced_segment_starts).astype(np.float32) * frame_duration
        rhythm_regularity = float(intervals.std())
    else:
        rhythm_regularity = 0.0

    return [n_silence_segments, mean_silence, std_silence, speech_rate,
            rhythm_regularity, longest_silence, n_voiced_segments]


def _jitter_shimmer_hnr_features(audio, sr):
    """Cycle-to-cycle microvariations via Praat (through parselmouth) --
    natural voices have small but nonzero jitter/shimmer and HNR in a
    characteristic 15-25dB band; synthetic speech tends to fall outside
    that band in one direction or the other. Praat can fail outright on
    very short/silent/unusual clips (no periodic point process to
    measure), so every failure path here returns zeros rather than
    propagating -- a missing microvariation reading isn't grounds to
    crash the whole feature extraction pipeline."""
    try:
        import parselmouth
        from parselmouth.praat import call

        snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)

        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3 = call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)

        def clean(x):
            return float(x) if x is not None and np.isfinite(x) else 0.0

        jitter_local = clean(jitter_local)
        jitter_rap = clean(jitter_rap)
        shimmer_local = clean(shimmer_local)
        shimmer_apq3 = clean(shimmer_apq3)
        hnr = clean(hnr)
        # NHR is HNR's inverse on a linear power scale, not another dB
        # value -- converting dB back to a ratio before inverting.
        nhr = float(1.0 / (10.0 ** (hnr / 10.0))) if hnr > 0 else 0.0

        return [jitter_local, jitter_rap, shimmer_local, shimmer_apq3, hnr, nhr]
    except Exception:
        return [0.0] * 6


def _speaking_style_features(rms_frames, audio, sr, mfcc):
    """Energy dynamics, spectral flux, and MFCC deltas -- how the voice's
    loudness and spectral shape change over time, which static per-
    coefficient means/stds (the original feature set) can't see at all."""
    if len(rms_frames) >= 2:
        x = np.arange(len(rms_frames), dtype=np.float32)
        energy_slope = float(np.polyfit(x, rms_frames, 1)[0])
    else:
        energy_slope = 0.0
    rms_mean = float(rms_frames.mean())
    energy_cv = float(rms_frames.std() / rms_mean) if rms_mean > 0 else 0.0

    stft_mag = np.abs(librosa.stft(audio))
    if stft_mag.shape[1] >= 2:
        flux = np.sqrt(np.sum(np.diff(stft_mag, axis=1) ** 2, axis=0))
        flux_mean = float(flux.mean())
        flux_std = float(flux.std())
    else:
        flux_mean, flux_std = 0.0, 0.0

    # Coefficients 1-5 (excluding MFCC[0], which is a log-energy term
    # already covered by the RMS/energy features above).
    mfcc_delta = librosa.feature.delta(mfcc)
    delta_subset = mfcc_delta[1:6]
    delta_mean = delta_subset.mean(axis=1)
    delta_std = delta_subset.std(axis=1)

    return [energy_slope, energy_cv, flux_mean, flux_std], delta_mean, delta_std


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

    # --- everything below is new: appended after the original feature
    # set above, which is untouched, so any model/cache built on the
    # original layout still reads its own portion of the vector correctly.

    # pyin's default hop_length (frame_length // 4 = 512) matches
    # librosa.feature.rms's default hop_length (512), so voiced_flag and
    # rms_frames below line up frame-for-frame with no reindexing needed.
    rms_frames = rms.flatten()

    feats.append(_pitch_contour_features(voiced_f0))
    feats.append(_pause_rhythm_features(rms_frames, voiced_flag, sr, hop_length=512))
    feats.append(_jitter_shimmer_hnr_features(audio, sr))

    style_feats, mfcc_delta_mean, mfcc_delta_std = _speaking_style_features(rms_frames, audio, sr, mfcc)
    feats.append(style_feats)
    feats.append(mfcc_delta_mean)
    feats.append(mfcc_delta_std)

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
