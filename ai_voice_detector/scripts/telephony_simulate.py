"""
Simulates G.711 mu-law telephony audio quality: downsample 16kHz -> 8kHz
(strips everything above 4kHz -- the Nyquist limit of standard telephony),
then mu-law compand + 8-bit quantize + expand (the actual lossy codec
step), then upsample back to 16kHz so it matches what the model expects.

Synthesis artifacts from neural vocoders often show up as subtle
high-frequency structure or fine quantization-noise patterns -- exactly
the kind of detail an 8-bit companded, 4kHz-bandlimited phone line would
destroy. This lets us measure that cost directly instead of assuming it.
"""
import librosa
import numpy as np

MU = 255


def mulaw_roundtrip(x, mu=MU):
    x = np.clip(x, -1.0, 1.0)
    magnitude = np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y_compressed = np.sign(x) * magnitude

    y_int8 = np.round(y_compressed * 127).astype(np.int8)
    y_compressed_r = y_int8.astype(np.float32) / 127.0

    x_reconstructed = np.sign(y_compressed_r) * (1.0 / mu) * (np.power(1.0 + mu, np.abs(y_compressed_r)) - 1.0)
    return x_reconstructed.astype(np.float32)


def simulate_telephony(audio, sr=16000, telephony_sr=8000):
    audio_8k = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=telephony_sr)
    degraded_8k = mulaw_roundtrip(audio_8k)
    audio_16k = librosa.resample(degraded_8k, orig_sr=telephony_sr, target_sr=sr)
    return audio_16k.astype(np.float32)


if __name__ == "__main__":
    import os
    import soundfile as sf

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(ROOT, "data", "real", sorted(
        f for f in os.listdir(os.path.join(ROOT, "data", "real")) if f.endswith(".wav"))[0])

    y, sr = librosa.load(src, sr=16000, mono=True)
    y_tel = simulate_telephony(y, sr=sr)
    out_path = os.path.join(ROOT, "scripts", "_telephony_test.wav")
    sf.write(out_path, y_tel, sr)
    print(f"wrote {out_path}, shape {y_tel.shape}, "
          f"peak before={np.max(np.abs(y)):.3f} after={np.max(np.abs(y_tel)):.3f}")
