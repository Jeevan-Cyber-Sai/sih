"""
Benchmarks real-time factor (RTF = processing_time / audio_duration),
per-chunk latency, and peak RAM for four SSL feature-extraction configs on
a single 2-second chunk (the unit of work in the live streaming path):

  (a) wav2vec2-base, full stack
  (b) XLS-R-300M, full stack
  (c) XLS-R-300M, truncated to only the layers we actually read
  (d) XLS-R-300M, truncated + dynamic int8 quantization (Linear layers)

First verifies (c)'s embeddings are numerically identical to (b)'s
(within float tolerance).

Each config runs in its OWN subprocess (_benchmark_one_config.py) so peak
RAM and timing for one config aren't contaminated by other models being
resident in the same process.
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features_ssl import SSL_LAYER, extract_ssl_features, extract_ssl_features_truncated  # noqa: E402
from preprocess import chunk_audio, load_and_preprocess  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(ROOT, "data", "real")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XLSR_MODEL = "facebook/wav2vec2-xls-r-300m"

N_RUNS = 25
N_WARMUP = 3

CONFIGS = [
    ("base", "(a) wav2vec2-base, full stack"),
    ("xlsr", "(b) XLS-R-300M, full stack"),
    ("truncated", "(c) XLS-R-300M, truncated"),
    ("quantized", "(d) XLS-R-300M, truncated + int8 quantized"),
]


def get_test_chunk():
    sample_file = sorted(f for f in os.listdir(REAL_DIR) if f.lower().endswith(".wav"))[0]
    audio = load_and_preprocess(os.path.join(REAL_DIR, sample_file))
    return chunk_audio(audio, chunk_seconds=2.0)[0]


def run_config_subprocess(config_key):
    script = os.path.join(SCRIPT_DIR, "_benchmark_one_config.py")
    result = subprocess.run(
        [sys.executable, script, config_key, str(N_RUNS), str(N_WARMUP)],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    print(result.stdout[-2000:])
    print(result.stderr[-2000:])
    raise RuntimeError(f"subprocess for config {config_key} produced no result")


def main():
    chunk = get_test_chunk()

    print("=" * 100)
    print("Numerical identity check: truncated vs untruncated XLS-R-300M, layer", SSL_LAYER)
    print("=" * 100)
    full_vec = extract_ssl_features(chunk, model_name=XLSR_MODEL, layer=SSL_LAYER)
    trunc_vec = extract_ssl_features_truncated(chunk, model_name=XLSR_MODEL, layer=SSL_LAYER, quantize=False)
    max_diff = float(np.max(np.abs(full_vec - trunc_vec)))
    identical = np.allclose(full_vec, trunc_vec, atol=1e-4)
    print(f"max abs diff: {max_diff:.8f}   numerically identical (atol=1e-4): {identical}")
    if not identical:
        print("WARNING: truncated embeddings do NOT match -- accuracy results for (c)/(d) "
              "using the existing classifier would not be valid.")

    print()
    print("=" * 100)
    print(f"RTF benchmark on a 2.0s chunk, {N_RUNS} runs each (after {N_WARMUP} warmup runs), "
          f"each config in its own isolated process")
    print("=" * 100)

    results = []
    for config_key, label in CONFIGS:
        print(f"\nrunning {label}...")
        r = run_config_subprocess(config_key)
        r["label"] = label
        results.append(r)
        print(f"  mean={r['mean_ms']:.1f}ms  median={r['median_ms']:.1f}ms  std={r['std_ms']:.1f}ms  "
              f"RTF={r['rtf']:.3f}  peak_RAM={r['peak_ram_mb']:.0f}MB")

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'config':<44}{'mean (ms)':>12}{'median (ms)':>14}{'RTF':>9}{'peak RAM (MB)':>16}{'speedup vs (b)':>16}")
    base_ms = next(r["mean_ms"] for r in results if r["config"] == "xlsr")
    for r in results:
        speedup = base_ms / r["mean_ms"]
        print(f"{r['label']:<44}{r['mean_ms']:>12.1f}{r['median_ms']:>14.1f}{r['rtf']:>9.3f}"
              f"{r['peak_ram_mb']:>16.0f}{speedup:>15.2f}x")


if __name__ == "__main__":
    main()
