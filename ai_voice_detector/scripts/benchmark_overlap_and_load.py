"""
Two additional real-time safety benchmarks for config (d) [XLS-R
truncated + int8 quantized], the production SSL variant:

1. Overlap impact. More overlap means more chunks per unit of audio, i.e.
   more total compute for the same audio timespan. Reports the aggregate
   RTF (total processing time / total audio duration) at 50% and 25%
   overlap, AND the real deadline check: in a continuous overlapping
   stream, a new chunk must be ready every `hop_seconds = chunk_seconds *
   (1 - overlap)` -- NOT every `chunk_seconds`. That hop interval is the
   actual constraint (this is also why backpressure was added to
   live_capture.py: if per-chunk latency exceeds the hop interval,
   backlog compounds).

2. RTF under artificial CPU load: spawns several CPU-bound busy processes
   to simulate a loaded demo machine, then re-measures single-chunk
   latency for config (d) to see how much real headroom exists.
"""
import multiprocessing
import os
import sys
import time

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features_ssl import extract_ssl_features_truncated, load_ssl_model_truncated  # noqa: E402
from preprocess import chunk_audio  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REALWORLD_REAL_DIR = os.path.join(ROOT, "data_realworld", "real")
CHUNK_SECONDS = 2.0
N_LOAD_WORKERS = 8
N_RUNS_UNDER_LOAD = 20
N_WARMUP = 3


def score(chunk):
    return extract_ssl_features_truncated(chunk, quantize=True)


def pick_test_files(n=5):
    files = sorted(f for f in os.listdir(REALWORLD_REAL_DIR) if f.lower().endswith(".wav"))
    with_len = []
    for f in files:
        try:
            info = sf.info(os.path.join(REALWORLD_REAL_DIR, f))
            with_len.append((f, info.duration))
        except Exception:
            continue
    with_len.sort(key=lambda x: -x[1])
    return [f for f, _ in with_len[:n]]


def load_trimmed(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y, sr


def benchmark_overlap(overlap, test_files):
    total_audio_s = 0.0
    total_proc_s = 0.0
    n_chunks = 0
    for fname in test_files:
        y, sr = load_trimmed(os.path.join(REALWORLD_REAL_DIR, fname))
        chunks = chunk_audio(y, sr=sr, chunk_seconds=CHUNK_SECONDS, overlap=overlap)
        total_audio_s += len(y) / sr
        for c in chunks:
            t0 = time.perf_counter()
            score(c)
            total_proc_s += time.perf_counter() - t0
            n_chunks += 1

    hop_s = CHUNK_SECONDS * (1 - overlap)
    avg_latency_s = total_proc_s / n_chunks
    return {
        "overlap": overlap,
        "hop_s": hop_s,
        "n_chunks": n_chunks,
        "total_audio_s": total_audio_s,
        "avg_latency_ms": avg_latency_s * 1000,
        "aggregate_rtf": total_proc_s / total_audio_s,
        "hop_rtf": avg_latency_s / hop_s,
        "keeps_up_vs_hop": avg_latency_s <= hop_s,
    }


def _busy_loop():
    x = 0.0001
    while True:
        for _ in range(200000):
            x = (x * 1.0000001 + 1) % 999999
        # no sleep -- deliberately CPU-bound


def benchmark_under_load(chunk):
    for _ in range(N_WARMUP):
        score(chunk)

    times = []
    for _ in range(N_RUNS_UNDER_LOAD):
        t0 = time.perf_counter()
        score(chunk)
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    return {
        "mean_ms": times.mean() * 1000,
        "median_ms": float(np.median(times) * 1000),
        "std_ms": times.std() * 1000,
        "rtf_vs_chunk": times.mean() / CHUNK_SECONDS,
    }


def main():
    print("preloading config (d) model...")
    load_ssl_model_truncated(quantize=True)

    test_files = pick_test_files(5)
    print(f"test files (longest available in data_realworld/real): {test_files}\n")

    print("=" * 100)
    print("1. OVERLAP IMPACT (config (d), no artificial load)")
    print("=" * 100)
    results_overlap = []
    for overlap in (0.5, 0.25):
        r = benchmark_overlap(overlap, test_files)
        results_overlap.append(r)
        print(f"overlap={overlap:.2f}  hop={r['hop_s']:.2f}s  chunks={r['n_chunks']}  "
              f"avg_latency={r['avg_latency_ms']:.1f}ms  aggregate_RTF={r['aggregate_rtf']:.3f}  "
              f"hop_RTF={r['hop_rtf']:.3f}  keeps_up_vs_hop_deadline={r['keeps_up_vs_hop']}")

    print()
    print("=" * 100)
    print("2. RTF UNDER ARTIFICIAL CPU LOAD")
    print("=" * 100)

    y, sr = load_trimmed(os.path.join(REALWORLD_REAL_DIR, test_files[0]))
    test_chunk = chunk_audio(y, sr=sr, chunk_seconds=CHUNK_SECONDS, overlap=0.5)[0]

    print("baseline (no artificial load)...")
    baseline = benchmark_under_load(test_chunk)
    print(f"  mean={baseline['mean_ms']:.1f}ms  median={baseline['median_ms']:.1f}ms  "
          f"std={baseline['std_ms']:.1f}ms  RTF(vs 2s chunk)={baseline['rtf_vs_chunk']:.3f}")

    print(f"\nspawning {N_LOAD_WORKERS} CPU-bound busy processes to simulate a loaded machine...")
    workers = [multiprocessing.Process(target=_busy_loop, daemon=True) for _ in range(N_LOAD_WORKERS)]
    for w in workers:
        w.start()
    time.sleep(2.0)  # let load ramp up

    try:
        print("under load...")
        loaded = benchmark_under_load(test_chunk)
        print(f"  mean={loaded['mean_ms']:.1f}ms  median={loaded['median_ms']:.1f}ms  "
              f"std={loaded['std_ms']:.1f}ms  RTF(vs 2s chunk)={loaded['rtf_vs_chunk']:.3f}")
    finally:
        for w in workers:
            w.terminate()
        for w in workers:
            w.join(timeout=2)

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"overlap=0.50 (hop=1.00s): avg_latency={results_overlap[0]['avg_latency_ms']:.0f}ms, "
          f"hop_RTF={results_overlap[0]['hop_rtf']:.3f}, keeps up: {results_overlap[0]['keeps_up_vs_hop']}")
    print(f"overlap=0.25 (hop=1.50s): avg_latency={results_overlap[1]['avg_latency_ms']:.0f}ms, "
          f"hop_RTF={results_overlap[1]['hop_rtf']:.3f}, keeps up: {results_overlap[1]['keeps_up_vs_hop']}")
    slowdown = loaded["mean_ms"] / baseline["mean_ms"]
    print(f"under {N_LOAD_WORKERS}-process CPU load: {baseline['mean_ms']:.0f}ms -> {loaded['mean_ms']:.0f}ms "
          f"({slowdown:.2f}x slower), RTF(vs 2s chunk) {baseline['rtf_vs_chunk']:.3f} -> {loaded['rtf_vs_chunk']:.3f}")


if __name__ == "__main__":
    main()
