"""
Benchmarks exactly ONE SSL config in isolation (own process), so peak RAM
and timing aren't contaminated by other models sharing the process.
Invoked as a subprocess by benchmark_ssl_configs.py; prints one JSON line.

argv: <config: base|xlsr|truncated|quantized> <n_runs> <n_warmup>
"""
import json
import os
import sys
import time

import numpy as np
import psutil
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features_ssl import (  # noqa: E402
    SSL_LAYER,
    extract_ssl_features,
    extract_ssl_features_truncated,
)
from preprocess import chunk_audio, load_and_preprocess  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(ROOT, "data", "real")
BASE_MODEL = "facebook/wav2vec2-base"
XLSR_MODEL = "facebook/wav2vec2-xls-r-300m"
CHUNK_SECONDS = 2.0

CONFIG_RUN_FNS = {
    "base": lambda chunk: extract_ssl_features(chunk, model_name=BASE_MODEL, layer=SSL_LAYER),
    "xlsr": lambda chunk: extract_ssl_features(chunk, model_name=XLSR_MODEL, layer=SSL_LAYER),
    "truncated": lambda chunk: extract_ssl_features_truncated(chunk, model_name=XLSR_MODEL,
                                                                layer=SSL_LAYER, quantize=False),
    "quantized": lambda chunk: extract_ssl_features_truncated(chunk, model_name=XLSR_MODEL,
                                                                layer=SSL_LAYER, quantize=True),
}


def get_test_chunk():
    sample_file = sorted(f for f in os.listdir(REAL_DIR) if f.lower().endswith(".wav"))[0]
    audio = load_and_preprocess(os.path.join(REAL_DIR, sample_file))
    return chunk_audio(audio, chunk_seconds=CHUNK_SECONDS)[0]


def main():
    config = sys.argv[1]
    n_runs = int(sys.argv[2])
    n_warmup = int(sys.argv[3])

    torch.set_num_threads(os.cpu_count())
    process = psutil.Process(os.getpid())
    chunk = get_test_chunk()
    run_fn = CONFIG_RUN_FNS[config]

    for _ in range(n_warmup):
        run_fn(chunk)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        run_fn(chunk)
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    info = process.memory_info()
    peak_ram_mb = getattr(info, "peak_wset", info.rss) / (1024 * 1024)

    result = {
        "config": config,
        "mean_ms": float(times.mean() * 1000),
        "std_ms": float(times.std() * 1000),
        "median_ms": float(np.median(times) * 1000),
        "rtf": float(times.mean() / CHUNK_SECONDS),
        "peak_ram_mb": peak_ram_mb,
    }
    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
