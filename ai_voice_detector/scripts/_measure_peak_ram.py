"""Measures peak RAM for exactly one model-construction method, in total
isolation (own process, nothing else ever loaded). argv: <method>
method in: full | truncated_deepcopy | truncated_direct | quantized_direct
"""
import json
import os
import sys

import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features_ssl import (  # noqa: E402
    SSL_LAYER,
    extract_ssl_features,
    extract_ssl_features_truncated,
    extract_ssl_features_truncated_direct,
)
from preprocess import chunk_audio, load_and_preprocess  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(ROOT, "data", "real")
XLSR_MODEL = "facebook/wav2vec2-xls-r-300m"


def get_test_chunk():
    f = sorted(x for x in os.listdir(REAL_DIR) if x.endswith(".wav"))[0]
    audio = load_and_preprocess(os.path.join(REAL_DIR, f))
    return chunk_audio(audio, chunk_seconds=2.0)[0]


def main():
    method = sys.argv[1]
    process = psutil.Process(os.getpid())
    chunk = get_test_chunk()

    if method == "full":
        extract_ssl_features(chunk, model_name=XLSR_MODEL, layer=SSL_LAYER)
    elif method == "truncated_deepcopy":
        extract_ssl_features_truncated(chunk, model_name=XLSR_MODEL, layer=SSL_LAYER, quantize=False)
    elif method == "truncated_direct":
        extract_ssl_features_truncated_direct(chunk, model_name=XLSR_MODEL, layer=SSL_LAYER, quantize=False)
    elif method == "quantized_direct":
        extract_ssl_features_truncated_direct(chunk, model_name=XLSR_MODEL, layer=SSL_LAYER, quantize=True)
    else:
        raise ValueError(method)

    info = process.memory_info()
    peak_ram_mb = getattr(info, "peak_wset", info.rss) / (1024 * 1024)
    print("RESULT_JSON:" + json.dumps({"method": method, "peak_ram_mb": peak_ram_mb}))


if __name__ == "__main__":
    main()
