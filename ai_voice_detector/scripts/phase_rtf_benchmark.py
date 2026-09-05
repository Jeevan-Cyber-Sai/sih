"""
Part 3, item 9: RTF (real-time factor) benchmark for the full four-layer
score_all_layers() call on a 2-second chunk, all three models loaded.
RTF = processing_time / audio_duration -- must stay under 1.0 for
real-time streaming use (a live call can't fall behind).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import score_all_layers  # noqa: E402
from preprocess import load_and_preprocess  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK_SECONDS = 2.0
N_RUNS = 20

# real speech content, not silence -- a representative benchmark input.
sample_path = os.path.join(ROOT, "test", "pc_bonafide.wav")
full_audio = load_and_preprocess(sample_path, duration=CHUNK_SECONDS)

# warm-up (model loading, JIT-ish caches) -- excluded from the timed runs.
score_all_layers(full_audio)

times = []
for i in range(N_RUNS):
    t0 = time.time()
    score_all_layers(full_audio)
    times.append(time.time() - t0)

times = np.array(times)
mean_t, std_t = times.mean(), times.std()
rtf = mean_t / CHUNK_SECONDS

print(f"chunk duration: {CHUNK_SECONDS}s, {N_RUNS} runs")
print(f"mean latency: {mean_t*1000:.1f}ms +/- {std_t*1000:.1f}ms")
print(f"RTF: {rtf:.4f}  ({'OK, under 1.0' if rtf < 1.0 else 'OVER 1.0 -- NOT real-time capable'})")
if rtf > 0.8:
    print("RTF > 0.8 -- per instructions, phase extraction should be optimized "
          "(e.g. n_fft=1024 instead of 2048, or cache STFT computation).")
