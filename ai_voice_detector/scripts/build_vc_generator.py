"""
Builds data_generators/knnvc/: genuine any-to-any VOICE CONVERSION samples
(not TTS) using kNN-VC (bshall/knn-vc: WavLM features + kNN matching +
HiFiGAN vocoder), the technique family most relevant to the actual threat
(impersonating a specific person's voice) and the one our detector was
weakest on in leave-one-generator-out testing when ASVspoof (our only
prior VC source) was excluded.

For each sample: picks a target "identity" (a LibriSpeech speaker, using
their real speaker ID grouping in data_realworld/real/librispeech_*.wav
filenames) and a source content utterance from a DIFFERENT speaker, then
converts the source's content into the target's voice -- genuine
zero-shot voice conversion using real reference voices from our own data,
exactly as requested. Same balanced channel degradation as every other
generator.
"""
import os
import random
import re
import sys
import time
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from degrade_audio import apply_random_degradation  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REALWORLD_REAL_DIR = os.path.join(ROOT, "data_realworld", "real")
OUT_DIR = os.path.join(ROOT, "data_generators", "knnvc")

N_TARGET_SPEAKERS = 20   # each contributes ~3 converted samples -> 60+
SAMPLES_PER_TARGET = 3
SEED = 555


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _patch_torchaudio_load():
    """torchaudio's default backend (torchcodec) fails to load its native
    DLL on this machine -- route through soundfile instead, which we
    already depend on and know works."""
    def patched_load(path, *args, **kwargs):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)
        return wav, sr
    torchaudio.load = patched_load


def group_speakers():
    by_speaker = defaultdict(list)
    for f in os.listdir(REALWORLD_REAL_DIR):
        if f.startswith("librispeech_") and f.endswith(".wav"):
            m = re.match(r"librispeech_(\d+)-", f)
            if m:
                by_speaker[m.group(1)].append(os.path.join(REALWORLD_REAL_DIR, f))
    return {k: v for k, v in by_speaker.items() if len(v) >= 2}


def main():
    _patch_torchaudio_load()
    os.makedirs(OUT_DIR, exist_ok=True)

    log("loading kNN-VC (WavLM-Large + HiFiGAN, CPU)...")
    knn_vc = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True,
                             trust_repo=True, pretrained=True, device="cpu")
    log("loaded.")

    speakers = group_speakers()
    log(f"{len(speakers)} speakers with >=2 utterances available as VC targets/sources")

    rng = random.Random(SEED)
    rng_np = np.random.default_rng(SEED)
    speaker_ids = list(speakers.keys())

    target_ids = rng.sample(speaker_ids, min(N_TARGET_SPEAKERS, len(speaker_ids)))

    i = 0
    n_ok = 0
    matching_set_cache = {}

    for target_id in target_ids:
        target_files = speakers[target_id][:3]  # up to 3 reference utterances
        if target_id not in matching_set_cache:
            try:
                matching_set_cache[target_id] = knn_vc.get_matching_set(target_files)
            except Exception as e:
                log(f"  SKIPPED target {target_id}: {e}")
                continue
        matching_set = matching_set_cache[target_id]

        other_ids = [s for s in speaker_ids if s != target_id]
        for _ in range(SAMPLES_PER_TARGET):
            out_path = os.path.join(OUT_DIR, f"knnvc_{i:04d}.wav")
            if os.path.exists(out_path):
                i += 1
                n_ok += 1
                continue
            try:
                source_id = rng.choice(other_ids)
                source_file = rng.choice(speakers[source_id])

                query_seq = knn_vc.get_features(source_file)
                out_wav = knn_vc.match(query_seq, matching_set, topk=4)
                audio = out_wav.numpy().astype(np.float32)

                degraded, _ = apply_random_degradation(audio, 16000, rng_np)
                sf.write(out_path, degraded, 16000)
                n_ok += 1
            except Exception as e:
                log(f"  SKIPPED sample {i} (target={target_id}): {e}")
            i += 1
            if i % 10 == 0:
                log(f"  progress {i}/{len(target_ids) * SAMPLES_PER_TARGET}")

    log(f"DONE. knnvc: {n_ok}/{i} generated in {OUT_DIR}")


if __name__ == "__main__":
    main()
