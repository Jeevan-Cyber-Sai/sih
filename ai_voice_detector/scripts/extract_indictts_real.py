"""
Pulls genuine (bonafide) Hindi/Tamil speech from IIT Madras's IndicTTS
corpus (mirrored on HuggingFace as SPRINGLab/IndicTTS-Hindi /
SPRINGLab/IndicTTS_Tamil), the fallback real-speech source since
SEA-Spoof (the originally requested dataset) is gated behind manual
author approval -- see readme discussion / conversation for why.

Downloads ONE parquet shard per language (not the whole multi-GB
dataset, not row-by-row streaming -- streaming this repo over HTTP
range requests was dropping connections repeatedly on this network,
same class of flakiness as the ASVspoof remote-zip WAF issue) and
extracts a capped number of samples as wav files, plus a
transcripts.json (filename -> original text) so build_multilingual_fake.py
can synthesize fake speech from the SAME sentences for a fair
real/fake comparison.

Usage:
    python extract_indictts_real.py hindi --n 250
    python extract_indictts_real.py tamil --n 250
"""
import argparse
import io
import json
import os

import soundfile as sf
from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LANG_CONFIG = {
    "hindi": {"repo": "SPRINGLab/IndicTTS-Hindi", "n_shards": 10},
    "tamil": {"repo": "SPRINGLab/IndicTTS_Tamil", "n_shards": 17},
}


def shard_name(i, n_shards):
    return f"data/train-{i:05d}-of-{n_shards:05d}.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("language", choices=sorted(LANG_CONFIG))
    parser.add_argument("--n", type=int, default=250, help="number of real samples to extract")
    args = parser.parse_args()

    cfg = LANG_CONFIG[args.language]
    dest_dir = os.path.join(ROOT, f"data_{args.language}", "real")
    os.makedirs(dest_dir, exist_ok=True)

    import pandas as pd

    transcripts = {}
    n_ok = 0
    for shard_i in range(cfg["n_shards"]):
        if n_ok >= args.n:
            break
        shard = shard_name(shard_i, cfg["n_shards"])
        print(f"downloading shard {shard} from {cfg['repo']} ...", flush=True)
        parquet_path = hf_hub_download(cfg["repo"], shard, repo_type="dataset")
        df = pd.read_parquet(parquet_path)
        print(f"shard has {len(df)} rows; need {args.n - n_ok} more")

        for i, row in df.iterrows():
            if n_ok >= args.n:
                break
            try:
                audio_bytes = row["audio"]["bytes"]
                y, sr = sf.read(io.BytesIO(audio_bytes))
                out_name = f"indictts_{args.language}_{n_ok:04d}.wav"
                out_path = os.path.join(dest_dir, out_name)
                sf.write(out_path, y, sr)
                transcripts[out_name] = str(row.get("text", ""))
                n_ok += 1
            except Exception as e:
                print(f"  SKIPPED shard {shard_i} row {i}: {e}")

    transcripts_path = os.path.join(ROOT, f"data_{args.language}", "transcripts.json")
    with open(transcripts_path, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)

    print(f"\nwrote {n_ok} real samples -> {dest_dir}")
    print(f"wrote transcripts -> {transcripts_path}")


if __name__ == "__main__":
    main()
