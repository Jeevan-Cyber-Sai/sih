"""
Pull a labeled subset of ASVspoof2019 LA (bonafide vs spoof) out of the
~7.1GB LA.zip hosted on datashare.ed.ac.uk, using HTTP range requests so we
never download the full archive -- only the dev-set protocol file plus the
individual flac entries we select.

Labels come strictly from the official protocol file
(ASVspoof2019.LA.cm.dev.trl.txt), never guessed from filenames.

Resumable: files already present in the output folder are skipped, so the
script can be re-run after a network interruption.
"""
import csv
import io
import os
import random
import sys
import time
import zipfile

import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from remote_zip import HTTPRangeFile

LA_ZIP_URL = "https://datashare.ed.ac.uk/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download"
DEV_PROTOCOL = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
DEV_FLAC_DIR = "LA/ASVspoof2019_LA_dev/flac/"

TARGET_PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 700
SEED = 42

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(ROOT, "data", "real")
FAKE_DIR = os.path.join(ROOT, "data", "fake")
MANIFEST = os.path.join(ROOT, "data", "download_manifest.csv")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_protocol(zf, rf):
    zinfo = zf.getinfo(DEV_PROTOCOL)
    guess_len = 30 + len(DEV_PROTOCOL.encode()) + 128 + zinfo.compress_size
    rf.prefetch(zinfo.header_offset, zinfo.header_offset + guess_len)
    text = zf.read(DEV_PROTOCOL).decode()
    bonafide, spoof = [], []
    for line in text.strip().split("\n"):
        parts = line.split()
        fname, key = parts[1], parts[-1]
        if key == "bonafide":
            bonafide.append(fname)
        elif key == "spoof":
            spoof.append(fname)
    return bonafide, spoof


def fetch_one(zf, rf, flac_name):
    zinfo = zf.getinfo(flac_name)
    guess_len = 30 + len(flac_name.encode()) + 128 + zinfo.compress_size
    rf.prefetch(zinfo.header_offset, zinfo.header_offset + guess_len)
    return zf.read(flac_name)


def main():
    os.makedirs(REAL_DIR, exist_ok=True)
    os.makedirs(FAKE_DIR, exist_ok=True)

    log("opening remote LA.zip (range requests)...")
    rf = HTTPRangeFile(LA_ZIP_URL)
    zf = zipfile.ZipFile(rf)
    log(f"remote zip length={rf.length}, entries={len(zf.namelist())}")

    bonafide, spoof = parse_protocol(zf, rf)
    log(f"dev protocol parsed: bonafide={len(bonafide)} spoof={len(spoof)}")

    rng = random.Random(SEED)
    sel_bona = rng.sample(bonafide, min(TARGET_PER_CLASS, len(bonafide)))
    sel_spoof = rng.sample(spoof, min(TARGET_PER_CLASS, len(spoof)))

    write_header = not os.path.exists(MANIFEST)
    manifest_f = open(MANIFEST, "a", newline="")
    writer = csv.writer(manifest_f)
    if write_header:
        writer.writerow(["label", "src_name", "out_path", "status", "detail"])

    jobs = [("real", f) for f in sel_bona] + [("fake", f) for f in sel_spoof]
    rng.shuffle(jobs)

    n_ok = n_skip = n_fail = 0
    for i, (label, fname) in enumerate(jobs, 1):
        out_dir = REAL_DIR if label == "real" else FAKE_DIR
        out_path = os.path.join(out_dir, fname + ".wav")

        if os.path.exists(out_path):
            n_skip += 1
            continue

        flac_entry = DEV_FLAC_DIR + fname + ".flac"
        try:
            flac_bytes = fetch_one(zf, rf, flac_entry)
            data, sr = sf.read(io.BytesIO(flac_bytes))
            sf.write(out_path, data, sr)
            writer.writerow([label, flac_entry, out_path, "ok", ""])
            n_ok += 1
        except Exception as e:
            writer.writerow([label, flac_entry, out_path, "fail", str(e)[:200]])
            n_fail += 1

        if i % 25 == 0 or i == len(jobs):
            manifest_f.flush()
            log(f"progress {i}/{len(jobs)}  ok={n_ok} skip={n_skip} fail={n_fail}")

    manifest_f.close()
    log(f"DONE. ok={n_ok} skip={n_skip} fail={n_fail}")
    log(f"real/: {len(os.listdir(REAL_DIR))} files, fake/: {len(os.listdir(FAKE_DIR))} files")


if __name__ == "__main__":
    main()
