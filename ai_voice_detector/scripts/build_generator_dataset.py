"""
Builds data_generators/<name>/ for each new TTS/voice-cloning generator
we've never trained on, so leave-one-generator-out evaluation can test
whether the detector learned genuine synthesis artifacts or just
memorized ASVspoof/ElevenLabs/gTTS's specific fingerprints.

Also writes/updates data_realworld/generator_manifest.json, tagging every
fake-class file (old and new) with which generator produced it, and
degrades a portion of each new generator's output the same way
data_realworld/fake/ was degraded, so no generator is systematically
"cleaner" than another (that would just reintroduce a shortcut).
"""
import io
import json
import os
import random
import sys
import tarfile
import wave

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from degrade_audio import apply_random_degradation  # noqa: E402
from ensure_ffmpeg import ensure_ffmpeg_on_path  # noqa: E402

ensure_ffmpeg_on_path()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(ROOT, "data_generators")
TARBALL_PATH = os.path.join(ROOT, "scripts", "_dev-clean.tar.gz")
MANIFEST_PATH = os.path.join(ROOT, "data_realworld", "generator_manifest.json")

N_PER_GENERATOR = 60
SEED = 202


def log(msg):
    print(msg, flush=True)


def load_sentences(n_needed, seed, skip=0):
    sentences = []
    with tarfile.open(TARBALL_PATH, "r:gz") as tar:
        trans_members = [m for m in tar.getmembers() if m.name.endswith(".trans.txt")]
        rng = random.Random(seed)
        rng.shuffle(trans_members)
        for m in trans_members:
            f = tar.extractfile(m)
            text = f.read().decode("utf-8", errors="ignore")
            for line in text.strip().split("\n"):
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    sentences.append(parts[1].strip())
            if len(sentences) >= (n_needed + skip) * 3:
                break

    rng.shuffle(sentences)
    seen, filtered = set(), []
    for s in sentences:
        if 25 <= len(s) <= 180 and s not in seen:
            seen.add(s)
            filtered.append(s)
    return filtered[skip:skip + n_needed]


def _get_voice_ids():
    import pyttsx3
    probe = pyttsx3.init()
    ids = [v.id for v in probe.getProperty("voices")]
    probe.stop()
    return ids


def _synth_one_sapi(text, voice_id, tmp_path, timeout=15):
    """Runs a single SAPI synthesis in its own subprocess with a timeout --
    pyttsx3 engines are known to hang on Windows if reused across many
    runAndWait() calls, and even a fresh-per-call engine can occasionally
    stall, so isolate + bound every call rather than risk the whole batch."""
    import subprocess
    code = (
        "import pyttsx3, sys\n"
        "e = pyttsx3.init()\n"
        f"e.setProperty('voice', {voice_id!r})\n"
        f"e.save_to_file({text!r}, {tmp_path!r})\n"
        "e.runAndWait()\n"
    )
    try:
        subprocess.run([sys.executable, "-c", code], timeout=timeout,
                        capture_output=True, check=True)
        return True
    except Exception:
        return False


def build_sapi(sentences, out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)
    voice_ids = _get_voice_ids()
    rng = random.Random(seed)
    rng2 = np.random.default_rng(seed)

    n_ok = 0
    for i, text in enumerate(sentences):
        out_path = os.path.join(out_dir, f"sapi_{i:04d}.wav")
        if os.path.exists(out_path):
            n_ok += 1
            continue
        try:
            voice_id = rng.choice(voice_ids)
            tmp_path = out_path + ".raw.wav"
            if not _synth_one_sapi(text, voice_id, tmp_path):
                log(f"  SKIPPED sapi {i}: synthesis timed out or failed")
                continue

            audio, sr = sf.read(tmp_path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            degraded, _ = apply_random_degradation(audio, sr, rng2)
            sf.write(out_path, degraded, sr)
            os.remove(tmp_path)
            n_ok += 1
        except Exception as e:
            log(f"  SKIPPED sapi {i}: {e}")
        if (i + 1) % 20 == 0:
            log(f"  sapi: {i + 1}/{len(sentences)}")
    return n_ok


def build_piper(sentences, out_dir, seed, voice_path):
    from piper import PiperVoice

    os.makedirs(out_dir, exist_ok=True)
    voice = PiperVoice.load(voice_path)
    rng = np.random.default_rng(seed)

    n_ok = 0
    for i, text in enumerate(sentences):
        out_path = os.path.join(out_dir, f"piper_{i:04d}.wav")
        if os.path.exists(out_path):
            n_ok += 1
            continue
        try:
            tmp_path = out_path + ".raw.wav"
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(voice.config.sample_rate)
                voice.synthesize_wav(text, wf)

            audio, sr = sf.read(tmp_path)
            audio = audio.astype(np.float32)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            degraded, _ = apply_random_degradation(audio, sr, rng)
            sf.write(out_path, degraded, sr)
            os.remove(tmp_path)
            n_ok += 1
        except Exception as e:
            log(f"  SKIPPED piper {i}: {e}")
        if (i + 1) % 20 == 0:
            log(f"  piper: {i + 1}/{len(sentences)}")
    return n_ok


EDGE_VOICES = ["en-US-AriaNeural", "en-US-GuyNeural", "en-GB-SoniaNeural", "en-IN-NeerjaNeural"]


def _synth_one_edge(text, voice, tmp_path, timeout=15):
    """Isolated subprocess per call, same rationale as SAPI: don't let one
    stalled network call hang the whole batch."""
    import subprocess
    code = (
        "import asyncio, edge_tts\n"
        "async def main():\n"
        f"    c = edge_tts.Communicate({text!r}, {voice!r})\n"
        f"    await c.save({tmp_path!r})\n"
        "asyncio.run(main())\n"
    )
    try:
        subprocess.run([sys.executable, "-c", code], timeout=timeout,
                        capture_output=True, check=True)
        return True
    except Exception:
        return False


def build_edge_tts(sentences, out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    rng2 = np.random.default_rng(seed)

    n_ok = 0
    for i, text in enumerate(sentences):
        out_path = os.path.join(out_dir, f"edgetts_{i:04d}.wav")
        if os.path.exists(out_path):
            n_ok += 1
            continue
        try:
            voice = rng.choice(EDGE_VOICES)
            tmp_path = out_path + ".raw.mp3"
            if not _synth_one_edge(text, voice, tmp_path):
                log(f"  SKIPPED edgetts {i}: synthesis timed out or failed")
                continue

            import librosa
            audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
            audio = audio.astype(np.float32)
            degraded, _ = apply_random_degradation(audio, sr, rng2)
            sf.write(out_path, degraded, sr)
            os.remove(tmp_path)
            n_ok += 1
        except Exception as e:
            log(f"  SKIPPED edgetts {i}: {e}")
        if (i + 1) % 20 == 0:
            log(f"  edgetts: {i + 1}/{len(sentences)}")
    return n_ok


def update_manifest(new_entries):
    manifest = []
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    existing_paths = {e["path"] for e in manifest}
    for e in new_entries:
        if e["path"] not in existing_paths:
            manifest.append(e)
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def seed_manifest_from_existing():
    """Tags the fake-class files we already have (from earlier phases)
    with their generator, so the manifest covers everything, not just the
    new generators added here."""
    entries = []

    fake_dir = os.path.join(ROOT, "data", "fake")
    for f in os.listdir(fake_dir):
        if f.lower().endswith(".wav"):
            entries.append({"path": f"data/fake/{f}", "generator": "asvspoof"})

    rw_fake_dir = os.path.join(ROOT, "data_realworld", "fake")
    if os.path.isdir(rw_fake_dir):
        for f in os.listdir(rw_fake_dir):
            if not f.lower().endswith(".wav"):
                continue
            if f.startswith("tts_gtts_"):
                entries.append({"path": f"data_realworld/fake/{f}", "generator": "gtts"})
            elif f.startswith("degraded_"):
                entries.append({"path": f"data_realworld/fake/{f}", "generator": "asvspoof"})

    elevenlabs_path = os.path.join(ROOT, "test", "ElevenLabs_Text_to_Speech_audio.mp3.mpeg")
    if os.path.exists(elevenlabs_path):
        entries.append({"path": "test/ElevenLabs_Text_to_Speech_audio.mp3.mpeg", "generator": "elevenlabs"})

    return entries


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    log("seeding manifest with existing fake-class generators...")
    existing_entries = seed_manifest_from_existing()
    update_manifest(existing_entries)
    log(f"  tagged {len(existing_entries)} existing files "
        f"(asvspoof/gtts/elevenlabs)")

    log("\nbuilding SAPI (Windows TTS) generator...")
    sapi_sentences = load_sentences(N_PER_GENERATOR, SEED, skip=0)
    sapi_dir = os.path.join(OUT_ROOT, "sapi")
    n_sapi = build_sapi(sapi_sentences, sapi_dir, SEED)
    log(f"  sapi: {n_sapi}/{len(sapi_sentences)} generated")
    update_manifest([
        {"path": f"data_generators/sapi/{f}", "generator": "sapi"}
        for f in os.listdir(sapi_dir) if f.lower().endswith(".wav")
    ])

    log("\nbuilding Piper (neural ONNX TTS) generator...")
    piper_sentences = load_sentences(N_PER_GENERATOR, SEED, skip=N_PER_GENERATOR)
    piper_dir = os.path.join(OUT_ROOT, "piper")
    voice_path = os.path.join(ROOT, "scripts", "_piper_voices", "en_US-lessac-medium.onnx")
    n_piper = build_piper(piper_sentences, piper_dir, SEED, voice_path)
    log(f"  piper: {n_piper}/{len(piper_sentences)} generated")
    update_manifest([
        {"path": f"data_generators/piper/{f}", "generator": "piper"}
        for f in os.listdir(piper_dir) if f.lower().endswith(".wav")
    ])

    log("\nbuilding Edge-TTS (Microsoft neural cloud TTS) generator...")
    edge_sentences = load_sentences(N_PER_GENERATOR, SEED, skip=2 * N_PER_GENERATOR)
    edge_dir = os.path.join(OUT_ROOT, "edgetts")
    n_edge = build_edge_tts(edge_sentences, edge_dir, SEED)
    log(f"  edgetts: {n_edge}/{len(edge_sentences)} generated")
    update_manifest([
        {"path": f"data_generators/edgetts/{f}", "generator": "edgetts"}
        for f in os.listdir(edge_dir) if f.lower().endswith(".wav")
    ])

    manifest = json.load(open(MANIFEST_PATH))
    from collections import Counter
    counts = Counter(e["generator"] for e in manifest)
    log(f"\nDONE. generator_manifest.json now covers {len(manifest)} files: {dict(counts)}")


if __name__ == "__main__":
    main()
