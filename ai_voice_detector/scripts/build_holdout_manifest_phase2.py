"""
Phase 2: adds the realworld_fake quadrant to the holdout manifest, now
that data_realworld/fake/'s modern-TTS samples exist. Reserves 20 of the
tts_gtts_*.wav samples (never the degraded_*.wav ASVspoof copies, to keep
this quadrant's provenance fully independent of ASVspoof) plus the
ElevenLabs sample from test/ as a bonus real-world "commercial voice
clone" case.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from holdout import load_manifest, save_manifest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REALWORLD_FAKE_DIR = os.path.join(ROOT, "data_realworld", "fake")
ELEVENLABS_PATH = os.path.join(ROOT, "test", "ElevenLabs_Text_to_Speech_audio.mp3.mpeg")

SEED = 123  # same seed as phase 1, continued
N_REALWORLD_FAKE = 20


def main():
    entries = load_manifest()
    already = {e["path"] for e in entries}

    tts_files = sorted(f for f in os.listdir(REALWORLD_FAKE_DIR) if f.startswith("tts_gtts_"))
    rng = random.Random(SEED + 1)
    selected = rng.sample(tts_files, min(N_REALWORLD_FAKE, len(tts_files)))

    new_entries = [{"path": f"data_realworld/fake/{f}", "quadrant": "realworld_fake"} for f in selected]

    if os.path.exists(ELEVENLABS_PATH):
        new_entries.append({"path": "test/ElevenLabs_Text_to_Speech_audio.mp3.mpeg", "quadrant": "realworld_fake"})

    new_entries = [e for e in new_entries if e["path"] not in already]
    entries += new_entries
    save_manifest(entries)

    print(f"phase 2 manifest updated: realworld_fake += {len(new_entries)} "
          f"(total manifest size: {len(entries)})")


if __name__ == "__main__":
    main()
