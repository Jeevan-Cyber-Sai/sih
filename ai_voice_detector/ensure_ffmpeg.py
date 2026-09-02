"""
Makes an `ffmpeg` executable available on PATH using the static binary
bundled by imageio-ffmpeg, so librosa/audioread can decode containers
libsndfile can't (AAC/Opus-in-mp4 WhatsApp voice notes, etc.) without
requiring a manual system-wide ffmpeg install.
"""
import os
import shutil

import imageio_ffmpeg

_BIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ffmpeg_bin")


def ensure_ffmpeg_on_path():
    found = shutil.which("ffmpeg")
    if found:
        return found

    os.makedirs(_BIN_DIR, exist_ok=True)
    target = os.path.join(_BIN_DIR, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not os.path.exists(target):
        src = imageio_ffmpeg.get_ffmpeg_exe()
        shutil.copy2(src, target)
        if os.name != "nt":
            os.chmod(target, 0o755)

    os.environ["PATH"] = _BIN_DIR + os.pathsep + os.environ.get("PATH", "")
    return target


if __name__ == "__main__":
    import subprocess

    path = ensure_ffmpeg_on_path()
    print(f"ffmpeg available at: {path}")
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    print(result.stdout.splitlines()[0])
