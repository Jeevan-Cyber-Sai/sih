"""
One-off organizer for the IndieFake Dataset (IFD): extracts the four
drive-download zip parts and sorts samples into data_indian/real/ and
data_indian/fake/ using ONLY the dataset's own folder labels
(Speaker-N/Bonafides/ -> real, Speaker-N/Deepfakes/ -> fake), never
guessing from filenames -- verified necessary since some deepfake files
carry another speaker's name in their filename (e.g.
Speaker-16/Deepfakes/Speaker-13_deepfake1.wav is genuinely Speaker-16's
sample; the folder is authoritative).

Destination filenames are prefixed with their source speaker/category
folder (e.g. "Speaker-16_Speaker-13_deepfake1.wav") because 69 basenames
collide across speakers once flattened -- without the prefix, later
extractions would silently overwrite earlier files of the same name.
"""
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Overridable so Colab (or any machine where the zips live somewhere
# other than next to the repo, e.g. a mounted Google Drive) can point
# this at the right place without editing the script.
SRC_DIR = os.environ.get("INDIEFAKE_ZIP_DIR", os.path.join(ROOT, "IndieFake  Dataset"))
DEST_REAL = os.path.join(ROOT, "data_indian", "real")
DEST_FAKE = os.path.join(ROOT, "data_indian", "fake")

ZIP_PARTS = [
    "drive-download-20260903T071406Z-1-001.zip",
    "drive-download-20260903T071406Z-1-002.zip",
    "drive-download-20260903T071406Z-1-003.zip",
    "drive-download-20260903T071406Z-1-004.zip",
]

LABEL_DIRS = {"Bonafides": DEST_REAL, "Deepfakes": DEST_FAKE}


def main():
    os.makedirs(DEST_REAL, exist_ok=True)
    os.makedirs(DEST_FAKE, exist_ok=True)

    n_real = n_fake = n_skipped = n_overwritten = 0
    seen = set()

    for zip_name in ZIP_PARTS:
        zip_path = os.path.join(SRC_DIR, zip_name)
        print(f"extracting {zip_name} ...", flush=True)
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                name = info.filename
                if not name.lower().endswith(".wav"):
                    continue
                parts = name.split("/")
                if len(parts) < 3:
                    n_skipped += 1
                    continue
                speaker_dir, category, fname = parts[0], parts[1], parts[-1]
                dest_dir = LABEL_DIRS.get(category)
                if dest_dir is None:
                    n_skipped += 1
                    continue

                dest_name = f"{speaker_dir}_{fname}"
                dest_path = os.path.join(dest_dir, dest_name)

                if dest_path in seen:
                    n_overwritten += 1
                seen.add(dest_path)

                with zf.open(info) as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())

                if category == "Bonafides":
                    n_real += 1
                else:
                    n_fake += 1

    print(f"\nreal (Bonafides) -> {DEST_REAL}: {n_real} files")
    print(f"fake (Deepfakes) -> {DEST_FAKE}: {n_fake} files")
    print(f"skipped (unrecognized path/label): {n_skipped}")
    if n_overwritten:
        print(f"WARNING: {n_overwritten} destination filename collisions occurred despite prefixing")


if __name__ == "__main__":
    main()
