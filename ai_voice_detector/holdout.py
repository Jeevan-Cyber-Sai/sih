"""
Shared held-out evaluation manifest: a fixed set of files reserved for the
four-quadrant test (clean real / clean fake / real-world real / real-world
fake) and excluded from ALL training runs, so per-quadrant accuracy
reflects genuine generalization rather than memorized training data.
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(ROOT, "data_realworld", "holdout_manifest.json")


def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return []
    with open(MANIFEST_PATH, "r") as f:
        return json.load(f)


def save_manifest(entries):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def get_exclude_set():
    """Absolute paths to skip during training dataset construction."""
    return {os.path.abspath(os.path.join(ROOT, e["path"])) for e in load_manifest()}


def get_quadrants():
    """quadrant name -> list of absolute file paths, for evaluation."""
    quadrants = {}
    for e in load_manifest():
        quadrants.setdefault(e["quadrant"], []).append(os.path.abspath(os.path.join(ROOT, e["path"])))
    return quadrants
