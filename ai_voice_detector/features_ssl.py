"""
Self-supervised speech embedding features (wav2vec2 / XLS-R) as an
alternative to the hand-crafted MFCC/spectral features in features.py.

Rationale: the MFCC + RandomForest model overfit to ASVspoof2019 LA's
narrow, clean studio recording conditions (see diagnose_mismatch.py) and
misclassifies real-world audio. Pretrained SSL speech models are trained
on very large, acoustically diverse speech corpora, so their internal
representations should be far less tied to any one dataset's recording
conditions.
"""
import copy

import numpy as np
import torch
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model

# Intermediate transformer layers (roughly 5-9 of ~12/24) tend to carry
# more phonetic/prosodic/spectral detail than the final layer, which
# drifts toward higher-level linguistic content -- prior spoofing-
# detection work (e.g. SSL-AASIST-style probing) finds mid layers more
# discriminative for this kind of task. We use layer 6 as a reasonable
# middle-of-that-range default.
SSL_LAYER = 6

# Auto-detects a GPU (e.g. on Colab) and uses it; falls back to CPU
# unchanged everywhere else. Without this, model(**inputs) below would
# silently run on CPU even with a GPU allocated -- torch tensors/modules
# stay on CPU unless explicitly moved.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODEL_CACHE = {}


def load_ssl_model(model_name="facebook/wav2vec2-xls-r-300m"):
    """Loads (and caches) a pretrained wav2vec2-family model + its
    matching feature extractor. Cached globally by model_name so repeated
    calls (e.g. once per file) don't reload the model."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    model.to(DEVICE)

    _MODEL_CACHE[model_name] = (feature_extractor, model)
    return feature_extractor, model


def extract_ssl_features(audio, sr=16000, model_name="facebook/wav2vec2-xls-r-300m", layer=SSL_LAYER):
    """Mean+std pool an intermediate hidden-state layer over time into a
    single fixed-length vector, regardless of input audio length."""
    feature_extractor, model = load_ssl_model(model_name)

    inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[layer][0]  # (time_steps, hidden_dim)
    mean_pooled = hidden.mean(dim=0)
    std_pooled = hidden.std(dim=0)

    vec = torch.cat([mean_pooled, std_pooled]).cpu().numpy()
    return vec.astype(np.float32)


_TRUNCATED_CACHE = {}


def load_ssl_model_truncated(model_name="facebook/wav2vec2-xls-r-300m", layer=SSL_LAYER, quantize=False):
    """A copy of the model with every encoder layer above `layer` dropped,
    so they never execute -- we only ever read hidden_states[layer], so
    those layers do nothing but burn CPU time in the full model.

    Only reading up to `layer` is *causal*: layer i's output never
    depends on layers > i, so hidden_states[layer] is unaffected by
    truncation -- EXCEPT for wav2vec2's "stable layer norm" variant (used
    by XLS-R), which applies one extra LayerNorm to the FINAL hidden
    state only, after the loop. Truncating naively would make our target
    layer "final" and pick up that extra norm, breaking numerical
    equality with the untruncated model -- so we neutralize it.
    """
    cache_key = (model_name, layer, quantize)
    if cache_key in _TRUNCATED_CACHE:
        return _TRUNCATED_CACHE[cache_key]

    feature_extractor, full_model = load_ssl_model(model_name)

    truncated = copy.deepcopy(full_model)
    truncated.encoder.layers = torch.nn.ModuleList(list(truncated.encoder.layers)[:layer])
    if getattr(truncated.config, "do_stable_layer_norm", False):
        truncated.encoder.layer_norm = torch.nn.Identity()
    truncated.eval()

    if quantize:
        # dynamic int8 quantization has no CUDA kernels -- this path
        # always runs on CPU regardless of DEVICE.
        truncated = torch.quantization.quantize_dynamic(truncated, {torch.nn.Linear}, dtype=torch.qint8)
        truncated.eval()
    else:
        truncated.to(DEVICE)

    _TRUNCATED_CACHE[cache_key] = (feature_extractor, truncated)
    return feature_extractor, truncated


_TRUNCATED_DIRECT_CACHE = {}


def load_ssl_model_truncated_direct(model_name="facebook/wav2vec2-xls-r-300m", layer=SSL_LAYER, quantize=False):
    """Same numerical result as load_ssl_model_truncated(), but never
    materializes the full 24-layer model at all: builds the architecture
    with num_hidden_layers=layer from the start, so from_pretrained only
    allocates the layers we keep (the checkpoint's extra layer weights
    are just skipped as "unexpected", not loaded). Use this -- not the
    deepcopy-based version -- when standalone memory footprint matters,
    e.g. measuring true production RAM."""
    cache_key = (model_name, layer, quantize)
    if cache_key in _TRUNCATED_DIRECT_CACHE:
        return _TRUNCATED_DIRECT_CACHE[cache_key]

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    config = Wav2Vec2Config.from_pretrained(model_name)
    config.num_hidden_layers = layer
    model = Wav2Vec2Model.from_pretrained(model_name, config=config)
    model.eval()
    if getattr(config, "do_stable_layer_norm", False):
        model.encoder.layer_norm = torch.nn.Identity()

    if quantize:
        # dynamic int8 quantization has no CUDA kernels -- this path
        # always runs on CPU regardless of DEVICE.
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        model.eval()
    else:
        model.to(DEVICE)

    _TRUNCATED_DIRECT_CACHE[cache_key] = (feature_extractor, model)
    return feature_extractor, model


def extract_ssl_features_truncated_direct(audio, sr=16000, model_name="facebook/wav2vec2-xls-r-300m",
                                           layer=SSL_LAYER, quantize=False):
    feature_extractor, model = load_ssl_model_truncated_direct(model_name, layer, quantize)
    device = torch.device("cpu") if quantize else DEVICE

    inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[layer][0]
    mean_pooled = hidden.mean(dim=0)
    std_pooled = hidden.std(dim=0)
    vec = torch.cat([mean_pooled, std_pooled]).cpu().numpy()
    return vec.astype(np.float32)


def extract_ssl_features_truncated(audio, sr=16000, model_name="facebook/wav2vec2-xls-r-300m",
                                    layer=SSL_LAYER, quantize=False):
    feature_extractor, model = load_ssl_model_truncated(model_name, layer, quantize)
    device = torch.device("cpu") if quantize else DEVICE

    inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[layer][0]  # now the LAST entry -- the model IS this deep
    mean_pooled = hidden.mean(dim=0)
    std_pooled = hidden.std(dim=0)

    vec = torch.cat([mean_pooled, std_pooled]).cpu().numpy()
    return vec.astype(np.float32)


if __name__ == "__main__":
    import os
    import random
    import time

    from preprocess import load_and_preprocess

    ROOT = os.path.dirname(os.path.abspath(__file__))
    REAL_DIR = os.path.join(ROOT, "data", "real")
    FAKE_DIR = os.path.join(ROOT, "data", "fake")

    def wavs(d):
        return [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".wav")]

    random.seed(0)
    sample = random.sample(wavs(REAL_DIR), 2) + random.sample(wavs(FAKE_DIR), 2)

    print("loading SSL model (facebook/wav2vec2-xls-r-300m)...")
    t0 = time.time()
    load_ssl_model()
    print(f"model loaded in {time.time() - t0:.1f}s")

    lengths = []
    for path in sample:
        audio = load_and_preprocess(path)
        t0 = time.time()
        vec = extract_ssl_features(audio)
        dt = time.time() - t0
        lengths.append(len(vec))
        print(f"{os.path.basename(path)} -> vector length {len(vec)}  ({dt:.2f}s)")

    print(f"all vector lengths consistent: {len(set(lengths)) == 1} (length={lengths[0]})")
