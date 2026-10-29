"""
compare_beatnet_plus.py

Evaluate the BeatNet+ beat tracker against ground-truth beats derived from the Slakh MIDI.

Outputs (OUT_DIR):
  tracks.csv / windows.csv  metrics per (track, guitar stem), and per 30 s window
  summary.png               distributions of the main metrics + chance baseline
  plots/*.png, plots/*.wav  worst and random windows, with a click track of GT (high) and tracker (low)
  cache/*.npz               cached tracker outputs

Usage examples:
  # Default: Evaluate using non-percussive / solo-instrument weights (recommended for guitar stems)
  python compare_beatnet_plus.py --beatnet_plus_variant af_non_percussive --num_workers 8

  # Evaluate using generic dual-branch weights
  python compare_beatnet_plus.py --beatnet_plus_variant generic --num_workers 8

  # Strictly causal LFO synthesis with octave-folded predictions
  python compare_beatnet_plus.py --causal_lfo --fold_pred --num_workers 8
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import types
import json
import math
import hashlib
import pathlib
import argparse
import warnings
import collections
import collections.abc
import multiprocessing as mp
import concurrent.futures

import numpy as np

# === Compatibility shims for BeatNet+ & madmom on Python>=3.10 / NumPy>=1.24 ===
try:
    import pyaudio  # noqa: F401
except ModuleNotFoundError:
    _pa = types.ModuleType("pyaudio")
    _pa.paInt16 = 8
    _pa.paFloat32 = 1
    _pa.PyAudio = lambda *args, **kwargs: None
    sys.modules["pyaudio"] = _pa

if not hasattr(collections, "MutableSequence"):
    collections.MutableSequence = collections.abc.MutableSequence
if not hasattr(collections, "Callable"):
    collections.Callable = collections.abc.Callable
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

import pandas as pd
import librosa
import soundfile as sf
import pretty_midi
import mir_eval
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf

warnings.filterwarnings("ignore", module="mir_eval")
warnings.filterwarnings("ignore", category=UserWarning)

# === Configuration ===
SPLIT = "test"
PROCESSED = pathlib.Path("/data/ssd0/rakec/slakh2100/processed")
DATA_PATH = PROCESSED / SPLIT
LATENT_DIR = PROCESSED / "latents" / SPLIT
DEFAULT_OUT_ROOT = pathlib.Path("/data/hdd0/rakec/test")
PRE_ENCODE_CFG = "configs/pre_encode_slakh2100.yaml"

BEAT_CHECK_REPORT = PROCESSED / "metadata" / f"beat_check_{SPLIT}.json"
MIN_GRID_SCORE = 1.3

DURATION_SEC = 30
STRIDE_SEC = 15
TRIM_SEC = 5.0
LAG_MAX_FRAMES = 6
N_PLOT_WORST = 30
N_PLOT_RANDOM = 20

BEATNET_PLUS_WEIGHT_MAP = {
    "af_non_percussive": "af_non_percussive_weights.pt",
    "generic": "generic_weights.pt",
    "generic_main": "generic_main_weights.pt",
}

DEFAULT_BEAT_CFG = {
    "beatnet_plus_variant": "af_non_percussive",
    "beatnet_mode": "online",
    "beatnet_inference": "PF",
    "beatnet_weights": None,
    "device": "cpu",
    "causal_lfo": False,
    "fold_pred": False,
    "low_bpm": 90,
    "high_bpm": 180,
    "lookahead": 7,
    "silence_db": -50.0,
    "fold_tempo": True,
}

TRACKER_KEYS = [
    "hop", "beatnet_plus_variant", "beatnet_mode", "beatnet_inference",
    "beatnet_weights", "silence_db",
]
EVENT_KEYS = ["F", "Cemgil", "CMLc", "CMLt", "AMLc", "AMLt"]

# Process-local cache for PyTorch models
_WORKER_BEATNET_PLUS = {}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--beatnet_plus_variant", choices=list(BEATNET_PLUS_WEIGHT_MAP.keys()),
                   default="af_non_percussive",
                   help="Pre-trained BeatNet+ weights variant (default: af_non_percussive)")
    p.add_argument("--beatnet_mode", choices=["online", "realtime", "offline"], default="online",
                   help="Inference mode ('online' uses causal CRNN + PF without real-time sleep)")
    p.add_argument("--beatnet_inference", choices=["PF", "DBN"], default="PF",
                   help="Inference algorithm ('PF' for causal Particle Filter, 'DBN' for offline)")
    p.add_argument("--beatnet_weights", type=str, default=None,
                   help="Optional explicit path to a .pt weights file (overrides --beatnet_plus_variant)")
    p.add_argument("--device", type=str, default="cpu",
                   help="PyTorch device (e.g. 'cpu' or 'cuda')")
    p.add_argument("--causal_lfo", action="store_true",
                   help="Synthesize LFO/tempo strictly causally from past beats only")
    p.add_argument("--fold_pred", action="store_true",
                   help="Octave-fold primary beat predictions (pk_times) into [low_bpm, high_bpm]")
    p.add_argument("--no_fold_gt", action="store_true",
                   help="Disable octave folding on MIDI ground-truth beats")
    p.add_argument("--stem_glob", type=str, default="guitar_*.wav",
                   help="Glob pattern for stems inside each track folder")
    p.add_argument("--out_dir", type=pathlib.Path, default=None,
                   help="Custom output directory")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_tracks", type=int, default=None)
    p.add_argument("--recompute", action="store_true", help="ignore cached tracker outputs")
    p.add_argument("--no_skip_bad_gt", action="store_true", help="also evaluate tracks flagged by the beat check")
    p.add_argument("--no_render", action="store_true", help="skip per-window plots and click tracks")
    return p.parse_args()


def load_cfg(path, args):
    cfg = dict(DEFAULT_BEAT_CFG)
    hop = 1600
    p = pathlib.Path(path)
    if p.exists():
        y = OmegaConf.to_container(OmegaConf.load(p), resolve=True)
        cfg.update(y.get("beat", {}) or {})
        hop = int(y.get("compression_ratio", hop))
    else:
        print(f"[WARNING] {path} not found, using hop={hop}.")
    cfg["hop"] = hop
    cfg["beatnet_plus_variant"] = args.beatnet_plus_variant
    cfg["beatnet_mode"] = args.beatnet_mode
    cfg["beatnet_inference"] = args.beatnet_inference
    cfg["beatnet_weights"] = args.beatnet_weights
    cfg["device"] = args.device
    cfg["causal_lfo"] = args.causal_lfo
    cfg["fold_pred"] = args.fold_pred
    if args.no_fold_gt:
        cfg["fold_tempo"] = False
    return cfg


# === Ground truth & Phase helpers ===
def fold_factor(beats, low_bpm, high_bpm):
    if beats.size < 2:
        return 1.0
    median_bpm = 60.0 / max(float(np.median(np.diff(beats))), 1e-6)
    f = 1.0
    while median_bpm * f < low_bpm:
        f *= 2.0
    while median_bpm * f > high_bpm:
        f *= 0.5
    return f


def beat_positions(beats, times):
    idx = np.arange(beats.size, dtype=np.float64)
    pos = np.interp(times, beats, idx)
    first_iv = max(beats[1] - beats[0], 1e-6)
    last_iv = max(beats[-1] - beats[-2], 1e-6)
    pos = np.where(times < beats[0], (times - beats[0]) / first_iv, pos)
    pos = np.where(times > beats[-1], (beats.size - 1) + (times - beats[-1]) / last_iv, pos)
    return pos


def folded_beat_times(beats, f, t_max):
    if beats.size < 2:
        return beats[beats <= t_max]
    idx = np.arange(beats.size, dtype=np.float64)
    x = np.arange(0.0, beats.size - 1 + 1e-9, 1.0 / f)
    t = np.interp(x, idx, beats)
    return t[t <= t_max]


def gt_tempo_per_frame(beats, f, times):
    if beats.size < 2:
        return np.zeros_like(times, dtype=np.float32)
    ivs = np.diff(beats)
    mid = 0.5 * (beats[:-1] + beats[1:])
    period = np.interp(times, mid, ivs)
    return (60.0 / np.maximum(period, 1e-6) * f).astype(np.float32)


def synthesize_lfo_and_tempo_from_beats(beat_times, frame_times, fold=1.0, causal=False):
    num_frames = len(frame_times)
    if beat_times.size < 2:
        return (
            np.zeros(num_frames, dtype=np.float32),
            np.zeros(num_frames, dtype=np.float32),
            np.zeros(num_frames, dtype=bool),
        )

    if not causal:
        pos = beat_positions(beat_times, frame_times) * fold
        lfo = np.cos(2.0 * np.pi * pos).astype(np.float32)
        tempo = gt_tempo_per_frame(beat_times, fold, frame_times)
        trusted = (frame_times >= beat_times[0]) & (frame_times <= beat_times[-1])
        return lfo, tempo, trusted

    lfo = np.zeros(num_frames, dtype=np.float32)
    tempo = np.zeros(num_frames, dtype=np.float32)
    trusted = np.zeros(num_frames, dtype=bool)

    idx = np.searchsorted(beat_times, frame_times, side="right") - 1
    valid = idx >= 1
    if not np.any(valid):
        return lfo, tempo, trusted

    k = idx[valid]
    t_last = beat_times[k]
    ibi = np.maximum(beat_times[k] - beat_times[k - 1], 1e-6)
    inst_bpm = (60.0 / ibi) * fold

    dt = frame_times[valid] - t_last
    within_hold = dt <= (2.5 * ibi)
    phase = (dt / ibi) * fold

    lfo[valid] = np.where(within_hold, np.cos(2.0 * np.pi * phase), 0.0).astype(np.float32)
    tempo[valid] = np.where(within_hold, inst_bpm, 0.0).astype(np.float32)
    trusted[valid] = within_hold
    return lfo, tempo, trusted


# === BeatNet+ Tracker Logic ===
def compute_frame_rms_db(mono, hop, num_frames):
    x = mono
    if x.shape[0] < num_frames * hop:
        x = np.pad(x, (0, num_frames * hop - x.shape[0]))
    frames = x[:num_frames * hop].reshape(num_frames, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-24)
    return (20.0 * np.log10(np.maximum(rms, 1e-12))).astype(np.float32)


def resolve_beatnet_plus_weights(cfg):
    if cfg.get("beatnet_weights"):
        p = pathlib.Path(cfg["beatnet_weights"])
        if not p.exists():
            raise FileNotFoundError(f"Custom BeatNet+ weights not found: {p}")
        return str(p)

    import BeatNetPlus
    import urllib.request

    pkg_dir = pathlib.Path(BeatNetPlus.__file__).resolve().parent
    fname = BEATNET_PLUS_WEIGHT_MAP[cfg["beatnet_plus_variant"]]

    # 1. Try local site-packages / source locations first
    candidates = [
        pkg_dir / "models" / fname,
        pkg_dir.parent / "BeatNetPlus" / "models" / fname,
        pathlib.Path("src/BeatNetPlus/models") / fname,
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    # 2. If stripped from the pip package, download it automatically to a local cache
    cache_dir = pathlib.Path.home() / ".beatnet_plus_weights"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / fname

    if out_path.exists():
        return str(out_path)

    url = f"https://raw.githubusercontent.com/mjhydri/BeatNet-Plus/main/src/BeatNetPlus/models/{fname}"
    print(f"\n[INFO] Model weights stripped from pip package. Downloading {fname} from GitHub...")
    print(f"       Saving to: {out_path}")

    try:
        urllib.request.urlretrieve(url, out_path)
        return str(out_path)
    except Exception as e:
        if out_path.exists():
            out_path.unlink() # Cleanup partial download
        raise FileNotFoundError(
            f"Failed to download {fname} from {url}\nError: {e}\n"
            f"Please download it manually and pass: --beatnet_weights /path/to/{fname}"
        )


def get_beatnet_plus_estimator(cfg):
    import torch
    import inspect
    torch.set_num_threads(1)

    from BeatNetPlus.inference import BeatNetPlusInference
    from BeatNetPlus.particle_filtering_cascade import particle_filter_cascade

    weights_path = resolve_beatnet_plus_weights(cfg)
    key = (weights_path, str(cfg["beatnet_mode"]),
           str(cfg["beatnet_inference"]), str(cfg.get("device", "cpu")))

    if key not in _WORKER_BEATNET_PLUS:
        w_path, mode, inf_model, device = key

        # 1. Dynamically build kwargs based on what BeatNetPlusInference accepts
        sig = inspect.signature(BeatNetPlusInference.__init__)
        kwargs = {}
        if "mode" in sig.parameters:
            kwargs["mode"] = mode
        if "inference_model" in sig.parameters:
            kwargs["inference_model"] = inf_model
        if "device" in sig.parameters:
            kwargs["device"] = device
        if "plot" in sig.parameters:
            kwargs["plot"] = []
        if "thread" in sig.parameters:
            kwargs["thread"] = False

        est = BeatNetPlusInference(w_path, **kwargs)
        _WORKER_BEATNET_PLUS[key] = est

    est = _WORKER_BEATNET_PLUS[key]

    # 2. Reset internal particle states to prevent crossover between tracks
    if cfg["beatnet_inference"] == "PF":
        # Dynamically build PF kwargs as well, since it also dropped 'plot'
        pf_sig = inspect.signature(particle_filter_cascade.__init__)
        pf_kwargs = {}
        if "beats_per_bar" in pf_sig.parameters:
            pf_kwargs["beats_per_bar"] = []
        if "fps" in pf_sig.parameters:
            pf_kwargs["fps"] = 50
        if "mode" in pf_sig.parameters:
            pf_kwargs["mode"] = cfg["beatnet_mode"]
        if "plot" in pf_sig.parameters:
            pf_kwargs["plot"] = []

        est.estimator = particle_filter_cascade(**pf_kwargs)

    return est


def run_beatnet_plus_tracker(mono, sr, wav_path, cfg):
    import torch
    torch.set_num_threads(1)

    hop = int(cfg["hop"])
    num_frames = int(math.ceil(len(mono) / hop))
    rms_db = compute_frame_rms_db(mono, hop, num_frames)
    active = rms_db > float(cfg["silence_db"])

    est = get_beatnet_plus_estimator(cfg)

    preds_50fps = None
    output = None
    if hasattr(est, "activation_extractor_online") and cfg["beatnet_mode"] in ("online", "realtime"):
        preds_50fps = est.activation_extractor_online(str(wav_path))
        output = est.estimator.process(preds_50fps)
    elif hasattr(est, "activation_extractor_offline") and cfg["beatnet_mode"] == "offline":
        preds_50fps = est.activation_extractor_offline(str(wav_path))
        output = est.estimator.process(preds_50fps)
    else:
        output = est.process(str(wav_path))

    if output is None or len(output) == 0:
        beat_times = np.zeros(0, dtype=np.float64)
        downbeat_times = np.zeros(0, dtype=np.float64)
    else:
        output = np.atleast_2d(output)
        beat_times = np.unique(np.sort(output[:, 0].astype(np.float64)))
        downbeat_times = np.unique(np.sort(output[output[:, 1] == 1, 0].astype(np.float64)))

    offset = cfg["lookahead"] + 0.5
    frame_times = (np.arange(num_frames) + offset) * hop / sr

    if preds_50fps is not None and len(preds_50fps) > 0:
        p_arr = np.asarray(preds_50fps, dtype=np.float32)
        if p_arr.ndim == 2 and p_arr.shape[1] >= 2:
            act_50fps = np.clip(p_arr[:, 0] + p_arr[:, 1], 0.0, 1.0)
        else:
            act_50fps = np.clip(p_arr.squeeze(), 0.0, 1.0)
        t_50fps = np.arange(len(act_50fps), dtype=np.float64) / 50.0
        conf = np.interp(frame_times, t_50fps, act_50fps).astype(np.float32)
    else:
        conf = np.ones(num_frames, dtype=np.float32)

    pred_fold = fold_factor(beat_times, cfg["low_bpm"], cfg["high_bpm"]) if cfg.get("fold_tempo", True) else 1.0
    lfo_fold = pred_fold if cfg.get("fold_pred", False) else 1.0
    lfo, tempo, trusted = synthesize_lfo_and_tempo_from_beats(
        beat_times,
        frame_times,
        fold=lfo_fold,
        causal=cfg.get("causal_lfo", False),
    )

    flag = np.zeros(num_frames, dtype=bool)
    if beat_times.size > 0:
        flag_idx = np.rint(beat_times * sr / hop - offset).astype(int)
        flag_idx = flag_idx[(flag_idx >= 0) & (flag_idx < num_frames)]
        flag[flag_idx] = True

    lfo = np.clip(np.nan_to_num(lfo, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    return dict(
        lfo=lfo,
        tempo=tempo,
        conf=conf,
        rms_db=rms_db,
        flag=flag,
        trusted=trusted,
        active=active,
        beat_times=beat_times,
        downbeat_times=downbeat_times,
    )


def tracker_cache_path(cache_dir, track, stem, cfg):
    key = json.dumps({k: cfg.get(k) for k in TRACKER_KEYS}, sort_keys=True, default=str)
    return cache_dir / f"{track}_{stem}_{hashlib.md5(key.encode()).hexdigest()[:8]}.npz"


def prepare_track(track, stem, cfg, cache_dir, recompute=False):
    wav_path = DATA_PATH / track / f"{stem}.wav"
    y, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    mono = y.mean(axis=1).astype(np.float64)
    hop = int(cfg["hop"])
    num_frames = int(math.ceil(len(mono) / hop))
    duration = len(mono) / sr
    offset = cfg["lookahead"] + 0.5
    frame_times = (np.arange(num_frames) + offset) * hop / sr

    cache = tracker_cache_path(cache_dir, track, stem, cfg)
    tr = None
    if cache.exists() and not recompute:
        loaded = dict(np.load(cache))
        if len(loaded["lfo"]) == num_frames:
            tr = loaded
            
    if tr is None:
        tr = run_beatnet_plus_tracker(mono, sr, wav_path, cfg)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **tr)

    midi = DATA_PATH / track / "all_src.mid"
    if not midi.exists():
        raise FileNotFoundError("all_src.mid missing (run convert.py --midi_only)")
    beats = np.asarray(pretty_midi.PrettyMIDI(str(midi)).get_beats(), dtype=np.float64)
    if beats.size < 2:
        raise ValueError("fewer than 2 beats in MIDI")
    fold = fold_factor(beats, cfg["low_bpm"], cfg["high_bpm"]) if cfg.get("fold_tempo", True) else 1.0

    raw_beats = np.asarray(tr.get("beat_times", np.array([])), dtype=np.float64)
    pred_fold = fold_factor(raw_beats, cfg["low_bpm"], cfg["high_bpm"]) if raw_beats.size >= 2 else 1.0
    folded_preds = folded_beat_times(raw_beats, pred_fold, duration) if raw_beats.size >= 2 else raw_beats

    lfo_fold = pred_fold if cfg.get("fold_pred", False) else 1.0
    tr["lfo"], tr["tempo"], tr["trusted"] = synthesize_lfo_and_tempo_from_beats(
        raw_beats,
        frame_times,
        fold=lfo_fold,
        causal=cfg.get("causal_lfo", False),
    )
    
    if cfg.get("fold_pred", False):
        pk_times = folded_preds
        flag_times = raw_beats
    else:
        pk_times = raw_beats
        flag_times = folded_preds

    gt_lfo = np.cos(2.0 * np.pi * beat_positions(beats, frame_times) * fold).astype(np.float32)
    P = dict(
        mono=mono, sr=sr, hop=hop, num_frames=num_frames, duration=duration, fold=fold,
        frame_times=frame_times, beats_raw=beats,
        gt_beats=folded_beat_times(beats, fold, duration),
        gt_lfo=gt_lfo,
        gt_tempo=gt_tempo_per_frame(beats, fold, frame_times),
        pk_times=pk_times,
        flag_times=flag_times,
    )
    P.update(tr)
    return P


# === Metrics ===
def event_metrics(gt, est, t0, t1, prefix):
    g = gt[(gt >= t0) & (gt < t1)]
    e = est[(est >= t0) & (est < t1)]
    out = {prefix + k: np.nan for k in EVENT_KEYS + ["off_med_ms", "off_mad_ms", "ibi_ratio"]}
    out[prefix + "n_gt"] = int(len(g))
    out[prefix + "n_est"] = int(len(e))
    offs = np.array([])
    if len(g) < 4:
        return out, offs
    if len(e) < 2:
        for k in EVENT_KEYS:
            out[prefix + k] = 0.0
        return out, offs
    try:
        out[prefix + "F"] = float(mir_eval.beat.f_measure(g, e))
        out[prefix + "Cemgil"] = float(mir_eval.beat.cemgil(g, e)[0])
        cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(g, e)
        out[prefix + "CMLc"], out[prefix + "CMLt"] = float(cmlc), float(cmlt)
        out[prefix + "AMLc"], out[prefix + "AMLt"] = float(amlc), float(amlt)
    except Exception:
        pass
    j = np.clip(np.searchsorted(g, e), 1, len(g) - 1)
    nearest = np.where(np.abs(e - g[j - 1]) < np.abs(e - g[j]), g[j - 1], g[j])
    d = e - nearest
    d = d[np.abs(d) < 0.15]
    if d.size:
        med = float(np.median(d))
        out[prefix + "off_med_ms"] = med * 1000.0
        out[prefix + "off_mad_ms"] = float(np.median(np.abs(d - med))) * 1000.0
        offs = d * 1000.0
    out[prefix + "ibi_ratio"] = float(np.median(np.diff(e)) / max(np.median(np.diff(g)), 1e-6))
    return out, offs


def _pearson(a, b):
    if a.size < 10 or a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def feature_metrics(pred, gt, frame_ms):
    corr0 = _pearson(pred, gt)
    best_c, best_d = -2.0, 0
    for d in range(-LAG_MAX_FRAMES, LAG_MAX_FRAMES + 1):
        if d >= 0:
            a, b = pred[d:], gt[:len(gt) - d]
        else:
            a, b = pred[:d], gt[-d:]
        c = _pearson(a, b)
        if c > best_c:
            best_c, best_d = c, d
    return {"feat_corr0": corr0, "feat_corr_best": best_c, "feat_lag_ms": best_d * frame_ms}


def tempo_metrics(pred, gt):
    v = (pred > 0) & (gt > 0)
    out = {"tempo_valid_frac": float(v.mean()), "tempo_acc1": np.nan, "tempo_acc2": np.nan, "tempo_ratio_med": np.nan}
    if v.sum() == 0:
        return out
    r = pred[v] / gt[v]
    out["tempo_acc1"] = float((np.abs(r - 1) < 0.04).mean())
    ok2 = np.zeros(r.shape, dtype=bool)
    for k in (1, 2, 0.5, 3, 1 / 3):
        ok2 |= np.abs(r / k - 1) < 0.04
    out["tempo_acc2"] = float(ok2.mean())
    out["tempo_ratio_med"] = float(np.median(r))
    return out


def evaluate_range(P, t0, t1):
    ft = P["frame_times"]
    t0e = max(t0, TRIM_SEC)
    f0, f1 = int(np.searchsorted(ft, t0e, "left")), int(np.searchsorted(ft, t1, "left"))
    gt_ev = P["gt_beats"]
    if f1 - f0 < 30 or ((gt_ev >= t0e) & (gt_ev < t1)).sum() < 4:
        return None
    row = {}
    m, off_pk = event_metrics(gt_ev, P["pk_times"], t0e, t1, "pk_")
    row.update(m)
    m, off_flag = event_metrics(gt_ev, P["flag_times"], t0e, t1, "flag_")
    row.update(m)
    row.update(feature_metrics(P["lfo"][f0:f1], P["gt_lfo"][f0:f1], P["hop"] / P["sr"] * 1000.0))
    row.update(tempo_metrics(P["tempo"][f0:f1], P["gt_tempo"][f0:f1]))
    row["active_frac"] = float(P["active"][f0:f1].mean())
    row["trusted_frac"] = float(P["trusted"][f0:f1].mean())
    row["lfo_nonzero_frac"] = float((np.abs(P["lfo"][f0:f1]) > 1e-6).mean())
    return row, {"pk": off_pk, "flag": off_flag}


def saved_feature_diffs(track, stem, lfo, gt_lfo):
    out = {"saved_beat_maxdiff": np.nan, "saved_gt_maxdiff": np.nan}
    try:
        import torch
    except ImportError:
        return out
    for key, suffix, ref in (("saved_beat_maxdiff", "_beat", lfo), ("saved_gt_maxdiff", "_beat_gt", gt_lfo)):
        p = LATENT_DIR / track / f"{stem}{suffix}.pt"
        if not p.exists():
            continue
        try:
            s = torch.load(p, weights_only=True).numpy()
            s = s[0] if s.ndim == 2 else s
            if s.shape[0] == ref.shape[0]:
                out[key] = float(np.abs(s - ref).max())
        except Exception:
            pass
    return out


# === Worker: one (track, guitar stem) ===
def process_item(args):
    track, stem, cfg, cache_dir, recompute = args
    try:
        P = prepare_track(track, stem, cfg, cache_dir, recompute)
    except Exception as e:
        return {"skip": f"{track}/{stem}: {type(e).__name__}: {e}"}

    sr = P["sr"]
    windows = []
    for start in range(0, len(P["mono"]) - DURATION_SEC * sr + 1, STRIDE_SEC * sr):
        res = evaluate_range(P, start / sr, start / sr + DURATION_SEC)
        if res is None:
            continue
        row, _ = res
        row.update(track=track, stem=stem, start_sec=start / sr, level="window")
        windows.append(row)

    track_row, payload = None, None
    res = evaluate_range(P, 0.0, P["duration"])
    if res is not None:
        row, offs = res
        row.update(
            track=track, stem=stem, level="track", duration=P["duration"], gt_fold=P["fold"],
            gt_bpm_raw=60.0 / float(np.median(np.diff(P["beats_raw"]))),
            n_tempo_changes=int(len(np.unique(np.round(np.diff(P["beats_raw"]), 3)))),
        )
        row.update(saved_feature_diffs(track, stem, P["lfo"], P["gt_lfo"]))
        track_row = row
        payload = dict(
            gt_beats=P["gt_beats"], gt_lfo=P["gt_lfo"], pk_times=P["pk_times"], lfo=P["lfo"],
            duration=P["duration"], fps=sr / P["hop"], frame_ms=P["hop"] / sr * 1000.0,
            off_pk=offs["pk"], off_flag=offs["flag"],
        )
    return {"windows": windows, "track": track_row, "payload": payload}


# === Chance baseline ===
def add_chance(track_rows, payloads, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(payloads.keys())
    for r in track_rows:
        r.update({"chance_pk_F": np.nan, "chance_pk_AMLt": np.nan, "chance_feat_corr_best": np.nan})
        others = [k for k in keys if k[0] != r["track"]]
        if not others:
            continue
        p, o = payloads[(r["track"], r["stem"])], payloads[others[int(rng.integers(len(others)))]]
        dur = min(p["duration"], o["duration"])
        m, _ = event_metrics(o["gt_beats"], p["pk_times"], TRIM_SEC, dur, "pk_")
        f0 = int(math.ceil(TRIM_SEC * p["fps"]))
        L = min(len(p["lfo"]), len(o["gt_lfo"]))
        fm = feature_metrics(p["lfo"][f0:L], o["gt_lfo"][f0:L], p["frame_ms"])
        r["chance_pk_F"], r["chance_pk_AMLt"], r["chance_feat_corr_best"] = (
            m["pk_F"], m["pk_AMLt"], fm["feat_corr_best"]
        )


# === Reporting ===
def summarize(trk, win, cfg):
    cols = [
        "pk_F", "pk_AMLt", "pk_CMLt", "flag_F", "flag_AMLt", "feat_corr0", "feat_corr_best", "feat_lag_ms",
        "pk_off_med_ms", "tempo_acc1", "tempo_acc2", "active_frac", "trusted_frac",
        "chance_pk_F", "chance_pk_AMLt", "chance_feat_corr_best",
        "saved_beat_maxdiff", "saved_gt_maxdiff",
    ]
    cols = [c for c in cols if c in trk.columns]
    pd.set_option("display.width", 160)
    print(f"\n=== Track-level summary ({len(trk)} tracks/stems, {len(win)} windows) [BeatNet+] ===")

    pk_desc = "folded beats" if cfg.get("fold_pred") else "raw beats"
    flag_desc = "raw beats" if cfg.get("fold_pred") else "folded beats"
    print(f"    (pk_* = {pk_desc} | flag_* = {flag_desc})")
    print(trk[cols].agg(["mean", "median"]).T.round(3))

    r = trk["pk_ibi_ratio"].dropna()
    if len(r):
        print("\nMetrical level of the tracker (median tracker beat interval / GT beat interval):")
        print(f"  same level    : {(np.abs(r - 1) < 0.1).mean():.1%}")
        print(f"  half-time     : {(np.abs(r - 2) < 0.2).mean():.1%}   (tracker slower than GT)")
        print(f"  double-time   : {(np.abs(r - 0.5) < 0.1).mean():.1%}   (tracker faster than GT)")
        print(f"  other         : {1 - ((np.abs(r - 1) < 0.1) | (np.abs(r - 2) < 0.2) | (np.abs(r - 0.5) < 0.1)).mean():.1%}")

    if len(win):
        bins = pd.cut(win["active_frac"], [-0.01, 0.5, 0.9, 1.01], labels=["<50% active", "50-90%", ">90%"])
        print("\nBy fraction of the window the guitar is active (input level gate):")
        print(win.groupby(bins, observed=True)[["pk_F", "flag_F", "feat_corr_best", "tempo_acc2"]].agg(["mean", "count"]).round(3))


def summary_plot(trk, win, offs_pk, path):
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    b = np.linspace(0, 1, 21)
    axs[0, 0].hist(trk["pk_F"].dropna(), b, alpha=0.7, label="BeatNet+ vs GT")
    axs[0, 0].hist(trk["chance_pk_F"].dropna(), b, alpha=0.5, label="chance (other track's GT)")
    axs[0, 0].set_title("Beat F-measure (+-70 ms) [BeatNet+]"); axs[0, 0].legend()

    bc = np.linspace(-0.5, 1, 31)
    axs[0, 1].hist(trk["feat_corr_best"].dropna(), bc, alpha=0.7, label="BeatNet+ vs GT")
    axs[0, 1].hist(trk["chance_feat_corr_best"].dropna(), bc, alpha=0.5, label="chance")
    axs[0, 1].set_title("Correlation of tracker LFO with GT LFO (best lag)"); axs[0, 1].legend()

    if offs_pk.size:
        axs[0, 2].hist(offs_pk, np.linspace(-150, 150, 61))
    axs[0, 2].axvline(0, color="k", lw=0.8)
    axs[0, 2].set_title("Signed timing offset of predicted beats (ms, > 0 = late)")

    lr = np.log2(trk["pk_ibi_ratio"].dropna())
    axs[1, 0].hist(lr, np.linspace(-2, 2, 41))
    axs[1, 0].set_title("log2(tracker beat interval / GT)  (0 ok, +1 half-time, -1 double-time)")

    axs[1, 1].scatter(win["active_frac"], win["pk_F"], s=6, alpha=0.3)
    axs[1, 1].set_xlabel("fraction of window with active guitar"); axs[1, 1].set_ylabel("F-measure")
    axs[1, 1].set_title("Window F vs guitar activity")

    axs[1, 2].hist(win["feat_lag_ms"].dropna(),
                   np.arange(-LAG_MAX_FRAMES - 0.5, LAG_MAX_FRAMES + 1.5) * trk.attrs.get("frame_ms", 33.3))
    axs[1, 2].set_title("Best lag of tracker LFO vs GT LFO (ms, > 0 = late)")

    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def render_window(row, cfg, out_dir, cache_dir, tag, rank):
    P = prepare_track(row["track"], row["stem"], cfg, cache_dir, recompute=False)
    sr = P["sr"]
    t0 = float(row["start_sec"])
    t1 = t0 + DURATION_SEC
    crop = P["mono"][int(round(t0 * sr)):int(round(t1 * sr))]
    t_axis = t0 + np.arange(len(crop)) / sr
    ft = P["frame_times"]
    m = (ft >= t0) & (ft < t1)
    gt_in = P["gt_beats"][(P["gt_beats"] >= t0) & (P["gt_beats"] < t1)]
    pk_in = P["pk_times"][(P["pk_times"] >= t0) & (P["pk_times"] < t1)]

    fig, axs = plt.subplots(3, 1, figsize=(16, 10), sharex=True, height_ratios=[1.4, 1, 1])
    axs[0].plot(t_axis, crop / (np.abs(crop).max() + 1e-9), color="lightgray", label="guitar waveform")
    axs[0].vlines(gt_in, 0, 1.1, color="green", lw=2, label=f"GT beats (MIDI, folded x{P['fold']:g})")
    axs[0].vlines(pk_in, -1.1, 0, color="red", lw=2, ls="--", label="BeatNet+ beats")
    axs[0].set_ylim(-1.1, 1.1); axs[0].legend(loc="upper right")
    axs[0].set_title(
        f"{row['track']}/{row['stem']} start={t0:.0f}s | F={row['pk_F']:.2f} (flag_F={row['flag_F']:.2f}) "
        f"AMLt={row['pk_AMLt']:.2f} corr={row['feat_corr_best']:.2f} lag={row['feat_lag_ms']:.0f}ms active={row['active_frac']:.2f}"
    )

    axs[1].plot(ft[m], P["gt_lfo"][m], color="green", label="GT LFO (MIDI)")
    axs[1].plot(ft[m], P["lfo"][m], color="red", ls="--", label="BeatNet+ LFO")
    axs[1].fill_between(ft[m], -1.1, 1.1, where=~P["trusted"][m], color="gray", alpha=0.18, step="post",
                        label="untrusted / outside beat range")
    axs[1].set_ylim(-1.1, 1.1); axs[1].legend(loc="upper right"); axs[1].set_ylabel("LFO")

    axs[2].plot(ft[m], P["gt_tempo"][m], color="green", label="GT tempo (folded)")
    axs[2].plot(ft[m], np.where(P["tempo"][m] > 0, P["tempo"][m], np.nan), color="red", ls="--",
                label="BeatNet+ tempo")
    axs[2].set_ylabel("BPM"); axs[2].set_xlabel("Time (seconds)"); axs[2].legend(loc="upper right")
    axs[2].set_xlim(t0, t1)

    plt.tight_layout()
    base = f"{tag}{rank:02d}_{row['track']}_{row['stem']}_start{int(t0)}s"
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "plots" / f"{base}.png", dpi=120)
    plt.close(fig)

    c_gt = librosa.clicks(times=gt_in - t0, sr=sr, length=len(crop), click_freq=1800.0, click_duration=0.05)
    c_pr = librosa.clicks(times=pk_in - t0, sr=sr, length=len(crop), click_freq=900.0, click_duration=0.05)
    mix = crop * 0.7 + 0.5 * c_gt + 0.5 * c_pr
    mix = mix / max(1.0, float(np.abs(mix).max()))
    sf.write(out_dir / "plots" / f"{base}.wav", mix.astype(np.float32), sr)


def main():
    args = parse_args()
    cfg = load_cfg(PRE_ENCODE_CFG, args)

    out_dir = args.out_dir or (DEFAULT_OUT_ROOT / f"beatnet_plus_{cfg['beatnet_plus_variant']}_eval")
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Tracker settings: { {k: cfg.get(k) for k in TRACKER_KEYS} } "
          f"fold_tempo={cfg.get('fold_tempo')} fold_pred={cfg.get('fold_pred')} causal_lfo={cfg.get('causal_lfo')}")

    grid_scores, bad = {}, set()
    if BEAT_CHECK_REPORT.exists():
        for k, v in json.load(open(BEAT_CHECK_REPORT)).items():
            gs = v.get("grid_score")
            grid_scores[k] = gs
            if v.get("status") in ("bad_midi", "no_midi") or (gs is not None and gs < MIN_GRID_SCORE):
                bad.add(k)
        print(f"[INFO] Beat-check report: {len(bad)} tracks below threshold "
              f"({'skipped' if not args.no_skip_bad_gt else 'kept'}).")
    else:
        print(f"[WARNING] {BEAT_CHECK_REPORT} not found; no GT quality filtering.")

    tracks = sorted(t.name for t in DATA_PATH.iterdir() if t.is_dir())
    if not args.no_skip_bad_gt:
        tracks = [t for t in tracks if t not in bad]
    if args.max_tracks:
        tracks = tracks[:args.max_tracks]

    items = []
    for t in tracks:
        for gp in sorted((DATA_PATH / t).glob(args.stem_glob)):
            items.append((t, gp.stem, cfg, cache_dir, args.recompute))
    print(f"[INFO] Evaluating {len(items)} stems ({args.stem_glob}) from {len(tracks)} tracks "
          f"({args.num_workers} workers) -> {out_dir}")

    mp_ctx = mp.get_context("spawn") if "cuda" in str(cfg["device"]) else None

    win_rows, trk_rows, payloads, skipped = [], [], {}, []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp_ctx) as ex:
        futs = [ex.submit(process_item, it) for it in items]
        for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), ncols=140):
            res = fut.result()
            if "skip" in res:
                skipped.append(res["skip"])
                continue
            win_rows += res["windows"]
            if res["track"] is not None:
                trk_rows.append(res["track"])
                payloads[(res["track"]["track"], res["track"]["stem"])] = res["payload"]

    for s in skipped[:20]:
        print(f"[WARNING] skipped {s}")
    if len(skipped) > 20:
        print(f"[WARNING] ... and {len(skipped) - 20} more skipped.")
    if not trk_rows:
        print("[ERROR] No tracks evaluated.")
        return

    add_chance(trk_rows, payloads)
    trk = pd.DataFrame(trk_rows)
    win = pd.DataFrame(win_rows)
    for df in (trk, win):
        df["grid_score"] = df["track"].map(grid_scores)
    trk.to_csv(out_dir / "tracks.csv", index=False)
    win.to_csv(out_dir / "windows.csv", index=False)
    trk.attrs["frame_ms"] = next(iter(payloads.values()))["frame_ms"]

    summarize(trk, win, cfg)
    offs = np.concatenate([p["off_pk"] for p in payloads.values() if p["off_pk"].size] or [np.array([])])
    summary_plot(trk, win, offs, out_dir / "summary.png")
    print(f"\n[INFO] Wrote tracks.csv, windows.csv, summary.png to {out_dir}")

    if not args.no_render and len(win):
        w = win.dropna(subset=["pk_F"]).sort_values(["pk_F", "feat_corr_best"])
        worst = w.head(N_PLOT_WORST)
        rand = w.drop(worst.index).sample(min(N_PLOT_RANDOM, max(len(w) - len(worst), 0)), random_state=0) \
            if len(w) > len(worst) else w.iloc[0:0]
        for tag, df in (("worst", worst), ("random", rand)):
            for rank, (_, r) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"render {tag}", ncols=140)):
                render_window(r, cfg, out_dir, cache_dir, tag, rank)
        print(f"[INFO] Plots and click tracks saved to {out_dir / 'plots'}")

if __name__ == "__main__":
    main()