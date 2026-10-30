"""
compare_full.py

Compare the real-time PLP beat tracker (settings identical to pre_encode.py) against ground-truth
beats derived from the Slakh MIDI (tempo map from all_src.mid).

Outputs (OUT_DIR):
  tracks.csv / windows.csv  metrics per (track, guitar stem), and per 30 s window
  summary.png               distributions of the main metrics + chance baseline
  plots/*.png, plots/*.wav  worst and random windows, with a click track of GT (high) and tracker (low)
  cache/*.npz               cached tracker outputs

Usage:
  python compare_full.py --num_workers 8
  python compare_full.py --max_tracks 20 --recompute
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import math
import hashlib
import pathlib
import argparse
import warnings
import concurrent.futures

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import pretty_midi
import mir_eval
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm
from omegaconf import OmegaConf

from realtimeplp_new import RealTimeBeatTracker

warnings.filterwarnings("ignore", module="mir_eval")

# === Configuration ===
SPLIT = "test"
PROCESSED = pathlib.Path("/data/ssd0/rakec/slakh2100/processed")
DATA_PATH = PROCESSED / SPLIT
LATENT_DIR = PROCESSED / "latents" / SPLIT                  # only used for the sanity check vs saved *_beat.pt
OUT_DIR = pathlib.Path("/data/hdd0/rakec/test/realtimeplp_eval")
CACHE_DIR = OUT_DIR / "cache"
PRE_ENCODE_CFG = "configs/pre_encode_slakh2100.yaml"        # beat settings + compression_ratio are read from here

BEAT_CHECK_REPORT = PROCESSED / "metadata" / f"beat_check_{SPLIT}.json"   # written by convert.py
MIN_GRID_SCORE = 1.3                                        # tracks below this are skipped (unreliable MIDI tempo map)

DURATION_SEC = 30
STRIDE_SEC = 15
TRIM_SEC = 5.0              # ignore the first seconds of each track (tracker warm-up; same as mir_eval's default)
LAG_MAX_FRAMES = 6          # lag search range for the feature correlation (+-6 frames = +-200 ms at 30 fps)
N_PLOT_WORST = 50
N_PLOT_BEST = 50
N_PLOT_RANDOM = 100

# Identical defaults to pre_encode.py (DEFAULT_BEAT_CFG); overridden by the yaml's `beat:` section
DEFAULT_BEAT_CFG = {
    "low_bpm": 90,
    "high_bpm": 180,
    "lookahead": 7,
    "silence_db": -50.0,
    "hold_frames": 30,
    "conf_threshold": 2.0,
    "switch_ratio": 1.25,
    "fold_tempo": True,
}
TRACKER_KEYS = ["hop", "low_bpm", "high_bpm", "lookahead", "silence_db",
                "hold_frames", "conf_threshold", "switch_ratio"]
EVENT_KEYS = ["F", "Cemgil", "CMLc", "CMLt", "AMLc", "AMLt"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_tracks", type=int, default=None)
    p.add_argument("--recompute", action="store_true", help="ignore cached tracker outputs")
    p.add_argument("--no_skip_bad_gt", action="store_true", help="also evaluate tracks flagged by the beat check")
    p.add_argument("--no_render", action="store_true", help="skip per-window plots and click tracks")
    return p.parse_args()


def load_cfg(path):
    cfg = dict(DEFAULT_BEAT_CFG)
    hop = 1600
    p = pathlib.Path(path)
    if p.exists():
        y = OmegaConf.to_container(OmegaConf.load(p), resolve=True)
        cfg.update(y.get("beat", {}) or {})
        hop = int(y.get("compression_ratio", hop))
    else:
        print(f"[WARNING] {path} not found, using the defaults of pre_encode.py (hop={hop}).")
    cfg["hop"] = hop
    return cfg


# === Ground truth from MIDI (same math as pre_encode.py / extract_beat_features_midi) ===
def fold_factor(beats, low_bpm, high_bpm):
    median_bpm = 60.0 / max(float(np.median(np.diff(beats))), 1e-6)
    f = 1.0
    while median_bpm * f < low_bpm:
        f *= 2.0
    while median_bpm * f > high_bpm:
        f *= 0.5
    return f


def beat_positions(beats, times):
    """Fractional beat index at each time (integer = on a beat), linear extrapolation outside."""
    idx = np.arange(beats.size, dtype=np.float64)
    pos = np.interp(times, beats, idx)
    first_iv = max(beats[1] - beats[0], 1e-6)
    last_iv = max(beats[-1] - beats[-2], 1e-6)
    pos = np.where(times < beats[0], (times - beats[0]) / first_iv, pos)
    pos = np.where(times > beats[-1], (beats.size - 1) + (times - beats[-1]) / last_iv, pos)
    return pos


def folded_beat_times(beats, f, t_max):
    """Beat times after octave folding: f=2 adds the half-beats, f=0.5 keeps every second beat (starting at beat 0)."""
    idx = np.arange(beats.size, dtype=np.float64)
    x = np.arange(0.0, beats.size - 1 + 1e-9, 1.0 / f)
    t = np.interp(x, idx, beats)
    return t[t <= t_max]


def gt_tempo_per_frame(beats, f, times):
    ivs = np.diff(beats)
    mid = 0.5 * (beats[:-1] + beats[1:])
    period = np.interp(times, mid, ivs)
    return 60.0 / np.maximum(period, 1e-6) * f


# === Real-time tracker (settings identical to pre_encode.extract_beat_features) ===
def run_tracker(mono, sr, cfg):
    hop = int(cfg["hop"])
    num_frames = int(math.ceil(len(mono) / hop))
    x = mono
    if x.shape[0] < num_frames * hop:
        x = np.pad(x, (0, num_frames * hop - x.shape[0]))

    tracker = RealTimeBeatTracker.from_args(
        N=2 * hop,
        H=hop,
        samplerate=sr,
        Theta=np.arange(cfg["low_bpm"], cfg["high_bpm"] + 1, 1),
        lookahead=cfg["lookahead"],
        silence_db=cfg["silence_db"],
        hold_frames=cfg["hold_frames"],
        conf_threshold=cfg["conf_threshold"],
        switch_ratio=cfg["switch_ratio"],
    )
    cs = tracker.cs

    lfo = np.zeros(num_frames, dtype=np.float32)
    tempo = np.zeros(num_frames, dtype=np.float32)
    conf = np.zeros(num_frames, dtype=np.float32)
    rms_db = np.zeros(num_frames, dtype=np.float32)
    flag = np.zeros(num_frames, dtype=bool)
    trusted = np.zeros(num_frames, dtype=bool)
    active = np.zeros(num_frames, dtype=bool)

    for n in range(num_frames):
        flag[n] = tracker.process(x[n * hop:(n + 1) * hop])
        lfo[n] = cs.gamma_lfo
        tempo[n] = tracker.plp.current_tempo
        conf[n] = tracker.plp.last_confidence
        trusted[n] = tracker.plp.last_trusted
        active[n] = tracker.active
        rms_db[n] = tracker.last_rms_db

    lfo = np.clip(np.nan_to_num(lfo, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    return dict(lfo=lfo, tempo=tempo, conf=conf, rms_db=rms_db, flag=flag, trusted=trusted, active=active)


def tracker_cache_path(track, stem, cfg):
    key = json.dumps({k: cfg[k] for k in TRACKER_KEYS}, sort_keys=True, default=str)
    return CACHE_DIR / f"{track}_{stem}_{hashlib.md5(key.encode()).hexdigest()[:8]}.npz"


def lfo_peak_times(lfo, offset_frames, hop, sr):
    """Beat events from the LFO feature itself: peaks, refined to sub-frame precision (parabolic)."""
    pk, _ = find_peaks(lfo, height=0.5, distance=6)
    if pk.size == 0:
        return np.array([])
    y0, y1, y2 = lfo[pk - 1], lfo[pk], lfo[pk + 1]
    den = y0 - 2 * y1 + y2
    delta = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / den, 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    return (pk + delta + offset_frames) * hop / sr


def prepare_track(track, stem, cfg, recompute=False):
    """Load audio, tracker output (cached) and GT for one guitar stem."""
    y, sr = sf.read(str(DATA_PATH / track / f"{stem}.wav"), dtype="float32", always_2d=True)
    mono = y.mean(axis=1).astype(np.float64)          # same as pre_encode: float32 -> mean over channels -> float64
    hop = int(cfg["hop"])
    num_frames = int(math.ceil(len(mono) / hop))

    cache = tracker_cache_path(track, stem, cfg)
    tr = None
    if cache.exists() and not recompute:
        tr = dict(np.load(cache))
        if len(tr["lfo"]) != num_frames:
            tr = None
    if tr is None:
        tr = run_tracker(mono, sr, cfg)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **tr)

    midi = DATA_PATH / track / "all_src.mid"
    if not midi.exists():
        raise FileNotFoundError("all_src.mid missing (run convert.py --midi_only)")
    beats = np.asarray(pretty_midi.PrettyMIDI(str(midi)).get_beats(), dtype=np.float64)
    if beats.size < 2:
        raise ValueError("fewer than 2 beats in MIDI")
    fold = fold_factor(beats, cfg["low_bpm"], cfg["high_bpm"]) if cfg.get("fold_tempo", True) else 1.0

    # Time convention (same as the GT feature in pre_encode.py): the feature at frame n describes the
    # beat phase at time (n + lookahead + 0.5) * hop / sr, because the pulse is read `lookahead` frames ahead.
    offset = cfg["lookahead"] + 0.5
    frame_times = (np.arange(num_frames) + offset) * hop / sr
    duration = len(mono) / sr

    gt_lfo = np.cos(2.0 * np.pi * beat_positions(beats, frame_times) * fold).astype(np.float32)
    P = dict(
        mono=mono, sr=sr, hop=hop, num_frames=num_frames, duration=duration, fold=fold,
        frame_times=frame_times, beats_raw=beats,
        gt_beats=folded_beat_times(beats, fold, duration),
        gt_lfo=gt_lfo,
        gt_tempo=gt_tempo_per_frame(beats, fold, frame_times),
        pk_times=lfo_peak_times(tr["lfo"], offset, hop, sr),
        flag_times=(np.nonzero(tr["flag"])[0] + offset) * hop / sr,
    )
    P.update(tr)
    return P


# === Metrics ===
def event_metrics(gt, est, t0, t1, prefix):
    """Standard beat tracking metrics (mir_eval) + timing offsets on events inside [t0, t1)."""
    g = gt[(gt >= t0) & (gt < t1)]
    e = est[(est >= t0) & (est < t1)]
    out = {prefix + k: np.nan for k in EVENT_KEYS + ["off_med_ms", "off_mad_ms", "ibi_ratio"]}
    out[prefix + "n_gt"] = int(len(g))
    out[prefix + "n_est"] = int(len(e))
    offs = np.array([])
    if len(g) < 4:
        return out, offs
    if len(e) < 2:                                   # tracker produced (almost) nothing: counts as failure
        for k in EVENT_KEYS:
            out[prefix + k] = 0.0
        return out, offs
    try:
        out[prefix + "F"] = float(mir_eval.beat.f_measure(g, e))                  # +-70 ms
        out[prefix + "Cemgil"] = float(mir_eval.beat.cemgil(g, e)[0])
        cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(g, e)
        out[prefix + "CMLc"], out[prefix + "CMLt"] = float(cmlc), float(cmlt)
        out[prefix + "AMLc"], out[prefix + "AMLt"] = float(amlc), float(amlt)
    except Exception:
        pass
    j = np.clip(np.searchsorted(g, e), 1, len(g) - 1)
    nearest = np.where(np.abs(e - g[j - 1]) < np.abs(e - g[j]), g[j - 1], g[j])
    d = e - nearest
    d = d[np.abs(d) < 0.15]                          # signed offset: > 0 means the tracker is late
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
    """Compare the LFO features frame by frame, exactly as the model would see them."""
    corr0 = _pearson(pred, gt)
    best_c, best_d = -2.0, 0
    for d in range(-LAG_MAX_FRAMES, LAG_MAX_FRAMES + 1):   # d > 0: tracker is late (pred[n] ~ gt[n - d])
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
    out["tempo_acc1"] = float((np.abs(r - 1) < 0.04).mean())            # within 4% of GT tempo
    ok2 = np.zeros(r.shape, dtype=bool)
    for k in (1, 2, 0.5, 3, 1 / 3):                                       # also accept metrical-level errors
        ok2 |= np.abs(r / k - 1) < 0.04
    out["tempo_acc2"] = float(ok2.mean())
    out["tempo_ratio_med"] = float(np.median(r))
    return out


def evaluate_range(P, t0, t1):
    """Evaluate event time range [t0, t1). Returns (row, offsets) or None if GT is too sparse there."""
    ft = P["frame_times"]
    t0e = max(t0, TRIM_SEC)
    f0, f1 = int(np.searchsorted(ft, t0e, "left")), int(np.searchsorted(ft, t1, "left"))
    gt_ev = P["gt_beats"]
    if f1 - f0 < 30 or ((gt_ev >= t0e) & (gt_ev < t1)).sum() < 4:
        return None
    row = {}
    m, off_pk = event_metrics(gt_ev, P["pk_times"], t0e, t1, "pk_")        # events from LFO peaks (the feature)
    row.update(m)
    m, off_flag = event_metrics(gt_ev, P["flag_times"], t0e, t1, "flag_")  # events from detect_beat() flags
    row.update(m)
    row.update(feature_metrics(P["lfo"][f0:f1], P["gt_lfo"][f0:f1], P["hop"] / P["sr"] * 1000.0))
    row.update(tempo_metrics(P["tempo"][f0:f1], P["gt_tempo"][f0:f1]))
    row["active_frac"] = float(P["active"][f0:f1].mean())
    row["trusted_frac"] = float(P["trusted"][f0:f1].mean())
    row["lfo_nonzero_frac"] = float((np.abs(P["lfo"][f0:f1]) > 1e-6).mean())
    return row, {"pk": off_pk, "flag": off_flag}


def saved_feature_diffs(track, stem, lfo, gt_lfo):
    """Sanity check: recomputed features vs the *_beat.pt / *_beat_gt.pt written by pre_encode.py."""
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
    track, stem, cfg, recompute = args
    try:
        P = prepare_track(track, stem, cfg, recompute)
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
        row.update(track=track, stem=stem, level="track", duration=P["duration"], gt_fold=P["fold"],
                   gt_bpm_raw=60.0 / float(np.median(np.diff(P["beats_raw"]))),
                   n_tempo_changes=int(len(np.unique(np.round(np.diff(P["beats_raw"]), 3)))))
        row.update(saved_feature_diffs(track, stem, P["lfo"], P["gt_lfo"]))
        track_row = row
        payload = dict(gt_beats=P["gt_beats"], gt_lfo=P["gt_lfo"], pk_times=P["pk_times"], lfo=P["lfo"],
                       duration=P["duration"], fps=sr / P["hop"], frame_ms=P["hop"] / sr * 1000.0,
                       off_pk=offs["pk"], off_flag=offs["flag"])
    return {"windows": windows, "track": track_row, "payload": payload}


# === Chance baseline: same tracker output scored against the GT of a different, random track ===
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
        r["chance_pk_F"], r["chance_pk_AMLt"], r["chance_feat_corr_best"] = m["pk_F"], m["pk_AMLt"], fm["feat_corr_best"]


# === Reporting ===
def summarize(trk, win):
    cols = ["pk_F", "pk_AMLt", "pk_CMLt", "flag_F", "feat_corr0", "feat_corr_best", "feat_lag_ms",
            "pk_off_med_ms", "tempo_acc1", "tempo_acc2", "active_frac", "trusted_frac",
            "chance_pk_F", "chance_pk_AMLt", "chance_feat_corr_best",
            "saved_beat_maxdiff", "saved_gt_maxdiff"]
    cols = [c for c in cols if c in trk.columns]
    pd.set_option("display.width", 160)
    print(f"\n=== Track-level summary ({len(trk)} tracks/stems, {len(win)} windows) ===")
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
        print(win.groupby(bins, observed=True)[["pk_F", "feat_corr_best", "tempo_acc2"]].agg(["mean", "count"]).round(3))

    if "saved_beat_maxdiff" in trk.columns and trk["saved_beat_maxdiff"].notna().any():
        bad = (trk["saved_beat_maxdiff"] > 1e-4).mean()
        print(f"\nRecomputed LFO vs saved *_beat.pt: {bad:.1%} of stems differ by > 1e-4 "
              f"(should be ~0 if the saved features used these settings).")


def summary_plot(trk, win, offs_pk, path):
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    b = np.linspace(0, 1, 21)
    axs[0, 0].hist(trk["pk_F"].dropna(), b, alpha=0.7, label="tracker vs GT")
    axs[0, 0].hist(trk["chance_pk_F"].dropna(), b, alpha=0.5, label="chance (other track's GT)")
    axs[0, 0].set_title("Beat F-measure (+-70 ms), LFO peaks"); axs[0, 0].legend()

    bc = np.linspace(-0.5, 1, 31)
    axs[0, 1].hist(trk["feat_corr_best"].dropna(), bc, alpha=0.7, label="tracker vs GT")
    axs[0, 1].hist(trk["chance_feat_corr_best"].dropna(), bc, alpha=0.5, label="chance")
    axs[0, 1].set_title("Correlation of gamma-LFO with GT LFO (best lag)"); axs[0, 1].legend()

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

    axs[1, 2].hist(win["feat_lag_ms"].dropna(), np.arange(-LAG_MAX_FRAMES - 0.5, LAG_MAX_FRAMES + 1.5) * trk.attrs.get("frame_ms", 33.3))
    axs[1, 2].set_title("Best lag of tracker LFO vs GT LFO (ms, > 0 = late)")

    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def render_window(row, cfg, tag, rank):
    P = prepare_track(row["track"], row["stem"], cfg, recompute=False)
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
    axs[0].vlines(pk_in, -1.1, 0, color="red", lw=2, ls="--", label="tracker beats (LFO peaks)")
    axs[0].set_ylim(-1.1, 1.1); axs[0].legend(loc="upper right")
    axs[0].set_title(
        f"{row['track']}/{row['stem']} start={t0:.0f}s | F={row['pk_F']:.2f} AMLt={row['pk_AMLt']:.2f} "
        f"corr={row['feat_corr_best']:.2f} lag={row['feat_lag_ms']:.0f}ms active={row['active_frac']:.2f}"
    )

    axs[1].plot(ft[m], P["gt_lfo"][m], color="green", label="GT LFO (MIDI)")
    axs[1].plot(ft[m], P["lfo"][m], color="red", ls="--", label=r"tracker $\gamma$-LFO")
    axs[1].fill_between(ft[m], -1.1, 1.1, where=~P["trusted"][m], color="gray", alpha=0.18, step="post",
                        label="tempo not updated (gated / flywheel)")
    axs[1].set_ylim(-1.1, 1.1); axs[1].legend(loc="upper right"); axs[1].set_ylabel("LFO")

    axs[2].plot(ft[m], P["gt_tempo"][m], color="green", label="GT tempo (folded)")
    axs[2].plot(ft[m], np.where(P["tempo"][m] > 0, P["tempo"][m], np.nan), color="red", ls="--", label="tracker tempo")
    axs[2].set_ylabel("BPM"); axs[2].set_xlabel("Time (seconds)"); axs[2].legend(loc="upper right")
    axs[2].set_xlim(t0, t1)

    plt.tight_layout()
    base = f"{tag}{rank:02d}_{row['track']}_{row['stem']}_start{int(t0)}s"
    (OUT_DIR / "plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_DIR / "plots" / f"{base}.png", dpi=120)
    plt.close(fig)

    # Click track: GT = high click (1800 Hz), tracker = low click (900 Hz)
    c_gt = librosa.clicks(times=gt_in - t0, sr=sr, length=len(crop), click_freq=900.0, click_duration=0.05)
    c_pr = librosa.clicks(times=pk_in - t0, sr=sr, length=len(crop), click_freq=900.0, click_duration=0.05)
    mix_gt = crop + 0.5 * c_gt
    mix_pr = crop + 0.5 * c_pr
    mix_gt = mix_gt / max(1.0, float(np.abs(mix_gt).max()))
    mix_pr = mix_pr / max(1.0, float(np.abs(mix_pr).max()))
    sf.write(OUT_DIR / "plots" / f"{base}_gt.wav", mix_gt.astype(np.float32), sr)
    sf.write(OUT_DIR / "plots" / f"{base}_pr.wav", mix_pr.astype(np.float32), sr)

def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(PRE_ENCODE_CFG)
    print(f"[INFO] Tracker settings: { {k: cfg[k] for k in TRACKER_KEYS} } fold_tempo={cfg.get('fold_tempo')}")

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
        for gp in sorted((DATA_PATH / t).glob("guitar_*.wav")):
            items.append((t, gp.stem, cfg, args.recompute))
    print(f"[INFO] Evaluating {len(items)} guitar stems from {len(tracks)} tracks ({args.num_workers} workers).")

    win_rows, trk_rows, payloads, skipped = [], [], {}, []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as ex:
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
    trk.to_csv(OUT_DIR / "tracks.csv", index=False)
    win.to_csv(OUT_DIR / "windows.csv", index=False)
    trk.attrs["frame_ms"] = next(iter(payloads.values()))["frame_ms"]

    summarize(trk, win)
    offs = np.concatenate([p["off_pk"] for p in payloads.values() if p["off_pk"].size] or [np.array([])])
    summary_plot(trk, win, offs, OUT_DIR / "summary.png")
    print(f"\n[INFO] Wrote tracks.csv, windows.csv, summary.png to {OUT_DIR}")

    if not args.no_render and len(win):
        w = win.dropna(subset=["pk_F"]).sort_values(["pk_F", "feat_corr_best"])
        worst = w.head(N_PLOT_WORST)
        best = w.tail(N_PLOT_BEST).iloc[::-1]
        pool = w.drop(worst.index).drop(best.index, errors="ignore")
        rand = pool.sample(min(N_PLOT_RANDOM, len(pool)), random_state=0) if len(pool) > 0 else w.iloc[0:0]
        for tag, df in (("best", best), ("worst", worst), ("random", rand)):
            for rank, (_, r) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"render {tag}", ncols=140)):
                render_window(r, cfg, tag, rank)
        print(f"[INFO] Plots and click tracks saved to {OUT_DIR / 'plots'}")

if __name__ == "__main__":
    main()