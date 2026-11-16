"""
compare_full.py  (Spectral Flux version, tracker direct at 30 FPS)

Compares the real-time PLP beat tracker (settings identical to what pre_encode.py must use) with ground-truth beat
information from the Slakh MIDI (tempo map of all_src.mid).

Differences to the previous version:
  1. Reverted to Spectral Flux (30 FPS direct at 48kHz, hop=1600).
  2. Tracker works directly at 30 FPS natively so lookahead is matched without model_hop / hop scaling.
  3. No causal downsampling needed anymore (1:1 feature output).
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
from scipy.ndimage import median_filter
from tqdm import tqdm
from omegaconf import OmegaConf

from realtimeplp_new import RealTimeBeatTracker, ControlSignals

warnings.filterwarnings("ignore", module="mir_eval")

# === Configuration (defaults, all overridable on the command line) ===
DEFAULT_SPLIT = "test"
DEFAULT_PROCESSED = "/data/ssd0/rakec/slakh2100/processed"
DEFAULT_OUT_DIR = "/data/hdd0/rakec/test/realtimeplp_eval_sincos"
DEFAULT_PRE_ENCODE_CFG = "configs/pre_encode_slakh2100.yaml"

MIN_GRID_SCORE = 1.3
SKIP_STATUSES = ("bad_midi", "no_midi", "bad_tempo", "tempo_glitch")

DURATION_SEC = 30
STRIDE_SEC = 15
TRIM_SEC = 5.0
LAG_MAX_FRAMES = 6
MIN_EVENT_GAP_FRAMES = 6
GT_MIN_BEAT_GAP_S = 0.15
LEVELS = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)
PLV_OK = 0.5
N_PLOT_WORST = 50
N_PLOT_BEST = 50
N_PLOT_RANDOM = 100
CACHE_VERSION = "specflux_lock_v1"

# Tracker locked directly to 30 FPS (H = 1600 at 48 kHz).
DEFAULT_BEAT_CFG = {
    "features": ["beat_sin", "beat_cos", "tempo_norm"],
    "tempo_scale": 180.0,
    "tempo_clip_bpm": 360.0,
    "low_bpm": 90,       # the tracker searches ONE octave [low_bpm, 2 * low_bpm - 1]; high_bpm is derived
    "high_bpm": 179,
    "lookahead": 7,      # in 30 FPS tracker frames
    "n_time": 10,
    "silence_db": -40.0,
    "hold_frames": 30,   # ~1s hold time at 30 FPS
    "conf_threshold": 2.0,
    "switch_ratio": 1.25,
    "energy_threshold": 0.05,
    "lock_enable": True,
    "lock_sec": 4.0,
    "lock_tol_bpm": 4.0,
    "lose_rho": 0.5,
    "lose_conf": 1.3,
    "lose_err": 0.4,
    "phase_gate": 0.15,
    "reacquire_sec": 4.0,
    "pll_tau_sec": 1.5,
    "pll_conf_hi": 3.0,
    "switch_persist_sec": 3.0,
    "harmonic_weight": 1.0,
    "hop": 1600,         # 30 FPS at 48 kHz
}
TRACKER_KEYS = ["hop", "n_time", "low_bpm", "high_bpm", "lookahead", "silence_db",
                "hold_frames", "conf_threshold", "switch_ratio", "energy_threshold",
                "lock_enable", "lock_sec", "lock_tol_bpm", "lose_rho", "lose_conf", "lose_err", "phase_gate",
                "reacquire_sec", "pll_tau_sec", "pll_conf_hi", "switch_persist_sec", "harmonic_weight"]
LOCK_KEYS = TRACKER_KEYS[TRACKER_KEYS.index("lock_enable"):]
EVENT_KEYS = ["F", "Cemgil", "CMLc", "CMLt", "AMLc", "AMLt"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    p.add_argument("--processed", type=str, default=DEFAULT_PROCESSED)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--pre_encode_cfg", type=str, default=DEFAULT_PRE_ENCODE_CFG)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_tracks", type=int, default=None)
    p.add_argument("--recompute", action="store_true", help="ignore cached tracker outputs")
    p.add_argument("--no_skip_bad_gt", action="store_true")
    p.add_argument("--no_render", action="store_true")
    p.add_argument("--lookahead", type=int, default=None, help="override the tracker lookahead (30FPS frames)")
    p.add_argument("--no_lock", action="store_true", help="disable lock mode (A/B comparison with the old behaviour)")
    p.add_argument("--low_bpm", type=int, default=None, help="lower edge of the one-octave tempo range (default 90)")
    return p.parse_args()


def make_paths(args):
    processed = pathlib.Path(args.processed)
    out = pathlib.Path(args.out_dir)
    return dict(
        data=processed / args.split,
        latent=processed / "latents" / args.split,
        report=processed / "metadata" / f"beat_check_{args.split}.json",
        out=out,
        cache=out / "cache",
    )


def load_cfg(path, lookahead=None, no_lock=False, low_bpm=None):
    cfg = dict(DEFAULT_BEAT_CFG)
    p = pathlib.Path(path)
    if p.exists():
        y = OmegaConf.to_container(OmegaConf.load(p), resolve=True)
        user = y.get("beat", {}) or {}
        if "fold_tempo" in user:
            print("[WARNING] beat.fold_tempo is not supported any more (tempos are never folded); ignored.")
        cfg.update(user)
    else:
        print(f"[WARNING] {path} not found, using the built-in defaults.")

    cfg["hop"] = 1600 # 30 FPS directly

    if low_bpm is not None:
        cfg["low_bpm"] = int(low_bpm)
    want_high = 2 * int(cfg["low_bpm"]) - 1
    if int(cfg["high_bpm"]) != want_high:
        print(f"[INFO] tempo range forced to ONE octave: [{cfg['low_bpm']}, {want_high}] "
              f"(config had high_bpm={cfg['high_bpm']}).")
    cfg["high_bpm"] = want_high
    if no_lock:
        cfg["lock_enable"] = False

    cfg["lookahead"] = int(lookahead) if lookahead is not None else int(cfg["lookahead"])
    return cfg


def midi_gt(midi_path, times):
    beats = np.asarray(pretty_midi.PrettyMIDI(str(midi_path)).get_beats(), dtype=np.float64)
    if beats.size < 2:
        raise ValueError("fewer than 2 beats in MIDI")

    idx = np.arange(beats.size, dtype=np.float64)
    pos = np.interp(times, beats, idx)

    intervals = np.maximum(np.diff(beats), 1e-6)
    first_iv = max(float(np.median(intervals[:4])), 1e-6)
    last_iv = max(float(np.median(intervals[-4:])), 1e-6)
    pos = np.where(times < beats[0], (times - beats[0]) / first_iv, pos)
    pos = np.where(times > beats[-1], (beats.size - 1) + (times - beats[-1]) / last_iv, pos)

    smooth_iv = median_filter(intervals, size=5, mode="mirror") if intervals.size >= 5 else intervals
    seg = np.clip(np.searchsorted(beats, times, side="right") - 1, 0, smooth_iv.size - 1)
    bpm = 60.0 / smooth_iv[seg]
    return beats, pos, bpm


def gt_fold_factor(beats, low_bpm):
    if beats.size < 2:
        return 1.0
    mb = 60.0 / max(float(np.median(np.diff(beats))), 1e-6)
    f = 1.0
    while mb * f < low_bpm:
        f *= 2.0
    while mb * f >= 2.0 * low_bpm:
        f *= 0.5
    return f


def folded_beat_times(beats, f, t_max):
    idx = np.arange(beats.size, dtype=np.float64)
    x = np.arange(0.0, beats.size - 1 + 1e-9, 1.0 / f)
    t = np.interp(x, idx, beats)
    return t[t <= t_max]


def clean_beats(beats, duration, min_gap=GT_MIN_BEAT_GAP_S):
    b = beats[beats <= duration]
    if b.size == 0:
        return b
    keep = [b[0]]
    for x in b[1:]:
        if x - keep[-1] >= min_gap:
            keep.append(x)
    return np.asarray(keep)


def run_tracker(mono, sr, cfg):
    hop = int(cfg["hop"])
    if sr != 48000:
        raise ValueError(f"tracker expects sr=48000, got {sr}")

    n_model = int(math.ceil(len(mono) / hop))
    x = mono
    if x.shape[0] < n_model * hop:
        x = np.pad(x, (0, n_model * hop - x.shape[0]))

    tracker = RealTimeBeatTracker.from_args(
        N=2 * hop,
        H=hop,
        samplerate=sr,
        N_time=cfg["n_time"],
        Theta=np.arange(cfg["low_bpm"], cfg["high_bpm"] + 1, 1),
        lookahead=cfg["lookahead"],
        silence_db=cfg["silence_db"],
        hold_frames=cfg["hold_frames"],
        conf_threshold=cfg["conf_threshold"],
        switch_ratio=cfg["switch_ratio"],
        energy_threshold=cfg["energy_threshold"],
        **{k: cfg[k] for k in LOCK_KEYS},
    )
    cs = tracker.cs

    sin_ = np.zeros(n_model, dtype=np.float32)
    cos_ = np.zeros(n_model, dtype=np.float32)
    tempo_ = np.zeros(n_model, dtype=np.float32)
    lfo_ = np.zeros(n_model, dtype=np.float32)
    conf_ = np.zeros(n_model, dtype=np.float32)
    rms_ = np.zeros(n_model, dtype=np.float32)
    flag_ = np.zeros(n_model, dtype=bool)
    trusted_ = np.zeros(n_model, dtype=bool)
    active_ = np.zeros(n_model, dtype=bool)
    lock_ = np.zeros(n_model, dtype=bool)
    lscore_ = np.zeros(n_model, dtype=np.float32)
    act_ = np.zeros(n_model, dtype=np.float32)

    for n in range(n_model):
        flag_[n] = tracker.process(x[n * hop:(n + 1) * hop])
        sin_[n] = cs.beat_sin
        cos_[n] = cs.beat_cos
        tempo_[n] = cs.tempo_bpm
        lfo_[n] = cs.gamma_lfo
        conf_[n] = tracker.plp.last_confidence
        trusted_[n] = tracker.plp.last_trusted
        active_[n] = tracker.active
        rms_[n] = tracker.last_rms_db
        lock_[n] = cs.lock_state
        lscore_[n] = cs.lock_score
        act_[n] = tracker.activation.activation_frame[0] if tracker.activation.activation_frame.size else 0.0

    sin_ = np.clip(np.nan_to_num(sin_, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    cos_ = np.clip(np.nan_to_num(cos_, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    lfo_ = np.clip(np.nan_to_num(lfo_, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)

    # tracker beat flags timed natively at 30 FPS
    flag_times = (np.nonzero(flag_)[0] + cfg["lookahead"] + 1) * hop / sr

    return dict(sin=sin_, cos=cos_, tempo=tempo_, lfo=lfo_, conf=conf_,
                rms_db=rms_, flag=flag_, trusted=trusted_, active=active_,
                lock=lock_, lock_score=lscore_,
                flag_times=flag_times.astype(np.float64), act30=act_,
                num_frames=n_model, vae_fps=sr/hop)


def tracker_cache_path(paths, track, stem, cfg):
    key = json.dumps({**{k: cfg[k] for k in TRACKER_KEYS}, "v": CACHE_VERSION}, sort_keys=True, default=str)
    return paths["cache"] / f"{track}_{stem}_{hashlib.md5(key.encode()).hexdigest()[:8]}.npz"


def phase_events(sin, cos, offset_sec, fps, min_gap_frames=MIN_EVENT_GAP_FRAMES):
    s, c = sin.astype(np.float64), cos.astype(np.float64)
    ok = np.hypot(s, c) > 1e-6
    ph = np.arctan2(s, c)
    n = np.where(ok[:-1] & ok[1:] & (ph[:-1] < 0) & (ph[1:] >= 0) & (c[:-1] > 0) & (c[1:] > 0))[0]
    if n.size == 0:
        return np.array([])
    frac = n + (-ph[n]) / np.maximum(ph[n + 1] - ph[n], 1e-9)
    keep = [0]
    for i in range(1, len(frac)):
        if frac[i] - frac[keep[-1]] >= min_gap_frames:
            keep.append(i)
    return (frac[keep] / fps) + offset_sec


def prepare_track(track, stem, cfg, paths, recompute=False):
    y, sr = sf.read(str(paths["data"] / track / f"{stem}.wav"), dtype="float32", always_2d=True)
    mono = y.mean(axis=1).astype(np.float64)

    cache = tracker_cache_path(paths, track, stem, cfg)
    tr = None
    if cache.exists() and not recompute:
        tr = dict(np.load(cache))
        if "act30" not in tr or "flag_times" not in tr:
            tr = None
    if tr is None:
        tr = run_tracker(mono, sr, cfg)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **tr)

    midi = paths["data"] / track / "all_src.mid"
    if not midi.exists():
        raise FileNotFoundError("all_src.mid missing")

    vae_fps = float(tr["vae_fps"])
    num_frames = int(tr["num_frames"])

    offset_sec = (cfg["lookahead"] + 0.5) * cfg["hop"] / sr
    frame_times = np.arange(num_frames) / vae_fps + offset_sec
    duration = len(mono) / sr

    beats, pos, bpm = midi_gt(midi, frame_times)
    fold = gt_fold_factor(beats, cfg["low_bpm"])
    pos, bpm = pos * fold, bpm * fold
    ang = 2.0 * np.pi * pos
    P = dict(
        mono=mono, sr=sr, duration=duration, gt_fold=fold,
        frame_times=frame_times, beats_raw=beats,
        gt_beats=clean_beats(folded_beat_times(beats, fold, duration), duration),
        gt_pos=pos, gt_bpm=bpm,
        gt_sin=np.sin(ang).astype(np.float32), gt_cos=np.cos(ang).astype(np.float32),
        pk_times=phase_events(tr["sin"], tr["cos"], offset_sec, vae_fps),
    )
    P.update(tr)
    P["vae_fps"], P["num_frames"] = vae_fps, num_frames
    return P

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

def phase_metrics(sin, cos, pos, gt_bpm):
    out = {"plv": np.nan, "plv_k1": np.nan, "level_k": np.nan, "phase_std_deg": np.nan, "phase_off_ms": np.nan,
           "locked_frac": float((np.hypot(sin, cos) > 1e-6).mean()) if len(sin) else np.nan}
    amp = np.hypot(sin.astype(np.float64), cos.astype(np.float64))
    ok = amp > 1e-6
    if ok.sum() < 30:
        return out
    z = (cos[ok].astype(np.float64) + 1j * sin[ok].astype(np.float64)) / amp[ok]
    p = pos[ok]
    best = None
    for k in LEVELS:
        m = (z * np.exp(-2j * np.pi * k * p)).mean()
        r = float(abs(m))
        if k == 1.0:
            out["plv_k1"] = r
        if best is None or r > best[0]:
            best = (r, k, m)
    r, k, m = best
    out["plv"], out["level_k"] = float(r), float(k)
    out["phase_std_deg"] = float(np.degrees(np.sqrt(max(-2.0 * np.log(max(r, 1e-9)), 0.0)))) + 0.0
    bpm = gt_bpm[ok]
    bpm_med = float(np.median(bpm[bpm > 0])) if (bpm > 0).any() else np.nan
    if np.isfinite(bpm_med):
        out["phase_off_ms"] = float(-np.angle(m) / (2.0 * np.pi) * 60000.0 / (bpm_med * k))
    return out

def _lagged(a, b, d):
    if d >= 0:
        return a[d:], b[:len(b) - d]
    return a[:d], b[-d:]

def feature_corr(sin, cos, pos, k, frame_ms):
    ang = 2.0 * np.pi * k * pos
    gs, gc = np.sin(ang), np.cos(ang)
    def corr(d):
        a1, b1 = _lagged(cos, gc, d)
        a2, b2 = _lagged(sin, gs, d)
        return 0.5 * (_pearson(a1, b1) + _pearson(a2, b2))
    best_c, best_d = -2.0, 0
    for d in range(-LAG_MAX_FRAMES, LAG_MAX_FRAMES + 1):
        c = corr(d)
        if c > best_c:
            best_c, best_d = c, d
    return {"feat_corr0": corr(0), "feat_corr_best": best_c, "feat_lag_ms": best_d * frame_ms}

def tempo_metrics(pred, gt):
    v = (pred > 0) & (gt > 0)
    out = {"tempo_valid_frac": float(v.mean()), "tempo_acc1": np.nan, "tempo_acc2": np.nan,
           "tempo_ratio_med": np.nan, "tempo_pred_med": np.nan}
    if v.sum() == 0:
        return out
    r = pred[v] / gt[v]
    out["tempo_acc1"] = float((np.abs(r - 1) < 0.04).mean())
    ok2 = np.zeros(r.shape, dtype=bool)
    for k in (1, 2, 0.5, 3, 1 / 3):
        ok2 |= np.abs(r / k - 1) < 0.04
    out["tempo_acc2"] = float(ok2.mean())
    out["tempo_ratio_med"] = float(np.median(r))
    out["tempo_pred_med"] = float(np.median(pred[v]))
    return out

def evaluate_range(P, t0, t1, cfg):
    ft = P["frame_times"]
    t0e = max(t0, TRIM_SEC)
    f0, f1 = int(np.searchsorted(ft, t0e, "left")), int(np.searchsorted(ft, t1, "left"))
    gt_ev = P["gt_beats"]
    if f1 - f0 < 30 or ((gt_ev >= t0e) & (gt_ev < t1)).sum() < 4:
        return None
    sl = slice(f0, f1)
    frame_ms = (1.0 / P["vae_fps"]) * 1000.0

    row = {}
    m, off_pk = event_metrics(gt_ev, P["pk_times"], t0e, t1, "pk_")
    row.update(m)
    m, off_flag = event_metrics(gt_ev, P["flag_times"], t0e, t1, "flag_")
    row.update(m)

    pm = phase_metrics(P["sin"][sl], P["cos"][sl], P["gt_pos"][sl], P["gt_bpm"][sl])
    row.update(pm)
    k = pm["level_k"] if np.isfinite(pm["level_k"]) else 1.0
    row.update(feature_corr(P["sin"][sl], P["cos"][sl], P["gt_pos"][sl], k, frame_ms))

    row.update(tempo_metrics(P["tempo"][sl], P["gt_bpm"][sl]))
    gt_med = float(np.median(P["gt_bpm"][sl]))
    row["tempo_gt_med"] = gt_med
    row["gt_fold_log2"] = float(np.log2(P["gt_fold"]))
    row["gt_in_range"] = bool(P["gt_fold"] == 1.0)
    row["active_frac"] = float(P["active"][sl].mean())
    row["trusted_frac"] = float(P["trusted"][sl].mean())
    return row, {"pk": off_pk, "flag": off_flag}

def lock_stats(lock, fps):
    lock = np.asarray(lock, dtype=bool)
    first = np.nonzero(lock)[0]
    t0 = int(math.ceil(TRIM_SEC * fps))
    return {"lock_on_sec": float(first[0] / fps) if first.size else np.nan,
            "lock_frac": float(lock[t0:].mean()) if lock.size > t0 else np.nan,
            "n_unlocks": int(np.sum(lock[:-1] & ~lock[1:]))}

def process_item(args):
    track, stem, cfg, paths, recompute = args
    try:
        P = prepare_track(track, stem, cfg, paths, recompute)
    except Exception as e:
        return {"skip": f"{track}/{stem}: {type(e).__name__}: {e}"}

    sr = P["sr"]
    windows = []
    for start in range(0, len(P["mono"]) - DURATION_SEC * sr + 1, STRIDE_SEC * sr):
        res = evaluate_range(P, start / sr, start / sr + DURATION_SEC, cfg)
        if res is None:
            continue
        row, _ = res
        row.update(track=track, stem=stem, start_sec=start / sr, level="window")
        windows.append(row)

    track_row, payload = None, None
    res = evaluate_range(P, 0.0, P["duration"], cfg)
    if res is not None:
        row, offs = res
        row.update(track=track, stem=stem, level="track", duration=P["duration"],
                   gt_bpm_raw=60.0 / float(np.median(np.diff(P["beats_raw"]))),
                   **lock_stats(P["lock"], P["vae_fps"]),
                   n_tempo_changes=int(len(np.unique(np.round(np.diff(P["beats_raw"]), 3)))))
        track_row = row
        payload = dict(gt_beats=P["gt_beats"], gt_pos=P["gt_pos"], gt_bpm=P["gt_bpm"], pk_times=P["pk_times"],
                       sin=P["sin"], cos=P["cos"], duration=P["duration"], fps=P["vae_fps"],
                       frame_ms=(1.0 / P["vae_fps"]) * 1000.0, off_pk=offs["pk"], off_flag=offs["flag"])
    return {"windows": windows, "track": track_row, "payload": payload}

def add_chance(track_rows, payloads, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(payloads.keys())
    for r in track_rows:
        r.update({"chance_pk_F": np.nan, "chance_pk_AMLt": np.nan, "chance_plv": np.nan,
                  "chance_plv_k1": np.nan, "chance_feat_corr_best": np.nan})
        others = [k for k in keys if k[0] != r["track"]]
        if not others:
            continue
        p, o = payloads[(r["track"], r["stem"])], payloads[others[int(rng.integers(len(others)))]]
        dur = min(p["duration"], o["duration"])
        m, _ = event_metrics(o["gt_beats"], p["pk_times"], TRIM_SEC, dur, "pk_")
        f0 = int(math.ceil(TRIM_SEC * p["fps"]))
        L = min(len(p["sin"]), len(o["gt_pos"]))
        pm = phase_metrics(p["sin"][f0:L], p["cos"][f0:L], o["gt_pos"][f0:L], o["gt_bpm"][f0:L])
        k = pm["level_k"] if np.isfinite(pm["level_k"]) else 1.0
        fm = feature_corr(p["sin"][f0:L], p["cos"][f0:L], o["gt_pos"][f0:L], k, p["frame_ms"])
        r["chance_pk_F"], r["chance_pk_AMLt"] = m["pk_F"], m["pk_AMLt"]
        r["chance_plv"], r["chance_plv_k1"] = pm["plv"], pm["plv_k1"]
        r["chance_feat_corr_best"] = fm["feat_corr_best"]

def summarize(trk, win, cfg):
    cols = ["pk_F", "pk_AMLt", "pk_CMLt", "flag_F", "plv", "plv_k1", "phase_std_deg", "phase_off_ms",
            "feat_corr0", "feat_corr_best", "feat_lag_ms", "pk_off_med_ms", "tempo_acc1", "tempo_acc2",
            "locked_frac", "active_frac", "trusted_frac", "lock_on_sec", "lock_frac", "n_unlocks",
            "chance_pk_F", "chance_pk_AMLt", "chance_plv", "chance_plv_k1", "chance_feat_corr_best"]
    cols = [c for c in cols if c in trk.columns]
    pd.set_option("display.width", 160)
    print(f"=== Track-level summary ({len(trk)} tracks/stems, {len(win)} windows) ===")
    print(trk[cols].agg(["mean", "median"]).T.round(3))

    if "pk_off_med_ms" in trk.columns and trk["pk_off_med_ms"].notna().any():
        med = float(trk["pk_off_med_ms"].median())
        hint = int(cfg["lookahead"] + round(med / ((cfg["hop"] / 48000) * 1000)))
        print(f"Median signed offset of the tracker beat events: {med:+.1f} ms (> 0 = tracker late), "
              f"lookahead = {cfg['lookahead']}.")
        if abs(med) > 15:
            print(f"  -> re-run with  --lookahead {hint} --recompute  (and use the same value in pre_encode.py)")

    r = trk["pk_ibi_ratio"].dropna()
    if len(r):
        print("Metrical level of the tracker events (median tracker beat interval / GT beat interval):")
        print(f"  same level    : {(np.abs(r - 1) < 0.1).mean():.1%}")
        print(f"  half-time     : {(np.abs(r - 2) < 0.2).mean():.1%}   (tracker slower than GT)")
        print(f"  double-time   : {(np.abs(r - 0.5) < 0.1).mean():.1%}   (tracker faster than GT)")

    lk = trk["level_k"].dropna()
    if len(lk):
        print("Best metrical level of the tracker PHASE vs the unfolded GT (tracker beat rate = k x GT beat rate):")
        for k in sorted(LEVELS):
            print(f"  k = {k:<5.3g}: {np.isclose(lk, k).mean():.1%}")
        print(f"  phase locked (PLV > {PLV_OK}) at the native level k=1 : {(trk['plv_k1'] > PLV_OK).mean():.1%}   ")
        print(f"  phase locked (PLV > {PLV_OK}) at any level            : {(trk['plv'] > PLV_OK).mean():.1%}")

    if "gt_fold_log2" in trk.columns and trk["gt_fold_log2"].notna().any():
        grp = np.where(trk["gt_fold_log2"] > 0, "folded UP (slow song, no info lost)",
                       np.where(trk["gt_fold_log2"] < 0, "folded DOWN (fast song, phase ambiguous)", "native (in range)"))
        print(f"By GT fold into the tracker's octave [{cfg['low_bpm']}, {cfg['high_bpm']}]:")
        keep = [c for c in ["pk_F", "plv", "plv_k1", "tempo_acc1", "tempo_acc2", "lock_frac"] if c in trk.columns]
        print(trk.groupby(grp)[keep].agg(["mean", "count"]).round(3))

def summary_plot(trk, win, offs_pk, path, frame_ms, cfg):
    fig, axs = plt.subplots(3, 3, figsize=(17, 13))
    b = np.linspace(0, 1, 21)
    axs[0, 0].hist(trk["pk_F"].dropna(), b, alpha=0.7, label="tracker vs GT")
    axs[0, 0].hist(trk["chance_pk_F"].dropna(), b, alpha=0.5, label="chance (other track's GT)")
    axs[0, 0].set_title("Beat F-measure (+-70 ms), phase-wrap events, native level"); axs[0, 0].legend()

    axs[0, 1].hist(trk["plv"].dropna(), b, alpha=0.7, label="best level")
    axs[0, 1].hist(trk["plv_k1"].dropna(), b, alpha=0.5, label="native level (k=1)")
    axs[0, 1].set_title("Phase-locking value of (sin, cos) vs GT phase"); axs[0, 1].legend()

    if offs_pk.size:
        axs[0, 2].hist(offs_pk, np.linspace(-150, 150, 61))
    axs[0, 2].axvline(0, color="k", lw=0.8)
    axs[0, 2].set_title("Signed timing offset of predicted beats (ms, > 0 = late)")

    lr = np.log2(trk["pk_ibi_ratio"].dropna())
    axs[1, 0].hist(lr, np.linspace(-2, 2, 41))
    axs[1, 0].set_title("log2(tracker beat interval / GT)  (0 ok, +1 half-time, -1 double-time)")

    if len(win):
        axs[1, 1].scatter(win["active_frac"], win["pk_F"], s=6, alpha=0.3)
    axs[1, 1].set_xlabel("fraction of window with active guitar"); axs[1, 1].set_ylabel("F-measure")

    axs[1, 2].hist(trk["phase_off_ms"].dropna(), np.linspace(-250, 250, 51))
    axs[1, 2].axvline(0, color="k", lw=0.8)
    axs[1, 2].set_title("Phase timing offset at the best level (ms, > 0 = late)")

    g = trk[["tempo_gt_med", "tempo_pred_med"]].dropna()
    if len(g):
        axs[2, 0].scatter(g["tempo_gt_med"], g["tempo_pred_med"], s=8, alpha=0.4)
        xs = np.array([20.0, 460.0])
        for kk, ls in ((1.0, "-"), (2.0, "--"), (0.5, "--")):
            axs[2, 0].plot(xs, xs * kk, color="k", lw=0.7, ls=ls)
        axs[2, 0].axvspan(cfg["low_bpm"], cfg["high_bpm"], color="green", alpha=0.08)
        axs[2, 0].axhspan(cfg["low_bpm"], cfg["high_bpm"], color="green", alpha=0.08)
        axs[2, 0].set_xscale("log"); axs[2, 0].set_yscale("log")
    axs[2, 0].set_title("Tempo (lines: k = 1, 2, 0.5; green: tracker range)")

    lk = trk["level_k"].dropna()
    ks = sorted(LEVELS)
    axs[2, 1].bar([f"{k:.3g}" for k in ks], [np.isclose(lk, k).mean() if len(lk) else 0 for k in ks])
    axs[2, 1].set_title("Best metrical level k (tracker rate = k x GT rate)")

    if len(win):
        axs[2, 2].hist(win["feat_lag_ms"].dropna(),
                       np.arange(-LAG_MAX_FRAMES - 0.5, LAG_MAX_FRAMES + 1.5) * frame_ms)
    axs[2, 2].set_title("Best lag of tracker sin/cos vs GT (ms, > 0 = late)")

    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)

def render_window(row, cfg, paths, tag, rank):
    P = prepare_track(row["track"], row["stem"], cfg, paths, recompute=False)
    sr = P["sr"]
    t0 = float(row["start_sec"])
    t1 = t0 + DURATION_SEC
    crop = P["mono"][int(round(t0 * sr)):int(round(t1 * sr))]
    t_axis = t0 + np.arange(len(crop)) / sr
    ft = P["frame_times"]
    m = (ft >= t0) & (ft < t1)
    gt_in = P["gt_beats"][(P["gt_beats"] >= t0) & (P["gt_beats"] < t1)]
    pk_in = P["pk_times"][(P["pk_times"] >= t0) & (P["pk_times"] < t1)]

    fig, axs = plt.subplots(4, 1, figsize=(16, 12), sharex=True, height_ratios=[1.4, 1, 1, 0.9])
    axs[0].plot(t_axis, crop / (np.abs(crop).max() + 1e-9), color="lightgray", label="guitar waveform")
    axs[0].vlines(gt_in, 0, 1.1, color="green", lw=2, label="GT beats (MIDI, folded)")
    axs[0].vlines(pk_in, -1.1, 0, color="red", lw=2, ls="--", label="tracker beats")
    axs[0].set_ylim(-1.1, 1.1); axs[0].legend(loc="upper right")
    axs[0].set_title(f"{row['track']}/{row['stem']} start={t0:.0f}s | F={row['pk_F']:.2f} PLV={row['plv']:.2f} "
                     f"(k={row['level_k']:.3g}, native {row['plv_k1']:.2f}) off={row['phase_off_ms']:.0f}ms "
                     f"active={row['active_frac']:.2f}")

    axs[1].plot(ft[m], P["gt_cos"][m], color="green", label="GT cos (MIDI)")
    axs[1].plot(ft[m], P["cos"][m], color="red", ls="--", label="tracker cos")
    axs[1].plot(ft[m], P["gt_sin"][m], color="green", lw=0.8, ls=":", alpha=0.7, label="GT sin")
    axs[1].plot(ft[m], P["sin"][m], color="red", lw=0.8, ls=":", alpha=0.7, label="tracker sin")
    axs[1].fill_between(ft[m], -1.1, 1.1, where=P["lock"][m], color="tab:blue", alpha=0.07, step="post",
                        label="locked (PLL)")
    axs[1].set_ylim(-1.1, 1.1); axs[1].legend(loc="upper right", ncol=3); axs[1].set_ylabel("sin / cos")

    axs[2].plot(ft[m], P["gt_bpm"][m], color="green", label="GT tempo (folded)")
    axs[2].plot(ft[m], np.where(P["tempo"][m] > 0, P["tempo"][m], np.nan), color="red", ls="--", label="tracker tempo")
    axs[2].set_ylabel("BPM"); axs[2].legend(loc="upper right")
    ax2b = axs[2].twinx()
    ax2b.plot(ft[m], P["lock_score"][m], color="tab:blue", lw=0.8, alpha=0.6)
    ax2b.set_ylim(0, 1.05); ax2b.set_ylabel("lock score")

    t30 = np.arange(len(P["act30"])) * cfg["hop"] / sr
    m30 = (t30 >= t0) & (t30 < t1)
    axs[3].plot(t30[m30], P["act30"][m30], color="purple", lw=0.8, label="beat activation (Spectral Flux, 30 FPS)")
    axs[3].vlines(gt_in, 0, 1.0, color="green", lw=1, alpha=0.6, label="GT beats")
    axs[3].set_ylim(0, 1.05); axs[3].set_ylabel("activation"); axs[3].set_xlabel("Time (seconds)")
    axs[3].legend(loc="upper right"); axs[3].set_xlim(t0, t1)

    plt.tight_layout()
    base = f"{tag}{rank:02d}_{row['track']}_{row['stem']}_start{int(t0)}s"
    (paths["out"] / "plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(paths["out"] / "plots" / f"{base}.png", dpi=120)
    plt.close(fig)

    c_gt = librosa.clicks(times=gt_in - t0, sr=sr, length=len(crop), click_freq=900.0, click_duration=0.05)
    c_pr = librosa.clicks(times=pk_in - t0, sr=sr, length=len(crop), click_freq=900.0, click_duration=0.05)
    mix_gt = crop + 0.5 * c_gt
    mix_pr = crop + 0.5 * c_pr
    sf.write(paths["out"] / "plots" / f"{base}_gt.wav", (mix_gt / max(1.0, float(np.abs(mix_gt).max()))).astype(np.float32), sr)
    sf.write(paths["out"] / "plots" / f"{base}_pr.wav", (mix_pr / max(1.0, float(np.abs(mix_pr).max()))).astype(np.float32), sr)

def main():
    args = parse_args()
    paths = make_paths(args)
    paths["out"].mkdir(parents=True, exist_ok=True)
    paths["cache"].mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(args.pre_encode_cfg, args.lookahead, args.no_lock, args.low_bpm)
    print(f"[INFO] Tracker settings: { {k: cfg[k] for k in TRACKER_KEYS} } features={cfg['features']}")

    grid_scores, bad = {}, set()
    if paths["report"].exists():
        for k, v in json.load(open(paths["report"])).items():
            gs = v.get("grid_score")
            grid_scores[k] = gs
            if v.get("status") in SKIP_STATUSES or (gs is not None and gs < MIN_GRID_SCORE):
                bad.add(k)

    tracks = sorted(t.name for t in paths["data"].iterdir() if t.is_dir())
    if not args.no_skip_bad_gt:
        tracks = [t for t in tracks if t not in bad]
    if args.max_tracks:
        tracks = tracks[:args.max_tracks]

    items = [(t, gp.stem, cfg, paths, args.recompute) for t in tracks for gp in sorted((paths["data"] / t).glob("guitar_*.wav"))]
    print(f"[INFO] Evaluating {len(items)} guitar stems ({args.num_workers} workers).")

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

    if not trk_rows:
        return

    add_chance(trk_rows, payloads)
    trk = pd.DataFrame(trk_rows)
    win = pd.DataFrame(win_rows)
    for df in (trk, win):
        if len(df):
            df["grid_score"] = df["track"].map(grid_scores)
    trk.to_csv(paths["out"] / "tracks.csv", index=False)
    win.to_csv(paths["out"] / "windows.csv", index=False)
    frame_ms = next(iter(payloads.values()))["frame_ms"]

    summarize(trk, win, cfg)
    offs = np.concatenate([p["off_pk"] for p in payloads.values() if p["off_pk"].size] or [np.array([])])
    summary_plot(trk, win, offs, paths["out"] / "summary.png", frame_ms, cfg)

    if not args.no_render and len(win):
        w = win.dropna(subset=["pk_F"]).sort_values(["pk_F", "feat_corr_best"])
        for tag, df in (("best", w.tail(N_PLOT_BEST).iloc[::-1]), ("worst", w.head(N_PLOT_WORST)), ("random", w.drop(w.head(N_PLOT_WORST).index).drop(w.tail(N_PLOT_BEST).iloc[::-1].index, errors="ignore").sample(min(N_PLOT_RANDOM, len(w) - N_PLOT_WORST - N_PLOT_BEST), random_state=0) if len(w) > N_PLOT_WORST + N_PLOT_BEST else w.iloc[0:0])):
            for rank, (_, r) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"render {tag}", ncols=140)):
                render_window(r, cfg, paths, tag, rank)

if __name__ == "__main__":
    main()